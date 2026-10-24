"""Export of one phase to the tidy tables ``runs``, ``curves`` and ``eval_episodes``.

`export_phase` reads the store read-only by ``run_uuid`` plus the runs' artifacts and
writes ``<out_dir>/<table>.parquet`` with a CSV copy each (contracts.md section 1).
Membership is the ``phases`` tag (R77); ``config_ref`` comes from `config_refs` in a
final or combo phase, else it is the hash. Eval files take their replay table (EV6)
when it exists, else their stored records with the replay-only columns null.
`log_export` logs the folder to run ``export-<phase>`` of experiment ``analysis``.
CLI: ``python -m wut_murl_craftax.evaluation.export --phase P --tracking-uri URI``.
@see [[2026-10-20-craftax-evaluation-and-metrics]]
@tested-by [[tests/evaluation/test_export.py]]
"""

import argparse
import contextlib
import json
import logging
import shutil
import sqlite3
import time
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from ..analysis.load import CURVES_COLUMNS, RUNS_COLUMNS
from ..orchestration.queue import PHASES_DIR, expand_phase
from ..orchestration.tracking import (
    _use_experiment,
    artifact_path,
    resolve_tracking_uri,
)
from ..training.checkpoints import load_index
from .episode_records import stored_records
from .hardware import SYSTEM_HI, SYSTEM_LO, hardware_summary
from .milestones import Game, game_of
from .rollout_metrics import (
    IDENTITY,
    MODES,
    OUT_ROOT,
    RUN_IDENTITY,
    arrow_table,
    episode_columns,
    out_paths,
)

logger = logging.getLogger(__name__)

ANALYSIS_EXPERIMENT = "analysis"
"""Experiment of the analysis runs (contracts.md section 8)."""
DEFAULT_URI = "sqlite:///mlruns.db"
STORE_TIMEOUT_S = 300.0
"""SQLite busy timeout of the store reads; ``?timeout=`` in the URI overrides it."""
TABLES = ("runs", "curves", "eval_episodes")
"""The export tables, by file stem."""
REPLAY_FRAMES = {"skill_ends": "skill_ends", "inventory": "eval_inventory"}
"""Replay file kind (`out_paths` key) -> phase table stem (contracts.md section 9)."""
REF_PHASES = ("final-", "combo-")
"""Phases whose ``config_ref`` comes from the phase file; elsewhere it is the hash."""


def _overrides(registry: str, **kw: Any) -> tuple[str, frozenset[tuple[str, Any]]]:
    return registry, frozenset(kw.items())


ABLATIONS = {
    _overrides("count-bonus-ppo-rnn", key="simhash", log2_k=24, max_occupancy=0.1): (
        "simhash"
    ),
    _overrides("sil", layout="single"): "single-buffer",
    _overrides("smdp-skills", eps_top_end=1.0): "random-top",
    _overrides("smdp-skills", monte_carlo=True): "monte-carlo",
    _overrides("smdp-skills", low_flags=False): "no-flags",
    _overrides("dyna-q", plan_kappa=0.0): "kappa0",
}
"""(registry, ``set`` of a phase-file item) -> its ``config_ref`` (contracts.md)."""
_NULL_TYPES = {"unique_positions": pa.int64()}
"""Arrow type of a replay-only column a stored record lacks; float64 otherwise."""
_EVAL_KEYS = {
    f"eval_{mode}_{what}": f"eval/{mode}/{key}"
    for mode in MODES
    for what, key in (("score", "score_pct"), ("return", "return"))
}
_CURVES_SQL = (
    "SELECT key, step, timestamp, value, is_nan FROM metrics "
    "WHERE run_uuid = ? AND (key < ? OR key >= ?)"
)
_CURVES_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("env_steps", pa.int64()),
    ("metric", pa.string()),
    ("value", pa.float64()),
])


class PhaseRun(NamedTuple):
    """A run of the store with the tags and facts the export needs."""

    run_id: str
    name: str
    status: str
    experiment: str
    dir: Path
    tags: dict[str, str]


def config_refs(phase_file: Path) -> dict[tuple[str, str], str]:
    """(registry, ``config_hash``) -> ``config_ref`` of a final or combo phase file.

    The first ``configs`` item of an entry is ``tuned``; every other item is named by
    its ``set`` in `ABLATIONS`.

    Raises:
        ValueError: If an item's ``set`` has no name, or an entry expands to another
            number of configs than it lists.
    """
    raw = yaml.safe_load(phase_file.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}
    for spec in expand_phase(phase_file):  # entries in file order, items in order
        hashes = found.setdefault(spec.registry_name, [])
        if spec.config_hash not in hashes:
            hashes.append(spec.config_hash)
    out = {}
    for registry, hashes in found.items():
        items = (raw["runs"][registry] or {}).get("configs", ["tuned"])
        if len(items) != len(hashes):
            msg = (
                f"{phase_file.name}: {registry} lists {len(items)} configs "
                f"but expands to {len(hashes)}"
            )
            raise ValueError(msg)
        out[registry, hashes[0]] = "tuned"
        for item, config_hash in zip(items[1:], hashes[1:], strict=True):
            sets = item.get("set", {}) if isinstance(item, dict) else {}
            if (name := ABLATIONS.get(_overrides(registry, **sets))) is None:
                msg = f"{phase_file.name}: {registry} config {item!r} has no name"
                raise ValueError(msg)
            out[registry, config_hash] = name
    return out


def _split_uri(tracking_uri: str) -> tuple[str, float]:
    """``sqlite:///`` URI without its query, and its ``timeout`` (s, default 300).

    Raises:
        ValueError: For another kind of store.
    """
    base, _, query = tracking_uri.partition("?")
    if not base.startswith("sqlite:///"):
        msg = f"expected a sqlite:/// tracking URI, got {tracking_uri}"
        raise ValueError(msg)
    timeout = parse_qs(query).get("timeout", [str(STORE_TIMEOUT_S)])[0]
    return base, float(timeout)


@contextlib.contextmanager
def _connect(tracking_uri: str) -> Generator[sqlite3.Connection]:
    """A short-lived read-only connection to the store of a ``sqlite:///`` URI.

    The store runs in rollback-journal mode, so a read blocks the writers' commits
    while it lasts (R103): callers open one per run and fetch every row at once.
    """
    base, timeout = _split_uri(tracking_uri)
    path = Path(base.removeprefix("sqlite:///")).expanduser().resolve()
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=timeout)
    try:
        yield con
    finally:
        con.close()


def phase_runs(con: sqlite3.Connection, phase: str) -> list[PhaseRun]:
    """Active runs whose ``phases`` tag lists ``phase``, by name (R77)."""
    tags: dict[str, dict[str, str]] = {}
    for rid, key, value in con.execute(
        "SELECT run_uuid, key, value FROM tags"
    ).fetchall():
        tags.setdefault(rid, {})[key] = value
    rows = con.execute(
        "SELECT r.run_uuid, r.name, r.status, e.name, r.artifact_uri FROM runs r "
        "JOIN experiments e ON e.experiment_id = r.experiment_id "
        "WHERE r.lifecycle_stage = 'active' ORDER BY r.name"
    ).fetchall()
    return [
        PhaseRun(rid, name, status, exp, artifact_path(uri), tags.get(rid, {}))
        for rid, name, status, exp, uri in rows
        if phase in tags.get(rid, {}).get("phases", "").split(",")
    ]


def run_curves(con: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Every stored series of a run but ``system/*``, long.

    A re-logged step (a resume) keeps its latest value; a NaN point stays NaN.
    """
    rows = con.execute(_CURVES_SQL, (run_id, SYSTEM_LO, SYSTEM_HI)).fetchall()
    df = pd.DataFrame(rows, columns=["metric", "env_steps", "ts", "value", "is_nan"])
    df["value"] = df["value"].astype(float).mask(df["is_nan"].astype(bool))
    df = df.sort_values(["metric", "env_steps", "ts"], kind="stable")
    df = df.drop_duplicates(["metric", "env_steps"], keep="last")
    df.insert(0, "run_id", run_id)
    return df[CURVES_COLUMNS].astype({"env_steps": "int64"}).reset_index(drop=True)


def _config_ref(run: PhaseRun, phase: str, refs: Mapping[tuple[str, str], str]) -> str:
    """``config_ref`` of a run in ``phase``; the hash outside final and combo phases."""
    method, config_hash = run.tags.get("method", ""), run.tags.get("config_hash", "")
    if not phase.startswith(REF_PHASES):
        return config_hash
    if (ref := refs.get((method, config_hash))) is None:
        msg = (
            f"run {run.name}: config_hash {config_hash} of {method} "
            f"is not produced by {phase}.yaml"
        )
        raise ValueError(msg)
    return ref


def run_row(
    con: sqlite3.Connection,
    run: PhaseRun,
    phase: str,
    ref: str,
    curves: pd.DataFrame,
    config_json: str,
) -> dict[str, Any]:
    """The ``runs`` row of one run (contracts.md section 1).

    Raises:
        ValueError: If the run has no ``train/env_steps`` series.
    """
    last = curves.sort_values("env_steps", kind="stable").drop_duplicates(
        "metric", keep="last"
    )
    at_end = dict(zip(last["metric"], last["value"], strict=True))
    if "train/env_steps" not in at_end:
        msg = f"run {run.name} has no train/env_steps series"
        raise ValueError(msg)
    t = run.tags
    hw = hardware_summary(con, run.run_id)
    row = {
        "run_id": run.run_id,
        "run_name": run.name,
        "experiment": run.experiment,
        "phase": phase,
        "phases": t.get("phases", ""),
        "method": t.get("method", ""),
        "budget": t.get("budget", ""),
        "config_hash": t.get("config_hash", ""),
        "seed": int(t.get("seed", -1)),
        "config_ref": ref,
        # tag `best` marks a tuning winner; it means nothing in a final phase
        "best": t.get("best") == "true" and not phase.startswith(REF_PHASES),
        "status": run.status,
        "config_json": config_json,
        "online_families": "train/action_entropy" in at_end,
        "final_score": at_end.get("final_score", np.nan),
        **{col: at_end.get(key, np.nan) for col, key in _EVAL_KEYS.items()},
        "env_steps": int(at_end["train/env_steps"]),
    }
    return row | {k: v for k, v in hw.items() if k not in row}


def _stored_table(
    path: Path, head: Mapping[str, Any], game: Game, order: Sequence[str]
) -> pa.Table:
    """Eval-episode rows of an eval file without a replay; replay-only columns null."""
    arrays = np.load(path)
    rows = [
        {**head, "eval_seed": int(seed)} | rec.row()
        for seed, rec in zip(arrays["seeds"], stored_records(arrays, game), strict=True)
    ]
    df = pd.DataFrame(rows)
    t = arrow_table(df[[c for c in order if c in df.columns]])
    n = len(df)
    return pa.table({
        c: t.column(c)
        if c in t.column_names
        else pa.nulls(n, _NULL_TYPES.get(c, pa.float64()))
        for c in order
    })


def _check_replay(t: pa.Table, head: Mapping[str, Any], where: Path) -> None:
    """Raise unless a replay table's identity columns match the run's tags."""
    for col in ("run_id", "method", "budget", "seed", "update", "mode"):
        if (got := t.column(col).unique().to_pylist()) != [head[col]]:
            msg = f"{where}: {col} {got} disagrees with the run's {head[col]!r}"
            raise ValueError(msg)


class RunTables(NamedTuple):
    """The eval-episode and replay tables of one run, one entry per eval file."""

    episodes: list[pa.Table]
    replay: dict[str, list[pa.Table]]
    """`REPLAY_FRAMES` key -> the replay files found."""


def run_episodes(run: PhaseRun, ref: str, game: Game, replay_root: Path) -> RunTables:
    """Every eval file of a run as eval-episode rows with ``config_ref`` inserted."""
    order = [c for c in episode_columns(game) if c in RUN_IDENTITY or c not in IDENTITY]
    steps = {e["update"]: e["env_steps"] for e in load_index(run.dir)}
    t = run.tags
    out = RunTables([], {k: [] for k in REPLAY_FRAMES})
    for mode in MODES:
        for path in sorted((run.dir / "eval" / mode).glob("*.npz")):
            update = int(path.stem)
            head = {
                "run_id": run.run_id, "method": t.get("method", ""),
                "budget": t.get("budget", ""), "seed": int(t.get("seed", -1)),
                "update": update, "env_steps": steps[update], "mode": mode,
            }  # fmt: skip
            replay = out_paths(replay_root, run.run_id, update, mode)
            if replay["episodes"].is_file():
                table = pq.read_table(replay["episodes"]).select(order)
                _check_replay(table, head, replay["episodes"])
                table = table.cast(_stored_schema(table.schema, order))
            else:
                table = _stored_table(path, head, game, order)
            refs = pa.array([ref] * len(table), pa.string())
            out.episodes.append(
                table.replace_schema_metadata(None).add_column(2, "config_ref", refs)
            )
            for kind in REPLAY_FRAMES:
                if replay[kind].is_file():
                    frame = pq.read_table(replay[kind]).replace_schema_metadata(None)
                    out.replay[kind].append(frame)
    return out


def _stored_schema(schema: pa.Schema, order: Sequence[str]) -> pa.Schema:
    """``schema`` with the replay-only columns typed as `_stored_table` types them."""
    fields = [schema.field(c) for c in order]
    return pa.schema([
        f.with_type(_NULL_TYPES[f.name]) if f.name in _NULL_TYPES else f for f in fields
    ])


def _json_lists(table: pa.Table) -> pa.Table:
    """``table`` with every list column as one JSON string per row (CSV copies)."""
    for i, field in enumerate(table.schema):
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            text = [json.dumps(v) for v in table.column(i).to_pylist()]
            table = table.set_column(i, field.name, pa.array(text, pa.string()))
    return table


def _replace(tmp: Path) -> None:
    """Move a finished ``<name>.tmp`` file over ``<name>``."""
    tmp.replace(tmp.with_suffix(""))


def write_table(table: pa.Table, out_dir: Path, stem: str) -> None:
    """``<stem>.parquet`` and its CSV copy, each written to a temp file then moved."""
    parquet, csv = out_dir / f"{stem}.parquet.tmp", out_dir / f"{stem}.csv.tmp"
    pq.write_table(table, parquet)
    pacsv.write_csv(_json_lists(table), csv)
    _replace(parquet)
    _replace(csv)


def _game(config_json: str) -> Game:
    return game_of(str(json.loads(config_json).get("env", "")))


def export_phase(
    phase: str,
    out_dir: Path,
    *,
    tracking_uri: str = DEFAULT_URI,
    replay_root: Path = OUT_ROOT,
    phases_dir: Path = PHASES_DIR,
    runs: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write the export tables of ``phase`` to ``out_dir``; return them by stem.

    ``runs`` (ids or names) restricts the export to those runs of the phase, for a
    partial look; a phase export lists them all. Curves stream to disk run by run.

    Raises:
        ValueError: If no run lists the phase, a final or combo run's hash is not one
            of the phase file's configs, or a replay table disagrees with its run.
    """
    refs = (
        config_refs(phases_dir / f"{phase}.yaml")
        if phase.startswith(REF_PHASES)
        else {}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, episodes = [], []
    replay: dict[str, list[pa.Table]] = {k: [] for k in REPLAY_FRAMES}
    curves_tmp = out_dir / "curves.parquet.tmp", out_dir / "curves.csv.tmp"
    t0 = time.time()
    with _connect(tracking_uri) as con:
        todo = [
            r
            for r in phase_runs(con, phase)
            if runs is None or {r.run_id, r.name} & set(runs)
        ]
    if not todo:
        msg = f"no run of the store lists phase {phase!r} in its `phases` tag"
        raise ValueError(msg)
    with (
        pq.ParquetWriter(curves_tmp[0], _CURVES_SCHEMA) as parquet,
        pacsv.CSVWriter(curves_tmp[1], _CURVES_SCHEMA) as csv,
    ):
        for k, run in enumerate(todo, 1):
            ref = _config_ref(run, phase, refs)
            config_json = (run.dir / "config.json").read_text(encoding="utf-8")
            with _connect(tracking_uri) as con:  # one short connection per run (R103)
                curves = run_curves(con, run.run_id)
                row = run_row(con, run, phase, ref, curves, config_json)
            rows.append(row)
            batch = pa.Table.from_pandas(curves, _CURVES_SCHEMA, preserve_index=False)
            parquet.write_table(batch)
            csv.write_table(batch)
            tables = run_episodes(run, ref, _game(config_json), replay_root)
            episodes += tables.episodes
            for kind, frames in tables.replay.items():
                replay[kind] += frames
            if k % 25 == 0 or k == len(todo):
                logger.info(
                    "%s: %d/%d runs (%.0fs)", phase, k, len(todo), time.time() - t0
                )
    for tmp in curves_tmp:
        _replace(tmp)
    runs_df = pd.DataFrame(rows)
    extra = [c for c in runs_df.columns if c not in RUNS_COLUMNS]
    write_table(
        pa.Table.from_pandas(runs_df[RUNS_COLUMNS + extra], preserve_index=False),
        out_dir,
        "runs",
    )
    write_table(pa.concat_tables(episodes), out_dir, "eval_episodes")
    written = {t: out_dir / f"{t}.parquet" for t in TABLES}
    for kind, stem in REPLAY_FRAMES.items():
        for old in (out_dir / f"{stem}.parquet", out_dir / f"{stem}.csv"):
            old.unlink(missing_ok=True)  # a rerun never keeps a stale replay frame
        if replay[kind]:
            frame = pa.concat_tables(replay[kind], promote_options="default")
            write_table(frame, out_dir, stem)
            written[stem] = out_dir / f"{stem}.parquet"
    logger.info(
        "%s: exported %d runs to %s (%.0fs)",
        phase,
        len(rows),
        out_dir,
        time.time() - t0,
    )
    return written


def log_export(folder: Path, phase: str, tracking_uri: str, *, csv: bool = True) -> str:
    """Log ``folder`` as the artifacts of run ``export-<phase>``; return its id.

    The run lives in experiment `ANALYSIS_EXPERIMENT` and carries ``analysis/*`` tags
    only (loaders select training runs by ``phases``, ``method`` ...). A rerun reuses
    the run of that name and replaces its artifacts. ``csv`` False logs the parquet
    files only: the CSV copies of a tuning phase's curves take gigabytes.
    """
    base, _, query = tracking_uri.partition("?")
    resolved = resolve_tracking_uri(base)
    uri = resolved + (f"?{query}" if query else "")  # SQLAlchemy takes ?timeout=
    mlflow.set_tracking_uri(uri)
    _use_experiment(mlflow, resolved, ANALYSIS_EXPERIMENT)
    client = mlflow.MlflowClient(uri)
    exp = client.get_experiment_by_name(ANALYSIS_EXPERIMENT)
    if exp is None:  # _use_experiment just made it
        msg = f"experiment {ANALYSIS_EXPERIMENT} is missing"
        raise RuntimeError(msg)
    name = f"export-{phase}"
    found = client.search_runs(
        [exp.experiment_id], filter_string=f"attributes.run_name = '{name}'"
    )
    if found:
        run_id, old = found[0].info.run_id, found[0].info.artifact_uri
        if old:  # never rmtree a path made from an empty URI
            shutil.rmtree(artifact_path(old), ignore_errors=True)
    else:
        run_id = client.create_run(exp.experiment_id, run_name=name).info.run_id
    budgets = sorted(
        set(pd.read_parquet(folder / "runs.parquet", columns=["budget"])["budget"])
    )
    for key, value in (
        ("analysis/kind", "export"),
        ("analysis/phases", phase),
        ("analysis/budget", ",".join(budgets)),
    ):
        client.set_tag(run_id, key, value)
    if csv:
        client.log_artifacts(run_id, str(folder))
    else:
        for path in sorted(folder.glob("*.parquet")):
            client.log_artifact(run_id, str(path))
    client.set_terminated(run_id)
    return str(run_id)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--phase", required=True, help="value of the runs' phases tag")
    p.add_argument("--out", type=Path, help="default outputs/metrics/<phase>")
    p.add_argument(
        "--tracking-uri",
        required=True,
        help="sqlite:/// store, read-only; ?timeout=<s> sets the busy timeout",
    )
    p.add_argument(
        "--replay-root", type=Path, default=OUT_ROOT, help="EV6 per-run tables"
    )
    p.add_argument("--no-log", action="store_true", help="write the tables only")
    p.add_argument(
        "--log-parquet-only",
        action="store_true",
        help="log the parquet files but not their CSV copies (disk space)",
    )
    p.add_argument(
        "--run", action="append", help="restrict to a run id or name (with --no-log)"
    )
    a = p.parse_args(argv)
    if a.run and not a.no_log:
        p.error("--run exports part of a phase: pass --no-log")
    return a


def main(argv: Sequence[str] | None = None) -> None:
    """Export a phase, then log it to the analysis experiment unless ``--no-log``."""
    a = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = a.out or OUT_ROOT / a.phase
    export_phase(
        a.phase, out, tracking_uri=a.tracking_uri, replay_root=a.replay_root, runs=a.run
    )
    if not a.no_log:
        run_id = log_export(out, a.phase, a.tracking_uri, csv=not a.log_parquet_only)
        logger.info("logged %s to run %s", out, run_id)


if __name__ == "__main__":
    main()
