"""Export of one phase to the tidy tables ``runs``, ``curves`` and ``eval_episodes``.

`export_phase` reads the store read-only by ``run_uuid`` plus the runs' artifacts and
writes ``<out_dir>/<table>.parquet`` with a CSV copy each, the curves one gzipped
(contracts.md section 1, R104); the folder is staged and swapped in as a whole.
Membership is the ``phases`` tag (R77); ``config_ref`` comes from `config_refs` in a
final or combo phase, else it is the hash. Eval files take their replay table (EV6)
when it exists, else their stored records with the replay-only columns null.
`log_export` logs the parquet files to run ``export-<phase>`` in ``analysis`` (CLI).
@see [[2026-10-20-craftax-evaluation-and-metrics]]
@tested-by [[tests/evaluation/test_export.py]]
"""

import argparse
import json
import logging
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from ..analysis.load import CURVES_COLUMNS, RUNS_COLUMNS
from ..orchestration.queue import PHASES_DIR, expand_phase
from ..orchestration.tracking import ANALYSIS_EXPERIMENT as ANALYSIS_EXPERIMENT
from ..orchestration.tracking import log_analysis_run
from ..training.checkpoints import load_index
from .episode_records import stored_records
from .hardware import SYSTEM_HI, SYSTEM_LO, hardware_summary, open_store
from .milestones import Game, game_of
from .rollout_metrics import (
    IDENTITY,
    MODES,
    OUT_ROOT,
    RUN_IDENTITY,
    RunRef,
    arrow_table,
    episode_columns,
    out_paths,
    phase_runs,
)

logger = logging.getLogger(__name__)

DEFAULT_URI = "sqlite:///mlruns.db"
TABLES = ("runs", "curves", "eval_episodes")
"""The export tables, by file stem."""
REPLAY_FRAMES = {"skill_ends": "skill_ends", "inventory": "eval_inventory"}
"""Replay file kind (`out_paths` key) -> phase table stem (contracts.md section 9)."""
REF_PHASES = ("final-", "combo-")
"""Phases whose ``config_ref`` comes from the phase file; elsewhere it is the hash."""
TUNED_ITEMS = ("tuned", "combined", "base")
"""Names the first ``configs`` item of an entry may have; it plays the tuned role:
``combined`` is the combo phases' one config and ``base`` the 1B controls' config."""
RUN_TAGS = ("phases", "method", "budget", "seed", "config_hash")
"""Runner tags every exported run must carry."""
STAGING, PREVIOUS = ".staging", ".previous"
"""Suffixes of the folder an export is built in and of the export it replaces."""


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
REPLAY_ONLY = ("new_tiles_per_100", "revisit_rate", "unique_positions")
"""With the ``coverage/*`` family: the columns only a replay fills (`coverage_summary`);
a stored record leaves them null."""
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


def config_refs(phase_file: Path) -> dict[tuple[str, str], str]:
    """(registry, ``config_hash``) -> ``config_ref`` of a final or combo phase file.

    The first ``configs`` item of an entry, a name in `TUNED_ITEMS`, is ``tuned``;
    every other item is named by its ``set`` in `ABLATIONS`.

    Raises:
        ValueError: If the first item is not a `TUNED_ITEMS` name, an item's ``set``
            has no name, or an entry expands to another number of configs than it
            lists.
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
        if items[0] not in TUNED_ITEMS:  # an ablation must never become `tuned`
            msg = f"{phase_file.name}: {registry} starts with {items[0]!r}, not tuned"
            raise ValueError(msg)
        out[registry, hashes[0]] = "tuned"
        for item, config_hash in zip(items[1:], hashes[1:], strict=True):
            sets = item.get("set", {}) if isinstance(item, dict) else {}
            if (name := ABLATIONS.get(_overrides(registry, **sets))) is None:
                msg = f"{phase_file.name}: {registry} config {item!r} has no name"
                raise ValueError(msg)
            out[registry, config_hash] = name
    return out


def run_curves(con: Any, run_id: str) -> pd.DataFrame:
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


def _config_ref(run: RunRef, phase: str, refs: Mapping[tuple[str, str], str]) -> str:
    """``config_ref`` of a run in ``phase``; the hash outside final and combo phases.

    Raises:
        ValueError: If a `RUN_TAGS` tag is missing, or a final or combo run's hash is
            not one of the phase file's configs.
    """
    if missing := [k for k in RUN_TAGS if not run.tags.get(k)]:
        msg = f"run {run.name} lacks the tags {', '.join(missing)}"
        raise ValueError(msg)
    method, config_hash = run.method, run.tags["config_hash"]
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
    con: Any,
    run: RunRef,
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
        "phases": t["phases"],
        "method": run.method,
        "budget": run.budget,
        "config_hash": t["config_hash"],
        "seed": run.seed,
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
    gaps = [c for c in order if c not in df.columns]
    if odd := [
        c for c in gaps if not c.startswith("coverage/") and c not in REPLAY_ONLY
    ]:
        msg = f"{path}: stored records lack the columns {odd}"
        raise ValueError(msg)
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
    stored: int = 0
    """Eval files without a replay table (their rows come from the stored events)."""


def run_episodes(run: RunRef, ref: str, game: Game, replay_root: Path) -> RunTables:
    """Every eval file of a run as eval-episode rows with ``config_ref`` inserted."""
    order = [c for c in episode_columns(game) if c in RUN_IDENTITY or c not in IDENTITY]
    steps = {e["update"]: e["env_steps"] for e in load_index(run.dir)}
    out = RunTables([], {k: [] for k in REPLAY_FRAMES})
    for mode in MODES:
        for path in sorted((run.dir / "eval" / mode).glob("*.npz")):
            update = int(path.stem)
            head = {
                "run_id": run.run_id, "method": run.method,
                "budget": run.budget, "seed": run.seed,
                "update": update, "env_steps": steps[update], "mode": mode,
            }  # fmt: skip
            replay = out_paths(replay_root, run.run_id, update, mode)
            if replay["episodes"].is_file():
                table = pq.read_table(replay["episodes"]).select(order)
                _check_replay(table, head, replay["episodes"])
                table = table.cast(_stored_schema(table.schema, order))
            else:
                table = _stored_table(path, head, game, order)
                out = out._replace(stored=out.stored + 1)
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


def write_table(table: pa.Table, out_dir: Path, stem: str) -> None:
    """``<stem>.parquet`` and its CSV copy (list columns as JSON) in ``out_dir``."""
    pq.write_table(table, out_dir / f"{stem}.parquet")
    pacsv.write_csv(_json_lists(table), out_dir / f"{stem}.csv")


def _game(config_json: str) -> Game:
    return game_of(str(json.loads(config_json).get("env", "")))


def _select(phase: str, tracking_uri: str, runs: Sequence[str] | None) -> list[RunRef]:
    """The runs of ``phase``, or those of them named in ``runs``.

    Raises:
        ValueError: If no run lists the phase, or a name in ``runs`` matches none.
    """
    todo = phase_runs(tracking_uri, [phase])
    if not todo:
        msg = f"no run of the store lists phase {phase!r} in its `phases` tag"
        raise ValueError(msg)
    if runs is None:
        return todo
    if unknown := set(runs) - {k for r in todo for k in (r.run_id, r.name)}:
        msg = f"no run of phase {phase} is {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    return [r for r in todo if {r.run_id, r.name} & set(runs)]


def _swap(stage: Path, out_dir: Path) -> None:
    """Move ``stage`` to ``out_dir``; the old export stays until the move succeeded."""
    previous = out_dir.with_name(out_dir.name + PREVIOUS)
    shutil.rmtree(previous, ignore_errors=True)
    if out_dir.exists():
        out_dir.rename(previous)
    try:
        stage.rename(out_dir)
    except BaseException:
        if previous.exists():
            previous.rename(out_dir)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _build(
    phase: str,
    todo: Sequence[tuple[RunRef, str]],
    stage: Path,
    tracking_uri: str,
    replay_root: Path,
) -> dict[str, pa.Table]:
    """Stream the curves into ``stage``; return the other tables of the phase."""
    rows, episodes, stored, files = [], [], 0, 0
    replay: dict[str, list[pa.Table]] = {k: [] for k in REPLAY_FRAMES}
    t0 = time.time()
    with (
        pq.ParquetWriter(stage / "curves.parquet", _CURVES_SCHEMA) as parquet,
        pa.CompressedOutputStream(stage / "curves.csv.gz", "gzip") as gz,
        pacsv.CSVWriter(gz, _CURVES_SCHEMA) as csv,
    ):
        for k, (run, ref) in enumerate(todo, 1):
            config_json = (run.dir / "config.json").read_text(encoding="utf-8")
            with open_store(tracking_uri) as con:  # one short connection per run, R103
                curves = run_curves(con, run.run_id)
                row = run_row(con, run, phase, ref, curves, config_json)
            rows.append(row)
            batch = pa.Table.from_pandas(curves, _CURVES_SCHEMA, preserve_index=False)
            parquet.write_table(batch)
            csv.write_table(batch)
            tables = run_episodes(run, ref, _game(config_json), replay_root)
            episodes += tables.episodes
            stored, files = stored + tables.stored, files + len(tables.episodes)
            for kind, frames in tables.replay.items():
                replay[kind] += frames
            if k % 25 == 0 or k == len(todo):
                logger.info(
                    "%s: %d/%d runs (%.0fs)", phase, k, len(todo), time.time() - t0
                )
    logger.warning(
        "%s: %d of %d eval files have no replay table in %s (stored events, "
        "replay-only columns null)",
        phase, stored, files, replay_root,
    )  # fmt: skip
    runs_df = pd.DataFrame(rows)
    extra = [c for c in runs_df.columns if c not in RUNS_COLUMNS]
    out = {
        "runs": pa.Table.from_pandas(
            runs_df[RUNS_COLUMNS + extra], preserve_index=False
        ),
        "eval_episodes": pa.concat_tables(episodes),
    }
    for kind, stem in REPLAY_FRAMES.items():
        if replay[kind]:
            out[stem] = pa.concat_tables(replay[kind], promote_options="default")
    return out


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

    Every run's tags and ``config_ref`` are checked before any work. The tables are
    built in ``<out_dir>.staging`` (curves streamed run by run), which then replaces
    ``out_dir`` as a whole: a failure leaves the previous export as it was and
    removes the staging folder. ``runs`` (ids or names) restricts the export to those
    runs of the phase, for a partial look in another folder.

    Raises:
        ValueError: If no run lists the phase, a name of ``runs`` matches none, a
            run lacks a `RUN_TAGS` tag, a final or combo run's hash is not one of the
            phase file's configs, or a replay table disagrees with its run.
    """
    refs = (
        config_refs(phases_dir / f"{phase}.yaml")
        if phase.startswith(REF_PHASES)
        else {}
    )
    todo = [
        (r, _config_ref(r, phase, refs)) for r in _select(phase, tracking_uri, runs)
    ]
    t0 = time.time()
    previous = out_dir.with_name(out_dir.name + PREVIOUS)
    if previous.exists() and not out_dir.exists():  # killed between the two moves
        previous.rename(out_dir)
    stage = out_dir.with_name(out_dir.name + STAGING)
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        tables = _build(phase, todo, stage, tracking_uri, replay_root)
        for stem, table in tables.items():
            write_table(table, stage, stem)
        _swap(stage, out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    logger.info(
        "%s: exported %d runs to %s (%.0fs)",
        phase,
        len(todo),
        out_dir,
        time.time() - t0,
    )
    return {stem: out_dir / f"{stem}.parquet" for stem in ("curves", *tables)}


def log_export(
    folder: Path, phase: str, tracking_uri: str, *, csv: bool = False
) -> str:
    """Log the export in ``folder`` as the artifacts of run ``export-<phase>``.

    The run lives in experiment `ANALYSIS_EXPERIMENT` and carries ``analysis/*`` tags
    only (loaders select training runs by ``phases``, ``method`` ...). Only the
    parquet files are logged unless ``csv`` (R104: the CSV copies take gigabytes). A
    rerun reuses the run of that name: it is RUNNING while the new files are logged,
    and the artifacts the new export lacks are deleted only after that succeeded.

    Returns:
        The run id.

    Raises:
        OSError: If an old artifact cannot be deleted.
    """
    budgets = sorted(
        set(pd.read_parquet(folder / "runs.parquet", columns=["budget"])["budget"])
    )
    tags = {
        "analysis/kind": "export",
        "analysis/phases": phase,
        "analysis/budget": ",".join(budgets),
    }
    globs = ("*.parquet", "*.csv", "*.csv.gz") if csv else ("*.parquet",)
    files = sorted(p for g in globs for p in folder.glob(g))
    return log_analysis_run(folder, files, f"export-{phase}", tags, tracking_uri)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--phase", required=True, help="value of the runs' phases tag")
    p.add_argument(
        "--out",
        type=Path,
        help="default outputs/metrics/<phase>, relative to the working directory: "
        "run from the main checkout",
    )
    p.add_argument(
        "--tracking-uri",
        required=True,
        help="sqlite:/// store, read-only; ?timeout=<s> sets the busy timeout",
    )
    p.add_argument(
        "--replay-root",
        type=Path,
        default=OUT_ROOT,
        help="EV6 per-run tables (default outputs/metrics, relative to the working "
        "directory)",
    )
    p.add_argument("--no-log", action="store_true", help="write the tables only")
    p.add_argument(
        "--log-csv",
        action="store_true",
        help="log the CSV copies too (default: parquet only, R104)",
    )
    p.add_argument(
        "--run",
        action="append",
        help="restrict to a run id or name (needs --no-log and --out)",
    )
    a = p.parse_args(argv)
    if a.run and not (a.no_log and a.out):
        p.error("--run exports part of a phase: pass --no-log and --out")
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
        run_id = log_export(out, a.phase, a.tracking_uri, csv=a.log_csv)
        logger.info("logged %s to run %s", out, run_id)


if __name__ == "__main__":
    main()
