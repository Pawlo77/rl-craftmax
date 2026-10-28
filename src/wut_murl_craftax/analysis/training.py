"""Training analysis of every method: hard scenarios, obstacles and case episodes.

``python -m wut_murl_craftax.analysis.training --phases final-1m combo-1m --method all
--out outputs/analysis --tracking-uri sqlite:///<db>?timeout=300`` writes, per budget
and method, ``<out>/<budget>/<method>/`` (``hard/``, ``plots/``, ``obstacles.md``,
``cases.csv``, ``cases/<id>.json`` and the `cases.export_case` files; units apart, R77)
and logs it as run ``training-analysis-<budget>-<method>`` of experiment ``analysis``.
Missing detector inputs give the "needs ..." status; a method with ``.complete`` is
skipped unless ``--force``. Store reads are short (R103); importing needs no JAX.

@see [[2026-10-20-craftax-training-analysis]]
"""

import argparse
import json
import logging
import re
import shutil
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..visualisation import charts
from . import hard, obstacles
from .cases import (
    CASE_COLUMNS,
    OUTPUTS,
    describe_case,
    export_case,
    find_cases,
    run_dirs,
    write_cases,
)
from .load import METRICS_ROOT, Frames, load_phases, select, units
from .obstacles import EVAL_MODE, INVENTORY, SKILL_ENDS, eval_actions

logger = logging.getLogger(__name__)

ANALYSIS_DIR = "analysis"
"""Name of the output folder: the case files are relative to its parent (contracts
section 4), so ``--out`` must end in it."""
COMPLETE, LOGGED, FAILED = ".complete", ".logged", "failed.json"
"""Markers of a finished method folder and of its finished MLflow logging, and the
failed cases of an unfinished one (id -> error)."""
LOG_SKIP = ("masters",)
"""Folder entries kept out of the MLflow artifacts (disk, R104)."""
UNIT_GB = 0.25
"""Disk per (method, config_ref) unit: about 20 cases of 9 to 17 MB each."""
DEFAULT_PHASES = ("final-1m", "combo-1m")
"""The phases that exist at 1M; the 1B ones are accepted but not assumed (R75, R86)."""
REPLAY_FRAMES = {"skill_ends": SKILL_ENDS, "inventory": INVENTORY}
"""Replay file stem (contracts section 10) -> name of the optional frame."""


@dataclass(slots=True)
class TrainingSummary:
    """What a run of `analyse_training` did, by ``(budget, method)``."""

    done: list[tuple[str, str]] = field(default_factory=list)
    """Methods whose files were computed."""
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """Methods whose complete folder was kept (a missing logging still ran)."""
    failed: list[str] = field(default_factory=list)
    """One ``<case id>: <error>`` per case that could not be replayed."""


def _root_of(out: Path) -> Path:
    """The folder the case files are relative to.

    Raises:
        ValueError: If ``out`` is not named `ANALYSIS_DIR`.
    """
    if out.name != ANALYSIS_DIR:
        msg = (
            f"--out must end in {ANALYSIS_DIR!r}: case paths are relative to its parent"
        )
        raise ValueError(msg)
    return out.parent


# ------------------------------------------------------------------ hard scenarios


def _write_hard(sel: Frames, folder: Path) -> None:
    """The `hard` tables of every unit of a method as CSV, and their plots."""
    eps, curves = sel["eval_episodes"], sel["curves"]
    tables = {
        "floor_reach": hard.floor_reach(eps),
        "achievement_status": hard.achievement_status(
            eps, curves if len(curves) else None
        ),
        "death_causes": hard.death_causes(eps),
        "stalls": hard.stall_table(eps),
        "stall_stats": hard.stall_stats(eps),
    }
    tables["stall_starts"] = hard.stall_starts(tables["stalls"])
    tables["stall_ccdf"] = hard.stall_ccdf(tables["stalls"])
    (folder / "hard").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / "hard" / f"{name}.csv", index=False)
    plots = folder / "plots"
    for old in plots.glob("*.png"):  # one panel a file now: the old stems are stale
        old.unlink()
    with charts.capture_captions() as captions:
        hard.plot_floor_reach(tables["floor_reach"], plots)
        hard.plot_achievement_status(tables["achievement_status"], plots)
        hard.plot_death_causes(tables["death_causes"], plots)
        hard.plot_stall_lengths(tables["stall_ccdf"], plots)
    charts.write_captions(plots, "HARD", captions)


# ---------------------------------------------------------------- detector inputs


def _final_updates(episodes: pd.DataFrame) -> dict[str, int]:
    """Each run's last checkpoint with stochastic eval episodes (`EVAL_MODE`)."""
    e = episodes[episodes["mode"] == EVAL_MODE]
    return {str(k): int(v) for k, v in e.groupby("run_id")["update"].max().items()}


def _read_replay(run_dir: Path, stem: str, final: int) -> pd.DataFrame | None:
    """The stochastic ``<stem>_<update>_stochastic`` files of one run, or None.

    None when the run's final checkpoint has no file: the detectors read the last
    checkpoint of the frame, which would then be an earlier one.
    """
    files = {
        int(m[1]): p
        for p in run_dir.glob(f"{stem}_*_{EVAL_MODE}.parquet")
        if (m := re.fullmatch(rf"{stem}_(\d+)_{EVAL_MODE}\.parquet", p.name))
    }
    if final not in files:
        return None
    columns = {
        "skill_ends": (*obstacles.SKILL_ENDS_COLUMNS, "pair_count"),
        "inventory": obstacles.INVENTORY_COLUMNS,
    }[stem]
    parts = []
    for update in sorted(files):
        have = set(pq.ParquetFile(files[update]).schema_arrow.names)
        parts.append(
            pd.read_parquet(files[update], columns=[c for c in columns if c in have])
        )
    return pd.concat(parts, ignore_index=True)


def replay_frames(
    runs: pd.DataFrame, finals: Mapping[str, int], root: Path
) -> dict[str, pd.DataFrame]:
    """The replay frames ``skill_ends`` and ``eval_inventory`` of ``runs`` (contract 9).

    Reads ``<root>/<run_id>/{skill_ends,inventory}_<update>_stochastic.parquet`` of
    every run whose final checkpoint (``finals``) has one; a frame no run has is left
    out, so its detectors read "needs replay (evaluation Task 6)".
    """
    found: dict[str, list[pd.DataFrame]] = {}
    for run_id in runs["run_id"]:
        for stem, frame in REPLAY_FRAMES.items():
            final = finals.get(str(run_id))
            part = (
                None if final is None else _read_replay(root / str(run_id), stem, final)
            )
            if part is not None:
                found.setdefault(frame, []).append(part)
    return {k: pd.concat(v, ignore_index=True) for k, v in found.items()}


def _action_frame(
    run_ids: Sequence[str], finals: Mapping[str, int], dirs: Mapping[str, Path]
) -> pd.DataFrame | None:
    """`obstacles.eval_actions` of each run's final stochastic file, or None."""
    parts = []
    for rid in run_ids:
        if rid not in finals:
            continue
        if (dirs[rid] / "eval" / EVAL_MODE / f"{finals[rid]:05d}.npz").is_file():
            parts.append(eval_actions(dirs[rid], rid, finals[rid]))
    return pd.concat(parts, ignore_index=True) if parts else None


def _unit_inputs(
    unit_runs: pd.DataFrame,
    finals: Mapping[str, int],
    dirs: Mapping[str, Path],
    metrics_root: Path,
) -> dict[str, pd.DataFrame]:
    """The optional `detect_all` frames of one unit's runs."""
    ids = [str(r) for r in unit_runs["run_id"]]
    inputs = replay_frames(unit_runs, finals, metrics_root)
    actions = _action_frame(ids, finals, dirs)
    return inputs | ({} if actions is None else {obstacles.ACTIONS: actions})


# --------------------------------------------------------------------- obstacles


def _refs(frames: Frames) -> list[str]:
    """The `config_ref` values of a method's runs, tuned first."""
    refs = frames["runs"]["config_ref"].unique()
    return sorted(map(str, refs), key=lambda r: (r != "tuned", r))


def _write_obstacles(
    frames: Frames,
    sel: Frames,
    method: str,
    budget: str,
    folder: Path,
    inputs: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    """``obstacles.md`` (a table per unit) and ``obstacles.csv`` (a row per run)."""
    lines = [
        f"# Obstacles of {method} ({budget.upper()})",
        "",
        (
            "Per-run verdicts: `obstacles.csv`. Mitigation, outcome and plot columns "
            "are filled in the analysis note."
        ),
    ]
    rows = []
    for ref in _refs(sel):
        title = ref if ref == "tuned" else f"{ref} (ablation)"
        if method in obstacles.DETECTORS:
            found = obstacles.detect_all(
                {**frames, **inputs.get(ref, {})}, method, budget, config_ref=ref
            )
        else:
            found = []
        table = obstacles.obstacles_table(obstacles.table_rows(found, budget))
        lines += ["", f"## {title}", "", table.rstrip()]
        rows += [
            {
                "config_ref": ref,
                "run_id": o.run,
                "obstacle": o.name,
                "status": o.status,
                "start": o.start,
                "end": o.end,
                "evidence": o.evidence,
            }
            for o in found
        ]
    (folder / "obstacles.md").write_text("\n".join(lines) + "\n")
    columns = ["config_ref", "run_id", "obstacle", "status", "start", "end", "evidence"]
    pd.DataFrame(rows, columns=columns).to_csv(folder / "obstacles.csv", index=False)


# ------------------------------------------------------------------------- cases


def _open_run(run_dir: Path) -> Any:
    """The replayable run of a run directory (imports JAX)."""
    from ..visualisation.eval_rollouts import EvalRun

    return EvalRun.load(run_dir)


def _json_default(value: Any) -> Any:
    """Numpy scalars and arrays as plain JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"case facts hold a {type(value).__name__}"
    raise TypeError(msg)


def _flat_rollout(root: Path, case_id: str) -> Path:
    """Where `export_case` copies a case's rollout for the video glob."""
    return root / "visualisation" / "runs" / f"case-{case_id}.npz"


def _replay_cases(
    cases: pd.DataFrame, folder: Path, root: Path, dirs: Mapping[str, Path]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Export and describe each case: the table of those that worked, the failures.

    One `_open_run` per run serves all its cases (its compiled replay is shared). A
    failed case's flat rollout, copied before a later step failed, is removed.
    """
    (folder / "cases").mkdir(parents=True, exist_ok=True)
    opened: dict[str, Any] = {}
    done, failed = [], {}
    for case in cases.to_dict("records"):
        cid, rid = str(case["id"]), str(case["run_id"])
        t0 = time.time()
        try:
            if rid not in opened:
                opened[rid] = _open_run(dirs[rid])
            row = export_case(case, folder, opened[rid], root=root)
            facts = describe_case(case, opened[rid])
        except Exception as e:  # one bad case must not stop the others
            logger.exception("case %s failed", cid)
            failed[cid] = f"{type(e).__name__}: {e}"
            _flat_rollout(root, cid).unlink(missing_ok=True)
            continue
        text = json.dumps(facts, indent=1, default=_json_default)
        (folder / "cases" / f"{cid}.json").write_text(text + "\n")
        done.append(row)
        logger.info("case %s (%s): %.0fs", cid, case["kind"], time.time() - t0)
    return pd.DataFrame(done, columns=list(CASE_COLUMNS)), failed


# ----------------------------------------------------------------------- one method


def _reset(folder: Path, root: Path) -> None:
    """Empty a method folder and drop its flat rollouts.

    Those named in its ``cases.csv`` and the ids of its ``failed.json``; only
    ``case-*.npz`` files directly under ``<root>/visualisation/runs``, so another
    method's rollouts and anything else stay.
    """
    table = folder / "cases.csv"
    flat = (root / "visualisation" / "runs").resolve()
    if table.is_file():
        names = pd.read_csv(table, keep_default_na=False, usecols=["rollout"])
        for rel in names["rollout"]:
            path = (root / str(rel)).resolve() if rel else None
            if path and path.parent == flat and path.name.startswith("case-"):
                path.unlink(missing_ok=True)
    if (folder / FAILED).is_file():
        for cid in json.loads((folder / FAILED).read_text()):
            _flat_rollout(root, cid).unlink(missing_ok=True)
    shutil.rmtree(folder, ignore_errors=True)


def _drop_rows(root: Path, budget: str, method: str) -> None:
    """Remove a method's rows from the budget's ``cases.csv`` (it exported no case)."""
    table = root / ANALYSIS_DIR / budget / "cases.csv"
    if table.is_file():
        rows = pd.read_csv(table, keep_default_na=False)
        rows[rows["method"] != method].to_csv(table, index=False)


def _analyse_method(
    frames: Frames,
    method: str,
    budget: str,
    *,
    folder: Path,
    root: Path,
    tracking_uri: str,
    metrics_root: Path,
) -> list[str]:
    """Write one method's folder; returns its case failures.

    ``.complete`` is written only when every case replayed.
    """
    sel = select(frames, method=method, budget=budget, pool=True)
    finals = _final_updates(sel["eval_episodes"])
    ids = [str(r) for r in sel["runs"]["run_id"]]
    t0 = time.time()
    dirs = run_dirs(ids, tracking_uri)  # one short read of the store per method
    folder.mkdir(parents=True, exist_ok=True)
    _write_hard(sel, folder)
    inputs = {
        ref: _unit_inputs(u["runs"], finals, dirs, metrics_root)
        for (_, ref), u in units(sel)
    }
    _write_obstacles(frames, sel, method, budget, folder, inputs)
    logger.info(
        "%s %s: tables, plots and obstacles in %.0fs", budget, method, time.time() - t0
    )
    found = [find_cases(frames, method, budget, ref) for ref in _refs(sel)]
    cases = pd.concat(
        [*found, pd.DataFrame(columns=list(CASE_COLUMNS))], ignore_index=True
    )
    table, failures = _replay_cases(cases, folder, root, dirs)
    if len(table):
        # all units of the method in one call: write_cases merges by method only
        write_cases(table, root)
    else:  # its old rows would name files that are gone
        folder.joinpath("cases.csv").write_text(",".join(CASE_COLUMNS) + "\n")
        _drop_rows(root, budget, method)
        logger.warning("%s %s: no case was exported", budget, method)
    failed = [f"{cid}: {error}" for cid, error in failures.items()]
    if failures:
        (folder / FAILED).write_text(json.dumps(failures, indent=1) + "\n")
    else:
        facts = {"units": _refs(sel), "cases": len(table), "runs": len(ids)}
        (folder / COMPLETE).write_text(json.dumps(facts) + "\n")
    return failed


def analyse_training(
    frames: Frames,
    *,
    methods: Sequence[str] | None = None,
    out: Path,
    tracking_uri: str,
    metrics_root: Path = METRICS_ROOT,
    phases: Sequence[str] = (),
    log: bool = True,
    force: bool = False,
    log_skip: Collection[str] = LOG_SKIP,
) -> TrainingSummary:
    """Analyse each (budget, method) of ``frames``: files first, then MLflow.

    ``methods`` None means every method of the frames; the base-control runs of the
    others stay in ``frames`` for the detectors. ``out`` ends in ``analysis``; its
    parent is the root of the case files. A complete folder is kept (``force`` empties
    and redoes it); a folder without ``.logged`` is only logged again, without the
    entries named in ``log_skip`` (default `LOG_SKIP`: ``masters`` holds a video per
    case and `vis` Task 2 renders them again). ``phases`` are the phases the frames
    were loaded from, for the run tag.

    Raises:
        ValueError: If ``out`` is not named ``analysis``, or ``methods`` names one the
            frames lack.
    """
    root = _root_of(out)
    runs = frames["runs"]
    if unknown := sorted(set(methods or ()) - set(runs["method"])):
        msg = f"no runs of {', '.join(unknown)} in the loaded phases"
        raise ValueError(msg)
    summary = TrainingSummary()
    for budget in sorted(runs["budget"].unique()):
        names = sorted(runs.loc[runs["budget"] == budget, "method"].unique())
        for method in [m for m in names if methods is None or m in methods]:
            folder, job = out / str(budget) / method, (str(budget), method)
            if force or not (folder / COMPLETE).is_file():
                _reset(folder, root)
                failed = _analyse_method(
                    frames, method, job[0], folder=folder, root=root,
                    tracking_uri=tracking_uri, metrics_root=metrics_root,
                )  # fmt: skip
                summary.failed += failed
                if failed:
                    continue
                summary.done.append(job)
            else:
                summary.skipped.append(job)
            if log and not (folder / LOGGED).is_file():
                from ..orchestration.tracking import log_analysis_run

                tags = {
                    "analysis/kind": "training",
                    "analysis/phases": ",".join(phases),
                    "analysis/budget": str(budget),
                    "analysis/method": method,
                }
                name = f"training-analysis-{budget}-{method}"
                paths = [c for c in sorted(folder.iterdir()) if c.name not in log_skip]
                run_id = log_analysis_run(folder, paths, name, tags, tracking_uri)
                (folder / LOGGED).write_text(run_id + "\n")
                logger.info("logged %s to run %s", folder, run_id)
    return summary


# ------------------------------------------------------------------------------ CLI


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        help="export phases to read",
    )
    p.add_argument(
        "--method", nargs="+", default=["all"], help="registry names, or all"
    )
    p.add_argument("--out", type=Path, default=OUTPUTS / ANALYSIS_DIR)
    p.add_argument("--metrics-root", type=Path, default=METRICS_ROOT)
    p.add_argument(
        "--tracking-uri",
        required=True,
        help="sqlite:/// store (run folders, logging); ?timeout=<s>: busy timeout",
    )
    p.add_argument(
        "--skip-missing", action="store_true", help="skip phases never exported"
    )
    p.add_argument("--force", action="store_true", help="redo complete methods")
    p.add_argument("--no-log", action="store_true", help="write the files only")
    p.add_argument(
        "--log-masters",
        action="store_true",
        help="also log masters/ (a video per case; off by default for disk space)",
    )
    a = p.parse_args(argv)
    if not a.tracking_uri.partition("?")[0].startswith("sqlite:///"):
        p.error(f"--tracking-uri must be a sqlite:/// store, got {a.tracking_uri}")
    try:
        _root_of(a.out)
    except ValueError as e:
        p.error(str(e))
    return a


def _check_disk(
    out: Path,
    runs: pd.DataFrame,
    methods: Sequence[str] | None,
    *,
    log: bool,
    masters: bool,
) -> None:
    """Refuse a run whose estimated output and MLflow copy exceed the free disk.

    Raises:
        SystemExit: If the free space of ``out``'s disk is below `UNIT_GB` per unit,
            times 2 with ``masters`` logged, 1.6 with other logging, 1 without.
    """
    sel = runs if methods is None else runs[runs["method"].isin(methods)]
    n = len(sel[["budget", "method", "config_ref"]].drop_duplicates())
    need = n * UNIT_GB * (1 + log * (1.0 if masters else 0.6))
    probe = next(p for p in (out, *out.parents) if p.exists())
    free = shutil.disk_usage(probe).free / 1e9
    if free < need:
        msg = f"{n} units need about {need:.1f} GB, {free:.1f} GB free on {probe}"
        raise SystemExit(msg)
    logger.info("%d units: about %.1f GB needed, %.1f GB free", n, need, free)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the training analysis of the given phases and methods.

    A case that cannot be replayed fails only its method (``failed.json`` in its
    folder); the other methods finish and the exit code is 1. Any other error (the
    store, a detector, a plot, no disk space) stops the command: rerun it, finished
    methods are skipped.

    Raises:
        SystemExit: If a method is unknown, the disk is too full, or some case could
            not be replayed.
    """
    a = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frames = load_phases(a.phases, a.metrics_root, skip_missing=a.skip_missing)
    methods = None if a.method == ["all"] else a.method
    _check_disk(a.out, frames["runs"], methods, log=not a.no_log, masters=a.log_masters)
    loaded = [p for p in a.phases if p in set(frames["runs"]["phase"])]
    try:
        summary = analyse_training(
            frames, methods=methods, out=a.out, tracking_uri=a.tracking_uri,
            metrics_root=a.metrics_root, phases=loaded, log=not a.no_log,
            force=a.force, log_skip=() if a.log_masters else LOG_SKIP,
        )  # fmt: skip
    except ValueError as e:
        raise SystemExit(str(e)) from e
    logger.info(
        "%d methods done, %d skipped, %d cases failed",
        len(summary.done), len(summary.skipped), len(summary.failed),
    )  # fmt: skip
    if summary.failed:
        msg = f"{len(summary.failed)} cases failed: {summary.failed}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
