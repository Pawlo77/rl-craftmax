"""Storage retention: delete run artifacts no later step reads (dry run by default).

FINISHED runs with a complete evaluation lose ``state/`` (class ``state``); tuning-only
runs every checkpoint (``ckpt_tuning``); final runs (``final-*``, ``combo-*``) every
checkpoint but the last, the case updates and those of the `seed_picks` seeds
(``ckpt_final``); a final run missing from the export tables keeps them all.
``ckpt/pruned.json`` lists checkpoints before they go (`mark_pruned`); ``index.json``
and eval files stay. The store is read-only. CLI: ``python -m
wut_murl_craftax.orchestration.prune [--tracking-uri URI] [--phases P ...]
[--metrics-root DIR] [--analysis-root DIR] [--apply]``.
"""

import argparse
import math
import shutil
import sqlite3
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ..config import load_experiment_config
from ..training.checkpoints import load_index, mark_pruned
from .queue import EXPERIMENT_CONFIG, TAG_EVAL_PENDING, TAG_PHASES
from .store import TAG_MERGE_PARTIAL, evaluation_complete
from .tracking import artifact_path, resolve_tracking_uri

FINAL_PREFIXES = ("final-", "combo-")
"""Phases whose runs the analysis, interpretation, videos and release read."""
CLASSES = ("state", "ckpt_tuning", "ckpt_final")
"""Deletion classes, in print order."""
REASON = "storage retention: no later step reads it (orchestration.prune)"
"""The ``reason`` written to ``ckpt/pruned.json``."""
METRICS_ROOT = Path("outputs/metrics")
ANALYSIS_ROOT = Path("outputs/analysis")
UNIT = ["budget", "method", "config_ref"]
"""A statistics unit (`analysis.load.UNIT_COLUMNS`) within one budget."""


class StoreRun(NamedTuple):
    """A run as the prune needs it, read straight from the store's tables."""

    run_id: str
    name: str
    status: str
    phases: tuple[str, ...]
    dir: Path
    pending: bool
    """Tagged ``eval/pending`` or ``merge/partial``: not whole, never pruned."""


class Item(NamedTuple):
    """One thing to delete: a ``state/`` folder or one checkpoint file."""

    run: str
    run_dir: Path
    kind: str
    path: Path
    size: int
    update: int | None


class Plan(NamedTuple):
    """The items to delete and the runs kept apart, by reason."""

    items: list[Item]
    kept_whole: list[str]
    unexported: list[str]
    incomplete: list[str]


def seed_picks(runs: pd.DataFrame) -> dict[str, set[str]]:
    """Run id -> roles of the seeds whose every checkpoint is kept, per `UNIT`.

    ``best`` and ``median`` rank by ``final_score`` (the training-episode score):
    the highest, and the lower median, ``(n - 1) // 2`` in ascending order, as
    interpretation's ``_median_run``; ``best-eval`` is the highest
    ``eval_stochastic_score`` (the mean stochastic eval score of the last
    checkpoint). Ties go to the larger run id; NaN scores never rank.
    """
    picks: dict[str, set[str]] = defaultdict(set)
    for _, group in runs.groupby(UNIT):
        scores = zip(group["final_score"], group["run_id"], strict=True)
        rows = [(s, r) for s, r in scores if not math.isnan(s)]
        if rows:
            ranked = [r for _, r in sorted(rows)]
            picks[ranked[-1]].add("best")
            picks[ranked[(len(ranked) - 1) // 2]].add("median")
        column = group.get("eval_stochastic_score", pd.Series(math.nan, group.index))
        evals = [
            (s, r)
            for s, r in zip(column, group["run_id"], strict=True)
            if not math.isnan(s)
        ]
        if evals:
            picks[max(evals)[1]].add("best-eval")
    return dict(picks)


def exported_runs(metrics_root: Path) -> pd.DataFrame:
    """The ``runs`` export tables of every final and combo phase, one row per run."""
    files = [
        path
        for prefix in FINAL_PREFIXES
        for path in sorted(metrics_root.glob(f"{prefix}*/runs.parquet"))
    ]
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame(columns=["run_id", *UNIT, "final_score"])
    return pd.concat(frames).drop_duplicates("run_id")


def case_updates(analysis_root: Path) -> dict[str, set[int]]:
    """Run id -> the checkpoint updates of its training-analysis cases."""
    cases: dict[str, set[int]] = defaultdict(set)
    for path in sorted(analysis_root.glob("*/cases.csv")):
        table = pd.read_csv(path, usecols=["run_id", "update"])
        for run_id, update in zip(table["run_id"], table["update"], strict=True):
            cases[str(run_id)].add(int(update))
    return dict(cases)


def _db_path(tracking_uri: str | None) -> Path:
    """The SQLite file of ``tracking_uri`` (default: ``configs/experiment.yaml``)."""
    uri = tracking_uri or load_experiment_config(EXPERIMENT_CONFIG).mlflow.tracking_uri
    base = uri.partition("?")[0]  # a ``?timeout=`` query is not part of the path
    return Path(resolve_tracking_uri(base).removeprefix("sqlite:///"))


def store_runs(tracking_uri: str | None) -> list[StoreRun]:
    """Every active run of the store, read through a read-only connection."""
    db = _db_path(tracking_uri)
    query = """
        select r.run_uuid, r.name, r.status, r.artifact_uri,
          (select value from tags t where t.run_uuid = r.run_uuid and t.key = ?),
          exists(select 1 from tags t where t.run_uuid = r.run_uuid
                 and t.key in (?, ?) and t.value = 'true')
        from runs r where r.lifecycle_stage = 'active'
    """
    keys = (TAG_PHASES, TAG_EVAL_PENDING, TAG_MERGE_PARTIAL)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=300)
    try:
        rows = con.execute(query, keys).fetchall()
    finally:
        con.close()
    return [
        StoreRun(
            rid,
            name,
            status,
            tuple(p for p in (phases or "").split(",") if p),
            artifact_path(uri),
            bool(pending),
        )
        for rid, name, status, uri, phases, pending in rows
    ]


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _checkpoints(run: StoreRun, keep: Collection[int], kind: str) -> list[Item]:
    """Every checkpoint file of ``run`` on disk whose update is not in ``keep``."""
    items = []
    for e in load_index(run.dir):
        path = run.dir / "ckpt" / e["file"]
        if e["update"] not in keep and path.is_file():
            items.append(Item(run.name, run.dir, kind, path, _size(path), e["update"]))
    return items


def _run_items(
    run: StoreRun,
    picks: Collection[str],
    listed: Collection[str],
    cases: dict[str, set[int]],
    plan: Plan,
) -> list[Item]:
    """What to delete of one run in scope; a run kept apart is noted in ``plan``."""
    final = any(p.startswith(FINAL_PREFIXES) for p in run.phases)
    state = run.dir / "state"
    has_files = state.is_dir() or any((run.dir / "ckpt").glob("*.msgpack"))
    if has_files and not evaluation_complete(run.dir, log_spaced=final):
        plan.incomplete.append(run.name)  # its evaluation may still need params
        return []
    items = [Item(run.name, run.dir, "state", state, _size(state), None)]
    items = items if state.is_dir() else []
    if not final:
        return items + _checkpoints(run, (), "ckpt_tuning")
    if run.run_id in picks:
        plan.kept_whole.append(run.name)
        return items
    if run.run_id not in listed:
        plan.unexported.append(run.name)
        return items
    keep = {
        *[e["update"] for e in load_index(run.dir)][-1:],
        *cases.get(run.run_id, ()),
    }
    return items + _checkpoints(run, keep, "ckpt_final")


def plan_prune(
    runs: Iterable[StoreRun],
    exported: pd.DataFrame,
    cases: dict[str, set[int]],
    phases: Collection[str] | None = None,
) -> Plan:
    """What to delete from ``runs`` (see the module docstring)."""
    picks, listed = seed_picks(exported), set(exported["run_id"])
    plan = Plan([], [], [], [])
    for run in sorted(runs, key=lambda r: r.name):
        if run.status != "FINISHED" or run.pending or not run.dir.is_dir():
            continue
        if phases is None or set(run.phases) & set(phases):
            plan.items.extend(_run_items(run, picks, listed, cases, plan))
    return plan


def apply_plan(plan: Plan, reason: str = REASON) -> None:
    """Delete the plan's items; a run's checkpoints are marked pruned first."""
    by_run: dict[Path, list[Item]] = defaultdict(list)
    for item in plan.items:
        by_run[item.run_dir].append(item)
    for run_dir, items in by_run.items():
        ckpts = [i for i in items if i.update is not None]
        if ckpts:
            mark_pruned(
                run_dir, [i.update for i in ckpts if i.update is not None], reason
            )
        for item in items:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink(missing_ok=True)


def summary(plan: Plan, applied: bool) -> list[str]:
    """Printable lines: size per class, then the runs kept apart."""
    head = "deleted" if applied else "dry run: nothing deleted (--apply deletes)"
    lines = [head, f"{'class':<12}{'items':>8}{'runs':>7}{'bytes':>20}{'GB':>9}"]
    for kind in (*CLASSES, "total"):
        items = [i for i in plan.items if kind in (i.kind, "total")]
        size = sum(i.size for i in items)
        runs = len({i.run_dir for i in items})
        lines.append(f"{kind:<12}{len(items):>8}{runs:>7}{size:>20,}{size / 1e9:>9.1f}")
    lines.append(f"kept whole: {len(plan.kept_whole)} runs (best and median seed by")
    lines.append("  final_score, best seed by eval_stochastic_score)")
    for label, names in (
        ("not in the export tables, every checkpoint kept", plan.unexported),
        ("evaluation incomplete, skipped", plan.incomplete),
    ):
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The CLI's arguments."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--tracking-uri", help="default: configs/experiment.yaml")
    p.add_argument("--phases", nargs="+", metavar="PHASE", help="default: every run")
    p.add_argument("--metrics-root", type=Path, default=METRICS_ROOT)
    p.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    p.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    return p.parse_args(argv)


def plan_from_args(args: argparse.Namespace) -> Plan:
    """`plan_prune` of the store and tables the arguments name."""
    return plan_prune(
        store_runs(args.tracking_uri),
        exported_runs(args.metrics_root),
        case_updates(args.analysis_root),
        args.phases,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    plan = plan_from_args(args)
    if args.apply:
        apply_plan(plan)
    print("\n".join(summary(plan, args.apply)))


if __name__ == "__main__":
    main()
