"""Storage retention: delete run artifacts no later step reads (dry run by default).

FINISHED runs with a complete evaluation lose ``state/`` (class ``state``); tuning
runs every checkpoint, smoke runs all but the last (``ckpt_tuning``); final runs
(``final-*``, ``combo-*``) all but the last, the case updates and those of the
`seed_picks` seeds (``ckpt_final``). Runs that may still need files are kept apart
(`_kept_apart`). ``ckpt/pruned.json`` lists checkpoints before they go; ``index.json``
and eval files stay; relative paths resolve against the repo root. CLI: ``python -m
wut_murl_craftax.orchestration.prune [--tracking-uri URI] [--phases P ...]
[--metrics-root DIR] [--analysis-root DIR] [--apply]``.
"""

import argparse
import math
import shutil
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ..config import load_experiment_config
from ..training.checkpoints import load_index, mark_pruned
from .queue import EXPERIMENT_CONFIG, REPO_ROOT, TAG_EVAL_PENDING, TAG_PHASES
from .runner import OWNER_FILE, TAG_CKPT_MISSING, TAG_WARN_ARTIFACTS
from .store import TAG_MERGE_PARTIAL, TAG_STATE_UNREADABLE, evaluation_complete
from .tracking import artifact_path

FINAL_PREFIXES = ("final-", "combo-")
"""Phases whose runs the analysis, interpretation, videos and release read."""
CLASSES = ("state", "ckpt_tuning", "ckpt_final")
"""Deletion classes, in print order."""
REASON = "storage retention: no later step reads it (orchestration.prune)"
"""The ``reason`` written to ``ckpt/pruned.json``."""
KEEP_LAST_PHASES = ("smoke",)
"""Non-final phases whose runs keep their last checkpoint (video plan, Classic PPO)."""
METRICS_ROOT = Path("outputs/metrics")
ANALYSIS_ROOT = Path("outputs/analysis")
"""Defaults, relative to the repo root like every relative path the CLI is given."""
GUARD_TAGS = (TAG_WARN_ARTIFACTS, TAG_CKPT_MISSING, TAG_STATE_UNREADABLE)
"""Run tags that mean a run may still need its files (a rebuild, a repair)."""
APART = {
    "guarded": "tagged warn/artifacts, ckpt/missing or warn/state_unreadable",
    "owned": "held by a live queue (owner.json)",
    "untagged": "no phases tag",
    "unsafe": "unsafe paths (links or outside ckpt/)",
    "incomplete": "evaluation incomplete, skipped",
    "unexported": "not in the export tables, every checkpoint kept",
}
"""Why a run is kept apart (`Plan.apart` key -> printed label)."""
UNIT = ["budget", "method", "config_ref"]
"""A statistics unit (`analysis.load.UNIT_COLUMNS`) within one budget."""


class StoreRun(NamedTuple):
    """A run as the prune needs it, read straight from the store's tables."""

    run_id: str
    name: str
    status: str
    phases: tuple[str, ...] | None
    """Its ``phases`` tag; None without one."""
    dir: Path
    pending: bool
    """Tagged ``eval/pending`` or ``merge/partial``: not whole, never pruned."""
    guarded: bool = False
    """Tagged with one of `GUARD_TAGS`."""


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
    apart: dict[str, list[str]]
    """`APART` key -> run names."""


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


def _from_repo(path: Path) -> Path:
    """``path``, relative ones taken from the repo root, not the working directory."""
    return path if path.is_absolute() else REPO_ROOT / path


def _db_path(tracking_uri: str | None) -> Path:
    """The SQLite file of ``tracking_uri`` (default: ``configs/experiment.yaml``)."""
    uri = tracking_uri or load_experiment_config(EXPERIMENT_CONFIG).mlflow.tracking_uri
    base = uri.partition("?")[0]  # a ``?timeout=`` query is not part of the path
    return _from_repo(Path(base.removeprefix("sqlite:///")))


def store_runs(tracking_uri: str | None) -> list[StoreRun]:
    """Every active run of the store, read through a read-only connection."""
    db = _db_path(tracking_uri)
    keys = (TAG_PHASES, TAG_EVAL_PENDING, TAG_MERGE_PARTIAL, *GUARD_TAGS)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=300)
    try:
        rows = con.execute(
            "select run_uuid, name, status, artifact_uri from runs"
            " where lifecycle_stage = 'active'"
        ).fetchall()
        tags = con.execute(
            "select run_uuid, key, value from tags where key in (?, ?, ?, ?, ?, ?)",
            keys,
        ).fetchall()
    finally:
        con.close()
    by_run: dict[str, dict[str, str]] = defaultdict(dict)
    for rid, key, value in tags:
        by_run[rid][key] = value
    out = []
    for rid, name, status, uri in rows:
        t = by_run[rid]
        phases = t.get(TAG_PHASES)
        pending = "true" in (t.get(TAG_EVAL_PENDING), t.get(TAG_MERGE_PARTIAL))
        listed = None if phases is None else tuple(p for p in phases.split(",") if p)
        guarded = any(k in t for k in GUARD_TAGS)
        run_dir = artifact_path(uri)
        out.append(StoreRun(rid, name, status, listed, run_dir, pending, guarded))
    return out


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _safe(run: StoreRun, root: Path | None) -> bool:
    """Whether every path the prune may delete stays inside ``run``'s folders.

    The run folder must sit under the store's folder ``root``; ``ckpt/``, ``state/``
    and the indexed checkpoint files must be real files and folders (no links)
    directly inside it.
    """
    here, ckpt = run.dir.resolve(), run.dir / "ckpt"
    if root is not None and not here.is_relative_to(root.resolve()):
        return False
    if run.dir.is_symlink() or ckpt.is_symlink() or (run.dir / "state").is_symlink():
        return False
    files = [ckpt / e["file"] for e in load_index(run.dir)]
    return all(
        not f.is_symlink() and f.resolve().parent == ckpt.resolve() for f in files
    )


def _checkpoints(run: StoreRun, keep: Collection[int], kind: str) -> list[Item]:
    """Every checkpoint file of ``run`` on disk whose update is not in ``keep``."""
    items = []
    for e in load_index(run.dir):
        path = run.dir / "ckpt" / e["file"]
        if e["update"] not in keep and path.is_file():
            items.append(Item(run.name, run.dir, kind, path, _size(path), e["update"]))
    return items


def _has_files(run: StoreRun) -> bool:
    """Whether the run has a ``state/`` folder or a checkpoint file."""
    return (run.dir / "state").is_dir() or any((run.dir / "ckpt").glob("*.msgpack"))


def _kept_apart(run: StoreRun, root: Path | None) -> str | None:
    """The `APART` reason a run in scope is never pruned, if any."""
    if run.guarded:
        return "guarded"
    if (run.dir / OWNER_FILE).exists():
        return "owned"
    if run.phases is None:
        return "untagged"
    return None if _safe(run, root) else "unsafe"


def _run_items(
    run: StoreRun,
    picks: Collection[str],
    listed: Collection[str],
    cases: dict[str, set[int]],
    plan: Plan,
) -> list[Item]:
    """What to delete of one run in scope; a run kept apart is noted in ``plan``."""
    phases = run.phases or ()
    final = any(p.startswith(FINAL_PREFIXES) for p in phases)
    state = run.dir / "state"
    if _has_files(run) and not evaluation_complete(run.dir, log_spaced=final):
        plan.apart["incomplete"].append(run.name)  # it may still need params
        return []
    items = [Item(run.name, run.dir, "state", state, _size(state), None)]
    items = items if state.is_dir() else []
    last = [e["update"] for e in load_index(run.dir)][-1:]
    if not final:
        smoke = set(phases) & set(KEEP_LAST_PHASES)
        return items + _checkpoints(run, last if smoke else [], "ckpt_tuning")
    if run.run_id in picks:
        plan.kept_whole.append(run.name)
        return items
    if run.run_id not in listed:
        plan.apart["unexported"].append(run.name)
        return items
    return items + _checkpoints(run, {*last, *cases.get(run.run_id, ())}, "ckpt_final")


def plan_prune(
    runs: Iterable[StoreRun],
    exported: pd.DataFrame,
    cases: dict[str, set[int]],
    phases: Collection[str] | None = None,
    root: Path | None = None,
) -> Plan:
    """What to delete from ``runs`` (see the module docstring).

    ``root``: the folder every run folder must be under (the store's).
    """
    picks, listed = seed_picks(exported), set(exported["run_id"])
    plan = Plan([], [], {key: [] for key in APART})
    for run in sorted(runs, key=lambda r: r.name):
        if run.status != "FINISHED" or run.pending or not run.dir.is_dir():
            continue
        if phases is not None and not set(run.phases or ()) & set(phases):
            continue
        if (reason := _kept_apart(run, root)) is not None:
            if reason != "untagged" or _has_files(run):  # bookkeeping runs: no files
                plan.apart[reason].append(run.name)
            continue
        plan.items.extend(_run_items(run, picks, listed, cases, plan))
    return plan


def apply_plan(plan: Plan, reason: str = REASON) -> list[str]:
    """Delete the plan's items, a run's checkpoints marked pruned first.

    A run a queue took since the plan (``owner.json``) is left alone; returns their
    names. A path that became a link is never followed.
    """
    by_run: dict[Path, list[Item]] = defaultdict(list)
    for item in plan.items:
        by_run[item.run_dir].append(item)
    skipped = []
    for run_dir, items in by_run.items():
        if (run_dir / OWNER_FILE).exists():
            skipped.append(items[0].run)
            continue
        updates = [i.update for i in items if i.update is not None]
        if updates:
            mark_pruned(run_dir, updates, reason)
        for item in items:
            if item.path.is_symlink():
                continue
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink(missing_ok=True)
    return skipped


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
    for key, label in APART.items():
        if names := plan.apart[key]:
            lines.append(f"{label}: {', '.join(names)}")
    return lines


def inputs(
    exported: pd.DataFrame, cases: dict[str, set[int]], args: argparse.Namespace
) -> list[str]:
    """Printable lines: what the keep rules were read from."""
    rows = sum(len(u) for u in cases.values())
    return [
        f"export tables: {len(exported)} runs from {args.metrics_root}",
        f"case updates: {rows} in {len(cases)} runs from {args.analysis_root}",
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The CLI's arguments; relative paths are made absolute from the repo root."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--tracking-uri", help="default: configs/experiment.yaml")
    p.add_argument("--phases", nargs="+", metavar="PHASE", help="default: every run")
    p.add_argument("--metrics-root", type=Path, default=METRICS_ROOT)
    p.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    p.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = p.parse_args(argv)
    args.metrics_root = _from_repo(args.metrics_root)
    args.analysis_root = _from_repo(args.analysis_root)
    return args


def _plan(
    args: argparse.Namespace,
) -> tuple[Plan, pd.DataFrame, dict[str, set[int]]]:
    exported, cases = exported_runs(args.metrics_root), case_updates(args.analysis_root)
    root = _db_path(args.tracking_uri).parent
    runs = store_runs(args.tracking_uri)
    return plan_prune(runs, exported, cases, args.phases, root), exported, cases


def plan_from_args(args: argparse.Namespace) -> Plan:
    """`plan_prune` of the store and tables the arguments name."""
    return _plan(args)[0]


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point.

    Raises:
        SystemExit: 2 for ``--apply`` when no export row or no case update loaded: a
            wrong path would otherwise delete checkpoints the keep rules protect.
    """
    args = parse_args(argv)
    plan, exported, cases = _plan(args)
    print("\n".join(inputs(exported, cases, args)))
    if args.apply and (exported.empty or not cases):
        print(
            f"refusing --apply: {len(exported)} export rows from {args.metrics_root}, "
            f"{len(cases)} case runs from {args.analysis_root}; check the paths",
            file=sys.stderr,
        )
        raise SystemExit(2)
    skipped = apply_plan(plan) if args.apply else []
    print("\n".join(summary(plan, args.apply)))
    if skipped:
        print(f"taken by a queue since the plan, left alone: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
