"""MLflow store tools: the status of a phase's runs and the merge of node stores.

``status`` lists each run of a phase file as pending, running, finished, killed or
failed, counting only runs whose tag ``phases`` contains the phase; without
``--expand-fine`` a ``configs: fine`` entry lists its runs in the store. ``merge``
copies the runs and artifact directories of another SQLite store (a cluster node's)
into the main one, keyed by run name: a run already there gets the ``phases`` merged
and a FINISHED node copy's evaluation files and ``eval/*`` points it lacks, unless the
node's copy is FINISHED and the main one is not (the finished run replaces it). The
runner seeds a node's store the same way from the main one (``names``). CLI: ``status
--phase P [--method M] [--expand-fine SLUG]``, ``merge SRC``.
"""

import argparse
import shutil
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Metric, Param

from ..config import load_experiment_config
from .queue import (
    EXPERIMENT_CONFIG,
    METHODS_DIR,
    TAG_EVAL_PENDING,
    TAG_FAIL_ERROR,
    TAG_METHOD,
    TAG_PHASES,
    fine_registries,
    phase_specs,
)
from .tracking import _artifact_location, _resolve_tracking_uri, artifact_path

STATES = ("pending", "running", "eval-pending", "finished", "killed", "failed")
"""Run states `phase_status` reports, in print order; ``eval-pending`` is FINISHED with
tag ``eval/pending`` (its evaluation was cut short, results readers skip it)."""
_BATCH = 1000
"""Most metrics MLflow accepts in one ``log_batch`` call."""
EVAL = "eval"
"""Run-directory folder and metric prefix of the evaluation results."""


def _main_uri(tracking_uri: str | None) -> str:
    """``tracking_uri``, else the one of ``configs/experiment.yaml``, made absolute."""
    uri = tracking_uri or load_experiment_config(EXPERIMENT_CONFIG).mlflow.tracking_uri
    return _resolve_tracking_uri(uri)


def _active_runs(client: Any) -> list[Any]:
    """Every active run of every experiment."""
    ids = [e.experiment_id for e in client.search_experiments()]
    runs: list[Any] = []
    token = None
    while ids:
        page = client.search_runs(ids, max_results=1000, page_token=token)
        runs += list(page)
        token = page.token
        if not token:
            break
    return runs


def _rank(run: Any) -> tuple[bool, int]:
    return run.info.status == "FINISHED", run.info.start_time


def _by_name(runs: list[Any]) -> dict[str, Any]:
    """One run per name: the FINISHED one, else the latest started."""
    best: dict[str, Any] = {}
    for run in runs:
        name = run.info.run_name
        if name not in best or _rank(run) > _rank(best[name]):
            best[name] = run
    return best


def _phases(run: Any | None) -> list[str]:
    """The names in a run's tag ``phases``."""
    tag = run.data.tags.get(TAG_PHASES, "") if run else ""
    return [p for p in tag.split(",") if p]


def _state(run: Any) -> str:
    """A run's `STATES` name."""
    if run.data.tags.get(TAG_EVAL_PENDING) == "true" and run.info.status == "FINISHED":
        return "eval-pending"
    return str(run.info.status).lower()


def phase_status(
    phase: Path,
    tracking_uri: str | None = None,
    *,
    method: str | None = None,
    methods_dir: Path = METHODS_DIR,
    expand_fine: str | None = None,
) -> dict[str, str]:
    """State of every run of ``phase``, by run name, in the phase file's order.

    A run counts only when its tag ``phases`` lists the phase (the file stem); a run
    of that name outside the phase, such as one yet to join it, is pending. Without
    ``expand_fine``, the runs of a ``configs: fine`` entry are the phase's runs with
    its registry name (tag ``method``), by name after the listed ones; none is pending.
    """
    args = argparse.Namespace(
        phase=phase,
        method=method,
        expand_fine=expand_fine,
        allow_partial=False,  # status shows an incomplete coarse stage as an error
        methods_dir=methods_dir,
    )
    specs = phase_specs(args, tracking_uri, skip_fine=True) or []
    fine = [] if expand_fine else fine_registries(phase, method)
    client = mlflow.MlflowClient(tracking_uri=_main_uri(tracking_uri))
    found = _by_name([r for r in _active_runs(client) if phase.stem in _phases(r)])
    states = {
        s.name: _state(found[s.name]) if s.name in found else "pending" for s in specs
    }
    for name in sorted(found):
        if found[name].data.tags.get(TAG_METHOD) in fine and name not in states:
            states[name] = _state(found[name])
    return states


def _source_artifacts(run: Any, experiment: str, db: Path) -> Path:
    """Artifact directory of a node run, also after its store was copied elsewhere.

    The recorded path is the node's; a copied store keeps the layout
    ``<db dir>/mlruns/<experiment>/<run id>/artifacts`` beside its database.
    """
    recorded = artifact_path(run.info.artifact_uri)
    moved = db.parent / "mlruns" / experiment / run.info.run_id / "artifacts"
    for path in (recorded, moved):
        if path.is_dir():
            return path
    msg = f"{run.info.run_name}: no artifacts at {recorded} or {moved}"
    raise FileNotFoundError(msg)


def _experiment_id(client: Any, name: str, tracking_uri: str) -> str:
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    location = _artifact_location(tracking_uri, name)
    return str(client.create_experiment(name, artifact_location=location))


def _points(client: Any, run: Any, keys: Collection[str]) -> list[Metric]:
    """Every logged point of the run's metrics ``keys``."""
    return [
        Metric(m.key, m.value, m.timestamp, m.step)
        for key in keys
        for m in client.get_metric_history(run.info.run_id, key)
    ]


def _log_points(target: Any, run_id: str, metrics: list[Metric]) -> None:
    for i in range(0, len(metrics), _BATCH):
        target.log_batch(run_id, metrics=metrics[i : i + _BATCH])


def _copy_run(
    target: Any,
    source: Any,
    run: Any,
    experiment_id: str,
    artifacts: Path,
    phases: list[str],
) -> None:
    """Copy a run with its params, metric histories, tags and artifact directory."""
    info = run.info
    tags = {**run.data.tags, TAG_PHASES: ",".join(phases)}
    new = target.create_run(
        experiment_id, start_time=info.start_time, tags=tags, run_name=info.run_name
    )
    run_id = new.info.run_id
    target.log_batch(run_id, params=[Param(k, v) for k, v in run.data.params.items()])
    _log_points(target, run_id, _points(source, run, run.data.metrics))
    shutil.copytree(artifacts, artifact_path(new.info.artifact_uri), dirs_exist_ok=True)
    target.set_terminated(run_id, info.status, info.end_time)


def _copy_missing(src: Path, dst: Path) -> None:
    """Copy each file under ``src`` that ``dst`` lacks; a file that exists is whole."""
    for path in sorted(src.rglob("*")):
        to = dst / path.relative_to(src)
        if path.is_file() and path.suffix != ".tmp" and not to.exists():
            to.parent.mkdir(parents=True, exist_ok=True)
            part = to.with_name(f"{to.name}.part")
            shutil.copy2(path, part)
            part.replace(to)


def _pending(run: Any) -> bool:
    return bool(run.data.tags.get(TAG_EVAL_PENDING) == "true")


def _join_evaluation(
    target: Any, source: Any, here: Any, run: Any, artifacts: Path
) -> None:
    """Give ``here`` the eval files and ``eval/*`` points of the FINISHED ``run``.

    Files and (metric, step) points ``here`` has stay as they are. Its evaluation is
    pending after the join if ``run``'s is, or if ``here``'s is and ``run`` was not
    listed in every phase of ``here`` (it never evaluated for those).
    """
    run_id = here.info.run_id
    _copy_missing(artifacts / EVAL, artifact_path(here.info.artifact_uri) / EVAL)
    keys = [k for k in run.data.metrics if k.startswith(f"{EVAL}/")]
    have = {(m.key, m.step) for m in _points(target, here, keys)}
    new = [m for m in _points(source, run, keys) if (m.key, m.step) not in have]
    _log_points(target, run_id, new)
    covered = set(_phases(here)) <= set(_phases(run))
    pending = _pending(run) or (_pending(here) and not covered)
    if pending and not _pending(here):
        target.set_tag(run_id, TAG_EVAL_PENDING, "true")
    elif not pending and _pending(here):
        for key in (TAG_EVAL_PENDING, TAG_FAIL_ERROR):  # as the runner's FINISHED close
            if key in here.data.tags:
                target.delete_tag(run_id, key)


def merge_stores(
    src: Path | str, dst: str | None = None, *, names: Collection[str] | None = None
) -> dict[str, int]:
    """Merge the runs of the SQLite store ``src`` into ``dst`` (default: the main one).

    Returns the runs ``copied`` (new to ``dst``, or FINISHED replacing an unfinished
    copy) and ``joined`` (already in ``dst``: the ``phases`` tag merged, plus the
    evaluation of a FINISHED ``src`` copy, `_join_evaluation`). ``names`` limits the
    merge to those run names. Every needed source artifact directory is found before
    anything is written.

    Raises:
        FileNotFoundError: ``src`` is no database, or a run to copy or a FINISHED run
            to join has no artifacts.
    """
    text = str(src)
    db = Path(text.removeprefix("sqlite:///")).resolve()
    if not db.is_file():
        msg = f"no MLflow store at {db}"
        raise FileNotFoundError(msg)
    source = mlflow.MlflowClient(tracking_uri=f"sqlite:///{db}")
    dst_uri = _main_uri(dst)
    target = mlflow.MlflowClient(tracking_uri=dst_uri)
    experiments = {e.experiment_id: e.name for e in source.search_experiments()}
    existing = _by_name(_active_runs(target))
    joins: list[tuple[Any, Any, Path | None, list[str]]] = []
    copies: list[tuple[Any, Any | None, str, Path, list[str]]] = []
    for name, run in _by_name(_active_runs(source)).items():
        if names is not None and name not in names:
            continue
        here = existing.get(name)
        phases = list(dict.fromkeys([*_phases(here), *_phases(run)]))
        experiment = experiments[run.info.experiment_id]
        finished = run.info.status == "FINISHED"
        if here is not None and not (finished and here.info.status != "FINISHED"):
            artifacts = _source_artifacts(run, experiment, db) if finished else None
            joins.append((here, run, artifacts, phases))
            continue
        artifacts = _source_artifacts(run, experiment, db)
        copies.append((run, here, experiment, artifacts, phases))
    for here, run, found, phases in joins:
        target.set_tag(here.info.run_id, TAG_PHASES, ",".join(phases))
        if found is not None:
            _join_evaluation(target, source, here, run, found)
    for run, here, experiment, artifacts, phases in copies:
        if here is not None:
            target.delete_run(here.info.run_id)  # soft: the old artifacts stay on disk
        experiment_id = _experiment_id(target, experiment, dst_uri)
        _copy_run(target, source, run, experiment_id, artifacts, phases)
    return {"copied": len(copies), "joined": len(joins)}


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="state of the runs of a phase")
    status.add_argument("--phase", type=Path, required=True)
    status.add_argument("--method", help="registry name or config slug to keep")
    status.add_argument("--expand-fine", metavar="SLUG")
    status.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    status.add_argument("--tracking-uri", help="default: configs/experiment.yaml")
    merge = commands.add_parser("merge", help="copy a node store into the main one")
    merge.add_argument("src", help="SQLite database (or sqlite:/// URI) of the node")
    merge.add_argument("--dst", help="tracking URI of the main store")
    args = parser.parse_args(argv)
    if args.command == "merge":
        counts = merge_stores(args.src, args.dst)
        print(f"copied {counts['copied']}, joined {counts['joined']}")
        return
    states = phase_status(
        args.phase,
        args.tracking_uri,
        method=args.method,
        methods_dir=args.methods_dir,
        expand_fine=args.expand_fine,
    )
    for name, state in states.items():
        print(f"{state:<8} {name}")
    totals = {s: list(states.values()).count(s) for s in STATES}
    shown = [
        f"{s} {n}"
        for s, n in totals.items()
        if n or s not in ("failed", "eval-pending")
    ]
    print(", ".join(shown))


if __name__ == "__main__":
    main()
