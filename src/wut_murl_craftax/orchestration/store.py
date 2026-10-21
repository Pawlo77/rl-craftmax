"""MLflow store tools: the status of a phase's runs and the merge of node stores.

``status`` lists each run of a phase file as pending, running, finished, killed or
failed, counting only runs whose tag ``phases`` contains the phase. ``merge`` copies
the runs and artifact directories of another SQLite store (a cluster node's) into the
main one, keyed by run name: a run already there only gets the ``phases`` merged,
unless the node's copy is FINISHED and the main one is not (the finished run replaces
it). CLI: ``status --phase P [--method M] [--expand-fine SLUG]``, ``merge SRC``.
"""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Metric, Param

from ..config import load_experiment_config
from .queue import EXPERIMENT_CONFIG, METHODS_DIR, TAG_PHASES, phase_specs
from .tracking import _artifact_location, _resolve_tracking_uri, artifact_path

STATES = ("pending", "running", "finished", "killed", "failed")
"""Run states `phase_status` reports, in print order."""
_BATCH = 1000
"""Most metrics MLflow accepts in one ``log_batch`` call."""


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
    of that name outside the phase, such as one yet to join it, is pending.
    """
    args = argparse.Namespace(
        phase=phase, method=method, expand_fine=expand_fine, methods_dir=methods_dir
    )
    specs = phase_specs(args, tracking_uri) or []
    client = mlflow.MlflowClient(tracking_uri=_main_uri(tracking_uri))
    found = _by_name([r for r in _active_runs(client) if phase.stem in _phases(r)])
    return {
        s.name: found[s.name].info.status.lower() if s.name in found else "pending"
        for s in specs
    }


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
    metrics = [
        Metric(m.key, m.value, m.timestamp, m.step)
        for key in run.data.metrics
        for m in source.get_metric_history(info.run_id, key)
    ]
    for i in range(0, len(metrics), _BATCH):
        target.log_batch(run_id, metrics=metrics[i : i + _BATCH])
    shutil.copytree(artifacts, artifact_path(new.info.artifact_uri), dirs_exist_ok=True)
    target.set_terminated(run_id, info.status, info.end_time)


def merge_stores(src: Path | str, dst: str | None = None) -> dict[str, int]:
    """Merge the runs of the SQLite store ``src`` into ``dst`` (default: the main one).

    Returns the runs ``copied`` (new to ``dst``, or FINISHED replacing an unfinished
    copy) and ``joined`` (already in ``dst``, only the ``phases`` tag merged). Every
    needed source artifact directory is found before anything is written.

    Raises:
        FileNotFoundError: ``src`` is no database, or a run to copy has no artifacts.
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
    joins: list[tuple[Any, list[str]]] = []
    copies: list[tuple[Any, Any | None, str, Path, list[str]]] = []
    for name, run in _by_name(_active_runs(source)).items():
        here = existing.get(name)
        phases = list(dict.fromkeys([*_phases(here), *_phases(run)]))
        if here is not None and not (_rank(run)[0] and not _rank(here)[0]):
            joins.append((here, phases))
            continue
        experiment = experiments[run.info.experiment_id]
        artifacts = _source_artifacts(run, experiment, db)
        copies.append((run, here, experiment, artifacts, phases))
    for here, phases in joins:
        target.set_tag(here.info.run_id, TAG_PHASES, ",".join(phases))
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
    shown = [f"{s} {n}" for s, n in totals.items() if n or s != "failed"]
    print(", ".join(shown))


if __name__ == "__main__":
    main()
