"""Local MLflow tracking helpers for resumable runs."""

import importlib
import json
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..config import ExperimentConfig, MLflowTrackingConfig
from .reproducibility import collect_reproducibility_report

_TRACKING_LOGGER = logging.getLogger(__name__)
_ACTIVE_MLFLOW_RUN_ID_ENV = "MLFLOW_ACTIVE_RUN_ID"
UNDER_PYTEST_ENV = "WUT_MURL_UNDER_PYTEST"
"""Set by the tests' conftest; the CLIs then refuse the repo-root MLflow store."""
_REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_EXPERIMENT = "analysis"
"""MLflow experiment of every analysis writer: export, training analysis, ... (contracts
section 8)."""


def _refuse_repo_store(local_path: Path) -> None:
    """Raise when the tests would open a store inside this checkout.

    Raises:
        RuntimeError: Under pytest (`UNDER_PYTEST_ENV`) for a path inside the repo,
            where a live queue keeps the real ``mlruns.db`` and ``mlruns/``.
    """
    if os.environ.get(UNDER_PYTEST_ENV) and local_path.is_relative_to(_REPO_ROOT):
        msg = (
            f"refusing the MLflow store {local_path} inside the repo under pytest: "
            "pass a temporary --tracking-uri or experiment config"
        )
        raise RuntimeError(msg)


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Resolve local tracking URIs to an absolute path when needed.

    Raises:
        RuntimeError: Under pytest, for a store inside the repo (`_refuse_repo_store`).
    """
    if tracking_uri.startswith("sqlite:"):
        sqlite_target = tracking_uri.removeprefix("sqlite:")
        if sqlite_target.lstrip("/") == ":memory:":
            return "sqlite:///:memory:"

        if sqlite_target.startswith("///"):
            db_target = sqlite_target[3:]
        elif sqlite_target.startswith("//"):
            db_target = sqlite_target[2:]
        elif sqlite_target.startswith("/"):
            db_target = sqlite_target[1:]
        else:
            db_target = sqlite_target

        if not db_target:
            db_target = "mlruns.db"

        resolved_db_path = Path(db_target).expanduser().resolve()
        _refuse_repo_store(resolved_db_path)
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{resolved_db_path.as_posix()}"

    if "://" in tracking_uri and not tracking_uri.startswith("file:"):
        return tracking_uri

    resolved_path = Path(tracking_uri).expanduser().resolve()
    _refuse_repo_store(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return str(resolved_path)


def resolve_store_uri(tracking_uri: str) -> str:
    """`resolve_tracking_uri` of the store part; a ``?timeout=<s>`` query is kept.

    SQLAlchemy takes the busy timeout (R103) from that query, so it is split off
    before the path is resolved and put back after.
    """
    base, _, query = tracking_uri.partition("?")
    return resolve_tracking_uri(base) + (f"?{query}" if query else "")


_resolve_tracking_uri = resolve_tracking_uri
"""The name the orchestration modules import (kept so they need no edit)."""


def _artifact_location(tracking_uri: str, experiment: str) -> str | None:
    """``<db dir>/mlruns/<experiment>`` for a SQLite file store, else MLflow's default.

    MLflow's default is relative to the working directory; beside the database the
    artifacts stay with their store wherever a process starts.
    """
    if not tracking_uri.startswith("sqlite:///") or tracking_uri.endswith(":memory:"):
        return None
    db = Path(tracking_uri.removeprefix("sqlite:///"))
    return (db.parent / "mlruns" / experiment).as_uri()


def artifact_path(artifact_uri: str) -> Path:
    """Local directory of a ``file:`` artifact URI.

    Raises:
        ValueError: For a remote artifact store.
    """
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in ("", "file"):
        msg = f"expected a local artifact store, got {artifact_uri}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))


def _use_experiment(mlflow: Any, tracking_uri: str, name: str) -> None:
    """Make ``name`` the active experiment, creating it if no other process has."""
    if mlflow.get_experiment_by_name(name) is None:
        location = _artifact_location(tracking_uri, name)
        try:
            mlflow.create_experiment(name, artifact_location=location)
        except mlflow.exceptions.MlflowException:
            # another worker created it between the lookup and the create
            if mlflow.get_experiment_by_name(name) is None:
                raise
    mlflow.set_experiment(name)


def _analysis_experiment(client: Any, uri: str) -> Any:
    """The `ANALYSIS_EXPERIMENT` through ``client`` alone, created if no process has."""
    import mlflow

    exp = client.get_experiment_by_name(ANALYSIS_EXPERIMENT)
    if exp is None:
        location = _artifact_location(uri.partition("?")[0], ANALYSIS_EXPERIMENT)
        try:
            client.create_experiment(ANALYSIS_EXPERIMENT, artifact_location=location)
        except mlflow.exceptions.MlflowException:  # another process created it
            if client.get_experiment_by_name(ANALYSIS_EXPERIMENT) is None:
                raise
        exp = client.get_experiment_by_name(ANALYSIS_EXPERIMENT)
    if exp is None:
        msg = f"experiment {ANALYSIS_EXPERIMENT} is missing"
        raise RuntimeError(msg)
    return exp


def log_analysis_run(
    folder: Path,
    paths: Sequence[Path],
    name: str,
    tags: Mapping[str, str],
    tracking_uri: str,
) -> str:
    """Log ``paths`` (files or folders in ``folder``) as the artifacts of run ``name``.

    The run lives in experiment `ANALYSIS_EXPERIMENT` (get-or-create) and carries the
    given tags only: loaders select training runs by ``phases``, ``method`` ... so an
    analysis run never sets those. ``tracking_uri`` may end in ``?timeout=<s>``. A
    rerun reuses the run of that name: it is RUNNING while the new files are logged,
    and the old artifacts the new upload lacks are deleted only after that succeeded.

    Returns:
        The run id.

    Raises:
        OSError: If an old artifact cannot be deleted.
    """
    # imported here: the file-only paths of the analysis CLIs need no MLflow
    import mlflow

    # only the client: the fluent set_tracking_uri / set_experiment write
    # MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID into os.environ, which then
    # reaches every later subprocess of the process
    uri = resolve_store_uri(tracking_uri)
    client = mlflow.MlflowClient(uri)
    exp = _analysis_experiment(client, uri)
    found = client.search_runs(
        [exp.experiment_id], filter_string=f"attributes.run_name = '{name}'"
    )
    if found:
        run_id = found[0].info.run_id
        client.update_run(run_id, status="RUNNING")
    else:
        run_id = client.create_run(exp.experiment_id, run_name=name).info.run_id
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    kept: set[Path] = set()
    for path in paths:
        rel = path.relative_to(folder)
        if path.is_dir():
            client.log_artifacts(run_id, str(path), rel.as_posix())
            kept |= {rel / f.relative_to(path) for f in path.rglob("*") if f.is_file()}
        else:
            client.log_artifact(run_id, str(path))
            kept.add(rel)
    if found and (uri_of := client.get_run(run_id).info.artifact_uri):
        root = artifact_path(uri_of)
        for old in sorted(root.rglob("*"), reverse=True):  # children before parents
            if old.is_file() and old.relative_to(root) not in kept:
                old.unlink()  # raises if it cannot delete
            elif old.is_dir() and not any(old.iterdir()):
                old.rmdir()
    client.set_terminated(run_id)
    return str(run_id)


@dataclass(slots=True)
class MlflowRunTracker:
    """Track a single run with MLflow while tolerating offline/local-only setups."""

    tracking: MLflowTrackingConfig
    experiment_config: ExperimentConfig
    run_name: str
    model_adapter: Any = None
    run_id: str | None = None
    """Run to resume; set to the started run's id by `start`."""
    tags: Mapping[str, str] = field(default_factory=dict)
    """Tags set on every start, also when resuming."""
    strict: bool = False
    """Raise when the run cannot start or reopen (never fall back to a new run), and
    ignore ``$MLFLOW_ACTIVE_RUN_ID``: a run's identity must not change silently."""

    _mlflow: Any = field(default=None, init=False, repr=False)
    _run_active: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        """Start a run and log static hyperparameters and efficiency metrics."""
        if not self.tracking.enabled:
            return

        try:
            mlflow = importlib.import_module("mlflow")
            tracking_uri = _resolve_tracking_uri(self.tracking.tracking_uri)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The filesystem tracking backend .*",
                    category=FutureWarning,
                )
                mlflow.set_tracking_uri(tracking_uri)
                env_run_id = (
                    None if self.strict else os.environ.get(_ACTIVE_MLFLOW_RUN_ID_ENV)
                )
                resume_run_id = self.run_id or env_run_id
                if not (resume_run_id and self._resume(mlflow, resume_run_id)):
                    _use_experiment(mlflow, tracking_uri, self.tracking.experiment_name)
                    mlflow.start_run(
                        run_name=self.run_name or self.tracking.run_name,
                        log_system_metrics=self.tracking.log_system_metrics,
                    )

            self._mlflow = mlflow
            self._run_active = True

            active_run = mlflow.active_run()
            if active_run is not None:
                self.run_id = active_run.info.run_id
                os.environ[_ACTIVE_MLFLOW_RUN_ID_ENV] = active_run.info.run_id

            mlflow.set_tag("pipeline.run_name", self.run_name)
            mlflow.set_tags(dict(self.tags))

            self._log_reproducibility_report()

        except Exception as e:
            if self.strict:
                raise
            _TRACKING_LOGGER.warning("Failed to start MLflow run: %s", e)
            self._run_active = False
            self._mlflow = None

    def _resume(self, mlflow: Any, run_id: str) -> bool:
        """Reopen ``run_id`` in its own experiment, which may be another phase's."""
        try:
            experiment_id = mlflow.get_run(run_id).info.experiment_id
            mlflow.set_experiment(experiment_id=experiment_id)
            mlflow.start_run(
                run_id=run_id, log_system_metrics=self.tracking.log_system_metrics
            )
        except Exception as e:
            if self.strict:
                msg = f"cannot reopen MLflow run {self.run_name} ({run_id}): {e}"
                raise RuntimeError(msg) from e
            _TRACKING_LOGGER.warning(
                "Failed to resume MLflow run_id=%s; starting a new run.", run_id
            )
            return False
        return True

    def _log_reproducibility_report(self) -> None:
        """Persist the reproducibility snapshot as an MLflow artifact."""
        report = collect_reproducibility_report()
        payload = report.to_dict()

        if self._run_active and self._mlflow is not None:
            self._mlflow.log_dict(payload, "reproducibility_report.json")
        else:
            _TRACKING_LOGGER.info(
                "MLflow inactive; reproducibility payload follows:\n%s",
                json.dumps(payload, indent=2, sort_keys=True),
            )

    @property
    def artifact_dir(self) -> Path:
        """Local artifact directory of the active run.

        Raises:
            RuntimeError: If no run is active (MLflow disabled or failed to start).
        """
        if not self._run_active or self._mlflow is None:
            msg = "no active MLflow run; see the warning logged by start()"
            raise RuntimeError(msg)
        return artifact_path(self._mlflow.active_run().info.artifact_uri)

    def close(self, status: str = "FINISHED") -> None:
        """End the active MLflow run, if one was started, with ``status``."""
        if self._run_active and self._mlflow is not None:
            self._mlflow.end_run(status=status)
            self._run_active = False
            self._mlflow = None
            os.environ.pop(_ACTIVE_MLFLOW_RUN_ID_ENV, None)


def build_mlflow_tracker(
    experiment_config: ExperimentConfig,
    run_name: str | None = None,
    model_adapter: Any = None,
    *,
    run_id: str | None = None,
    tags: Mapping[str, str] | None = None,
    strict: bool = False,
) -> MlflowRunTracker:
    """Build an MLflow tracker from an experiment configuration."""
    return MlflowRunTracker(
        tracking=experiment_config.mlflow,
        experiment_config=experiment_config,
        run_name=run_name or experiment_config.name,
        model_adapter=model_adapter,
        run_id=run_id,
        tags=dict(tags or {}),
        strict=strict,
    )
