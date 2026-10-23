"""Orchestration: JAX runtime, run queue, resumable runner, MLflow tracking, finals."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .abort import first_nonfinite, first_out_of_bounds, flag_hits
    from .finals import kappa_ablation
    from .profiles import Profile, load_profile
    from .queue import (
        RunSpec,
        budget_steps,
        coarse_results,
        config_hash,
        expand_fine,
        expand_phase,
        load_method,
        mark_best,
        mark_tuned,
        permanently_ended,
        resolve_config,
        tuning_means,
    )
    from .reproducibility import ReproducibilityReport, collect_reproducibility_report
    from .runner import ChunkMetrics, Trainer, final_score, worker_specs
    from .runtime import (
        BackendError,
        DeviceRoles,
        check_backend,
        configure_runtime,
        device_roles,
        env_mesh,
        import_craftax,
        select_platforms,
        shard_envs,
        warm_texture_cache,
    )
    from .store import merge_stores, phase_status
    from .tracking import MlflowRunTracker, build_mlflow_tracker
    from .train_metrics import log_every, log_points, metric_points

_EXPORTS: dict[str, str] = {
    "BackendError": "runtime",
    "ChunkMetrics": "runner",
    "DeviceRoles": "runtime",
    "MlflowRunTracker": "tracking",
    "Profile": "profiles",
    "ReproducibilityReport": "reproducibility",
    "RunSpec": "queue",
    "Trainer": "runner",
    "budget_steps": "queue",
    "build_mlflow_tracker": "tracking",
    "check_backend": "runtime",
    "coarse_results": "queue",
    "collect_reproducibility_report": "reproducibility",
    "config_hash": "queue",
    "configure_runtime": "runtime",
    "device_roles": "runtime",
    "env_mesh": "runtime",
    "expand_fine": "queue",
    "expand_phase": "queue",
    "final_score": "runner",
    "first_nonfinite": "abort",
    "first_out_of_bounds": "abort",
    "flag_hits": "abort",
    "import_craftax": "runtime",
    "kappa_ablation": "finals",
    "load_method": "queue",
    "load_profile": "profiles",
    "log_every": "train_metrics",
    "log_points": "train_metrics",
    "mark_best": "queue",
    "mark_tuned": "queue",
    "merge_stores": "store",
    "metric_points": "train_metrics",
    "permanently_ended": "queue",
    "phase_status": "store",
    "resolve_config": "queue",
    "select_platforms": "runtime",
    "shard_envs": "runtime",
    "tuning_means": "queue",
    "warm_texture_cache": "runtime",
    "worker_specs": "runner",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "BackendError",
    "ChunkMetrics",
    "DeviceRoles",
    "MlflowRunTracker",
    "Profile",
    "ReproducibilityReport",
    "RunSpec",
    "Trainer",
    "budget_steps",
    "build_mlflow_tracker",
    "check_backend",
    "coarse_results",
    "collect_reproducibility_report",
    "config_hash",
    "configure_runtime",
    "device_roles",
    "env_mesh",
    "expand_fine",
    "expand_phase",
    "final_score",
    "first_nonfinite",
    "first_out_of_bounds",
    "flag_hits",
    "import_craftax",
    "kappa_ablation",
    "load_method",
    "load_profile",
    "log_every",
    "log_points",
    "mark_best",
    "mark_tuned",
    "merge_stores",
    "metric_points",
    "permanently_ended",
    "phase_status",
    "resolve_config",
    "select_platforms",
    "shard_envs",
    "tuning_means",
    "warm_texture_cache",
    "worker_specs",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
