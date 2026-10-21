"""Orchestration: fast JAX runtime, run queue, resumable runner, MLflow tracking."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .profiles import Profile, load_profile
    from .queue import (
        RunSpec,
        budget_steps,
        coarse_results,
        config_hash,
        expand_fine,
        expand_phase,
        load_method,
        mark_tuned,
        resolve_config,
    )
    from .reproducibility import ReproducibilityReport, collect_reproducibility_report
    from .runner import ChunkMetrics, Trainer, final_score, worker_specs
    from .runtime import (
        DeviceRoles,
        configure_runtime,
        device_roles,
        env_mesh,
        select_platforms,
        shard_envs,
        warm_texture_cache,
    )
    from .store import merge_stores, phase_status
    from .tracking import MlflowRunTracker, build_mlflow_tracker

_EXPORTS: dict[str, str] = {
    "ChunkMetrics": "runner",
    "DeviceRoles": "runtime",
    "MlflowRunTracker": "tracking",
    "Profile": "profiles",
    "ReproducibilityReport": "reproducibility",
    "RunSpec": "queue",
    "Trainer": "runner",
    "budget_steps": "queue",
    "build_mlflow_tracker": "tracking",
    "coarse_results": "queue",
    "collect_reproducibility_report": "reproducibility",
    "config_hash": "queue",
    "configure_runtime": "runtime",
    "device_roles": "runtime",
    "env_mesh": "runtime",
    "expand_fine": "queue",
    "expand_phase": "queue",
    "final_score": "runner",
    "load_method": "queue",
    "load_profile": "profiles",
    "mark_tuned": "queue",
    "merge_stores": "store",
    "phase_status": "store",
    "resolve_config": "queue",
    "select_platforms": "runtime",
    "shard_envs": "runtime",
    "warm_texture_cache": "runtime",
    "worker_specs": "runner",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ChunkMetrics",
    "DeviceRoles",
    "MlflowRunTracker",
    "Profile",
    "ReproducibilityReport",
    "RunSpec",
    "Trainer",
    "budget_steps",
    "build_mlflow_tracker",
    "coarse_results",
    "collect_reproducibility_report",
    "config_hash",
    "configure_runtime",
    "device_roles",
    "env_mesh",
    "expand_fine",
    "expand_phase",
    "final_score",
    "load_method",
    "load_profile",
    "mark_tuned",
    "merge_stores",
    "phase_status",
    "resolve_config",
    "select_platforms",
    "shard_envs",
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
