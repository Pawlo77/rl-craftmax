"""Orchestration utilities: fast JAX runtime, run queue and MLflow tracking."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
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
    from .runtime import (
        DeviceRoles,
        configure_runtime,
        device_roles,
        env_mesh,
        shard_envs,
        warm_texture_cache,
    )
    from .tracking import MlflowRunTracker, build_mlflow_tracker

_EXPORTS: dict[str, str] = {
    "DeviceRoles": "runtime",
    "MlflowRunTracker": "tracking",
    "ReproducibilityReport": "reproducibility",
    "RunSpec": "queue",
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
    "load_method": "queue",
    "mark_tuned": "queue",
    "resolve_config": "queue",
    "shard_envs": "runtime",
    "warm_texture_cache": "runtime",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "DeviceRoles",
    "MlflowRunTracker",
    "ReproducibilityReport",
    "RunSpec",
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
    "load_method",
    "mark_tuned",
    "resolve_config",
    "shard_envs",
    "warm_texture_cache",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
