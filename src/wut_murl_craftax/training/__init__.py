"""Training loops and their records: baseline PPO, checkpoints, metrics, episodes."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .checkpoints import (
        checkpoint_schedule,
        load_index,
        load_params,
        save_checkpoint,
    )
    from .ppo import ActorCritic, PPOConfig, make_network, train
    from .records import (
        Episodes,
        RunLog,
        TrainingMetrics,
        achievement_names,
        achievement_weights,
        first_unlocks,
        load_episodes,
        load_metrics,
    )

_EXPORTS: dict[str, str] = {
    "ActorCritic": "ppo",
    "Episodes": "records",
    "PPOConfig": "ppo",
    "RunLog": "records",
    "TrainingMetrics": "records",
    "achievement_names": "records",
    "achievement_weights": "records",
    "checkpoint_schedule": "checkpoints",
    "first_unlocks": "records",
    "load_episodes": "records",
    "load_index": "checkpoints",
    "load_metrics": "records",
    "load_params": "checkpoints",
    "make_network": "ppo",
    "save_checkpoint": "checkpoints",
    "train": "ppo",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ActorCritic",
    "Episodes",
    "PPOConfig",
    "RunLog",
    "TrainingMetrics",
    "achievement_names",
    "achievement_weights",
    "checkpoint_schedule",
    "first_unlocks",
    "load_episodes",
    "load_index",
    "load_metrics",
    "load_params",
    "make_network",
    "save_checkpoint",
    "train",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
