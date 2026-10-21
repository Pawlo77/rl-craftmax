"""Training: registry, PPO, PPO-RNN, hooks, checkpoints, resume state, metrics."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .batch_renorm import BatchRenorm
    from .checkpoints import (
        checkpoint_schedule,
        load_index,
        load_params,
        load_state,
        save_checkpoint,
        save_state,
    )
    from .hooks import (
        HOOK_NAMES,
        HookFields,
        HookSpec,
        ResolvedHooks,
        Traj,
        check_metric_names,
        config_dict,
        resolve,
    )
    from .interface import METHODS, Policy
    from .ppo import ActorCritic, PPOConfig, PPOTrainer, make_network, train
    from .ppo_rnn import (
        ActorCriticRNN,
        PPORNNConfig,
        PPORNNPolicy,
        PPORNNTrainer,
        build_network,
        make_trainer,
    )
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
    "HOOK_NAMES": "hooks",
    "HookFields": "hooks",
    "HookSpec": "hooks",
    "METHODS": "interface",
    "ActorCritic": "ppo",
    "ActorCriticRNN": "ppo_rnn",
    "BatchRenorm": "batch_renorm",
    "Episodes": "records",
    "PPOConfig": "ppo",
    "PPOTrainer": "ppo",
    "PPORNNConfig": "ppo_rnn",
    "PPORNNPolicy": "ppo_rnn",
    "PPORNNTrainer": "ppo_rnn",
    "Policy": "interface",
    "ResolvedHooks": "hooks",
    "RunLog": "records",
    "TrainingMetrics": "records",
    "Traj": "hooks",
    "achievement_names": "records",
    "achievement_weights": "records",
    "build_network": "ppo_rnn",
    "check_metric_names": "hooks",
    "checkpoint_schedule": "checkpoints",
    "config_dict": "hooks",
    "first_unlocks": "records",
    "load_episodes": "records",
    "load_index": "checkpoints",
    "load_metrics": "records",
    "load_params": "checkpoints",
    "load_state": "checkpoints",
    "make_network": "ppo",
    "make_trainer": "ppo_rnn",
    "resolve": "hooks",
    "save_checkpoint": "checkpoints",
    "save_state": "checkpoints",
    "train": "ppo",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "HOOK_NAMES",
    "METHODS",
    "ActorCritic",
    "ActorCriticRNN",
    "BatchRenorm",
    "Episodes",
    "HookFields",
    "HookSpec",
    "PPOConfig",
    "PPORNNConfig",
    "PPORNNPolicy",
    "PPORNNTrainer",
    "PPOTrainer",
    "Policy",
    "ResolvedHooks",
    "RunLog",
    "TrainingMetrics",
    "Traj",
    "achievement_names",
    "achievement_weights",
    "build_network",
    "check_metric_names",
    "checkpoint_schedule",
    "config_dict",
    "first_unlocks",
    "load_episodes",
    "load_index",
    "load_metrics",
    "load_params",
    "load_state",
    "make_network",
    "make_trainer",
    "resolve",
    "save_checkpoint",
    "save_state",
    "train",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
