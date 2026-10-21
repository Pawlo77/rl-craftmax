"""Training: registry, PPO(-RNN), PQN-RNN, hooks, wrappers, checkpoints, metrics."""

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
        HookMethod,
        HookSpec,
        ResolvedHooks,
        Traj,
        check_metric_names,
        config_dict,
        resolve,
        with_method,
    )
    from .interface import METHODS, Policy
    from .ppo import (
        ActorCritic,
        PPOConfig,
        PPOTrainer,
        gae,
        make_env,
        make_network,
        make_train_env,
        ppo_terms,
        train,
    )
    from .ppo_rnn import ActorCriticRNN, PPORNNConfig, PPORNNPolicy, PPORNNTrainer
    from .pqn_rnn import (
        PQNRNNConfig,
        PQNRNNPolicy,
        PQNRNNTrainer,
        RNNQNetwork,
        q_lambda_targets,
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
    from .wrappers import (
        BatchEnvWrapper,
        GymnaxWrapper,
        LogWrapper,
        OptimisticResetVecEnvWrapper,
        unwrap_state,
    )

_EXPORTS: dict[str, str] = {
    "HOOK_NAMES": "hooks",
    "HookFields": "hooks",
    "HookMethod": "hooks",
    "HookSpec": "hooks",
    "METHODS": "interface",
    "ActorCritic": "ppo",
    "ActorCriticRNN": "ppo_rnn",
    "BatchEnvWrapper": "wrappers",
    "BatchRenorm": "batch_renorm",
    "Episodes": "records",
    "GymnaxWrapper": "wrappers",
    "LogWrapper": "wrappers",
    "OptimisticResetVecEnvWrapper": "wrappers",
    "PPOConfig": "ppo",
    "PPOTrainer": "ppo",
    "PPORNNConfig": "ppo_rnn",
    "PPORNNPolicy": "ppo_rnn",
    "PPORNNTrainer": "ppo_rnn",
    "PQNRNNConfig": "pqn_rnn",
    "PQNRNNPolicy": "pqn_rnn",
    "PQNRNNTrainer": "pqn_rnn",
    "Policy": "interface",
    "RNNQNetwork": "pqn_rnn",
    "ResolvedHooks": "hooks",
    "RunLog": "records",
    "TrainingMetrics": "records",
    "Traj": "hooks",
    "achievement_names": "records",
    "achievement_weights": "records",
    "check_metric_names": "hooks",
    "checkpoint_schedule": "checkpoints",
    "config_dict": "hooks",
    "first_unlocks": "records",
    "gae": "ppo",
    "load_episodes": "records",
    "load_index": "checkpoints",
    "load_metrics": "records",
    "load_params": "checkpoints",
    "load_state": "checkpoints",
    "make_env": "ppo",
    "make_network": "ppo",
    "make_train_env": "ppo",
    "ppo_terms": "ppo",
    "q_lambda_targets": "pqn_rnn",
    "resolve": "hooks",
    "save_checkpoint": "checkpoints",
    "save_state": "checkpoints",
    "train": "ppo",
    "unwrap_state": "wrappers",
    "with_method": "hooks",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "HOOK_NAMES",
    "METHODS",
    "ActorCritic",
    "ActorCriticRNN",
    "BatchEnvWrapper",
    "BatchRenorm",
    "Episodes",
    "GymnaxWrapper",
    "HookFields",
    "HookMethod",
    "HookSpec",
    "LogWrapper",
    "OptimisticResetVecEnvWrapper",
    "PPOConfig",
    "PPORNNConfig",
    "PPORNNPolicy",
    "PPORNNTrainer",
    "PPOTrainer",
    "PQNRNNConfig",
    "PQNRNNPolicy",
    "PQNRNNTrainer",
    "Policy",
    "RNNQNetwork",
    "ResolvedHooks",
    "RunLog",
    "TrainingMetrics",
    "Traj",
    "achievement_names",
    "achievement_weights",
    "check_metric_names",
    "checkpoint_schedule",
    "config_dict",
    "first_unlocks",
    "gae",
    "load_episodes",
    "load_index",
    "load_metrics",
    "load_params",
    "load_state",
    "make_env",
    "make_network",
    "make_train_env",
    "ppo_terms",
    "q_lambda_targets",
    "resolve",
    "save_checkpoint",
    "save_state",
    "train",
    "unwrap_state",
    "with_method",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
