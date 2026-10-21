"""A hook method on PPO-RNN whose YAML knobs change its reward and its observations."""

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from wut_murl_craftax.training.hooks import HookFields, HookMethod, HookSpec
from wut_murl_craftax.training.ppo_rnn import PPORNNConfig, make_trainer
from wut_murl_craftax.training.wrappers import GymnaxWrapper

TINY = {
    "env": "Craftax-Classic-Symbolic-v1",
    "envs": 8,
    "num_steps": 8,
    "minibatches": 2,
    "width": 16,
    "rnn_width": 16,
}
"""Base keys of a toy run config: 64 env steps per update."""


@dataclass(frozen=True, slots=True)
class ToyConfig:
    """The toy method's own keys."""

    toy_bonus: float = 0.0
    """Intrinsic reward of every env step."""
    toy_features: int = 0
    """Ones the env wrapper appends to every observation (0: no wrapper)."""


class Widen(GymnaxWrapper):
    """Appends ``k`` features ``fill`` to each observation; the state passes through."""

    def __init__(self, env: Any, k: int, fill: float) -> None:
        super().__init__(env)
        self.k, self.fill = k, fill

    def _pad(self, obs: jax.Array) -> jax.Array:
        pad = jnp.full((*obs.shape[:-1], self.k), self.fill, obs.dtype)
        return jnp.concatenate([obs, pad], -1)

    def reset(self, key: jax.Array, params: Any = None) -> Any:
        obs, state = self._env.reset(key, params)
        return self._pad(obs), state

    def step(self, key: jax.Array, state: Any, action: Any, params: Any = None) -> Any:
        obs, state, r, d, info = self._env.step(key, state, action, params)
        return self._pad(obs), state, r, d, info

    def observation_space(self, params: Any) -> Any:
        space = self._env.observation_space(params)
        return type(space)(0.0, 1.0, (space.shape[0] + self.k,), space.dtype)


def _widen(k: int, fill: float, env: Any, params: Any) -> tuple[Widen, Any]:
    return Widen(env, k, fill), params


def toy_hooks(cfg: ToyConfig, base: PPORNNConfig) -> HookFields:
    """Reward ``toy_bonus * gamma`` per step, summed in the state; append
    ``toy_features`` features ``gamma`` (both read from the resolved base config)."""
    r_int = cfg.toy_bonus * base.gamma

    def bonus(
        s: jax.Array, e: Any, e2: Any, o: Any, d: jax.Array, t: Any
    ) -> tuple[jax.Array, jax.Array]:
        r = jnp.full(d.shape, r_int, jnp.float32)
        return r, s + r.sum()

    def init(key: jax.Array, spec: HookSpec) -> jax.Array:
        return jnp.zeros((), jnp.float32)

    k = cfg.toy_features
    return HookFields(
        init_state_fn=init,
        intrinsic_reward_fn=bonus,
        env_wrapper_fn=functools.partial(_widen, k, base.gamma) if k else None,
        metrics_fn=lambda s: {"toy/bonus_sum": s},
    )


TOY = make_trainer(PPORNNConfig(), method=HookMethod(ToyConfig, toy_hooks))
"""The toy registry entry."""
