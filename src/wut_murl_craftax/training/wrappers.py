# Vendored from Craftax_Baselines wrappers.py
# (https://github.com/MichaelTMatthews/Craftax_Baselines, commit 7ce36fa).
# Copyright (c) 2024 Michael Matthews. MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions: The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS
# IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""Craftax_Baselines env wrappers (commit 7ce36fa): the one copy every method imports.

``GymnaxWrapper``, ``BatchEnvWrapper``, ``LogWrapper`` and
``OptimisticResetVecEnvWrapper`` as in the original, with type annotations and
docstrings. The only behaviour change: the optimistic-reset step also returns the
`StepEvents` of every env, computed before the reset, as ``info["events"]``; plus the
helper `unwrap_state`. Obs, states, rewards and dones are the original's.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from ..evaluation.events import step_events
from ..evaluation.milestones import game_of


def unwrap_state(state: Any) -> Any:
    """Follow ``.env_state`` down to the Craftax ``EnvState``."""
    while hasattr(state, "env_state"):
        state = state.env_state
    return state


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env: Any) -> None:
        """Wrap ``env``."""
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name: str) -> Any:
        """Attributes of the wrapped env."""
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions."""

    def __init__(self, env: Any, num_envs: int) -> None:
        """Batch ``num_envs`` copies of ``env``."""
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng: jax.Array, params: Any = None) -> tuple[jax.Array, Any]:
        """Reset every env with its own key."""
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self, rng: jax.Array, state: Any, action: jax.Array, params: Any = None
    ) -> Any:
        """Step every env with its own key."""
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """Provides efficient 'optimistic' resets.

    The wrapper also necessarily handles the batching of environment steps and
    resetting. reset_ratio: the number of environment workers per environment reset.
    Higher means more efficient but a higher chance of duplicate resets.
    """

    def __init__(self, env: Any, num_envs: int, reset_ratio: int) -> None:
        """Batch ``num_envs`` envs that share ``num_envs // reset_ratio`` resets."""
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        msg = "Reset ratio must perfectly divide num envs."
        assert num_envs % reset_ratio == 0, msg  # nosec B101  # the original's check
        self.num_resets = self.num_envs // reset_ratio
        self.game = game_of(env.name)

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng: jax.Array, params: Any = None) -> tuple[jax.Array, Any]:
        """Reset every env with its own key."""
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self, rng: jax.Array, state: Any, action: jax.Array, params: Any = None
    ) -> Any:
        """Step every env; done envs restart from one of ``num_resets`` new worlds."""
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)
        # before the resets below replace the done envs' terminal states
        events = jax.vmap(partial(step_events, self.game))(
            unwrap_state(state), action, unwrap_state(state_st)
        )
        info["events"] = events

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset_random = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )

        being_reset_deterministic = jnp.argsort(done)[-self.num_resets :]

        being_reset = jax.lax.select(
            done.astype(jnp.int32).sum() < self.num_resets,
            being_reset_deterministic,
            being_reset_random,
        )

        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(
            done: jax.Array,
            state_re: Any,
            state_st: Any,
            obs_re: jax.Array,
            obs_st: jax.Array,
        ) -> tuple[Any, jax.Array]:
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    """Env state plus the running and the last finished episode's return and length."""

    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: Any) -> None:
        """Wrap ``env``."""
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: jax.Array, params: Any = None
    ) -> tuple[jax.Array, LogEnvState]:
        """Reset ``env`` with zeroed episode counters."""
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float | jax.Array,
        params: Any = None,
    ) -> Any:
        """Step ``env``; the done step moves the counters to ``returned_*``."""
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
