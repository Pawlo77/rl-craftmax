"""A deterministic 3-update trainer with the shape of a registry entry."""

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wut_murl_craftax.evaluation.episode_metrics import ONLINE_LAYOUT
from wut_murl_craftax.training.records import LOSS_COLUMNS

STEPS, ENVS, N_ACH = 4, 2, 22
"""Rollout shape of one update; 22 achievements as on Craftax-Classic."""
STEPS_PER_UPDATE = 333_333
"""A 1m budget is 3 updates by default."""
N_ACTIONS = 17
"""Craftax-Classic actions, for the evaluation of the fake policy."""


class RandomPolicy:
    """Uniform random actions, NOOP when greedy; feed-forward (empty carry)."""

    def init_carry(self, n: int) -> tuple[()]:
        return ()

    def step(
        self,
        params: Any,
        carry: tuple[()],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[()]]:
        sampled = jax.random.randint(key, obs.shape[:1], 0, N_ACTIONS)
        return jnp.where(greedy, 0, sampled), carry


class FakeTrainer:
    """Params random walk, Adam-like moments, env counters, typed keys, BatchRenorm.

    ``signal_at`` sends that signal to this process during the given update, as
    Ctrl-C or a scheduler's SIGTERM would in the middle of a chunk; ``to_parent``
    sends it to the parent process instead (a runner child's), then waits a second
    for the parent to forward the stop. ``extra`` maps an
    update number to the method metrics reported at it (``{"ppo/entropy": 0.1}``);
    ``loss_columns`` makes the trainer report only those loss columns (NaN in the
    others, as PQN-RNN leaves the PPO-only ones) and ``loss_nan_at`` turns the
    reported ones NaN at that update.
    """

    def __init__(
        self,
        signal_at: tuple[int, int] | None = None,
        steps_per_update: int = STEPS_PER_UPDATE,
        extra: Callable[[int], Mapping[str, float]] | None = None,
        loss_columns: tuple[str, ...] | None = None,
        loss_nan_at: int | None = None,
        to_parent: bool = False,
    ) -> None:
        self.signal_at, self.to_parent = signal_at, to_parent
        self.spu = steps_per_update
        self.extra = extra
        self.loss_nan_at = loss_nan_at
        if loss_columns is not None:
            self.loss_columns = loss_columns
        self.updates: list[int] = []
        self.cpu_devices = 0
        self.init_steps: list[int | None] = []
        self.init_configs: list[dict[str, Any]] = []

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        return self.spu

    def init(self, config: Mapping[str, Any], key: jax.Array) -> dict[str, Any]:
        self.cpu_devices = len(jax.devices("cpu"))
        self.init_steps.append(config.get("steps"))
        self.init_configs.append(dict(config))
        k_w, key = jax.random.split(key)
        return {
            "params": {"w": jax.random.normal(k_w, (3,)) * config["lr"]},
            "opt": {"mu": jnp.zeros(3), "count": jnp.zeros((), jnp.int32)},
            "env": {"t": jnp.zeros(ENVS, jnp.int32), "alive": jnp.ones(ENVS, bool)},
            "key": key,
            "batch_stats": {"mean": jnp.zeros(3), "var": jnp.ones(3)},
            "method": np.zeros(2),
        }

    def _update(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        key, k_g, k_d, k_a, k_r = jax.random.split(state["key"], 5)
        grad = jax.random.normal(k_g, (3,))
        mu = 0.9 * state["opt"]["mu"] + 0.1 * grad
        t = state["env"]["t"] + STEPS
        done = jax.random.bernoulli(k_d, 0.5, (STEPS, ENVS))
        new = {
            "params": {"w": state["params"]["w"] - 1e-3 * mu},
            "opt": {"mu": mu, "count": state["opt"]["count"] + 1},
            "env": {"t": t, "alive": ~state["env"]["alive"]},
            "key": key,
            "batch_stats": {
                "mean": 0.99 * state["batch_stats"]["mean"] + 0.01 * grad,
                "var": state["batch_stats"]["var"] * 1.01,
            },
            "method": state["method"] + np.array([1.0, done.sum()]),
        }
        metrics = {
            "losses": grad[:1].repeat(len(LOSS_COLUMNS)),
            "done": done,
            "ach": jax.random.bernoulli(k_a, 0.3, (STEPS, ENVS, N_ACH)),
            "ret": jax.random.uniform(k_r, (STEPS, ENVS)) * N_ACH,
            "length": jnp.broadcast_to(t, (STEPS, ENVS)),
            # no online families: their per-update values stay NaN, never logged
            "online": jnp.zeros(max(s.stop for s in ONLINE_LAYOUT.values())),
            "first_unlock": jnp.full(N_ACH, -1, jnp.int32),
        }
        return new, metrics

    def _reported(self, losses: jax.Array, u: int) -> dict[str, Any]:
        """The losses of update ``u`` as reported, plus its method metrics."""
        reported = getattr(self, "loss_columns", LOSS_COLUMNS)
        keep = np.array([c in reported for c in LOSS_COLUMNS])
        if u == self.loss_nan_at:
            losses = losses * jnp.nan
        out: dict[str, Any] = {"losses": jnp.where(keep, losses, jnp.nan)}
        if self.extra is not None:
            out |= {k: jnp.asarray(v, jnp.float32) for k, v in self.extra(u).items()}
        return out

    def train_chunk(
        self, state: dict[str, Any], n_updates: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = []
        for _ in range(n_updates):
            state, metrics = self._update(state)
            self.updates.append(u := int(state["opt"]["count"]))
            metrics |= self._reported(metrics["losses"], u)
            rows.append(metrics)
            if self.signal_at and self.signal_at[0] == self.updates[-1]:
                os.kill(
                    os.getppid() if self.to_parent else os.getpid(), self.signal_at[1]
                )
                if self.to_parent:
                    time.sleep(1.0)
        return state, {k: jnp.stack([r[k] for r in rows]) for k in rows[0]}

    def params(self, state: dict[str, Any]) -> Any:
        return state["params"]

    def policy(self, state: dict[str, Any]) -> RandomPolicy:
        return RandomPolicy()

    def env_wrapper(self, state: dict[str, Any]) -> None:
        return None
