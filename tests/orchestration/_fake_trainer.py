"""A deterministic 3-update trainer with the shape of a registry entry."""

import os
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wut_murl_craftax.training.records import LOSS_COLUMNS

STEPS, ENVS, N_ACH = 4, 2, 22
"""Rollout shape of one update; 22 achievements as on Craftax-Classic."""
STEPS_PER_UPDATE = 333_333
"""A 1m budget is 3 updates by default."""


class FakeTrainer:
    """Params random walk, Adam-like moments, env counters, typed keys, BatchRenorm.

    ``signal_at`` sends that signal to this process during the given update, as
    Ctrl-C or a scheduler's SIGTERM would in the middle of a chunk.
    """

    def __init__(
        self,
        signal_at: tuple[int, int] | None = None,
        steps_per_update: int = STEPS_PER_UPDATE,
    ) -> None:
        self.signal_at = signal_at
        self.spu = steps_per_update
        self.updates: list[int] = []
        self.cpu_devices = 0
        self.init_steps: list[int | None] = []

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        return self.spu

    def init(self, config: Mapping[str, Any], key: jax.Array) -> dict[str, Any]:
        self.cpu_devices = len(jax.devices("cpu"))
        self.init_steps.append(config.get("steps"))
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
        }
        return new, metrics

    def train_chunk(
        self, state: dict[str, Any], n_updates: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = []
        for _ in range(n_updates):
            state, metrics = self._update(state)
            rows.append(metrics)
            self.updates.append(int(state["opt"]["count"]))
            if self.signal_at and self.signal_at[0] == self.updates[-1]:
                os.kill(os.getpid(), self.signal_at[1])
        return state, {k: jnp.stack([r[k] for r in rows]) for k in rows[0]}

    def params(self, state: dict[str, Any]) -> Any:
        return state["params"]

    def policy(self, state: dict[str, Any]) -> None:
        return None
