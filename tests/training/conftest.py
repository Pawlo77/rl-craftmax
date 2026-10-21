"""Shared fixtures of the training tests: recurrent configs, equality, CPU runs."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from wut_murl_craftax.training.ppo_rnn import PPORNNConfig
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig

TEST_ENV = os.environ.get("CRAFTAX_TEST_ENV", "Craftax-Classic-Symbolic-v1")
"""Env of the recurrent tests; Craftax-Symbolic reruns them at the real scale."""


@pytest.fixture(scope="session")
def rnn_cfg() -> Callable[..., PPORNNConfig]:
    """Tiny PPO-RNN config (8 envs x 8 steps, published widths, 4 updates of budget)."""

    def make(**overrides: Any) -> PPORNNConfig:
        base = {"env": TEST_ENV, "envs": 8, "num_steps": 8, "width": 512}
        base |= {"minibatches": 2, "steps": 4 * 8 * 8}
        return PPORNNConfig(**(base | overrides))

    return make


@pytest.fixture(scope="session")
def pqn_cfg() -> Callable[..., PQNRNNConfig]:
    """Tiny PQN-RNN config (8 envs x 8 steps, the repo widths, 4 updates of budget)."""

    def make(**overrides: Any) -> PQNRNNConfig:
        base = {"env": TEST_ENV, "envs": 8, "num_steps": 8, "minibatches": 2}
        return PQNRNNConfig(**(base | {"steps": 4 * 8 * 8} | overrides))

    return make


def _same(a: Any, b: Any) -> bool:

    def raw(x: Any) -> bytes:
        if isinstance(x, jax.Array) and jax.dtypes.issubdtype(
            x.dtype, jax.dtypes.prng_key
        ):
            x = jax.random.key_data(x)
        return np.asarray(x).tobytes()

    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        raw(x) == raw(y) for x, y in zip(la, lb, strict=True)
    )


@pytest.fixture(scope="session")
def same() -> Callable[[Any, Any], bool]:
    """Every leaf equal bit for bit; typed keys compared by their key data."""
    return _same


def _start_on_cpus(script: str, devices: int, out: Path) -> subprocess.Popen[str]:
    flags = f"--xla_force_host_platform_device_count={devices}" if devices > 1 else ""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(out)],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture(scope="session")
def start_on_cpus() -> Callable[[str, int, Path], subprocess.Popen[str]]:
    """Start ``python -c script out`` on ``devices`` host CPUs (JAX fixes the count
    when it starts, hence a subprocess)."""
    return _start_on_cpus
