"""Shared fixtures of the evaluation tests: real start states of both games."""

from typing import Any

import jax
import pytest
from craftax.craftax_env import make_craftax_env_from_name


def _start(name: str) -> Any:
    env = make_craftax_env_from_name(name, auto_reset=False)
    return jax.jit(env.reset)(jax.random.PRNGKey(0), env.default_params)[1]


@pytest.fixture(scope="session")
def craftax_start() -> Any:
    """Craftax state after reset with key 0 (player at (24, 24) on the overworld)."""
    return _start("Craftax-Symbolic-v1")


@pytest.fixture(scope="session")
def classic_start() -> Any:
    """Craftax-Classic state after reset with key 0."""
    return _start("Craftax-Classic-Symbolic-v1")
