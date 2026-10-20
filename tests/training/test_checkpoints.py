"""Tests for the checkpoint schedule and params round trip."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from wut_murl_craftax.training.checkpoints import (
    checkpoint_schedule,
    load_index,
    load_params,
    save_checkpoint,
)


def test_schedule_is_log_spaced_with_first_and_last() -> None:
    assert checkpoint_schedule(122, 12) == [0, 1, 2, 4, 6, 9, 14, 21, 33, 51, 79, 122]


def test_schedule_tiny_run_keeps_unique_updates() -> None:
    sched = checkpoint_schedule(3, 12)
    assert sched == [0, 1, 2, 3]


def test_params_round_trip(tmp_path: Path) -> None:
    params = {"dense": {"kernel": jnp.arange(6.0).reshape(2, 3), "bias": jnp.ones(3)}}
    save_checkpoint(tmp_path, 0, 0, 0.0, params)
    save_checkpoint(tmp_path, 5, 640, 1.5, jax.tree.map(lambda x: x * 2, params))
    index = load_index(tmp_path)
    assert [entry["update"] for entry in index] == [0, 5]
    assert index[1]["env_steps"] == 640
    template = jax.tree.map(jnp.zeros_like, params)
    loaded = load_params(tmp_path, 5, template)
    np.testing.assert_array_equal(
        loaded["dense"]["kernel"], params["dense"]["kernel"] * 2
    )
