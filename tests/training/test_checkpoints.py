"""Tests for the checkpoint schedule, params round trip and full-state saves."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from wut_murl_craftax.training import checkpoints
from wut_murl_craftax.training.checkpoints import (
    checkpoint_schedule,
    drop_missing_checkpoints,
    load_index,
    load_params,
    load_state,
    save_checkpoint,
    save_state,
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


def _state() -> dict[str, Any]:
    """A trainer-like state: params, optax state, env state, keys, stats, numpy logs."""
    params = {"dense": {"kernel": jnp.arange(6.0).reshape(2, 3), "bias": jnp.ones(3)}}
    tx = optax.adam(1e-3)
    opt = tx.update(params, tx.init(params), params)[1]
    env = {
        "pos": jnp.array([[1, 2], [3, 4]], jnp.int32),
        "alive": jnp.array([True, False]),
        "map": jnp.arange(8, dtype=jnp.uint8).reshape(2, 4),
        "half": jnp.array([0.5, 1.5], jnp.bfloat16),
    }
    return {
        "params": params,
        "opt": opt,
        "env": env,
        "key": jax.random.split(jax.random.key(3), 2),
        "rbg": jax.random.key(4, impl="rbg"),
        "batch_stats": {"mean": jnp.full(3, 0.25), "var": jnp.full(3, 2.0)},
        "episodes": {"ret": np.array([1.5, 2.5], np.float32), "n": np.int64(7)},
        "rows": np.random.default_rng(0).normal(size=(4, 3)),
        "update": 2,
    }


def _zeros_like(state: dict[str, Any]) -> dict[str, Any]:
    """Same structure and leaf types, other values (keys from another seed)."""

    def zero(x: Any) -> Any:
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            key = jax.random.key(99, impl=jax.random.key_impl(x))
            return jax.random.split(key, x.shape) if x.shape else key
        if isinstance(x, np.ndarray):
            return np.zeros((0, *x.shape[1:]), x.dtype)
        return jnp.zeros_like(x) if isinstance(x, jax.Array) else type(x)(0)

    return jax.tree.map(zero, state)


def _assert_identical(got: Any, want: Any) -> None:
    got_leaves, got_def = jax.tree.flatten(got)
    want_leaves, want_def = jax.tree.flatten(want)
    assert got_def == want_def
    for g, w in zip(got_leaves, want_leaves, strict=True):
        assert type(g) is type(w)
        if isinstance(w, jax.Array) and jnp.issubdtype(w.dtype, jax.dtypes.prng_key):
            assert jax.random.key_impl(g) == jax.random.key_impl(w)
            g, w = jax.random.key_data(g), jax.random.key_data(w)
        g, w = np.asarray(g), np.asarray(w)
        assert g.dtype == w.dtype
        assert g.shape == w.shape
        assert g.tobytes() == w.tobytes()


def test_state_round_trips_bit_for_bit(tmp_path: Path) -> None:
    state = _state()
    save_state(tmp_path, state)
    loaded = load_state(tmp_path, _zeros_like(state))
    _assert_identical(loaded, state)
    # the restored keys draw what the saved ones draw
    np.testing.assert_array_equal(
        jax.random.uniform(loaded["key"][1]), jax.random.uniform(state["key"][1])
    )


def test_a_save_killed_half_way_leaves_the_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _state()
    save_state(tmp_path, first)

    def killed(file: Any, **_: Any) -> None:
        file.write(b"PK\x03\x04 half a zip")
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoints.np, "savez", killed)
    second = {**first, "update": 3}
    with pytest.raises(KeyboardInterrupt):
        save_state(tmp_path, second)
    _assert_identical(load_state(tmp_path, _zeros_like(first)), first)
    monkeypatch.undo()
    save_state(tmp_path, second)
    assert load_state(tmp_path, _zeros_like(first))["update"] == 3


def test_load_state_refuses_another_structure(tmp_path: Path) -> None:
    save_state(tmp_path, {"a": jnp.ones(2)})
    with pytest.raises(ValueError, match="template has 2"):
        load_state(tmp_path, {"a": jnp.ones(2), "b": jnp.ones(2)})


@pytest.mark.parametrize(
    ("like", "error"),
    [
        ({"a": jnp.ones(3), "b": jnp.zeros(2, jnp.int32)}, r"\['a'\].*\(2,\).*\(3,\)"),
        ({"a": jnp.ones(2), "b": jnp.zeros(2)}, r"\['b'\].*int32.*float32"),
        ({"a": jnp.ones(2), "c": jnp.zeros(2, jnp.int32)}, "other tree"),
        ({"a": np.ones((0, 4)), "b": jnp.zeros(2, jnp.int32)}, r"\['a'\].*\(2,\)"),
    ],
)
def test_load_state_refuses_other_shapes_dtypes_and_fields(
    tmp_path: Path, like: dict[str, Any], error: str
) -> None:
    """Same leaf count, but the code that saved it had another state layout."""
    save_state(tmp_path, {"a": jnp.ones(2), "b": jnp.zeros(2, jnp.int32)})
    with pytest.raises(ValueError, match=error):
        load_state(tmp_path, like)


def test_a_state_saved_without_the_tree_paths_still_loads(tmp_path: Path) -> None:
    """Runs saved before the paths were stored resume; leaves are still checked."""
    state = _state()
    save_state(tmp_path, state)
    path = tmp_path / "state" / "state.npz"
    with np.load(path) as z:
        old = {k: z[k] for k in z.files if k != "paths"}
    np.savez(path, **old)
    _assert_identical(load_state(tmp_path, _zeros_like(state)), state)


def test_a_checkpoint_killed_half_way_leaves_the_previous_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Params and index are renamed into place: a kill never leaves half a file."""
    params = {"w": jnp.ones(3)}
    save_checkpoint(tmp_path, 0, 0, 0.0, params)
    before = (tmp_path / "ckpt" / "index.json").read_bytes()

    def killed(*_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoints.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(tmp_path, 1, 10, 1.0, params)
    assert not (tmp_path / "ckpt" / "00001.msgpack").exists()
    (tmp_path / "ckpt" / "00000.msgpack").unlink()
    with pytest.raises(KeyboardInterrupt):  # cleaning the index is atomic too
        drop_missing_checkpoints(tmp_path)
    assert (tmp_path / "ckpt" / "index.json").read_bytes() == before


def test_params_checkpoints_and_state_live_apart(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, 0, 0, 0.0, {"w": jnp.ones(2)})
    save_state(tmp_path, _state())
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == [
        "00000.msgpack",
        "index.json",
    ]
    assert [p.name for p in (tmp_path / "state").iterdir()] == ["state.npz"]
