"""Tests for the Dyna-Q skill-outcome model and its planning updates on toy chains."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.training.dyna_q import (
    ModelState,
    Outcome,
    init_model,
    model_accuracy,
    pack_mask,
    unpack_mask,
    write,
)

S = 3
"""Skills of the toy model."""


def outcome(
    k: list[int],
    o: list[int],
    ret: list[float] | None = None,
    k_next: list[int] | None = None,
    done: list[bool] | None = None,
    disc: float = 0.95,
) -> Outcome:
    n = len(k)
    return Outcome(
        k=jnp.array(k, jnp.int32),
        o=jnp.array(o, jnp.int32),
        ret=jnp.array(ret or [0.0] * n, jnp.float32),
        disc=jnp.full(n, disc, jnp.float32),
        k_next=jnp.array(k_next or [0] * n, jnp.int32),
        done=jnp.array(done or [False] * n),
        avail=jnp.ones((n, S), bool),
    )


def put(m: ModelState, rec: Outcome, t: float = 0.0, ended: Any = None) -> ModelState:
    ended = jnp.ones(rec.k.shape, bool) if ended is None else jnp.asarray(ended)
    return write(m, rec, ended, jnp.float32(t))


def test_init_model_shapes_and_env_guard() -> None:
    m = init_model(4, S, 8, 4)
    assert m.pair_id.shape == (16, S)
    assert (np.asarray(m.pair_id) == -1).all()
    assert m.m_R.shape == m.m_next.shape == (8, 4)
    assert int(m.n_pairs) == 0
    assert not bool(m.full)
    with pytest.raises(ValueError, match="envs"):
        init_model(3, S, 16, 4, envs=8)  # pair index rows read as the env axis
    with pytest.raises(ValueError, match="envs"):
        init_model(4, S, 8, 4, envs=8)  # pair capacity read as the env axis


def test_first_visits_take_slots_in_order_and_repeats_reuse_them() -> None:
    m = init_model(4, S, 8, 4)
    m = put(m, outcome([3], [0]))
    m = put(m, outcome([5], [2]))
    m = put(m, outcome([3, 7, 5], [1, 0, 2]))
    assert int(m.n_pairs) == 4
    ids = np.asarray(m.pair_id)
    assert (ids[3, 0], ids[5, 2], ids[3, 1], ids[7, 0]) == (0, 1, 2, 3)
    np.testing.assert_array_equal(np.asarray(m.pair_k)[:4], [3, 5, 3, 7])
    np.testing.assert_array_equal(np.asarray(m.pair_o)[:4], [0, 2, 1, 0])
    np.testing.assert_array_equal(np.asarray(m.count)[:4], [1, 2, 1, 1])


def test_a_ring_keeps_the_last_m_outcomes() -> None:
    m = init_model(4, S, 8, 8)
    for i in range(10):
        m = put(m, outcome([2], [1], ret=[float(i)]))
    p = int(m.pair_id[2, 1])
    assert int(m.count[p]) == 10
    np.testing.assert_array_equal(m.m_R[p], [8, 9, 2, 3, 4, 5, 6, 7])


def test_two_envs_on_one_new_pair_share_the_pair_not_the_slot() -> None:
    m = put(init_model(4, S, 8, 4), outcome([6, 6], [2, 2], ret=[1.0, 2.0]))
    assert int(m.n_pairs) == 1
    assert int(m.count[0]) == 2
    np.testing.assert_array_equal(m.m_R[0, :2], [1.0, 2.0])


def test_unfinished_skills_write_nothing_and_last_tried_is_the_latest() -> None:
    m = init_model(4, S, 8, 4)
    m = put(m, outcome([1, 2], [0, 0]), t=3.0, ended=[True, False])
    assert int(m.n_pairs) == 1
    assert int(m.pair_id[2, 0]) == -1
    m = put(m, outcome([1], [0]), t=7.5)
    assert float(m.last_tried[0]) == 7.5


def test_reaching_capacity_sets_full_and_drops_new_pairs() -> None:
    m = put(init_model(4, S, 2, 4), outcome([1, 2, 3], [0, 0, 0]))
    assert int(m.n_pairs) == 2
    assert bool(m.full)
    assert int(m.pair_id[3, 0]) == -1
    assert int(m.count.sum()) == 2


def test_masks_pack_into_one_int() -> None:
    avail = jnp.array([[True, False, True], [False, False, False]])
    packed = pack_mask(avail)
    assert packed.dtype == jnp.int32
    np.testing.assert_array_equal(packed, [5, 0])
    np.testing.assert_array_equal(unpack_mask(packed, S), avail)
    full = jnp.ones(25, bool)
    np.testing.assert_array_equal(unpack_mask(pack_mask(full), 25), full)


def test_model_accuracy() -> None:
    m = put(init_model(4, S, 8, 4), outcome([1], [0], k_next=[5]))
    same = model_accuracy(m, outcome([1], [0], k_next=[5]), jnp.array([True]))
    new = model_accuracy(m, outcome([1], [0], k_next=[6]), jnp.array([True]))
    unseen = model_accuracy(m, outcome([2], [0], k_next=[5]), jnp.array([True]))
    assert [int(x) for x in same] == [1, 1]
    assert [int(x) for x in new] == [0, 1]
    assert [int(x) for x in unseen] == [0, 1]


def test_write_jits_and_vmaps() -> None:
    m, rec = init_model(4, S, 8, 4), outcome([3, 3, 9], [0, 0, 1], ret=[1.0, 2.0, 3.0])
    ended, t = jnp.array([True, True, False]), jnp.float32(2.0)
    eager = write(m, rec, ended, t)
    jitted = jax.jit(write)(m, rec, ended, t)
    for x, y in zip(jax.tree.leaves(eager), jax.tree.leaves(jitted), strict=True):
        np.testing.assert_array_equal(x, y)
    two = jax.tree.map(lambda x: jnp.stack([x, x]), m)
    both = jax.vmap(write, in_axes=(0, None, None, None))(two, rec, ended, t)
    np.testing.assert_array_equal(both.m_R[1], eager.m_R)
    acc = jax.jit(model_accuracy)(eager, rec, ended)
    assert [int(x) for x in acc] == [2, 2]
