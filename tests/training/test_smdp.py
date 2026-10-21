"""Tests for the top-level SMDP table on a hand-computed 3-skill toy.

Toy: keys k0 = 3 and k1 = 5 of an 8-row table, gamma 0.9, alpha 0.5, the table zero
except ``Q[k1] = [2, 0, 1]``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.training.progress_key import hash_fields
from wut_murl_craftax.training.skills import decode_obs, key_fields, low_stats
from wut_murl_craftax.training.smdp import (
    MCRecords,
    accumulate,
    init_table,
    mc_init,
    mc_push,
    mc_update,
    select_skill,
    smdp_update,
    top_key,
)

K0, K1, GAMMA, ALPHA = 3, 5, 0.9, 0.5
ALL = jnp.ones(3, bool)


def toy() -> jax.Array:
    return init_table(3, 3, 0.0).at[K1].set(jnp.array([2.0, 0.0, 1.0]))


def env_a() -> tuple[jax.Array, jax.Array]:
    """``o0`` from k0 for 3 steps with native rewards 0, 0, 1."""
    ret, disc = jnp.float32(0.0), jnp.float32(1.0)
    for r in (0.0, 0.0, 1.0):
        ret, disc = accumulate(ret, disc, jnp.float32(r), GAMMA)
    return ret, disc


def update(
    q: jax.Array,
    ret: list[float],
    disc: list[float],
    done: list[bool],
    ended: list[bool],
    avail: jax.Array = ALL,
    k: list[int] | None = None,
    o: list[int] | None = None,
) -> jax.Array:
    n = len(ret)
    return smdp_update(
        q,
        jnp.array(k or [K0] * n),
        jnp.array(o or [0] * n),
        jnp.array(ret, jnp.float32),
        jnp.array(disc, jnp.float32),
        jnp.full(n, K1),
        jnp.tile(avail, (n, 1)),
        jnp.array(done),
        jnp.array(ended),
        ALPHA,
    )


def test_init_table() -> None:
    q = init_table(4, 25, 10.0)
    assert q.shape == (16, 25)
    assert q.dtype == jnp.float32
    assert float(q.min()) == float(q.max()) == 10.0
    with pytest.raises(ValueError, match="envs"):
        init_table(3, 25, 10.0, envs=8)  # rows would be read as the env axis


def test_accumulator() -> None:
    ret, disc = env_a()
    assert float(ret) == pytest.approx(0.81)
    assert float(disc) == pytest.approx(0.729)


def test_update_bootstraps_from_the_best_available_skill() -> None:
    ret, disc = env_a()
    q = update(toy(), [float(ret)], [float(disc)], [False], [True])
    assert float(q[K0, 0]) == pytest.approx(0.5 * 2.268)
    masked = jnp.array([False, True, True])
    q = update(toy(), [0.81], [0.729], [False], [True], avail=masked)
    assert float(q[K0, 0]) == pytest.approx(0.7695)


def test_death_does_not_bootstrap() -> None:
    q = update(toy(), [0.81], [0.729], [True], [True])
    assert float(q[K0, 0]) == pytest.approx(0.405)


def test_same_pair_in_one_step_takes_the_mean_td() -> None:
    q = update(toy(), [0.81, 0.0], [0.729, 0.729], [False, True], [True, True])
    assert float(q[K0, 0]) == pytest.approx(0.567)  # a plain scatter-add: 1.134


def test_running_skills_leave_the_table_unchanged() -> None:
    q0 = toy()
    q = update(q0, [0.81, 5.0], [0.729, 0.5], [False, False], [False, False])
    np.testing.assert_array_equal(q, q0)
    # an unfinished env on the same pair neither updates nor dilutes the mean
    q = update(q0, [0.81, 5.0], [0.729, 0.5], [False, False], [True, False])
    assert float(q[K0, 0]) == pytest.approx(1.134)
    mixed = update(q0, [0.81, 9.0], [0.729, 1.0], [True, False], [True, False],
                   k=[K0, K1], o=[0, 2])  # fmt: skip
    np.testing.assert_array_equal(np.asarray(mixed)[K1], np.asarray(q0)[K1])


def push(rec: MCRecords, ended: list[bool], k: list[int], o: list[int],
         r: list[float], disc: list[float]) -> MCRecords:  # fmt: skip
    return mc_push(rec, *(jnp.array(x) for x in (ended, k, o, r, disc)))


def test_monte_carlo_returns_without_bootstrap() -> None:
    rec = mc_init(envs=1, length=4)
    rec = push(rec, [True], [K0], [0], [0.81], [0.729])
    rec = push(rec, [True], [K1], [1], [1.0], [0.9])
    q, rec = mc_update(toy(), rec, jnp.array([True]), ALPHA)
    assert float(q[K0, 0]) == pytest.approx(0.7695)  # target 0.81 + 0.729 x 1
    assert float(q[K1, 1]) == pytest.approx(0.5)  # target 1
    assert int(rec.n[0]) == 0


def test_monte_carlo_waits_for_the_episode_end() -> None:
    rec = mc_init(envs=2, length=4)
    rec = push(rec, [True, False], [K0, K0], [0, 0], [1.0, 1.0], [0.5, 0.5])
    q, rec2 = mc_update(toy(), rec, jnp.array([False, False]), ALPHA)
    np.testing.assert_array_equal(q, toy())
    np.testing.assert_array_equal(rec2.n, [1, 0])


def test_monte_carlo_keeps_the_last_records_when_full() -> None:
    """An episode longer than the buffer: the oldest records are dropped, the kept
    ones still get their exact returns."""
    rec = mc_init(envs=1, length=2)
    for k, o, r in ((K0, 0, 7.0), (K0, 1, 0.81), (K1, 1, 1.0)):
        rec = push(rec, [True], [k], [o], [r], [0.729])
    q, _ = mc_update(toy(), rec, jnp.array([True]), ALPHA)
    assert float(q[K0, 0]) == 0.0
    assert float(q[K0, 1]) == pytest.approx(0.7695)
    assert float(q[K1, 1]) == pytest.approx(0.5)


def test_select_skill_is_masked_epsilon_greedy() -> None:
    q = toy()
    avail = jnp.array([False, True, True])
    assert int(select_skill(q, jnp.int32(K1), avail, 0.0, jax.random.key(0))) == 2
    assert int(select_skill(q, jnp.int32(K1), ALL, 0.0, jax.random.key(0))) == 0
    keys = jax.random.split(jax.random.key(1), 10_000)
    draws = jax.vmap(lambda key: select_skill(q, jnp.int32(K1), avail, 1.0, key))(keys)
    assert set(np.unique(np.asarray(draws)).tolist()) == {1, 2}


def test_select_skill_breaks_ties_at_random() -> None:
    """With the optimistic start every entry is equal; greedy picks any of them."""
    q = init_table(3, 3, 10.0)
    keys = jax.random.split(jax.random.key(2), 1000)
    draws = jax.vmap(lambda key: select_skill(q, jnp.int32(0), ALL, 0.0, key))(keys)
    assert set(np.unique(np.asarray(draws)).tolist()) == {0, 1, 2}


def test_top_key_hashes_the_fields_and_low_stat_flags() -> None:
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    s = env.reset(jax.random.key(0), env.default_params)[1]
    v, low = (
        decode_obs(env.get_obs(s)),
        decode_obs(env.get_obs(s.replace(player_food=2))),
    )
    flags = low_stats(v).astype(jnp.int32)
    want = hash_fields(jnp.concatenate([key_fields(v, "log2"), flags]), 20)
    assert int(top_key(v, "log2", 20)) == int(want)
    assert int(top_key(low, "log2", 20)) != int(top_key(v, "log2", 20))
    no_flags = top_key(v, "log2", 20, low_flags=False)
    assert int(no_flags) == int(hash_fields(key_fields(v, "log2"), 20))
    assert int(top_key(low, "log2", 20, low_flags=False)) == int(no_flags)


def test_jit_and_vmap() -> None:
    q = jax.jit(smdp_update)(
        toy(), jnp.array([K0, K0]), jnp.array([0, 0]), jnp.array([0.81, 0.0]),
        jnp.array([0.729, 0.729]), jnp.array([K1, K1]), jnp.ones((2, 3), bool),
        jnp.array([False, True]), jnp.array([True, True]), ALPHA,
    )  # fmt: skip
    assert float(q[K0, 0]) == pytest.approx(0.567)
    keys = jax.random.split(jax.random.key(3), 4)
    ks = jnp.array([K1, K1, 0, 0])
    picks = jax.jit(jax.vmap(select_skill, in_axes=(None, 0, None, None, 0)))(
        toy(), ks, ALL, 0.0, keys
    )
    assert picks.tolist()[:2] == [0, 0]
