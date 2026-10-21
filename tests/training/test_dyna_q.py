"""Tests for the Dyna-Q skill-outcome model and its planning updates on toy chains."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.training.dyna_q import (
    ModelState,
    Outcome,
    full_backup,
    init_model,
    model_accuracy,
    pack_mask,
    plan,
    sample_backup,
    uniform_batch,
    unpack_mask,
    write,
)
from wut_murl_craftax.training.smdp import init_table, smdp_update

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


# --- planning on toy deterministic chains (one skill per key, tau = 1) ---

G2, ONE = 0.95, jnp.ones((1, 2), bool)


PLAN = {"batch": 1024, "theta": 1e-4, "tol": 1e-3, "max_sweeps": 50}
"""The plan's fixed planning settings; `plan` takes them from the config."""


def plan_(*args: Any, **kw: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
    return plan(*args, **(PLAN | kw))


def one(k: int, o: int, r: float = 0.0, k_next: int = 0, done: bool = False) -> Outcome:
    """A single finished skill (gamma^tau 0.95, every next skill available)."""
    a = jnp.array
    return Outcome(a([k]), a([o]), a([r]), a([G2]), a([k_next]), a([done]), ONE)


def chain(n_keys: int, alpha: float) -> tuple[jax.Array, ModelState]:
    """Real episode ``k0 -> ... -> k_{n-1} -> goal``: table after the direct updates
    in visit order, and the model of its outcomes (skill 0, R 1 at the goal)."""
    q, m = init_table(4, 2, 0.0), init_model(4, 2, 8, 4)
    for i in range(n_keys):
        last = i == n_keys - 1
        rec = one(i, 0, 1.0 if last else 0.0, i + 1, last)
        m = write(m, rec, jnp.array([True]), jnp.float32(i))
        q = smdp_update(q, rec.k, rec.o, rec.ret, rec.disc, rec.k_next, rec.avail,
                        rec.done, jnp.array([True]), alpha)  # fmt: skip
    return q, m


def test_book_corridor_backs_up_one_key_per_planning_step() -> None:
    q, m = chain(3, 0.1)  # A -> B -> C -> G
    assert float(q[2, 0]) == pytest.approx(0.1)
    assert float(q[1, 0]) == float(q[0, 0]) == 0.0
    for key in (1, 0):  # B, then A
        p = m.pair_id[key, 0][None]
        q = sample_backup(q, m, p, jnp.array([0]), jnp.array([True]), 0.0, 0.0, 0.1)
    assert round(float(q[1, 0]), 5) == 0.0095
    assert round(float(q[0, 0]), 5) == 0.00090


def test_prioritized_sweeping_carries_the_goal_back_in_one_episode() -> None:
    q, m = chain(5, 1.0)
    np.testing.assert_array_equal(q[:5, 0], [0, 0, 0, 0, 1])  # model-free, n = 0
    order = []
    for i in range(4):
        q2, stats = plan_(q, m, 1, "prioritized", 0.0, 5.0, jax.random.key(i),
                         alpha=1.0, batch=1)  # fmt: skip
        changed = np.flatnonzero(np.asarray(q2[:, 0] != q[:, 0]))
        order.append(int(changed[0]))
        assert int(stats["backups"]) == 1
        q = q2
    assert order == [3, 2, 1, 0]
    np.testing.assert_allclose(
        q[:5, 0], [0.81450625, 0.857375, 0.9025, 0.95, 1.0], rtol=1e-6
    )
    q0, m0 = chain(5, 1.0)
    q4, _ = plan_(q0, m0, 4, "prioritized", 0.0, 5.0, jax.random.key(0), alpha=1.0,
                 batch=1)  # fmt: skip
    np.testing.assert_array_equal(q4, q)


def test_value_iteration_converges_to_the_discounted_goal() -> None:
    q, m = chain(5, 1.0)
    q, stats = plan_(q, m, 0, "value_iteration", 0.0, 5.0, jax.random.key(0),
                    alpha=1.0)  # fmt: skip
    want = [G2 ** (4 - i) for i in range(5)]
    np.testing.assert_allclose(q[:5, 0], want, atol=1e-3)
    assert 4 <= int(stats["sweeps"]) <= 50


def test_full_backup_averages_the_stored_outcomes_under_the_next_mask() -> None:
    m = init_model(4, 2, 8, 4)
    both = Outcome(
        jnp.array([1, 1, 5]), jnp.array([0, 0, 0]), jnp.array([1.0, 0.0, 0.0]),
        jnp.array([G2] * 3), jnp.array([0, 0, 6]), jnp.array([True, True, False]),
        jnp.array([[True, True], [True, True], [False, True]]),
    )  # fmt: skip
    m = write(m, both, jnp.ones(3, bool), jnp.float32(0.0))
    q = init_table(4, 2, 0.0).at[6].set(jnp.array([5.0, 1.0]))
    pairs = jnp.array([m.pair_id[1, 0], m.pair_id[5, 0]])
    q = full_backup(q, m, pairs, jnp.array([True, True]), 0.0, 0.0, 1.0)
    assert float(q[1, 0]) == pytest.approx(0.5)
    assert float(q[5, 0]) == pytest.approx(G2 * 1.0)  # o0 at k6 is masked out


def test_dyna_q_plus_bonus_on_simulated_backups_only() -> None:
    m = init_model(4, 2, 8, 4)
    rec = one(2, 1, done=True)
    m = write(m, rec, jnp.array([True]), jnp.float32(4.0))
    q0 = init_table(4, 2, 0.0)
    q = sample_backup(q0, m, jnp.array([0]), jnp.array([0]), jnp.array([True]), 0.01,
                      20.0, 1.0)  # fmt: skip
    assert float(q[2, 1]) == pytest.approx(0.04)
    planned, _ = plan_(q0, m, 1, "uniform", 0.01, 20.0, jax.random.key(0), alpha=1.0)
    assert float(planned[2, 1]) == pytest.approx(0.04)
    assert "kappa" not in inspect.signature(smdp_update).parameters
    real = smdp_update(q0, rec.k, rec.o, rec.ret, rec.disc, rec.k_next, rec.avail,
                       rec.done, jnp.array([True]), 1.0)  # fmt: skip
    assert float(real[2, 1]) == 0.0


@pytest.mark.parametrize("variant", ["uniform", "prioritized"])
def test_no_planning_is_the_smdp_skills_table(variant: str) -> None:
    q, m = chain(5, 0.5)
    q2, stats = plan_(q, m, 0, variant, 0.03, 5.0, jax.random.key(1), alpha=0.5)
    assert np.asarray(q2).tobytes() == np.asarray(q).tobytes()
    assert int(stats["backups"]) == 0


def test_uniform_samples_visited_pairs_and_filled_slots() -> None:
    m = init_model(4, 2, 8, 4)
    for i, n in enumerate((1, 3, 6)):  # pairs with 1, 3 and 6 (> M) outcomes
        for _ in range(n):
            m = write(m, one(i, 1), jnp.array([True]), jnp.float32(0.0))
    p, slot, valid = uniform_batch(m, jax.random.key(2), 10_000)
    p, slot = np.asarray(p), np.asarray(slot)
    assert bool(valid.all())
    assert set(p.tolist()) == {0, 1, 2}
    assert (slot < np.minimum(np.asarray(m.count)[p], 4)).all()
    assert set(slot[p == 2].tolist()) == {0, 1, 2, 3}
    empty = uniform_batch(init_model(4, 2, 8, 4), jax.random.key(2), 4)[2]
    assert not bool(empty.any())


def test_plan_jits_with_a_traced_backup_count() -> None:
    q, m = chain(5, 1.0)
    run = jax.jit(plan, static_argnames=("variant", "batch"))
    for variant in ("uniform", "prioritized", "value_iteration"):
        kw = PLAN | {"batch": 2}
        got, _ = run(q, m, jnp.int32(3), variant, 0.0, 5.0, jax.random.key(0),
                     alpha=1.0, **kw)  # fmt: skip
        eager, _ = plan_(q, m, 3, variant, 0.0, 5.0, jax.random.key(0), alpha=1.0,
                        batch=2)  # fmt: skip
        np.testing.assert_allclose(got, eager, rtol=1e-6)


def test_prioritized_sweeping_skips_pairs_below_theta() -> None:
    q, m = chain(5, 1.0)
    q = q.at[3, 0].set(0.95 - 5e-5)  # the only priority left is 5e-5
    for i in (2, 1, 0):  # the rest already consistent with it
        q = q.at[i, 0].set(0.95 * q[i + 1, 0])
    key = jax.random.key(0)
    gated, stats = plan_(q, m, 4, "prioritized", 0.0, 5.0, key, alpha=1.0)
    assert int(stats["backups"]) == 0
    np.testing.assert_array_equal(gated, q)
    loose, stats = plan_(q, m, 1, "prioritized", 0.0, 5.0, key, alpha=1.0, theta=1e-5)
    assert int(stats["backups"]) == 1
    assert float(loose[3, 0]) == pytest.approx(0.95)


def test_value_iteration_stops_at_max_sweeps() -> None:
    q, m = chain(5, 1.0)
    q, stats = plan_(q, m, 0, "value_iteration", 0.0, 5.0, jax.random.key(0),
                     alpha=1.0, max_sweeps=2)  # fmt: skip
    assert int(stats["sweeps"]) == 2
    np.testing.assert_allclose(q[:5, 0], [0, 0, 0.9025, 0.95, 1], rtol=1e-6)


def test_plan_needs_every_planning_setting() -> None:
    q, m = chain(2, 1.0)
    with pytest.raises(TypeError, match="theta"):
        plan(q, m, 1, "uniform", 0.0, 0.0, jax.random.key(0), alpha=1.0, batch=4,
             tol=1e-3, max_sweeps=50)  # fmt: skip


def test_more_than_m_envs_on_one_pair_keep_the_last_m_whole() -> None:
    """12 envs end one pair with M = 8: the last 8 envs fill the ring, each row from
    one env (R, next key and done all encode the env index)."""
    n = 12
    rec = Outcome(
        k=jnp.full(n, 4), o=jnp.full(n, 1), ret=jnp.arange(n, dtype=jnp.float32),
        disc=jnp.full(n, G2), k_next=jnp.arange(n), done=jnp.arange(n) % 2 == 0,
        avail=jnp.ones((n, 2), bool),
    )  # fmt: skip
    m = write(init_model(4, 2, 8, 8), rec, jnp.ones(n, bool), jnp.float32(0.0))
    p = int(m.pair_id[4, 1])
    assert int(m.count[p]) == n
    r = np.asarray(m.m_R[p]).astype(int)
    assert sorted(r.tolist()) == list(range(4, 12))
    np.testing.assert_array_equal(np.asarray(m.m_next[p]), r)
    np.testing.assert_array_equal(np.asarray(m.m_done[p]), r % 2 == 0)
    # ring order: slot (count_before + rank) % M, as twelve sequential writes
    np.testing.assert_array_equal(r, [8, 9, 10, 11, 4, 5, 6, 7])


def test_pack_mask_refuses_more_than_31_skills() -> None:
    with pytest.raises(ValueError, match="31"):
        pack_mask(jnp.ones(32, bool))


# --- planning statistics (the dyna/* metrics) ---


def test_plan_reports_backups_td_and_the_bonus() -> None:
    """One pair 16 units old, R 0, terminal, kappa 0.01: target 0.04 from Q 0."""
    m = write(init_model(4, 2, 8, 4), one(2, 1, done=True), jnp.array([True]), 4.0)
    q, stats = plan_(init_table(4, 2, 0.0), m, 1, "uniform", 0.01, 20.0,
                     jax.random.key(0), alpha=1.0)  # fmt: skip
    assert float(q[2, 1]) == pytest.approx(0.04)
    assert (int(stats["backups"]), int(stats["td_n"])) == (1, 1)
    assert float(stats["td_abs"]) == pytest.approx(0.04)
    assert float(stats["bonus"]) == pytest.approx(0.04)
    assert float(stats["q_upd"]) == pytest.approx(0.04)
    assert not bool(stats["q_bad"])
    assert (int(stats["sweeps"]), int(stats["above_theta"])) == (0, 0)
    assert float(stats["residual"]) == 0.0


def test_prioritized_stats_count_the_pairs_above_theta_before_planning() -> None:
    q, m = chain(5, 1.0)  # only k3 is off its target (0.95 against 0)
    _, stats = plan_(q, m, 4, "prioritized", 0.0, 5.0, jax.random.key(0), alpha=1.0,
                     batch=1)  # fmt: skip
    assert int(stats["above_theta"]) == 1
    assert int(stats["td_n"]) == 4
    want = 0.95 + 0.9025 + 0.857375 + 0.81450625  # each backup from 0 to its value
    assert float(stats["td_abs"]) == pytest.approx(want, rel=1e-6)
    assert float(stats["q_upd"]) == pytest.approx(0.95)


def test_value_iteration_stats_keep_the_last_change() -> None:
    q, m = chain(5, 1.0)
    _, stats = plan_(q, m, 0, "value_iteration", 0.0, 5.0, jax.random.key(0),
                     alpha=1.0)  # fmt: skip
    sweeps = int(stats["sweeps"])
    assert float(stats["residual"]) < 1e-3
    assert int(stats["td_n"]) == 5 * sweeps  # one stored outcome per pair and sweep
    assert int(stats["backups"]) == 5 * sweeps
    assert float(stats["q_upd"]) == pytest.approx(1.0)


def test_a_non_finite_planned_value_is_reported() -> None:
    m = write(init_model(4, 2, 8, 4), one(1, 0, k_next=3), jnp.array([True]), 0.0)
    q = init_table(4, 2, 0.0).at[3].set(jnp.nan)
    _, stats = plan_(q, m, 1, "uniform", 0.0, 0.0, jax.random.key(0), alpha=1.0)
    assert bool(stats["q_bad"])
    assert float(stats["q_upd"]) == -np.inf


def test_full_backup_equals_sample_backups_of_every_filled_slot() -> None:
    """The sort-free full backup against the duplicate-mean rule of `smdp_update`."""
    rng = np.random.default_rng(0)
    m = init_model(4, S, 16, 4)
    for _ in range(30):
        k, o = rng.integers(0, 6, 5).tolist(), rng.integers(0, S, 5).tolist()
        k_next, done = rng.integers(0, 6, 5).tolist(), (rng.random(5) < 0.2).tolist()
        rec = outcome(k, o, rng.random(5).tolist(), k_next, done)
        m = write(m, rec, jnp.asarray(rng.random(5) < 0.7), jnp.float32(0.0))
    q = jnp.asarray(rng.random((16, S)), jnp.float32)
    p = jnp.arange(16)
    valid = (p < m.n_pairs) & (p % 3 != 0)
    got = full_backup(q, m, p, valid, 0.01, 5.0, 0.3)
    slots = jnp.arange(4)
    mask = (valid[:, None] & (slots < jnp.minimum(m.count, 4)[:, None])).ravel()
    want = sample_backup(q, m, jnp.repeat(p, 4), jnp.tile(slots, 16), mask, 0.01, 5.0,
                         0.3)  # fmt: skip
    assert int(m.n_pairs) > 8
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(got, q)
