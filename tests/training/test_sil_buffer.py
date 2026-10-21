"""Tests for the SIL window buffer: compression, sizes, window copy, masks, FIFO."""

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import StepEvents
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.hooks import HookSpec, Traj
from wut_murl_craftax.training.sil import (
    MAP_DIMS,
    SILConfig,
    SILState,
    buffer_bytes,
    check_memory,
    compress_obs,
    decompress_obs,
    init,
    observe,
)

T, E, A, D = 16, 2, 22, 1345
"""Rollout steps, envs, achievements and obs length of the scripted Classic runs."""
M = MAP_DIMS[D]
L = 128
MB, GB = 1e6, 1e9


def _events(unlocked: jax.Array) -> StepEvents:
    zeros = {
        f.name: jnp.zeros(unlocked.shape[:2]) for f in dataclasses.fields(StepEvents)
    }
    return StepEvents(**(zeros | {"unlocked": unlocked}))


def _traj(start: int, unlocks=(), dones=(), rewards=None) -> Traj:
    """Rollout of global steps ``start..start+T-1``.

    tail[0] and hidden[0] hold the step, tail[1] and hidden[1] the env; the map
    one-hot marks ``step % M``.
    """
    g = jnp.broadcast_to(jnp.arange(start, start + T)[:, None], (T, E))
    env = jnp.broadcast_to(jnp.arange(E)[None], (T, E)).astype(jnp.float32)
    tail = jnp.zeros((T, E, D - M)).at[..., 0].set(g).at[..., 1].set(env)
    obs = jnp.concatenate([jax.nn.one_hot(g % M, M), tail], axis=-1)
    hidden = jnp.zeros((T, E, 512)).at[..., 0].set(g).at[..., 1].set(env)
    unl = np.zeros((T, E, A), bool)
    for step, i, a in unlocks:
        unl[step - start, i, a] = True
    done = np.zeros((T, E), bool)
    for step, i in dones:
        if start <= step < start + T:
            done[step - start, i] = True
    r = jnp.zeros((T, E)) if rewards is None else rewards
    extra = {"r_int": jnp.zeros((T, E)), "values": jnp.zeros((T, E))}
    return Traj(
        obs, g % 17, r, jnp.asarray(done), hidden, _events(jnp.asarray(unl)), extra
    )


_observe = jax.jit(observe)


def _run(cfg: SILConfig, n: int, last=None, dones=()) -> SILState:
    """``n`` rollouts with no unlock, then ``last(start)`` as the next rollout."""
    st = init(cfg, E, A, D)
    for k in range(n):
        st = _observe(st, _traj(k * T, dones=dones))
    return st if last is None else _observe(st, last(n * T))


def _stored(st: SILState) -> np.ndarray:
    return np.flatnonzero(np.asarray(st.buffer["stamp"]) >= 0)


def _obs_of(env_name: str) -> jax.Array:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    key = jax.random.key(0)
    obs, st = env.reset(key, env.default_params)

    def step(c: Any, k: jax.Array) -> Any:
        o, s = c
        a = jax.random.randint(k, (), 0, env.action_space(env.default_params).n)
        o, s, *_ = env.step(k, s, a, env.default_params)
        return (o, s), o

    _, seq = jax.lax.scan(step, (obs, st), jax.random.split(key, 30))
    return jnp.concatenate([obs[None], seq])


@pytest.mark.parametrize(
    ("name", "dim"),
    [("Craftax-Symbolic-v1", 8268), ("Craftax-Classic-Symbolic-v1", 1345)],
)
def test_compress_round_trips_real_observations(name: str, dim: int) -> None:
    obs = _obs_of(name)
    assert obs.shape[-1] == dim
    mp, tail = compress_obs(obs)
    assert (mp.dtype, tail.dtype) == (jnp.uint8, jnp.float32)
    assert mp.shape[-1] == MAP_DIMS[dim]
    assert mp.any()
    back = decompress_obs(mp, tail)
    assert back.dtype == obs.dtype
    np.testing.assert_array_equal(np.asarray(back), np.asarray(obs))


def test_compress_refuses_an_unknown_obs_length() -> None:
    with pytest.raises(ValueError, match="obs length"):
        compress_obs(jnp.zeros((2, 100)))


def test_init_is_empty_and_sized_as_the_design_table() -> None:
    cfg = SILConfig(slots_windows=2)
    st = init(cfg, 4, 67, 8268)
    buf = st.buffer
    assert buf["map"].shape == (134, L, 8217)
    assert buf["map"].dtype == jnp.uint8
    assert buf["tail"].shape == (134, L, 51)
    assert buf["h0"].shape == (134, 512)
    assert (np.asarray(buf["stamp"]) == -1).all()
    assert not np.asarray(buf["mask"]).any()
    assert st.staging["map"].shape == (4, L, 8217)
    assert st.staging["hidden"].dtype == jnp.float16
    assert np.asarray(st.staging["done"]).all()  # before the run: masked everywhere
    assert int(st.ptr) == 0
    leaves = jax.tree.leaves(st)
    assert buffer_bytes(cfg, 4, 67, 8268).total == sum(x.nbytes for x in leaves)
    # design note, "Memory": 8,432 B per stored step, 1.21 MB staging per env,
    # 1.08 MB per window, buffers of 145 MB / 579 MB / 1.16 GB
    step = (buffer_bytes(SILConfig(window=2), 4, 67).buffer
            - buffer_bytes(SILConfig(window=1), 4, 67).buffer) / 134  # fmt: skip
    assert step == 8432
    one = buffer_bytes(SILConfig(), 1, 67)
    assert round(one.staging / MB, 2) == 1.21
    for envs, staging in ((256, 310), (1024, 1240)):
        sizes = [buffer_bytes(SILConfig(slots_windows=s), envs, 67) for s in (2, 8, 16)]
        assert {round(b.staging / MB, -1 if envs > 256 else 0) for b in sizes} == {
            staging
        }
        assert [round(b.buffer / MB) for b in sizes[:2]] == [145, 579]
        assert round(sizes[2].buffer / GB, 2) == 1.16
        assert round(sizes[2].buffer / 134 / 8 / MB, 2) == 1.08
    peaks = [
        buffer_bytes(SILConfig(slots_windows=16), n, 67).total for n in (256, 1024)
    ]
    assert [round(p / GB, 2) for p in peaks] == [1.47, 2.40]


def test_check_memory_refuses_a_buffer_over_the_limit() -> None:
    spec = HookSpec(n_envs=256, obs_dim=8268, n_actions=43, n_achievements=67,
                    total_steps=10**6)  # fmt: skip
    cfg = SILConfig(slots_windows=2)
    assert check_memory(cfg, spec, 32 * 2**30) == buffer_bytes(cfg, 256, 67, 8268)
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        check_memory(cfg, spec, 100 * 10**6)


def test_unlock_copies_the_window_from_staging_and_rollout() -> None:
    a = 5
    st = _run(SILConfig(), 8, lambda s: _traj(s, unlocks=[(130, 1, a)]))
    rows = _stored(st)
    assert rows.tolist() == [a * 2]
    assert int(st.inserted) == 1
    buf = jax.tree.map(lambda x: np.asarray(x[rows[0]]), st.buffer)
    steps = np.arange(3, 131)
    np.testing.assert_array_equal(buf["tail"][:, 0], steps)
    np.testing.assert_array_equal(buf["tail"][:, 1], 1)
    np.testing.assert_array_equal(buf["map"].argmax(-1), steps % M)
    np.testing.assert_array_equal(buf["act"], steps % 17)
    assert buf["h0"][0] == 3
    assert buf["h0"][1] == 1
    assert buf["mask"].all()
    assert buf["ach"] == a
    assert buf["prio"] == pytest.approx(float(st.max_prio))


def test_done_step_unlock_masks_the_previous_episode() -> None:
    def last(s):
        return _traj(s, unlocks=[(130, 1, 0)], dones=[(130, 1)])

    st = _run(SILConfig(), 8, last, dones=[(50, 1)])
    mask = np.asarray(st.buffer["mask"][_stored(st)[0]])
    np.testing.assert_array_equal(mask, np.arange(3, 131) > 50)


def test_window_before_the_run_start_is_masked() -> None:
    st = _run(SILConfig(), 0, lambda s: _traj(s, unlocks=[(5, 0, 2)]))
    buf = jax.tree.map(lambda x: np.asarray(x[_stored(st)[0]]), st.buffer)
    np.testing.assert_array_equal(buf["mask"], np.arange(-122, 6) >= 0)
    np.testing.assert_array_equal(buf["tail"][-6:, 0], np.arange(6))


def test_returns_are_discounted_and_cut_at_the_unlock() -> None:
    r = jnp.zeros((T, E)).at[2, 0].set(1.0).at[3, 0].set(5.0)
    st = _run(SILConfig(gamma=0.9), 8, lambda s: _traj(s, [(130, 0, 0)], rewards=r))
    ret = np.asarray(st.buffer["ret"][_stored(st)[0]])
    np.testing.assert_allclose(ret[-3:], [0.81, 0.9, 1.0], rtol=1e-6)
    # the reward of step 131, after the unlock, is cut off
    np.testing.assert_allclose(ret, 0.9 ** (130 - np.arange(3, 131)), rtol=1e-5)


def test_slot_keeps_the_newest_windows() -> None:
    cfg, a = SILConfig(slots_windows=2), 3
    st = init(cfg, E, A, D)
    for k, ends in enumerate([[0], [16 + 1, 16 + 2, 16 + 3], [32 + 4]]):
        st = _observe(st, _traj(16 * k, unlocks=[(s, 0, a) for s in ends]))
        last = sorted(int(x) for x in st.buffer["tail"][_stored(st), -1, 0])
        assert last == [[0], [18, 19], [19, 36]][k]
    assert set(_stored(st)) == {2 * a, 2 * a + 1}


def test_insert_cap_takes_64_and_counts_the_overflow() -> None:
    unl = [(0, i, a) for i in range(E) for a in range(A)]
    unl += [(1, 0, a) for a in range(A)] + [(1, 1, a) for a in range(4)]
    assert len(unl) == 70
    st = _run(SILConfig(slots_windows=4), 0, lambda s: _traj(s, unlocks=unl))
    assert (int(st.inserted), int(st.overflow)) == (64, 6)
    assert len(_stored(st)) == 64


def test_single_layout_shares_one_fifo() -> None:
    cfg = SILConfig(slots_windows=1, layout="single")
    unl = [(t, 0, t % 3) for t in range(T)]
    st = _run(cfg, 0, lambda s: _traj(s, unlocks=unl))
    assert st.buffer["stamp"].shape == (A,)
    assert len(_stored(st)) == T
    unl = [(T + t, 1, 7) for t in range(10)]
    st = _observe(st, _traj(T, unlocks=unl))
    ach = np.asarray(st.buffer["ach"])
    assert len(_stored(st)) == A
    assert (ach == 7).sum() == 10


def test_observe_reads_only_the_state_and_the_traj() -> None:
    assert list(inspect.signature(observe).parameters) == ["state", "traj"]


@pytest.mark.parametrize("n_envs", [4, 2 * A])
def test_only_the_staging_ring_is_per_env(n_envs: int) -> None:
    st = init(SILConfig(slots_windows=2), n_envs, A, D)  # 44 rows meet 44 envs
    for x in jax.tree.leaves(st.staging):
        assert x.shape[0] == n_envs
    rest = jax.tree.leaves(st.replace(staging={}))
    assert all(x.ndim == 0 or x.shape[0] != n_envs for x in rest)


def test_state_round_trips_save_state(tmp_path: Path) -> None:
    cfg = SILConfig()
    st = _run(cfg, 2, lambda s: _traj(s, unlocks=[(40, 1, 4)]))
    save_state(tmp_path, st)
    back = load_state(tmp_path, init(cfg, E, A, D))
    for got, want in zip(jax.tree.leaves(back), jax.tree.leaves(st), strict=True):
        np.testing.assert_array_equal(got, want)
    assert back.cfg == cfg


@pytest.mark.parametrize(
    "bad",
    [{"slots_windows": 0}, {"layout": "shared"}, {"gamma": 0.0}, {"insert_cap": 0},
     {"lambda_sil": -1.0}],
)  # fmt: skip
def test_config_refuses_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        SILConfig(**bad)
