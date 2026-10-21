"""Tests for the online score and achievement metrics of training episodes."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Achievement, Action
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.episode_metrics import (
    episode_step,
    events_of,
    first_unlock_offsets,
    init_acc,
    score_pct,
    update_summary,
)
from wut_murl_craftax.training.ppo import PPOConfig, init_state
from wut_murl_craftax.training.ppo import train_chunk as ppo_train_chunk
from wut_murl_craftax.training.records import achievement_weights
from wut_murl_craftax.training.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

CLASSIC, CRAFTAX = "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"
ENVS, EPISODE = 16, 200
TABLE = Achievement.PLACE_TABLE.value


@pytest.fixture(scope="module")
def scripted() -> dict[str, Any]:
    """200 steps of 16 Classic envs through the optimistic-reset wrapper.

    Nothing spawns, so every episode lasts exactly 200 steps. Env 1 places a table at
    step 100; env 0 places one on step 200, the done step, where the wrapper has
    already replaced its state with a fresh episode. The others do nothing.
    """
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    vec = OptimisticResetVecEnvWrapper(env, ENVS, 16)
    params = vec.default_params.replace(
        max_timesteps=EPISODE,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    _, st = vec.reset(jax.random.key(0), params)
    s = st.env_state
    wood = s.inventory.wood.at[:2].set(5)
    s = s.replace(
        inventory=s.inventory.replace(wood=wood),
        zombies=s.zombies.replace(mask=jnp.zeros_like(s.zombies.mask)),
        skeletons=s.skeletons.replace(mask=jnp.zeros_like(s.skeletons.mask)),
    )
    st = st.replace(env_state=s)
    acc, step = init_acc(ENVS, 22), jax.jit(episode_step)
    out: dict[str, list[Any]] = {k: [] for k in ("done", "ret", "len", "ach", "reward")}
    for t in range(1, EPISODE + 1):
        act = np.zeros(ENVS, np.int32)
        act[1] = Action.PLACE_TABLE.value if t == 100 else 0
        act[0] = Action.PLACE_TABLE.value if t == EPISODE else 0
        _, st, r, d, info = vec.step(jax.random.key(t), st, jnp.asarray(act), params)
        acc, end = step(acc, r, d, events_of(info).unlocked)
        for k, v in zip(out, (d, end.ret, end.length, end.ach, r), strict=True):
            out[k].append(np.asarray(v))
    stacked = {k: np.stack(v) for k, v in out.items()}
    stacked["logged"] = np.asarray(info["returned_episode_returns"])
    stacked["fresh"] = np.asarray(st.env_state.achievements)
    return stacked


def test_the_unlock_on_the_done_step_is_counted(scripted: dict[str, Any]) -> None:
    done, ach = scripted["done"], scripted["ach"]
    assert np.flatnonzero(done.any(1)).tolist() == [EPISODE - 1]
    assert ach[-1, 0, TABLE], "env 0 unlocked on its last step"
    assert ach[-1, 1, TABLE], "env 1 unlocked 100 steps earlier and keeps it"
    assert ach[-1].sum() == 2
    # what a comparison of states across the done step sees: the fresh episode
    assert not scripted["fresh"].any()


def test_episode_return_and_length_cover_the_done_step(
    scripted: dict[str, Any],
) -> None:
    ret, length = scripted["ret"][-1], scripted["len"][-1]
    assert (length == EPISODE).all()
    np.testing.assert_allclose(ret, scripted["reward"].sum(0), atol=1e-5)
    np.testing.assert_allclose(ret, scripted["logged"], atol=1e-5)
    assert ret[:2].tolist() == [1.0, 1.0]  # one achievement, reward 1 in Classic


def test_the_update_summary_counts_finished_episodes(scripted: dict[str, Any]) -> None:
    got = update_summary(
        scripted["done"], *(scripted[k] for k in ("ret", "len", "ach"))
    )
    assert got.n_episodes == ENVS
    assert got.ret == pytest.approx(2 / ENVS)
    assert got.length == EPISODE
    want = np.zeros(22)
    want[TABLE] = 100 * 2 / ENVS
    np.testing.assert_allclose(got.success, want)
    pct = score_pct(got.ret, achievement_weights(CLASSIC).sum())
    assert pct == pytest.approx(2 / ENVS / 22 * 100)


def test_a_step_before_the_end_only_counts_finished_episodes(
    scripted: dict[str, Any],
) -> None:
    mid = update_summary(
        scripted["done"][:-1], *(scripted[k][:-1] for k in ("ret", "len", "ach"))
    )
    assert mid.n_episodes == 0
    assert np.isnan([mid.ret, mid.length]).all()
    assert np.isnan(mid.success).all()


def test_the_achievements_match_craftaxs_own_report_on_done_steps() -> None:
    """A random policy for 400 steps: our episode flags equal ``Achievements/*``."""
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    vec = OptimisticResetVecEnvWrapper(env, ENVS, 16)
    params = vec.default_params
    _, st = vec.reset(jax.random.key(0), params)
    acc, step = init_acc(ENVS, 22), jax.jit(episode_step)
    names = [a.name.lower() for a in sorted(Achievement, key=lambda a: a.value)]
    finished = 0
    for t in range(400):
        k_a, k_s = jax.random.split(jax.random.key(t))
        act = jax.random.randint(k_a, (ENVS,), 0, 17)
        _, st, r, d, info = vec.step(k_s, st, act, params)
        acc, end = step(acc, r, d, events_of(info).unlocked)
        theirs = np.stack(
            [np.asarray(info[f"Achievements/{n}"]) > 0 for n in names], -1
        )
        d = np.asarray(d)
        assert np.array_equal(np.asarray(end.ach)[d], theirs[d]), t
        finished += d.sum()
    assert finished > 0
    assert np.asarray(end.ach).any(), "the random policy unlocked something"


def test_accumulators_restart_after_a_done_step() -> None:
    """Env 0 ends on its 2nd step with a new unlock; env 1 runs on."""
    acc, step = init_acc(2, 4), jax.jit(episode_step)
    unlocked = np.zeros((2, 4), bool)
    unlocked[0, 1] = True
    acc, end = step(acc, jnp.array([1.0, 0.5]), jnp.array([False, False]), unlocked)
    unlocked = np.zeros((2, 4), bool)
    unlocked[0, 3] = True
    acc, end = step(acc, jnp.array([2.0, 0.0]), jnp.array([True, False]), unlocked)
    assert np.asarray(end.ach).tolist() == [[0, 1, 0, 1], [0, 0, 0, 0]]
    assert np.asarray(end.ret).tolist() == [3.0, 0.5]
    assert np.asarray(end.length).tolist() == [2, 2]
    acc, end = step(acc, jnp.array([1.0, 1.0]), jnp.array([False, True]), unlocked * 0)
    assert np.asarray(end.ach).tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert np.asarray(end.ret).tolist() == [1.0, 1.5]
    assert np.asarray(end.length).tolist() == [1, 3]


def test_first_unlock_is_the_earliest_env_step_of_the_update() -> None:
    """(steps 3, envs 2): achievement 1 unlocks first at step 2, 0 at step 1."""
    unlocked = np.zeros((3, 2, 4), bool)
    unlocked[0, 1, 0] = unlocked[1, 0, 1] = unlocked[2, 1, 1] = True
    got = jax.jit(first_unlock_offsets)(unlocked)
    assert np.asarray(got).tolist() == [2, 4, -1, -1]  # (t + 1) * envs


@pytest.mark.parametrize(("env", "top"), [(CLASSIC, 22), (CRAFTAX, 226)])
def test_score_is_the_return_as_a_share_of_the_maximum(env: str, top: int) -> None:
    assert achievement_weights(env).sum() == top
    assert score_pct(top / 2, achievement_weights(env).sum()) == 50.0
    both = score_pct(np.array([0.0, top / 4]), top)
    np.testing.assert_allclose(both, [0.0, 25.0])


def test_info_without_events_fails_loudly() -> None:
    with pytest.raises(ValueError, match=r"info.*events.*optimistic"):
        events_of({"returned_episode_returns": 0.0})


def test_auto_reset_ppo_refuses_to_log_rather_than_logging_zeros() -> None:
    """Craftax's own auto-reset gives no events, so the first update raises."""
    cfg = PPOConfig(env=CLASSIC, envs=8, num_steps=4, width=8, resets="auto")
    state = init_state(cfg, jax.random.key(0))
    with pytest.raises(ValueError, match="optimistic"):
        ppo_train_chunk(state, 1)
