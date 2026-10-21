"""Tests for the vendored Craftax_Baselines wrappers and their step events."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement, BlockType
from craftax.craftax.constants import Action as CraftaxAction
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.events import step_events
from wut_murl_craftax.training.ppo import PPOConfig, make_train_env
from wut_murl_craftax.training.wrappers import (
    BatchEnvWrapper,
    GymnaxWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    unwrap_state,
)

ENVS, RATIO = 32, 16
CLASSIC, CRAFTAX = "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"
NECRO = Achievement.DEFEAT_NECROMANCER.value
TABLE = Achievement.PLACE_TABLE.value
BOSS_T = 777
"""Timestep that makes `BossEnv` end the episode with the necromancer defeated."""


def _raw(x: Any) -> np.ndarray:
    if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
    return np.asarray(x)


def _same(a: Any, b: Any) -> bool:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    pairs = zip(la, lb, strict=True)
    return len(la) == len(lb) and all(
        np.array_equal(_raw(x), _raw(y)) for x, y in pairs
    )


def _env_i(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def _timeout(st: Any, envs: list[int], max_t: int) -> Any:
    """Envs ``envs`` one step before the time limit, so this step ends them."""
    inner = st.env_state
    t = inner.timestep.at[jnp.array(envs)].set(max_t - 1)
    return st.replace(env_state=inner.replace(timestep=t))


def _plain(vec: OptimisticResetVecEnvWrapper, rng: jax.Array, st: Any, act: Any) -> Any:
    """The batched step and resets the wrapper draws, with its key splits."""
    rng, k = jax.random.split(rng)
    out = vec.step_fn(jax.random.split(k, vec.num_envs), st, act, vec.default_params)
    _, k = jax.random.split(rng)
    _, resets = vec.reset_fn(jax.random.split(k, vec.num_resets), vec.default_params)
    return out, resets


class BossEnv(GymnaxWrapper):
    """Craftax whose step at ``BOSS_T`` defeats the necromancer and ends the episode,
    as the real boss kill does (``is_game_over``), with its tier reward."""

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: Any, state: Any, action: Any, params: Any = None) -> Any:
        obs, s2, r, done, info = self._env.step(key, state, action, params)
        boss = state.timestep == BOSS_T
        ach = s2.achievements.at[NECRO].set(s2.achievements[NECRO] | boss)
        r = r + jnp.where(boss, ACHIEVEMENT_REWARD_MAP[NECRO], 0.0)
        return obs, s2.replace(achievements=ach), r, done | boss, info


@pytest.fixture(scope="module")
def classic() -> OptimisticResetVecEnvWrapper:
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    return OptimisticResetVecEnvWrapper(env, num_envs=ENVS, reset_ratio=RATIO)


@pytest.mark.parametrize("done_envs", [[5], [0, 7, 20]], ids=["issue-11", "3-done"])
def test_only_done_envs_restart_and_from_a_fresh_world(
    classic: OptimisticResetVecEnvWrapper, done_envs: list[int]
) -> None:
    """Fewer done envs than resets (issue #11) and more: the rest match the step."""
    max_t = classic.default_params.max_timesteps
    _, st = classic.reset(jax.random.key(0), classic.default_params)
    st = _timeout(st, done_envs, max_t)
    act, rng = jnp.zeros(ENVS, jnp.int32), jax.random.key(1)
    obs, st2, _, done, _ = classic.step(rng, st, act, classic.default_params)
    (obs_p, st_p, *_), resets = _plain(classic, rng, st, act)
    assert np.flatnonzero(done).tolist() == done_envs
    for i in range(ENVS):
        if i not in done_envs:
            assert _same(_env_i(st2, i), _env_i(st_p, i)), i
            assert np.array_equal(obs[i], obs_p[i])
            continue
        fresh = unwrap_state(_env_i(st2, i))
        assert int(fresh.timestep) == 0
        assert not np.asarray(fresh.achievements).any()
        assert all(not np.asarray(x).any() for x in jax.tree.leaves(fresh.inventory))
        starts = [_env_i(resets, j) for j in range(classic.num_resets)]
        assert any(_same(_env_i(st2, i), s) for s in starts), i


def test_reset_ratio_must_divide_the_envs() -> None:
    env = LogWrapper(make_craftax_env_from_name(CLASSIC, auto_reset=False))
    with pytest.raises(AssertionError, match="perfectly divide"):
        OptimisticResetVecEnvWrapper(env, num_envs=30, reset_ratio=RATIO)


@pytest.mark.parametrize(("envs", "ratio"), [(32, 16), (8, 8)])
def test_the_training_env_uses_ratio_min_16_envs(envs: int, ratio: int) -> None:
    env, _ = make_train_env(PPOConfig(env=CLASSIC, envs=envs))
    assert isinstance(env, OptimisticResetVecEnvWrapper)
    assert (env.num_envs, env.reset_ratio) == (envs, ratio)


def test_auto_resets_keep_craftax_auto_reset() -> None:
    """The debug comparison: Craftax resets each done env itself, without events."""
    env, params = make_train_env(PPOConfig(env=CLASSIC, envs=8, resets="auto"))
    assert isinstance(env, BatchEnvWrapper)
    _, st = env.reset(jax.random.key(0), params)
    st = _timeout(st, [2], params.max_timesteps)
    act = jnp.zeros(8, jnp.int32)
    _, st2, _, done, info = env.step(jax.random.key(1), st, act, params)
    assert np.flatnonzero(done).tolist() == [2]
    assert int(unwrap_state(st2).timestep[2]) == 0
    assert "events" not in info


@pytest.mark.parametrize("ratio", [0, -2, 1.5, True])
def test_bad_reset_ratios_fail(ratio: Any) -> None:
    with pytest.raises(ValueError, match="reset_ratio"):
        PPOConfig(reset_ratio=ratio)


@pytest.fixture(scope="module")
def boss_step() -> dict[str, Any]:
    """One Craftax step of 32 envs: env 0 places a table on its last step, env 3
    defeats the necromancer, env 5 times out on floor 1 with 3 health."""
    base = make_craftax_env_from_name(CRAFTAX, auto_reset=False)
    vec = OptimisticResetVecEnvWrapper(LogWrapper(BossEnv(base)), ENVS, RATIO)
    params = vec.default_params
    _, st = vec.reset(jax.random.key(0), params)
    s = st.env_state
    r, c = (int(x) for x in s.player_position[0])
    up = (r - 1, c)  # the player starts facing up
    s = s.replace(
        map=s.map.at[0, 0, *up].set(BlockType.GRASS.value),
        inventory=s.inventory.replace(wood=s.inventory.wood.at[0].set(2)),
        timestep=s.timestep.at[3].set(BOSS_T),
        player_level=s.player_level.at[5].set(1),
        player_health=s.player_health.at[5].set(3.0),
    )
    st = _timeout(st.replace(env_state=s), [0, 5], params.max_timesteps)
    act = jnp.zeros(ENVS, jnp.int32).at[0].set(CraftaxAction.PLACE_TABLE.value)
    rng = jax.random.key(2)
    _, st2, rew, done, info = vec.step(rng, st, act, params)
    (_, plain_st, *_), _ = _plain(vec, rng, st, act)
    s, s2 = unwrap_state(st), unwrap_state(plain_st)
    plain = jax.vmap(functools.partial(step_events, "craftax"))(s, act, s2)
    return {
        "st2": st2,
        "rew": rew,
        "done": done,
        "info": info,
        "plain": plain,
        "s": s,
    }


def test_events_equal_the_plain_step_events(boss_step: dict[str, Any]) -> None:
    ev, plain = boss_step["info"]["events"], boss_step["plain"]
    assert np.flatnonzero(boss_step["done"]).tolist() == [0, 3, 5]
    keep = np.arange(ENVS) != 3  # BossEnv edits env 3 after the plain step
    for a, b in zip(jax.tree.leaves(ev), jax.tree.leaves(plain), strict=True):
        assert np.array_equal(np.asarray(a)[keep], np.asarray(b)[keep])


def test_an_unlock_on_the_timeout_step_is_reported(boss_step: dict[str, Any]) -> None:
    unlocked = np.asarray(boss_step["info"]["events"].unlocked)
    assert np.flatnonzero(unlocked[0]).tolist() == [TABLE]
    fresh = unwrap_state(_env_i(boss_step["st2"], 0))
    assert int(fresh.timestep) == 0
    assert not np.asarray(fresh.achievements).any()


def test_the_necromancer_kill_is_reported_and_scored(boss_step: dict[str, Any]) -> None:
    info, rew = boss_step["info"], np.asarray(boss_step["rew"])
    assert np.asarray(info["events"].unlocked)[3, NECRO]
    assert ACHIEVEMENT_REWARD_MAP[NECRO] == 8
    assert rew[3] > 7.5  # 8 points plus at most a small health change
    # the episode started at the reset, so its return is this step's reward
    assert info["returned_episode_returns"][3] == pytest.approx(rew[3])
    assert not np.asarray(unwrap_state(_env_i(boss_step["st2"], 3)).achievements).any()


def test_floor_and_health_of_a_done_env_are_terminal(boss_step: dict[str, Any]) -> None:
    ev, s = boss_step["info"]["events"], boss_step["s"]
    fresh = unwrap_state(_env_i(boss_step["st2"], 5))
    assert int(ev.floor[5]) == 1
    assert int(fresh.player_level) == 0
    reset_delta = float(fresh.player_health - s.player_health[5])
    assert float(ev.health_delta[5]) != pytest.approx(reset_delta)
    assert float(ev.health_delta[5]) == float(boss_step["plain"].health_delta[5])
