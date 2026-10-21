"""Tests for the count table and bonus: counts, masking, decay, SimHash, metrics."""

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.count_bonus import (
    CountBonusConfig,
    init,
    intrinsic_reward,
    metrics,
    simhash_key,
)
from wut_murl_craftax.training.hooks import HookSpec
from wut_murl_craftax.training.progress_key import fields, hash_fields
from wut_murl_craftax.training.wrappers import LogEnvState

CLASSIC = "Craftax-Classic-Symbolic-v1"
BETA = 0.1
SPEC = HookSpec(
    n_envs=2, obs_dim=1345, n_actions=17, n_achievements=22, total_steps=1000
)
KNOBS = {
    "beta": BETA,
    "granularity": "recipe",
    "decay": "constant",
    "log2_k": 20,
    "key": "progress",
}
"""Every method knob (the config has no defaults): smoke config A."""


def _cfg(**over: Any) -> CountBonusConfig:
    return CountBonusConfig(**(KNOBS | over))


@pytest.fixture(scope="module")
def env() -> Any:
    """Craftax-Classic env without auto-reset."""
    return make_craftax_env_from_name(CLASSIC, auto_reset=False)


@pytest.fixture(scope="module")
def start(env: Any) -> Any:
    """Craftax-Classic start state (key a)."""
    return env.reset(jax.random.key(0), env.default_params)[1]


@pytest.fixture(scope="module")
def picked(start: Any) -> Any:
    """The start state with a wood pickaxe (key b)."""
    return start.replace(inventory=start.inventory.replace(wood_pickaxe=1))


def _batch(*states: Any) -> LogEnvState:
    """Stack per-env inner states under a log state, as the trainers see them."""
    inner = jax.tree.map(lambda *x: jnp.stack(x), *states)
    zeros = jnp.zeros(len(states))
    return LogEnvState(inner, zeros, zeros, zeros, zeros, zeros)


def _call(state: Any, nxt: Any, done: list[bool], t: int = 0, env_state: Any = None):
    obs = jnp.zeros((len(done), SPEC.obs_dim))
    return jax.jit(intrinsic_reward)(
        state, env_state, nxt, obs, jnp.array(done), jnp.int32(t)
    )


def _key(s: Any) -> int:
    return int(hash_fields(fields(s, "recipe"), 20))


def test_scripted_counts_and_bonuses(start: Any, picked: Any) -> None:
    a, b = _key(start), _key(picked)
    assert a != b
    st = init(_cfg(), jax.random.key(1), SPEC)
    script = [
        ((start, start), [False, False], 2, [1 / math.sqrt(2)] * 2),
        ((picked, start), [False, False], 3, [1.0, 1 / math.sqrt(3)]),
        ((start, start), [False, True], 4, [0.5, 0.0]),
    ]
    for states, done, n_a, bonus in script:
        r, st = _call(st, _batch(*states), done)
        assert int(st.table[a]) == n_a
        np.testing.assert_allclose(r, BETA * np.array(bonus), rtol=1e-6)
    assert int(st.table[b]) == 1
    assert int(st.table.sum()) == 5  # the done step counted nothing


def test_bonus_ignores_the_pre_step_state(start: Any, picked: Any) -> None:
    st = init(_cfg(), jax.random.key(1), SPEC)
    nxt = _batch(picked, start)
    r1, s1 = _call(st, nxt, [False, True], env_state=_batch(start, start))
    r2, s2 = _call(st, nxt, [False, True], env_state=jnp.full(7, -1))
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(s1.table, s2.table)


def test_init_sizes_from_the_spec() -> None:
    spec = HookSpec(
        n_envs=4, obs_dim=8268, n_actions=43, n_achievements=67, total_steps=10**6
    )
    st = init(_cfg(), jax.random.key(2), spec)
    assert st.table.shape == (2**20,)
    assert st.table.dtype == jnp.int32
    assert not st.table.any()
    assert st.proj.shape == (8268, 64)
    assert st.total_steps == 1_000_000
    # per-env leaves (leading dim == envs) are sharded over the mesh: none here
    assert all(np.ndim(x) == 0 or np.shape(x)[0] != 4 for x in jax.tree.leaves(st))


def test_linear_decay_halves_beta_at_half_budget(start: Any) -> None:
    cfg = _cfg(decay="linear")
    st = init(cfg, jax.random.key(1), SPEC)
    r, _ = _call(st, _batch(start, start), [False, False], t=SPEC.total_steps // 2)
    np.testing.assert_allclose(r, BETA / 2 / math.sqrt(2), rtol=1e-6)
    r, _ = _call(st, _batch(start, start), [False, False], t=SPEC.total_steps)
    np.testing.assert_array_equal(r, 0.0)


def test_zero_beta_pays_nothing_and_calls_repeat(start: Any, picked: Any) -> None:
    st = init(_cfg(beta=0.0), jax.random.key(1), SPEC)
    nxt = _batch(start, picked)
    r1, s1 = _call(st, nxt, [False, False])
    r2, s2 = _call(st, nxt, [False, False])
    np.testing.assert_array_equal(r1, 0.0)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(s1.table, s2.table)
    assert s1.table.dtype == jnp.int32
    assert int(s1.table.sum()) == 2


def test_simhash_key_separates_one_map_tile(env: Any, start: Any) -> None:
    proj = init(_cfg(key="simhash"), jax.random.key(3), SPEC).proj
    pos = start.player_position + jnp.array([1, 0])
    tile = int(start.map[pos[0], pos[1]])
    new = (
        BlockType.STONE.value
        if tile != BlockType.STONE.value
        else BlockType.WATER.value
    )
    obs = env.get_obs(start)
    moved = env.get_obs(start.replace(map=start.map.at[pos[0], pos[1]].set(new)))
    assert not np.array_equal(obs, moved)
    k = jax.jit(simhash_key, static_argnums=2)
    assert int(k(obs, proj, 20)) == int(k(obs + 0.0, proj, 20))
    assert int(k(obs, proj, 20)) != int(k(moved, proj, 20))
    batch = k(jnp.stack([obs, moved]), proj, 20)
    np.testing.assert_array_equal(batch, [k(obs, proj, 20), k(moved, proj, 20)])


def test_simhash_counts_observations(env: Any, start: Any) -> None:
    cfg = _cfg(key="simhash")
    st = init(cfg, jax.random.key(3), SPEC)
    obs = jnp.stack([env.get_obs(start)] * 2)
    r, st = jax.jit(intrinsic_reward)(
        st, None, None, obs, jnp.array([False, False]), jnp.int32(0)
    )
    np.testing.assert_allclose(r, BETA / math.sqrt(2), rtol=1e-6)
    assert int(st.table[simhash_key(obs[0], st.proj, 20)]) == 2


def test_metrics_match_numpy() -> None:
    st = init(_cfg(), jax.random.key(4), SPEC)
    keys = np.random.default_rng(5).integers(0, 2**20, 3000)
    table = st.table.at[keys].add(1)
    got = jax.jit(metrics)(st.replace(table=table))
    ref = np.zeros(2**20, np.int64)
    np.add.at(ref, keys, 1)
    distinct = np.count_nonzero(ref)
    assert set(got) == {"count/distinct_keys", "count/occupancy"}
    assert int(got["count/distinct_keys"]) == distinct
    np.testing.assert_allclose(got["count/occupancy"], distinct / 2**20, rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"beta": -0.1},
        {"beta": math.nan},
        {"granularity": "fine"},
        {"decay": "cosine"},
        {"key": "rnd"},
    ],
)
def test_config_refuses_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        _cfg(**bad)


def test_state_round_trips_save_state(tmp_path: Path, start: Any) -> None:
    cfg = _cfg(decay="linear")
    st = init(cfg, jax.random.key(1), SPEC)
    _, st = _call(st, _batch(start, start), [False, False], t=40)
    save_state(tmp_path, st)
    back = load_state(tmp_path, init(cfg, jax.random.key(9), SPEC))
    for got, want in zip(jax.tree.leaves(back), jax.tree.leaves(st), strict=True):
        np.testing.assert_array_equal(got, want)
    assert (back.cfg, back.total_steps) == (cfg, SPEC.total_steps)
