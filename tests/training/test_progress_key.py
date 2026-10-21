"""Tests for the progress key: unwrap, bucketed fields, material buckets, hash."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import MONSTERS_KILLED_TO_CLEAR_LEVEL
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.training.progress_key import (
    CLASSIC_FIELDS,
    FIELDS,
    LADDER_KILLS,
    bucket,
    fields,
    hash_fields,
    unwrap,
)
from wut_murl_craftax.training.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

CLASSIC, CRAFTAX = "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"


def _reset(name: str) -> Any:
    env = make_craftax_env_from_name(name, auto_reset=False)
    return env.reset(jax.random.key(0), env.default_params)[1]


@pytest.fixture(scope="module")
def craftax() -> Any:
    """Craftax start state."""
    return _reset(CRAFTAX)


@pytest.fixture(scope="module")
def classic() -> Any:
    """Craftax-Classic start state."""
    return _reset(CLASSIC)


def _inv(state: Any, **items: Any) -> Any:
    return state.replace(inventory=state.inventory.replace(**items))


def _rich(state: Any) -> Any:
    """A Craftax state with every key field away from its start value."""
    inv = state.inventory
    state = _inv(
        state,
        wood=9, stone=2, coal=7, iron=1, diamond=5, sapling=3, pickaxe=3, sword=2,
        bow=1, arrows=0, torches=4, ruby=1, sapphire=0, books=2,
        armour=inv.armour.at[0].set(2).at[2].set(1),
        potions=inv.potions.at[1].set(2),
    )  # fmt: skip
    kills = state.monsters_killed.at[3].set(8)
    return state.replace(
        player_level=jnp.int32(3),
        monsters_killed=kills,
        learned_spells=jnp.array([True, False]),
        sword_enchantment=jnp.int32(1),
        armour_enchantments=jnp.array([0, 0, 2, 0], jnp.int32),
    )


def test_ladder_kills_is_craftax_constant() -> None:
    assert LADDER_KILLS == MONSTERS_KILLED_TO_CLEAR_LEVEL


def test_unwrap_reaches_inner_state_through_the_wrappers(classic: Any) -> None:
    env = make_craftax_env_from_name(CLASSIC, auto_reset=False)
    vec = OptimisticResetVecEnvWrapper(LogWrapper(env), 2, 2)
    _, st = vec.reset(jax.random.key(1), env.default_params)
    assert unwrap(st) is st.env_state
    assert unwrap(classic) is classic


def test_fields_recipe_is_the_key_table(craftax: Any) -> None:
    got = np.asarray(fields(_rich(craftax), "recipe"))
    want = {
        "floor": 3, "ladder_open": 1, "pickaxe": 3, "sword": 2, "bow": 1,
        "armour": 3, "spell_fireball": 1, "spell_iceball": 0,
        "ench_sword": 1, "ench_bow": 0, "ench_armour": 1,
        "wood": 4, "stone": 2, "coal": 3, "iron": 1, "diamond": 3, "sapphire": 0,
        "ruby": 1, "sapling": 1, "arrows": 0, "torches": 1, "books": 1, "potions": 1,
    }  # fmt: skip
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, [want[n] for n in FIELDS])


@pytest.mark.parametrize("granularity", ["tiers", "recipe", "log2"])
def test_fields_ignore_scenery(craftax: Any, granularity: str) -> None:
    s = _rich(craftax)
    mobs = s.melee_mobs.replace(health=s.melee_mobs.health + 1.0)
    scenery = s.replace(
        map=s.map.at[3, 10, 10].set(1),
        player_position=s.player_position + 1,
        melee_mobs=mobs,
        light_level=jnp.float32(0.1),
        player_health=jnp.float32(2.0),
    )
    np.testing.assert_array_equal(fields(s, granularity), fields(scenery, granularity))


def test_fields_see_pickaxe_and_floor(craftax: Any) -> None:
    base = fields(craftax, "tiers")
    picked = fields(_inv(craftax, pickaxe=1), "tiers")
    down = fields(craftax.replace(player_level=jnp.int32(1)), "tiers")
    assert not np.array_equal(base, picked)
    assert not np.array_equal(base, down)


@pytest.mark.parametrize(("kills", "open_"), [(0, 0), (7, 0), (8, 1), (12, 1)])
def test_ladder_open_needs_eight_kills_on_this_floor(
    craftax: Any, kills: int, open_: int
) -> None:
    s = craftax.replace(
        player_level=jnp.int32(2),
        monsters_killed=craftax.monsters_killed.at[2].set(kills).at[1].set(9),
    )
    assert int(fields(s, "recipe")[FIELDS.index("ladder_open")]) == open_


def test_classic_fields(classic: Any) -> None:
    s = _inv(classic, wood=6, stone=1, coal=0, iron=2, diamond=9, sapling=2)
    s = _inv(s, wood_pickaxe=1, stone_pickaxe=0, iron_pickaxe=0)
    s = _inv(s, wood_sword=3, stone_sword=1, iron_sword=0)
    want = {
        "wood": 4, "stone": 1, "coal": 0, "iron": 2, "diamond": 3, "sapling": 1,
        "wood_pickaxe": 1, "stone_pickaxe": 0, "iron_pickaxe": 0,
        "wood_sword": 1, "stone_sword": 1, "iron_sword": 0,
    }  # fmt: skip
    np.testing.assert_array_equal(
        fields(s, "recipe"), [want[n] for n in CLASSIC_FIELDS]
    )
    tiers = np.asarray(fields(s, "tiers"))
    assert not tiers[:6].any()
    np.testing.assert_array_equal(tiers[6:], [1, 0, 0, 1, 1, 0])


@pytest.mark.parametrize(
    ("name", "granularity", "counts", "want"),
    [
        ("wood", "recipe", [0, 1, 4, 9], [0, 1, 4, 4]),
        ("stone", "log2", [0, 1, 2, 3, 99], [0, 1, 1, 2, 6]),
        ("stone", "recipe", [0, 3, 7], [0, 3, 3]),
        ("ruby", "recipe", [0, 1, 5], [0, 1, 1]),
        ("diamond", "tiers", [0, 1, 50], [0, 0, 0]),
        ("wood", "tiers", [0, 1, 50], [0, 0, 0]),
    ],
)
def test_bucket(
    name: str, granularity: str, counts: list[int], want: list[int]
) -> None:
    got = bucket(name, jnp.asarray(counts), granularity)
    assert got.dtype == jnp.int32
    np.testing.assert_array_equal(got, want)


def test_bucket_refuses_unknown_names() -> None:
    with pytest.raises(ValueError, match="material"):
        bucket("gold", jnp.int32(1), "recipe")
    with pytest.raises(ValueError, match="granularity"):
        bucket("wood", jnp.int32(1), "fine")


@pytest.mark.parametrize("bits", [16, 20])
def test_hash_fields_range_and_equality(bits: int) -> None:
    f = jax.random.randint(jax.random.key(2), (500, len(FIELDS)), 0, 10)
    keys = np.asarray(hash_fields(f, bits))
    assert keys.dtype == np.int32
    assert keys.min() >= 0
    assert keys.max() < 2**bits
    np.testing.assert_array_equal(keys, hash_fields(f, bits))


def test_hash_fields_takes_appended_flags() -> None:
    f = jnp.arange(len(FIELDS), dtype=jnp.int32)
    flags = jnp.concatenate([f, jnp.array([1, 0, 1, 1], jnp.int32)])
    assert hash_fields(flags).shape == ()
    assert int(hash_fields(flags)) != int(hash_fields(f))


def test_hash_fields_collide_rarely() -> None:
    """10,000 distinct vectors in 2^20 slots: expected share of collisions n / 2K."""
    rng = np.random.default_rng(3)
    f = np.unique(rng.integers(0, 9, (10_500, len(FIELDS))), axis=0)[:10_000]
    assert len(f) == 10_000
    keys = np.asarray(hash_fields(jnp.asarray(f, jnp.int32), 20))
    assert 1 - len(np.unique(keys)) / len(f) < 0.01


def test_all_four_jit_and_vmap(craftax: Any) -> None:
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), _rich(craftax))
    batch = batch.replace(player_level=jnp.array([3, 0], jnp.int32))
    per_env = jax.jit(jax.vmap(functools.partial(fields, granularity="log2")))
    f = per_env(unwrap(batch))
    assert f.shape == (2, len(FIELDS))
    np.testing.assert_array_equal(f[0], fields(_rich(craftax), "log2"))
    keys = jax.jit(jax.vmap(functools.partial(hash_fields, bits=20)))(f)
    np.testing.assert_array_equal(keys, hash_fields(f, 20))
    b = jax.jit(jax.vmap(lambda n: bucket("coal", n, "log2")))(jnp.arange(5))
    np.testing.assert_array_equal(b, [0, 1, 1, 2, 2])
    inner = jax.jit(unwrap)(batch)
    np.testing.assert_array_equal(inner.player_level, [3, 0])
