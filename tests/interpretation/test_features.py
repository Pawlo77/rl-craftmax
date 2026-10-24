"""Tests for the named observation features of decision-tree distillation."""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax.constants import BlockType
from craftax.craftax.renderer import render_craftax_symbolic as craftax_symbolic
from craftax.craftax_classic.renderer import (
    render_craftax_symbolic as classic_symbolic,
)
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.interpretation import FEATURES, featurize
from wut_murl_craftax.visualisation.rollout import ENV_NAMES, GameKind

GAMES: tuple[GameKind, ...] = ("craftax", "classic")
SENTINEL = {"craftax": 10, "classic": 8}


def _reset(game: GameKind, seed: int = 0) -> Any:
    env = make_craftax_env_from_name(ENV_NAMES[game], auto_reset=False)
    return env.reset(jax.random.PRNGKey(seed), env.default_params)[1]


def _obs(game: GameKind, *states: Any) -> np.ndarray:
    render = craftax_symbolic if game == "craftax" else classic_symbolic
    return np.stack([np.asarray(render(s)) for s in states])


def _row(game: GameKind, state: Any) -> dict[str, float]:
    return dict(zip(FEATURES[game], featurize(_obs(game, state), game)[0], strict=True))


def _no_mobs(state: Any, game: GameKind) -> Any:
    names = (
        ("melee_mobs", "passive_mobs", "ranged_mobs", "mob_projectiles")
        if game == "craftax"
        else ("zombies", "cows", "skeletons", "arrows")
    )
    return state.replace(**{
        n: getattr(state, n).replace(mask=jnp.zeros_like(getattr(state, n).mask))
        for n in names
    })


def _grass_world(game: GameKind) -> Any:
    """Reset state on an all-grass, fully lit map without creatures."""
    s = _no_mobs(_reset(game), game)
    s = s.replace(map=jnp.full_like(s.map, BlockType.GRASS.value))
    return s.replace(light_map=jnp.ones_like(s.light_map)) if game == "craftax" else s


def _put(state: Any, game: GameKind, offset: tuple[int, int], block: int) -> Any:
    r, c = np.asarray(state.player_position) + offset
    if game == "craftax":
        return state.replace(map=state.map.at[state.player_level, r, c].set(block))
    return state.replace(map=state.map.at[r, c].set(block))


@pytest.mark.parametrize("game", GAMES)
def test_reset_intrinsics_and_inventory_counts_round_trip(game: GameKind) -> None:
    s = _reset(game)
    if game == "craftax":
        inv = s.inventory.replace(
            wood=7, stone=12, torches=3, arrows=5, books=1, pickaxe=2, sword=3, bow=1,
            potions=jnp.array([1, 2, 0, 4, 0, 9]), armour=jnp.array([1, 2, 0, 1]),
        )  # fmt: skip
        s = s.replace(inventory=inv, player_health=jnp.float32(7.5), player_xp=2)
        s = s.replace(sword_enchantment=1, armour_enchantments=jnp.array([0, 0, 2, 0]))
        expected = {
            "health": 7.5, "food": s.player_food, "drink": s.player_drink,
            "energy": s.player_energy, "mana": s.player_mana, "xp": 2,
            "dexterity": s.player_dexterity, "strength": s.player_strength,
            "intelligence": s.player_intelligence, "wood": 7, "stone": 12,
            "torches": 3, "arrows": 5, "books": 1, "pickaxe_level": 2,
            "sword_level": 3, "bow": 1, "potion_red": 1, "potion_green": 2,
            "potion_pink": 4, "potion_yellow": 9, "helmet_level": 1,
            "chestplate_level": 2, "boots_level": 1, "sword_enchantment": 1,
            "armour_enchanted": 1, "floor": 0, "direction": int(s.player_direction),
        }  # fmt: skip
    else:
        inv = s.inventory.replace(wood=7, stone=9, sapling=1, stone_pickaxe=1)
        s = s.replace(
            inventory=inv.replace(wood_pickaxe=1, wood_sword=1), player_food=6
        )
        expected = {
            "health": s.player_health, "food": 6, "drink": s.player_drink,
            "energy": s.player_energy, "wood": 7, "stone": 9, "sapling": 1,
            "pickaxe_level": 2, "sword_level": 1, "direction": int(s.player_direction),
        }  # fmt: skip
    row = _row(game, s)
    assert {k: row[k] for k in expected} == pytest.approx({
        k: float(v) for k, v in expected.items()
    })


@pytest.mark.parametrize("game", GAMES)
def test_tree_three_tiles_left_and_facing_stone(game: GameKind) -> None:
    s = _put(_grass_world(game), game, (0, -3), BlockType.TREE.value)
    s = _put(s, game, (0, 1), BlockType.STONE.value).replace(player_direction=2)
    row = _row(game, s)
    assert (row["tree_dist"], row["tree_row_sign"], row["tree_col_sign"]) == (3, 0, -1)
    assert (row["stone_dist"], row["stone_col_sign"]) == (1, 1)
    assert (row["diamond_dist"], row["diamond_row_sign"]) == (SENTINEL[game], 0)
    assert row["direction"] == 2
    assert row["facing_block"] == BlockType.STONE.value
    assert (row["walkable_right"], row["walkable_left"]) == (0, 1)
    assert (row["melee_count"], row["hostile_dist"]) == (0, SENTINEL[game])


def test_zombie_down_right_is_the_nearest_hostile() -> None:
    s = _grass_world("classic")
    z = s.zombies
    pos = z.position.at[0].set(s.player_position + jnp.array([2, 1]))
    s = s.replace(zombies=z.replace(position=pos, mask=z.mask.at[0].set(True)))
    row = _row("classic", s)
    assert row["melee_count"] == 1
    assert (row["hostile_dist"], row["hostile_row_sign"], row["hostile_col_sign"]) == (
        3,
        1,
        1,
    )
    assert row["hostile_adjacent"] == 0


def test_an_invisible_tile_is_ignored() -> None:
    s = _put(_grass_world("craftax"), "craftax", (0, -3), BlockType.TREE.value)
    s = _put(s, "craftax", (2, 3), BlockType.TREE.value)
    r, c = np.asarray(s.player_position) + np.array([0, -3])
    s = s.replace(light_map=s.light_map.at[0, r, c].set(0.0))
    row = _row("craftax", s)
    assert (row["tree_dist"], row["tree_row_sign"], row["tree_col_sign"]) == (5, 1, 1)


@pytest.mark.parametrize("game", GAMES)
def test_one_name_per_column_and_any_leading_shape(game: GameKind) -> None:
    obs = _obs(game, _reset(game))
    assert len(FEATURES[game]) == featurize(obs, game).shape[-1]
    assert len(set(FEATURES[game])) == len(FEATURES[game])
    np.testing.assert_array_equal(featurize(obs[0], game), featurize(obs, game)[0])


def _rollout_obs(game: GameKind, n: int) -> np.ndarray:
    """Observations of a short random-action episode (mobs, terrain, darkness vary)."""
    env = make_craftax_env_from_name(ENV_NAMES[game], auto_reset=True)
    params = env.default_params
    obs, state = env.reset(jax.random.PRNGKey(3), params)

    def step(carry: Any, key: jax.Array) -> tuple[Any, jax.Array]:
        o, s = carry
        a = jax.random.randint(key, (), 0, env.num_actions)
        o2, s2, *_ = env.step(key, s, a, params)
        return (o2, s2), o

    keys = jax.random.split(jax.random.PRNGKey(4), n)
    return np.asarray(jax.lax.scan(step, (obs, state), keys)[1])


@pytest.mark.parametrize("game", GAMES)
def test_numpy_and_jitted_jax_give_identical_features(game: GameKind) -> None:
    obs = _rollout_obs(game, 64)
    want = featurize(obs, game)
    got = jax.jit(lambda o: featurize(o, game, jnp))(jnp.asarray(obs))
    assert isinstance(got, jax.Array)
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("game", GAMES)
def test_ten_thousand_observations_are_featurized_at_once(game: GameKind) -> None:
    obs = _rollout_obs(game, 50)
    many = np.tile(obs, (200, 1))
    start = time.perf_counter()
    out = featurize(many, game)
    elapsed = time.perf_counter() - start
    np.testing.assert_array_equal(out, np.tile(featurize(obs, game), (200, 1)))
    # a Python loop over 10,000 states takes tens of seconds
    assert elapsed < 3.0
