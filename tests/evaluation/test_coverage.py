"""Tests for exploration coverage: the seen-tile update and the episode summary."""

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.evaluation.coverage import (
    coverage_summary,
    init_seen,
    update_seen,
)


class Craftax(NamedTuple):
    """The state fields the update reads."""

    map: jax.Array
    light_map: jax.Array
    player_level: jax.Array
    player_position: jax.Array
    timestep: jax.Array


class Classic(NamedTuple):
    map: jax.Array
    player_position: jax.Array
    timestep: jax.Array


def craftax(
    pos: tuple[int, int], t: int = 0, floor: int = 0, light: Any = None
) -> Craftax:
    lit = jnp.ones((9, 48, 48)) if light is None else light
    return Craftax(
        jnp.zeros((9, 48, 48), jnp.int32),
        lit,
        jnp.int32(floor),
        jnp.array(pos, jnp.int32),
        jnp.int32(t),
    )


def walk(states: list[Any], game: str) -> np.ndarray:
    seen = init_seen(states[0], game)
    for s in states[1:]:
        seen = update_seen(seen, s, game)
    return np.asarray(seen)


def test_ten_steps_right_on_an_open_overworld_add_ten_view_columns() -> None:
    states = [craftax((24, 10 + t), t) for t in range(11)]
    start = np.asarray(init_seen(states[0], "craftax"))
    seen = walk(states, "craftax")
    assert (start >= 0).sum() == 9 * 11
    assert (seen >= 0).sum() - (start >= 0).sum() == 10 * 9
    assert np.flatnonzero((seen[0] >= 0).any(0)).tolist() == list(range(5, 26))
    pos = np.array([[24, 10 + t] for t in range(11)])
    out = coverage_summary(seen, pos, np.zeros(11, int))
    assert out["new_tiles_per_100"] == pytest.approx(100 * 90 / 10)
    assert out["coverage/overworld"] == pytest.approx(189 / 48**2)


def test_dark_tiles_on_a_dark_floor_are_not_counted() -> None:
    light = jnp.zeros((9, 48, 48)).at[1, 23:26, 23:26].set(0.06)
    light = light.at[1, 20, 24].set(0.05)  # in view but not above the threshold
    light = light.at[1, 40, 40].set(1.0)  # lit but out of view
    seen = np.asarray(init_seen(craftax((24, 24), floor=1, light=light), "craftax"))
    assert (seen >= 0).sum() == 9
    assert (seen[1, 23:26, 23:26] == 0).all()


def test_returning_to_a_visited_position_raises_the_revisit_count() -> None:
    seen = walk([craftax((24, 24))], "craftax")
    there = np.array([[24, 24], [24, 25], [24, 26], [24, 27]])
    back = np.array([[24, 24], [24, 25], [24, 26], [24, 25]])
    floors = np.zeros(4, int)
    away = coverage_summary(seen, there, floors)
    home = coverage_summary(seen, back, floors)
    assert (away["revisit_rate"], away["unique_positions"]) == (0.0, 4)
    assert (home["revisit_rate"], home["unique_positions"]) == (pytest.approx(1 / 3), 3)
    deeper = coverage_summary(seen, back, np.array([0, 0, 0, 1]))
    assert deeper["revisit_rate"] == 0.0  # the same tile on another floor is new


def test_coverage_per_floor_is_a_share_and_nan_for_unvisited_floors() -> None:
    rng = np.random.default_rng(0)
    pos, states = np.array([24, 24]), []
    for t in range(300):
        pos = np.clip(pos + rng.integers(-1, 2, 2), 0, 47)
        states.append(craftax(tuple(pos), t, floor=int(t >= 150)))
    seen = walk(states, "craftax")
    positions = np.array([s.player_position for s in states])
    floors = np.array([int(s.player_level) for s in states])
    out = coverage_summary(seen, positions, floors)
    for f in ("overworld", "dungeon"):
        assert 0 < out[f"coverage/{f}"] <= 1
    assert math.isnan(out["coverage/sewers"])


def test_classic_uses_its_own_map_size() -> None:
    s = Classic(jnp.zeros((64, 64), jnp.int32), jnp.array([32, 32]), jnp.int32(0))
    seen = np.asarray(init_seen(s, "classic"))
    assert seen.shape == (1, 64, 64)
    assert (seen >= 0).sum() == 7 * 9
    out = coverage_summary(seen, np.array([[32, 32]]), np.zeros(1, int))
    assert out["coverage/overworld"] == pytest.approx(63 / 64**2)
    assert [k for k in out if k.startswith("coverage/")] == ["coverage/overworld"]


def test_the_update_runs_under_jit_and_clips_the_view_at_the_map_edge() -> None:
    seen = init_seen(craftax((0, 0)), "craftax")
    step = jax.jit(update_seen, static_argnums=2)
    got = step(seen, craftax((0, 1), 1), "craftax")
    np.testing.assert_array_equal(got, update_seen(seen, craftax((0, 1), 1), "craftax"))
    assert (np.asarray(got) >= 0).sum() == 5 * 7


def test_an_invalid_padded_step_leaves_seen_unchanged() -> None:
    seen = init_seen(craftax((24, 24)), "craftax")
    later = craftax((24, 40), 5)
    kept = update_seen(seen, later, "craftax", jnp.bool_(False))
    np.testing.assert_array_equal(kept, seen)
    batch = jax.vmap(update_seen, (0, 0, None, 0))(
        jnp.stack([seen, seen]),
        jax.tree.map(lambda x: jnp.stack([x, x]), later),
        "craftax",
        jnp.array([True, False]),
    )
    assert (batch[0] >= 0).sum() > (batch[1] >= 0).sum() == 99


def test_real_start_states_see_the_whole_view(
    craftax_start: Any, classic_start: Any
) -> None:
    assert (np.asarray(init_seen(craftax_start, "craftax")) >= 0).sum() == 9 * 11
    assert (np.asarray(init_seen(classic_start, "classic")) >= 0).sum() == 7 * 9
