"""Exploration coverage of an episode: tiles seen per floor, new tiles, revisits.

``seen`` is ``(floors, rows, cols)`` int32: the ``timestep`` a tile was first inside
the view window and lit (``light_map > 0.05``, as the renderer masks it), -1 if never;
``seen >= 0`` is the seen mask. Classic has one floor and no light map, so every tile
in its 7 x 9 view counts. `update_seen` is pure and runs under ``jit``; a replay scans
it over the states after `init_seen` of the first. Definitions: "Exploration
coverage", "New tiles per 100 steps" and "Revisit rate" of the evaluation plan.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .milestones import FLOORS, Game, constants

VISIBLE_LIGHT = 0.05
"""A tile with ``light_map`` at most this is dark (Craftax renderer threshold)."""


def update_seen(seen: jax.Array, s2: Any, game: Game) -> jax.Array:
    """``seen`` with the lit tiles of ``s2``'s view stamped by ``s2.timestep``."""
    rows, cols = constants(game).OBS_DIM
    level = 0 if game == "classic" else s2.player_level
    p = s2.player_position
    r = jnp.arange(seen.shape[1])[:, None]
    c = jnp.arange(seen.shape[2])[None, :]
    view = (jnp.abs(r - p[0]) <= rows // 2) & (jnp.abs(c - p[1]) <= cols // 2)
    if game == "craftax":
        view &= s2.light_map[level] > VISIBLE_LIGHT
    floor = seen[level]
    return seen.at[level].set(jnp.where(view & (floor < 0), s2.timestep, floor))


def init_seen(s0: Any, game: Game) -> jax.Array:
    """The view of the first state ``s0``, on a map shaped like ``s0.map``."""
    shape = s0.map.shape if game == "craftax" else (1, *s0.map.shape)
    return update_seen(jnp.full(shape, -1, jnp.int32), s0, game)


def coverage_summary(
    seen: Any, positions: np.ndarray, floors: np.ndarray
) -> dict[str, float | int]:
    """Coverage per floor, new tiles per 100 steps, revisit rate, unique positions.

    ``positions`` and ``floors`` are the player's in each of the ``n + 1`` states.
    Coverage is the seen share of the floor's map, NaN on a floor never visited. New
    tiles are those first seen after the first state, per 100 steps: the time-weighted
    mean of the plan's 100-step windows. A step revisits when its position on its floor
    was held in an earlier state (standing still counts).
    """
    seen = np.asarray(seen)
    f, h, w = seen.shape
    floors = np.asarray(floors, np.int64)
    pos = np.asarray(positions, np.int64)
    steps = len(pos) - 1
    known = seen >= 0
    first = seen[known].min() if known.any() else 0
    unique = len(np.unique((floors * h + pos[:, 0]) * w + pos[:, 1]))
    visited = set(np.unique(floors).tolist())
    cover = {
        f"coverage/{FLOORS[k][0]}": float(known[k].mean()) if k in visited else math.nan
        for k in range(f)
    }
    nan = math.nan
    return cover | {
        "new_tiles_per_100": 100 * float((seen > first).sum()) / steps
        if steps
        else nan,
        "revisit_rate": (steps - unique + 1) / steps if steps else nan,
        "unique_positions": unique,
    }
