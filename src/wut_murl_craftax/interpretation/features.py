"""Named, human-readable features of symbolic observations for tree distillation.

`featurize` maps flat observations ``(..., D)`` of either game to the columns of
`FEATURES` (layout: [[Craftax observations and actions]]). It is array-API agnostic:
with ``xp=jax.numpy`` it jits inside closed-loop tree play. The map comes from
`visualisation.overlay.decode_obs`, flat offsets from `CRAFTAX_BARS`; this module only
adds the item and visibility channels the overlay drops. Categorical columns:
``direction`` (`Action` id 1-4), ``facing_block`` (`BlockType` id), ``facing_creature``
(`CREATURES` index). Distances are Manhattan over visible tiles; with nothing in view
they are the sentinel (one above the largest in-view distance) and both signs are 0.
"""

import functools
from dataclasses import dataclass
from types import ModuleType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax import constants as craftax_constants
from craftax.craftax_classic import constants as classic_constants

from ..visualisation.overlay import CLASSIC_BARS, CRAFTAX_BARS, decode_obs
from ..visualisation.rollout import MOVES, GameKind

type Array = np.ndarray | jax.Array

CREATURES = ("none", "melee", "passive", "ranged", "projectile")
"""``facing_creature`` codes: the creature classes both games share, Craftax order."""
_CHUNK = 4096
"""Rows per numpy pass; bounds the float32 copy ``decode_obs`` makes (about 135 MB)."""
_TARGETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "tree": (("TREE", "FIRE_TREE", "ICE_SHRUB"), ()),  # all three give wood
    "water": (("WATER", "FOUNTAIN"), ()),  # both quench thirst
    "stone": (("STONE",), ()),
    "coal": (("COAL",), ()),
    "iron": (("IRON",), ()),
    "diamond": (("DIAMOND",), ()),
    "sapphire": (("SAPPHIRE",), ()),
    "ruby": (("RUBY",), ()),
    "crafting_table": (("CRAFTING_TABLE",), ()),
    "furnace": (("FURNACE",), ()),
    "lava": (("LAVA",), ()),
    "chest": (("CHEST",), ()),
    "ripe_plant": (("RIPE_PLANT",), ()),
    "down_ladder": ((), ("LADDER_DOWN", "LADDER_DOWN_BLOCKED")),  # open: ladder_open
    "up_ladder": ((), ("LADDER_UP",)),
    "fire_table": (("ENCHANTMENT_TABLE_FIRE",), ()),
    "ice_table": (("ENCHANTMENT_TABLE_ICE",), ()),
}
"""Nearest-target name -> (block types, item types); a game keeps the types it has."""
_CONSTANTS = {"craftax": craftax_constants, "classic": classic_constants}
_GROUPS: dict[GameKind, tuple[str, ...]] = {
    "craftax": (
        *("health", "food", "drink", "energy", "mana", "xp"),
        *("dexterity", "strength", "intelligence"),
        *("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling"),
        *("torches", "arrows", "books", "pickaxe_level", "sword_level"),
        *("sword_enchantment", "bow_enchantment", "bow", "helmet_level"),
        *("chestplate_level", "leggings_level", "boots_level", "armour_enchanted"),
        *(f"potion_{c}" for c in ("red", "green", "blue", "pink", "cyan", "yellow")),
        *("light_level", "sleeping", "resting", "learned_fireball"),
        *("learned_iceball", "floor", "ladder_open", "boss_vulnerable"),
    ),
    "classic": (
        *("health", "food", "drink", "energy"),
        *("wood", "stone", "coal", "iron", "diamond", "sapling"),
        *("pickaxe_level", "sword_level", "light_level", "sleeping"),
    ),
}
"""Scalar columns read from the flat tail of the observation, per game."""
_MAP_COLUMNS = (
    *("direction", "facing_block", "facing_creature"),
    *("melee_count", "passive_count", "ranged_count", "projectile_count"),
    *("hostile_dist", "hostile_row_sign", "hostile_col_sign", "hostile_adjacent"),
    *("passive_dist", "walkable_left", "walkable_right", "walkable_up"),
    "walkable_down",
)
"""Columns decoded from the map view (after the scalars, before the targets)."""


@dataclass(frozen=True, slots=True)
class _Layout:
    """Per-game constants of the decoding, numpy arrays indexed by flat tile."""

    targets: tuple[str, ...]
    block_hits: np.ndarray
    """Bool ``(n_blocks, targets)``: block type counts as that target."""
    item_hits: np.ndarray | None
    """Bool ``(n_items, targets)``; None in Classic (no item channels)."""
    walkable: np.ndarray
    """Bool per block type: safe to step on (not solid, water, lava or out of view)."""
    dist: np.ndarray
    drow: np.ndarray
    dcol: np.ndarray
    neighbours: np.ndarray
    """Flat tile index of the left, right, up and down neighbour of the player."""
    sentinel: int
    map_size: int


@functools.cache
def _layout(game: GameKind) -> _Layout:
    consts = _CONSTANTS[game]
    blocks, items = consts.BlockType.__members__, getattr(consts, "ItemType", None)
    item_names = items.__members__ if items else {}
    kept = {
        name: (
            [blocks[b].value for b in bs if b in blocks],
            [item_names[i].value for i in its if i in item_names],
        )
        for name, (bs, its) in _TARGETS.items()
    }
    kept = {name: hits for name, hits in kept.items() if hits[0] or hits[1]}
    block_hits = np.zeros((len(blocks), len(kept)), bool)
    item_hits = np.zeros((len(item_names), len(kept)), bool)
    for k, (bs, its) in enumerate(kept.values()):
        block_hits[bs, k], item_hits[its, k] = True, True
    walkable = ~np.isin(np.arange(len(blocks)), np.asarray(consts.SOLID_BLOCKS))
    walkable[
        [blocks[b].value for b in ("INVALID", "OUT_OF_BOUNDS", "WATER", "LAVA")]
    ] = False
    rows, cols = consts.OBS_DIM
    drow, dcol = (
        g.ravel()
        for g in np.indices((rows, cols))
        - np.array([rows // 2, cols // 2])[:, None, None]
    )
    centre = (rows // 2) * cols + cols // 2
    channels = len(blocks) + (4 if game == "classic" else len(item_names) + 41)
    return _Layout(
        targets=tuple(kept),
        block_hits=block_hits,
        item_hits=item_hits if items else None,
        walkable=walkable,
        dist=np.abs(drow) + np.abs(dcol),
        drow=drow,
        dcol=dcol,
        neighbours=np.array([
            centre + r * cols + c for r, c in (MOVES[a] for a in sorted(MOVES))
        ]),
        sentinel=rows // 2 + cols // 2 + 1,
        map_size=rows * cols * channels,
    )


def _names(game: GameKind) -> list[str]:
    targets = _layout(game).targets
    near = [f"{t}_{part}" for t in targets for part in ("dist", "row_sign", "col_sign")]
    return [*_GROUPS[game], *_MAP_COLUMNS, *near]


FEATURES: dict[GameKind, list[str]] = {g: _names(g) for g in ("craftax", "classic")}
"""Column names of `featurize`, per game (111 Craftax, 60 Classic)."""


def featurize(obs: Array, game: GameKind, xp: ModuleType | None = None) -> Array:
    """Float32 features ``(..., len(FEATURES[game]))`` of observations ``(..., D)``.

    ``xp`` is ``numpy`` or ``jax.numpy`` (default: the module of ``obs``); with
    ``jax.numpy`` the function is traceable by ``jax.jit``.
    """
    xp = xp or (jnp if isinstance(obs, jax.Array) else np)
    flat = xp.reshape(obs, (-1, obs.shape[-1]))
    out: Array
    if xp is np and len(flat) > _CHUNK:
        chunks = range(0, len(flat), _CHUNK)
        out = np.concatenate([
            _featurize(flat[i : i + _CHUNK], game, xp) for i in chunks
        ])
    else:
        out = _featurize(flat, game, xp)
    return cast(Array, xp.reshape(out, (*obs.shape[:-1], out.shape[-1])))


def _featurize(obs: Array, game: GameKind, xp: ModuleType) -> Array:
    lay = _layout(game)
    view = decode_obs(cast(np.ndarray, obs), game == "classic")  # jax arrays work too
    n = obs.shape[0]
    blocks = xp.reshape(view.blocks, (n, -1))
    mobs = xp.reshape(view.mobs, (n, blocks.shape[1], -1))[..., :4]
    hits = xp.asarray(lay.block_hits)[blocks]
    if lay.item_hits is not None:  # Craftax: items and the visibility channel
        tiles = xp.reshape(obs[:, : lay.map_size], (n, blocks.shape[1], -1))
        items = xp.argmax(tiles[..., lay.block_hits.shape[0] : -41], axis=-1)
        hits = (hits | xp.asarray(lay.item_hits)[items]) & (tiles[..., -1:] > 0)
    if game == "craftax":
        scalars, facing = _craftax_scalars(obs, xp)
    else:
        scalars, facing = _classic_scalars(obs, view.bars, lay.map_size, xp)
    near_blocks = blocks[:, lay.neighbours]
    near_mobs = mobs[:, lay.neighbours]
    faced = xp.sum(near_mobs * facing[..., None], axis=1) > 0
    creature = xp.where(xp.any(faced, axis=-1), xp.argmax(faced, axis=-1) + 1, 0)
    hostile = mobs[..., 0] | mobs[..., 2]
    dist, rs, cs = _nearest(xp.stack([hostile, mobs[..., 1]], axis=-1), lay, xp)
    columns = [
        scalars,
        xp.argmax(facing, axis=-1)[:, None] + 1,
        xp.sum(near_blocks * facing, axis=-1)[:, None],
        creature[:, None],
        xp.sum(mobs, axis=1),
        dist[:, :1],
        rs[:, :1],
        cs[:, :1],
        xp.any(hostile & (xp.asarray(lay.dist) == 1), axis=-1)[:, None],
        dist[:, 1:],
        xp.asarray(lay.walkable)[near_blocks] & ~xp.any(near_mobs[..., :3], axis=-1),
        xp.reshape(xp.stack(_nearest(hits, lay, xp), axis=-1), (n, -1)),
    ]
    return cast(Array, xp.concatenate([xp.asarray(c, xp.float32) for c in columns], 1))


def _nearest(hits: Array, lay: _Layout, xp: ModuleType) -> tuple[Array, Array, Array]:
    """Distance and row and column offset signs of the nearest hit, per target."""
    d = xp.where(hits, xp.asarray(lay.dist)[:, None], lay.sentinel)
    idx = xp.argmin(d, axis=1)  # first in row-major order among ties
    dist = xp.min(d, axis=1)
    seen = dist < lay.sentinel
    sign = [
        xp.where(seen, xp.sign(xp.asarray(o)[idx]), 0) for o in (lay.drow, lay.dcol)
    ]
    return dist, sign[0], sign[1]


def _craftax_scalars(obs: Array, xp: ModuleType) -> tuple[Array, Array]:
    """Craftax tail columns in `_GROUPS` order, and the facing one-hot (L, R, U, D)."""
    inv, vit = CRAFTAX_BARS["wood"], CRAFTAX_BARS["health"]
    tail = xp.asarray(obs[:, inv:], dtype=xp.float32)
    vit -= inv
    columns = [
        tail[:, vit : vit + 9] * 10,
        xp.round((tail[:, :10] * 10) ** 2),  # sqrt(n) / 10 encoding
        xp.round(tail[:, 10:13] * xp.asarray([2.0, 4.0, 4.0])),  # books, pickaxe, sword
        tail[:, 13:16],
        xp.round(tail[:, vit + 13 : vit + 17] * 2),  # armour levels
        xp.any(tail[:, vit + 17 : vit + 21] > 0, axis=-1)[:, None],
        xp.round((tail[:, 16:22] * 10) ** 2),
        tail[:, vit + 21 : vit + 26],
        xp.round(tail[:, vit + 26 : vit + 27] * 10),  # floor
        tail[:, vit + 27 : vit + 29],
    ]
    out = xp.concatenate([xp.asarray(c, dtype=xp.float32) for c in columns], axis=1)
    return out, tail[:, vit + 9 : vit + 13]


def _classic_scalars(
    obs: Array, bars: Array, map_size: int, xp: ModuleType
) -> tuple[Array, Array]:
    """Classic tail columns in `_GROUPS` order, and the facing one-hot (L, R, U, D)."""
    counts = xp.round(bars * 10)  # n / 10, through float16
    tools = xp.asarray([1.0, 2.0, 3.0])
    rest = xp.asarray(obs[:, map_size + len(CLASSIC_BARS) :], dtype=xp.float32)
    columns = [
        counts[:, 12:16],
        counts[:, :6],
        xp.max((counts[:, 6:9] > 0) * tools, axis=-1)[:, None],
        xp.max((counts[:, 9:12] > 0) * tools, axis=-1)[:, None],
        rest[:, 4:6],
    ]
    out = xp.concatenate([xp.asarray(c, dtype=xp.float32) for c in columns], axis=1)
    return out, rest[:, :4]
