"""Progress key of a Craftax state: bucketed integer fields and their hash.

``fields`` reads floor, ladder, tools, armour, spells, enchantments and bucketed
materials from the inner ``EnvState`` (Classic: materials and one bit per tool);
scenery (map, position, mobs, daylight, survival stats) is left out. ``hash_fields``
folds a field vector of any length to ``bits`` bits. Key table and caps:
[[Count-Based Progress Bonus design]]. Everything is pure, for ``jit`` and ``vmap``;
no Craftax import (its constants start the JAX backends).
"""

from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from .wrappers import unwrap_state

type Granularity = Literal["tiers", "recipe", "log2"]
"""Key granularity knob: no materials, recipe-capped counts, or log2 buckets."""

unwrap = unwrap_state
"""The inner Craftax ``EnvState`` under ``LogWrapper`` (and the optimistic resets,
which keep the log state); the identity on an inner state."""

LADDER_KILLS = 8
"""``MONSTERS_KILLED_TO_CLEAR_LEVEL`` of craftax 1.6.1: kills that open the ladder."""
CAPS: dict[str, int] = {
    "wood": 4, "stone": 3, "coal": 3, "iron": 3, "diamond": 3, "sapphire": 1,
    "ruby": 1, "sapling": 1, "arrows": 1, "torches": 1, "books": 1, "potions": 1,
}  # fmt: skip
"""Largest amount of a material one recipe needs (``recipe`` buckets cap there)."""
LOG2_MAX = 6
"""Top ``log2`` bucket: counts 32-99."""
MATERIALS = tuple(CAPS)
"""Craftax materials in field order; ``potions`` is the sum over the six colours."""
FIELDS: tuple[str, ...] = (
    "floor", "ladder_open", "pickaxe", "sword", "bow", "armour",
    "spell_fireball", "spell_iceball", "ench_sword", "ench_bow", "ench_armour",
    *MATERIALS,
)  # fmt: skip
"""Craftax field names in ``fields`` order."""
CLASSIC_MATERIALS = ("wood", "stone", "coal", "iron", "diamond", "sapling")
"""Craftax-Classic materials in field order."""
CLASSIC_TOOLS = (
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)  # fmt: skip
"""Craftax-Classic tools, one bit each (more than one only stacks sword damage)."""
CLASSIC_FIELDS = (*CLASSIC_MATERIALS, *CLASSIC_TOOLS)
"""Craftax-Classic field names in ``fields`` order."""
KNUTH = 2654435761
"""Knuth's multiplicative-hash constant (2^32 over the golden ratio, odd)."""


def _splitmix64(i: int) -> int:
    z = (i + 1) * 0x9E3779B97F4A7C15 % 2**64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 % 2**64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB % 2**64
    return z ^ (z >> 31)


MULTIPLIERS = np.array([_splitmix64(i) % 2**32 | 1 for i in range(128)], np.uint32)
"""Fixed odd 32-bit weight per field position (SplitMix64, no RNG stream to drift),
so saved count tables keep their keys; 128 covers the 64 SimHash bits."""


def bucket(name: str, count: Any, granularity: Granularity) -> jax.Array:
    """Bucket of a material count, int32, broadcast over ``count``.

    Raises:
        ValueError: On a name outside `CAPS` or an unknown granularity.
    """
    if name not in CAPS:
        msg = f"unknown material {name!r}; expected one of {MATERIALS}"
        raise ValueError(msg)
    n = jnp.asarray(count, jnp.int32)
    match granularity:
        case "tiers":
            return jnp.zeros_like(n)
        case "recipe":
            return jnp.minimum(n, CAPS[name])
        case "log2":  # floor(log2(1 + n)) exactly, from the leading zeros
            return jnp.minimum(31 - jax.lax.clz(n + 1), LOG2_MAX)
        case _:
            msg = f"unknown granularity {granularity!r}"
            raise ValueError(msg)


def fields(state: Any, granularity: Granularity = "recipe") -> jax.Array:
    """Field vector of one inner state, int32 (`FIELDS`; Classic `CLASSIC_FIELDS`)."""
    inv = state.inventory

    def mats(names: tuple[str, ...]) -> list[jax.Array]:
        return [bucket(n, getattr(inv, n), granularity) for n in names]

    if not hasattr(inv, "pickaxe"):  # Classic keeps one inventory slot per tool
        tools = [jnp.minimum(getattr(inv, n), 1) for n in CLASSIC_TOOLS]
        return jnp.stack([*mats(CLASSIC_MATERIALS), *tools]).astype(jnp.int32)
    floor = state.player_level
    enchanted = jnp.stack([
        state.sword_enchantment > 0,
        state.bow_enchantment > 0,
        jnp.any(state.armour_enchantments > 0),
    ]) & (granularity != "tiers")
    inv = inv.replace(potions=jnp.sum(inv.potions))
    head = [
        floor,
        state.monsters_killed[floor] >= LADDER_KILLS,
        inv.pickaxe,
        inv.sword,
        jnp.minimum(inv.bow, 1),
        jnp.sum(inv.armour),
    ]
    parts = [
        jnp.stack(head),
        state.learned_spells,
        enchanted,
        jnp.stack(mats(MATERIALS)),
    ]
    return jnp.concatenate([p.astype(jnp.int32) for p in parts])


def hash_fields(fields: jax.Array, bits: int = 20) -> jax.Array:
    """Key in ``[0, 2^bits)`` of the field vectors on the last axis, int32.

    Multiply-add over `MULTIPLIERS` mod 2^32, then the Knuth fold to the top ``bits``.

    Raises:
        ValueError: On ``bits`` outside 1-31 or more fields than `MULTIPLIERS`.
    """
    n = fields.shape[-1]
    if not 0 < bits < 32 or n > len(MULTIPLIERS):
        msg = f"need 0 < bits < 32 and <= {len(MULTIPLIERS)} fields, got {bits}, {n}"
        raise ValueError(msg)
    w = MULTIPLIERS[:n]
    h = jnp.sum(fields.astype(jnp.uint32) * w, axis=-1, dtype=jnp.uint32)
    return cast(jax.Array, ((h * jnp.uint32(KNUTH)) >> (32 - bits)).astype(jnp.int32))
