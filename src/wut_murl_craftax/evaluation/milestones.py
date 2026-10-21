"""Game tables the metrics share: tech ladder, floors, creatures, actions, radar axes.

Tables hold names only, so importing this module never imports Craftax (its constants
build JAX arrays and start the backends); ``ladder`` looks ids up on first use.
Facts: ``knowledge/game/`` and [[Craftax observations and actions]].
"""

import importlib
from typing import Literal

from ..training.records import is_classic

type Game = Literal["craftax", "classic"]

TECH_LADDER: tuple[str, ...] = (
    "COLLECT_WOOD",
    "PLACE_TABLE",
    "MAKE_WOOD_PICKAXE",
    "COLLECT_STONE",
    "MAKE_STONE_PICKAXE",
    "PLACE_FURNACE",
    "COLLECT_IRON",
    "MAKE_IRON_PICKAXE",
    "COLLECT_DIAMOND",
    "MAKE_DIAMOND_PICKAXE",
)
"""Progress-funnel achievements in tech order; Classic stops at ``COLLECT_DIAMOND``."""
FLOORS: tuple[tuple[str, str | None], ...] = (
    ("overworld", None),
    ("dungeon", "ENTER_DUNGEON"),
    ("gnomish_mines", "ENTER_GNOMISH_MINES"),
    ("sewers", "ENTER_SEWERS"),
    ("vaults", "ENTER_VAULT"),
    ("troll_mines", "ENTER_TROLL_MINES"),
    ("fire_realm", "ENTER_FIRE_REALM"),
    ("ice_realm", "ENTER_ICE_REALM"),
    ("graveyard", "ENTER_GRAVEYARD"),
)
"""(name, achievement for entering it) per ``player_level`` 0-8; Classic is floor 0."""
_TYPES: dict[str, tuple[str, ...]] = {
    "melee": (
        "zombie",
        "gnome_warrior",
        "orc_soldier",
        "lizard",
        "knight",
        "troll",
        "pigman",
        "frost_troll",
    ),
    "passive": ("cow", "bat", "snail"),
    "ranged": (
        "skeleton",
        "gnome_archer",
        "orc_mage",
        "kobold",
        "knight_archer",
        "deep_thing",
        "fire_elemental",
        "ice_elemental",
    ),
    "projectile": (
        "arrow",
        "dagger",
        "fireball",
        "iceball",
        "arrow2",
        "slimeball",
        "fireball2",
        "iceball2",
    ),
}
CREATURES: tuple[tuple[str, int, str], ...] = tuple(
    (cls, type_id, name)
    for cls, names in _TYPES.items()
    for type_id, name in enumerate(names)
)
"""(class, ``type_id``, name) in observation class order; Classic zombies, cows,
skeletons and arrows are type 0 of melee, passive, ranged and projectile."""
CREATURE_OFFSET: dict[str, int] = {
    cls: [c for c, _, _ in CREATURES].index(cls) for cls in _TYPES
}
"""Index in ``CREATURES`` of type 0 of each class: index = offset + ``type_id``."""
HOSTILE_CLASSES: tuple[str, ...] = ("melee", "ranged", "projectile")
"""Classes that damage the player (enemy projectiles only; player ones never hit it)."""
ADVANCED_ACTIONS: tuple[str, ...] = (
    "CAST_FIREBALL",
    "CAST_ICEBALL",
    "READ_BOOK",
    "DRINK_POTION_RED",
    "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE",
    "DRINK_POTION_PINK",
    "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW",
    "ENCHANT_SWORD",
    "ENCHANT_ARMOUR",
    "ENCHANT_BOW",
    "MAKE_ARROW",
    "SHOOT_ARROW",
    "LEVEL_UP_DEXTERITY",
    "LEVEL_UP_STRENGTH",
    "LEVEL_UP_INTELLIGENCE",
)
"""Craftax actions counted by the advanced-mechanics family ([[Craftax magic potions
and enchanting]])."""
RADAR_AXES: tuple[str, ...] = (
    "explorer",
    "survivor",
    "fighter",
    "crafter",
    "diver",
    "efficient",
    "mage",
    "precise",
)
"""Playstyle radar axes, in drawing order."""


def game_of(env_name: str) -> Game:
    """Game of a Craftax env name."""
    return "classic" if is_classic(env_name) else "craftax"


def ladder(game: Game) -> tuple[int, ...]:
    """Achievement ids of the tech ladder by ``.value`` (enum order is not id order)."""
    module = "craftax_classic" if game == "classic" else "craftax"
    enum = importlib.import_module(f"craftax.{module}.constants").Achievement
    names = TECH_LADDER[:9] if game == "classic" else TECH_LADDER
    return tuple(enum[name].value for name in names)
