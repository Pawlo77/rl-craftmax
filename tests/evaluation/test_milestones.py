"""Tests for the game tables shared by every metric family."""

import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from craftax.craftax import constants as craftax_constants
from craftax.craftax_classic import constants as classic_constants

from wut_murl_craftax.evaluation.milestones import (
    ADVANCED_ACTIONS,
    CREATURE_OFFSET,
    CREATURES,
    FLOORS,
    HOSTILE_CLASSES,
    RADAR_AXES,
    TECH_LADDER,
    Game,
    achievement_names,
    game_of,
    ladder,
)
from wut_murl_craftax.training.records import achievement_weights

NOTE_CREATURES = {
    "melee": [
        "zombie",
        "gnome warrior",
        "orc soldier",
        "lizard",
        "knight",
        "troll",
        "pigman",
        "frost troll",
    ],
    "passive": ["cow", "bat", "snail"],
    "ranged": [
        "skeleton",
        "gnome archer",
        "orc mage",
        "kobold",
        "knight archer",
        "deep thing",
        "fire elemental",
        "ice elemental",
    ],
    "projectile": [
        "arrow",
        "dagger",
        "fireball",
        "iceball",
        "arrow2",
        "slimeball",
        "fireball2",
        "iceball2",
    ],
}
"""Creature ``type_id`` table of [[Craftax observations and actions]], by class."""


@pytest.mark.parametrize(
    ("game", "consts", "n"),
    [("craftax", craftax_constants, 10), ("classic", classic_constants, 9)],
)
def test_ladder_ids_index_the_achievement_array_by_value(
    game: Game, consts: Any, n: int
) -> None:
    ids = ladder(game)
    assert len(ids) == n
    assert list(ids) == [consts.Achievement[name].value for name in TECH_LADDER[:n]]
    assert TECH_LADDER[8:] == ("COLLECT_DIAMOND", "MAKE_DIAMOND_PICKAXE")


def test_tier_rewards_sum_to_226_and_classic_to_22() -> None:
    assert achievement_weights("Craftax-Symbolic-v1").sum() == 226
    assert achievement_weights("Craftax-Classic-Symbolic-v1").sum() == 22


def test_floors_match_the_level_achievement_map() -> None:
    enum = craftax_constants.Achievement
    level_map = np.asarray(craftax_constants.LEVEL_ACHIEVEMENT_MAP)
    assert len(FLOORS) == 9
    assert FLOORS[0] == ("overworld", None)
    for floor, (_, enter) in enumerate(FLOORS[1:], start=1):
        assert enter is not None
        assert enter.startswith("ENTER_")
        assert enum[enter].value == level_map[floor]


def test_creatures_follow_the_observation_table() -> None:
    by_class = {
        cls: [name for c, _, name in CREATURES if c == cls] for cls in NOTE_CREATURES
    }
    assert by_class == {
        cls: [n.replace(" ", "_") for n in names]
        for cls, names in NOTE_CREATURES.items()
    }
    for i, (cls, type_id, _) in enumerate(CREATURES):
        assert CREATURE_OFFSET[cls] + type_id == i
    projectiles = [p.name.lower() for p in craftax_constants.ProjectileType]
    assert by_class["projectile"] == projectiles
    assert set(HOSTILE_CLASSES) == {"melee", "ranged", "projectile"}


def test_advanced_actions_and_radar_axes() -> None:
    names = {a.name for a in craftax_constants.Action}
    assert set(ADVANCED_ACTIONS) <= names
    assert len(ADVANCED_ACTIONS) == len(set(ADVANCED_ACTIONS)) == 17
    assert len(RADAR_AXES) == len(set(RADAR_AXES)) == 8


def test_game_of_env_name() -> None:
    assert game_of("Craftax-Symbolic-v1") == "craftax"
    assert game_of("Craftax-Classic-Symbolic-v1") == "classic"


def test_importing_the_tables_leaves_craftax_unimported() -> None:
    code = (
        "import sys, wut_murl_craftax.evaluation.milestones\n"
        "print(any(m.startswith('craftax') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_achievement_names_are_in_value_order() -> None:
    names = achievement_names("craftax")
    enum = craftax_constants.Achievement
    assert len(names) == 67
    for a in (
        enum.COLLECT_SAPPHIRE,
        enum.MAKE_DIAMOND_SWORD,
        enum.DEFEAT_KNIGHT,
        enum.DEFEAT_ARCHER,
        enum.ENCHANT_ARMOUR,
    ):
        assert a.value >= 25
        assert names[a.value] == a.name.lower()
    classic = [a.name.lower() for a in classic_constants.Achievement]
    assert list(achievement_names("classic")) == classic
