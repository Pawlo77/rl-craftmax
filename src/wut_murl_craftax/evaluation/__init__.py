"""Evaluation and metrics: game tables shared by every metric family."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .milestones import (
        ADVANCED_ACTIONS,
        CREATURE_OFFSET,
        CREATURES,
        FLOORS,
        HOSTILE_CLASSES,
        RADAR_AXES,
        TECH_LADDER,
        Game,
        game_of,
        ladder,
    )

_EXPORTS: dict[str, str] = {
    "ADVANCED_ACTIONS": "milestones",
    "CREATURES": "milestones",
    "CREATURE_OFFSET": "milestones",
    "FLOORS": "milestones",
    "Game": "milestones",
    "HOSTILE_CLASSES": "milestones",
    "RADAR_AXES": "milestones",
    "TECH_LADDER": "milestones",
    "game_of": "milestones",
    "ladder": "milestones",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ADVANCED_ACTIONS",
    "CREATURES",
    "CREATURE_OFFSET",
    "FLOORS",
    "HOSTILE_CLASSES",
    "RADAR_AXES",
    "TECH_LADDER",
    "Game",
    "game_of",
    "ladder",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
