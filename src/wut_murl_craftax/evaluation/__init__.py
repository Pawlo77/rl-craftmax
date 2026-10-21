"""Evaluation and metrics: game tables and the per-step events every metric reads."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .events import (
        DAMAGE_SOURCES,
        LAVA,
        NO_SOURCE,
        UNKNOWN,
        StepEvents,
        step_events,
    )
    from .milestones import (
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

_EXPORTS: dict[str, str] = {
    "ADVANCED_ACTIONS": "milestones",
    "CREATURES": "milestones",
    "CREATURE_OFFSET": "milestones",
    "DAMAGE_SOURCES": "events",
    "FLOORS": "milestones",
    "Game": "milestones",
    "HOSTILE_CLASSES": "milestones",
    "LAVA": "events",
    "NO_SOURCE": "events",
    "RADAR_AXES": "milestones",
    "StepEvents": "events",
    "TECH_LADDER": "milestones",
    "UNKNOWN": "events",
    "achievement_names": "milestones",
    "game_of": "milestones",
    "ladder": "milestones",
    "step_events": "events",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ADVANCED_ACTIONS",
    "CREATURES",
    "CREATURE_OFFSET",
    "DAMAGE_SOURCES",
    "FLOORS",
    "HOSTILE_CLASSES",
    "LAVA",
    "NO_SOURCE",
    "RADAR_AXES",
    "TECH_LADDER",
    "UNKNOWN",
    "Game",
    "StepEvents",
    "achievement_names",
    "game_of",
    "ladder",
    "step_events",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
