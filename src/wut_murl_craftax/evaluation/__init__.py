"""Evaluation and metrics: game tables, per-step events and the reproduction check."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .episode_metrics import (
        EpisodeAcc,
        UpdateSummary,
        episode_step,
        events_of,
        first_unlock_offsets,
        init_acc,
        score_pct,
        update_summary,
    )
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
    from .reproduction import Reproduction, band

_EXPORTS: dict[str, str] = {
    "ADVANCED_ACTIONS": "milestones",
    "CREATURES": "milestones",
    "CREATURE_OFFSET": "milestones",
    "DAMAGE_SOURCES": "events",
    "EpisodeAcc": "episode_metrics",
    "FLOORS": "milestones",
    "Game": "milestones",
    "HOSTILE_CLASSES": "milestones",
    "LAVA": "events",
    "NO_SOURCE": "events",
    "RADAR_AXES": "milestones",
    "Reproduction": "reproduction",
    "StepEvents": "events",
    "TECH_LADDER": "milestones",
    "UNKNOWN": "events",
    "UpdateSummary": "episode_metrics",
    "achievement_names": "milestones",
    "band": "reproduction",
    "episode_step": "episode_metrics",
    "events_of": "episode_metrics",
    "first_unlock_offsets": "episode_metrics",
    "game_of": "milestones",
    "init_acc": "episode_metrics",
    "ladder": "milestones",
    "score_pct": "episode_metrics",
    "step_events": "events",
    "update_summary": "episode_metrics",
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
    "EpisodeAcc",
    "Game",
    "Reproduction",
    "StepEvents",
    "UpdateSummary",
    "achievement_names",
    "band",
    "episode_step",
    "events_of",
    "first_unlock_offsets",
    "game_of",
    "init_acc",
    "ladder",
    "score_pct",
    "step_events",
    "update_summary",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
