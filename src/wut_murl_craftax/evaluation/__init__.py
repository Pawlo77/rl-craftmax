"""Evaluation and metrics: game tables, per-step events and the reproduction check."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .episode_metrics import (
        ADDED_FIELDS,
        CAUSES,
        ONLINE_KEYS,
        ONLINE_LAYOUT,
        EpisodeAcc,
        UpdateSummary,
        added_paths,
        chunk_series,
        close_update,
        episode_step,
        events_of,
        first_unlock_offsets,
        init_acc,
        metric_keys,
        new_first_unlocks,
        resume_from,
        score_pct,
        stack_online,
        update_metrics,
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
        ADVANCED_ACHIEVEMENTS,
        ADVANCED_ACTIONS,
        CREATURE_OFFSET,
        CREATURES,
        FLOORS,
        HOSTILE_CLASSES,
        RADAR_AXES,
        TECH_LADDER,
        Game,
        achievement_names,
        action_names,
        game_of,
        ladder,
        reward_weights,
    )
    from .reproduction import Reproduction, band

_EXPORTS: dict[str, str] = {
    "ADDED_FIELDS": "episode_metrics",
    "ADVANCED_ACHIEVEMENTS": "milestones",
    "ADVANCED_ACTIONS": "milestones",
    "CAUSES": "episode_metrics",
    "CREATURES": "milestones",
    "CREATURE_OFFSET": "milestones",
    "DAMAGE_SOURCES": "events",
    "EpisodeAcc": "episode_metrics",
    "FLOORS": "milestones",
    "Game": "milestones",
    "HOSTILE_CLASSES": "milestones",
    "LAVA": "events",
    "NO_SOURCE": "events",
    "ONLINE_KEYS": "episode_metrics",
    "ONLINE_LAYOUT": "episode_metrics",
    "RADAR_AXES": "milestones",
    "Reproduction": "reproduction",
    "StepEvents": "events",
    "TECH_LADDER": "milestones",
    "UNKNOWN": "events",
    "UpdateSummary": "episode_metrics",
    "achievement_names": "milestones",
    "action_names": "milestones",
    "added_paths": "episode_metrics",
    "band": "reproduction",
    "chunk_series": "episode_metrics",
    "close_update": "episode_metrics",
    "episode_step": "episode_metrics",
    "events_of": "episode_metrics",
    "first_unlock_offsets": "episode_metrics",
    "game_of": "milestones",
    "init_acc": "episode_metrics",
    "ladder": "milestones",
    "metric_keys": "episode_metrics",
    "new_first_unlocks": "episode_metrics",
    "resume_from": "episode_metrics",
    "reward_weights": "milestones",
    "score_pct": "episode_metrics",
    "stack_online": "episode_metrics",
    "step_events": "events",
    "update_metrics": "episode_metrics",
    "update_summary": "episode_metrics",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ADDED_FIELDS",
    "ADVANCED_ACHIEVEMENTS",
    "ADVANCED_ACTIONS",
    "CAUSES",
    "CREATURES",
    "CREATURE_OFFSET",
    "DAMAGE_SOURCES",
    "FLOORS",
    "HOSTILE_CLASSES",
    "LAVA",
    "NO_SOURCE",
    "ONLINE_KEYS",
    "ONLINE_LAYOUT",
    "RADAR_AXES",
    "TECH_LADDER",
    "UNKNOWN",
    "EpisodeAcc",
    "Game",
    "Reproduction",
    "StepEvents",
    "UpdateSummary",
    "achievement_names",
    "action_names",
    "added_paths",
    "band",
    "chunk_series",
    "close_update",
    "episode_step",
    "events_of",
    "first_unlock_offsets",
    "game_of",
    "init_acc",
    "ladder",
    "metric_keys",
    "new_first_unlocks",
    "resume_from",
    "reward_weights",
    "score_pct",
    "stack_online",
    "step_events",
    "update_metrics",
    "update_summary",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
