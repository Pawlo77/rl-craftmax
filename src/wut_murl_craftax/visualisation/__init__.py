"""Gameplay and training videos: rollouts, frames, highlights, review, ffmpeg cuts.

Shared building blocks: ``ffmpeg`` (encoding), ``text`` (captions), ``rollout``
(record and replay). Gameplay steps: ``render``, ``highlights``, ``review``, ``cuts``.
Learning-process steps: ``eval_rollouts``, ``grid``, ``curves``, ``timeline``,
``overlay``, ``montage``. Each step runs as
``python -m wut_murl_craftax.visualisation.<step>``.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .curves import render_curves
    from .cuts import gif, long_cut, short_cut
    from .eval_rollouts import (
        Q_FAMILIES,
        EvalRun,
        episode_outputs,
        evaluate,
        evaluate_spec,
        load_eval,
        network_family,
        network_outputs,
        replay_episode,
    )
    from .grid import build_grid
    from .highlights import candidates, detect
    from .montage import sequence
    from .overlay import render_overlay, value_label
    from .render import render_kept, render_master
    from .review import build_assets, write_page
    from .rollout import (
        HeuristicPolicy,
        NetworkPolicy,
        Policy,
        RandomPolicy,
        Rollout,
        game_for,
        play,
        replay,
    )
    from .timeline import render_timeline

_EXPORTS: dict[str, str] = {
    "EvalRun": "eval_rollouts",
    "Q_FAMILIES": "eval_rollouts",
    "HeuristicPolicy": "rollout",
    "NetworkPolicy": "rollout",
    "Policy": "rollout",
    "RandomPolicy": "rollout",
    "Rollout": "rollout",
    "build_assets": "review",
    "build_grid": "grid",
    "candidates": "highlights",
    "detect": "highlights",
    "episode_outputs": "eval_rollouts",
    "evaluate": "eval_rollouts",
    "evaluate_spec": "eval_rollouts",
    "game_for": "rollout",
    "gif": "cuts",
    "load_eval": "eval_rollouts",
    "long_cut": "cuts",
    "network_family": "eval_rollouts",
    "network_outputs": "eval_rollouts",
    "play": "rollout",
    "render_curves": "curves",
    "render_kept": "render",
    "render_master": "render",
    "render_overlay": "overlay",
    "render_timeline": "timeline",
    "replay": "rollout",
    "replay_episode": "eval_rollouts",
    "sequence": "montage",
    "short_cut": "cuts",
    "value_label": "overlay",
    "write_page": "review",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "Q_FAMILIES",
    "EvalRun",
    "HeuristicPolicy",
    "NetworkPolicy",
    "Policy",
    "RandomPolicy",
    "Rollout",
    "build_assets",
    "build_grid",
    "candidates",
    "detect",
    "episode_outputs",
    "evaluate",
    "evaluate_spec",
    "game_for",
    "gif",
    "load_eval",
    "long_cut",
    "network_family",
    "network_outputs",
    "play",
    "render_curves",
    "render_kept",
    "render_master",
    "render_overlay",
    "render_timeline",
    "replay",
    "replay_episode",
    "sequence",
    "short_cut",
    "value_label",
    "write_page",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
