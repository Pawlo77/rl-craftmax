"""Gameplay and training videos: rollouts, frames, highlights, review, ffmpeg cuts.

Shared building blocks: ``ffmpeg`` (encoding), ``text`` (captions), ``rollout``
(record and replay). Gameplay steps: ``render``, ``highlights``, ``review``, ``cuts``.
Learning-process steps: ``eval_rollouts``, ``grid``, ``curves``, ``timeline``,
``overlay``, ``montage``. Each step runs as
``python -m wut_murl_craftax.visualisation.<step>``.
"""

from .curves import render_curves
from .cuts import gif, long_cut, short_cut
from .eval_rollouts import EvalRun, evaluate, load_eval
from .grid import build_grid
from .highlights import candidates, detect
from .montage import sequence
from .overlay import render_overlay
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

__all__ = [
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
    "evaluate",
    "game_for",
    "gif",
    "load_eval",
    "long_cut",
    "play",
    "render_curves",
    "render_kept",
    "render_master",
    "render_overlay",
    "render_timeline",
    "replay",
    "sequence",
    "short_cut",
    "write_page",
]
