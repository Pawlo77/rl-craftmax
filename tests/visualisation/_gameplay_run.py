"""One short Classic heuristic rollout and its 16 px master, built once per session."""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import pytest

from wut_murl_craftax.visualisation.render import render_master
from wut_murl_craftax.visualisation.rollout import (
    HeuristicPolicy,
    Rollout,
    game_for,
    play,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@functools.cache
def classic_rollout() -> Rollout:
    """60 heuristic steps on Craftax-Classic, seed 1."""
    game = game_for("classic")
    return play(game, HeuristicPolicy(game), seed=1, max_steps=60)


@functools.cache
def classic_master() -> Path:
    """``classic_heur_1.mkv``: the rollout above as a lossless 16 px master."""
    root = Path(tempfile.mkdtemp(prefix="gameplay_run_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    out = root / "classic_heur_1.mkv"
    render_master(classic_rollout(), out)
    return out
