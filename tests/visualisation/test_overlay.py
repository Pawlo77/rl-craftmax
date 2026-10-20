"""Tests for the agent internals overlay."""

from pathlib import Path

import jax
import numpy as np
import pytest
from _learning_run import needs_ffmpeg, tiny_run
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.overlay import (
    block_colors,
    decode_obs,
    discounted_return,
    render_overlay,
)


@pytest.mark.parametrize(
    ("env_name", "classic", "rows", "cols", "n_mobs"),
    [
        ("Craftax-Classic-Symbolic-v1", True, 7, 9, 4),
        ("Craftax-Symbolic-v1", False, 9, 11, 5),
    ],
)
def test_decode_obs_matches_the_map_view(
    env_name: str, classic: bool, rows: int, cols: int, n_mobs: int
) -> None:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    obs, _ = env.reset(jax.random.PRNGKey(0), env.default_params)
    view = decode_obs(np.asarray(obs)[None], classic)
    assert view.blocks.shape == (1, rows, cols)
    assert view.mobs.shape == (1, rows, cols, n_mobs)
    assert len(view.bar_labels) == view.bars.shape[1]
    assert view.bars.min() >= 0
    assert len(block_colors(classic)) > view.blocks.max()
    # the agent stands on a walkable tile at the centre of the view
    assert view.blocks[0, rows // 2, cols // 2] > 1


def test_discounted_return_stops_at_episode_end() -> None:
    r = np.array([1.0, 0.0, 1.0, 5.0])
    np.testing.assert_allclose(discounted_return(r, 3, 0.5), [1.25, 0.5, 1.0, 0.0])


@needs_ffmpeg
def test_overlay_is_game_plus_panel_one_frame_per_step(tmp_path: Path) -> None:
    out = render_overlay(
        tiny_run(),
        update=4,
        seed=0,
        steps=(0, 6),
        fps=5,
        game_px=144,
        panel=(200, 144),
        out=tmp_path / "o.mp4",
    )
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (344, 144)
    assert ffmpeg.count_frames(out) == 6
