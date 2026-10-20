"""Tests for the checkpoint comparison grid."""

from pathlib import Path

import numpy as np
from _learning_run import EVAL_T, needs_ffmpeg, tiny_run

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.grid import build_grid, panel_label


def test_panel_label_has_colour_key_and_text() -> None:
    img = np.asarray(panel_label("0.5M steps", "eval score 21% of max", 270, "#3987e5"))
    assert img.shape[1] == 270
    assert tuple(img[5, 2, :3]) == (0x39, 0x87, 0xE5)
    assert (img[:, 20:, :3] > 200).any()  # text drawn


@needs_ffmpeg
def test_grid_is_2x2_synced_to_the_longest_episode(tmp_path: Path) -> None:
    res = build_grid(tiny_run(), [0, 1, 2, 4], panel=96, fps=10, out=tmp_path / "g.mp4")
    assert len(res.lens) == 4
    assert all(1 <= n <= EVAL_T for n in res.lens)
    info = ffmpeg.probe(res.out)["streams"][0]
    assert (info["width"], info["height"]) == (192, 192)
    assert ffmpeg.count_frames(res.out) == max(res.lens)
