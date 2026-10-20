"""Tests for the achievement timeline."""

from pathlib import Path

import numpy as np
from _learning_run import needs_ffmpeg, tiny_run

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.timeline import first_unlock_order, render_timeline


def test_rows_sorted_by_first_unlock_never_last() -> None:
    first = np.array([5, -1, 1, 5, -1, 3])
    assert first_unlock_order(first).tolist() == [2, 5, 0, 3, 1, 4]


@needs_ffmpeg
def test_timeline_frames_and_thumbnail(tmp_path: Path) -> None:
    out = render_timeline(
        tiny_run(), tmp_path / "t.mkv", seconds=1, fps=6, hold=0.5, size=(480, 300)
    )
    assert ffmpeg.count_frames(out) == 9
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (480, 300)
    assert out.with_suffix(".png").exists()
