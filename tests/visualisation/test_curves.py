"""Tests for the animated training curves: cursor sync and rendered clip."""

from pathlib import Path

import numpy as np
from _learning_run import needs_ffmpeg, tiny_run

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.curves import grid_cursor, render_curves, seq_cursor


def test_grid_cursor_reveals_then_holds() -> None:
    cur = grid_cursor(10, n_updates=5, reveal=0.7)
    assert cur.tolist() == [0, 0, 1, 2, 2, 3, 4, 4, 4, 4]


def test_seq_cursor_jumps_to_the_checkpoint_on_screen() -> None:
    cur, seg = seq_cursor([0, 2, 4], [3, 2, 1])
    assert cur.tolist() == [0, 0, 0, 1, 1, 3]
    assert seg.tolist() == [0, 0, 0, 1, 1, 2]


@needs_ffmpeg
def test_render_curves_writes_one_frame_per_cursor(tmp_path: Path) -> None:
    run = tiny_run()
    cur = grid_cursor(8, n_updates=4)
    out = render_curves(run, [0, 1, 2, 4], cur, None, tmp_path / "c.mkv", (320, 360))
    assert ffmpeg.count_frames(out) == 8
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (320, 360)
    first, last = ffmpeg.read_frames(out, 320, 360)[[0, -1]]
    assert not np.array_equal(first, last)  # the curve grew
