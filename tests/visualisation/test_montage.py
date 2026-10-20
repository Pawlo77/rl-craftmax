"""Tests for the long-cut montage and the Short composite (grid over curves)."""

from pathlib import Path

from _learning_run import needs_ffmpeg, tiny_run

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.curves import grid_cursor, render_curves
from wut_murl_craftax.visualisation.grid import build_grid
from wut_murl_craftax.visualisation.montage import sequence, short


@needs_ffmpeg
def test_sequence_plays_checkpoints_one_after_another(tmp_path: Path) -> None:
    out = sequence(
        tiny_run(),
        [0, 4],
        seconds=1,
        fps=6,
        game_px=96,
        curves_size=(120, 96),
        out=tmp_path / "seq.mp4",
    )
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (216, 96)
    assert ffmpeg.count_frames(out) == 12


@needs_ffmpeg
def test_short_stacks_grid_over_curves_with_gif(tmp_path: Path) -> None:
    run = tiny_run()
    grid = build_grid(run, [0, 1, 2, 4], panel=64, fps=10, out=tmp_path / "g.mp4")
    n = ffmpeg.count_frames(grid.out)
    curves = render_curves(
        run,
        [0, 1, 2, 4],
        grid_cursor(n, 4),
        None,
        tmp_path / "c.mkv",
        (128, 96),
        fps=10,
    )
    mp4, gif = short(grid.out, curves, tmp_path / "s.mp4", gif_fps=5, gif_width=64)
    info = ffmpeg.probe(mp4)["streams"][0]
    assert (info["width"], info["height"]) == (128, 224)
    assert ffmpeg.count_frames(mp4) == n
    assert gif is not None
    assert ffmpeg.probe(gif)["streams"][0]["width"] == 64
