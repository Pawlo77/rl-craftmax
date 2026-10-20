"""Tests for the ffmpeg/ffprobe wrappers on tiny synthetic clips."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from _gameplay_run import needs_ffmpeg

from wut_murl_craftax.visualisation import ffmpeg


def _frames(n: int, h: int = 32, w: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (n, h // 8, w // 8, 3), dtype=np.uint8)
    return blocks.repeat(8, 1).repeat(8, 2)


def _master(path: Path, n: int = 20, fps: int = 10, seed: int = 0) -> np.ndarray:
    frames = _frames(n, seed=seed)
    with ffmpeg.FrameWriter(path, width=48, height=32, fps=fps) as w:
        w.write(frames)
    return frames


def test_missing_executable_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ffmpeg.FfmpegNotFoundError, match="ffmpeg"):
        ffmpeg.executable("ffmpeg")


@needs_ffmpeg
def test_failed_command_reports_stderr(tmp_path: Path) -> None:
    with pytest.raises(ffmpeg.FfmpegError, match="No such file"):
        ffmpeg.run(["-i", tmp_path / "missing.mkv", tmp_path / "out.mp4"])


@needs_ffmpeg
def test_lossless_master_round_trips_bit_exact(tmp_path: Path) -> None:
    frames = _master(tmp_path / "m.mkv")
    back = ffmpeg.read_frames(tmp_path / "m.mkv", 48, 32)
    assert np.array_equal(back, frames)
    assert ffmpeg.duration(tmp_path / "m.mkv") == pytest.approx(2.0)


@needs_ffmpeg
def test_youtube_encode_passes_upload_checks(tmp_path: Path) -> None:
    _master(tmp_path / "m.mkv", n=30)
    graph = f"[0]fps=60,scale=iw*4:ih*4:flags=neighbor,{ffmpeg.BT709}[v]"
    out = tmp_path / "yt.mp4"
    ffmpeg.run([
        "-i",
        tmp_path / "m.mkv",
        *ffmpeg.SILENT_AUDIO,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "1:a",
        "-shortest",
        *ffmpeg.youtube_args(),
        out,
    ])
    check = ffmpeg.check_youtube(out)
    assert check.ok, check
    assert check.keyframe_spacing == {30}


@needs_ffmpeg
def test_gif_has_one_frame_per_step(tmp_path: Path) -> None:
    _master(tmp_path / "m.mkv", n=15)
    out = tmp_path / "a.gif"
    ffmpeg.write_gif(
        ["-i", tmp_path / "m.mkv"], "[0]scale=iw*2:ih*2:flags=neighbor[g]", out
    )
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (96, 64)
    assert ffmpeg.count_frames(out) == 15


@needs_ffmpeg
def test_xstack_grid_and_layout(tmp_path: Path) -> None:
    assert (
        ffmpeg.xstack_layout(4, cols=2, cell_w=10, cell_h=20) == "0_0|10_0|0_20|10_20"
    )
    for i in range(4):
        _master(tmp_path / f"m{i}.mkv", n=10, seed=i)
    inputs = [arg for i in range(4) for arg in ("-i", tmp_path / f"m{i}.mkv")]
    graph = ffmpeg.stack_filter(
        ["[0]", "[1]", "[2]", "[3]"], "x", cols=2, cell=(48, 32)
    )
    ffmpeg.run([
        *inputs,
        "-filter_complex",
        graph,
        *ffmpeg.LOSSLESS_YUV,
        tmp_path / "g.mp4",
    ])
    info = ffmpeg.probe(tmp_path / "g.mp4")["streams"][0]
    assert (info["width"], info["height"]) == (96, 64)


@needs_ffmpeg
def test_concat_and_xfade_durations(tmp_path: Path) -> None:
    clips = []
    for i in range(3):
        _master(tmp_path / f"m{i}.mkv", n=20, seed=i)
        clip = tmp_path / f"c{i}.mp4"
        ffmpeg.run([
            "-i",
            tmp_path / f"m{i}.mkv",
            "-vf",
            f"fps=60,{ffmpeg.BT709}",
            *ffmpeg.LOSSLESS_YUV,
            clip,
        ])
        clips.append(clip)
    enc = ffmpeg.youtube_args(crf=23, preset="veryfast")
    ffmpeg.concat(clips, tmp_path / "cat.mp4", enc)
    ffmpeg.xfade(clips, tmp_path / "xf.mp4", enc, seconds=0.5)
    assert ffmpeg.duration(tmp_path / "cat.mp4") == pytest.approx(6.0, abs=0.05)
    assert ffmpeg.duration(tmp_path / "xf.mp4") == pytest.approx(5.0, abs=0.05)
    assert ffmpeg.check_youtube(tmp_path / "cat.mp4").ok


def test_overlay_filter_enable_window() -> None:
    assert ffmpeg.overlay_filter("[g]", "[1]", "[c]", start=1.5, end=2.5) == (
        "[g][1]overlay=0:0:shortest=1:enable='between(t,1.5,2.5)'[c]"
    )
    assert (
        ffmpeg.overlay_filter("[g]", "[1]", "[c]") == "[g][1]overlay=0:0:shortest=1[c]"
    )
