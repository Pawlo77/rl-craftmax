"""Tests for lossless masters and final-size window renders."""

from pathlib import Path

import numpy as np
from _gameplay_run import classic_master, classic_rollout, needs_ffmpeg

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.render import render_kept, render_master
from wut_murl_craftax.visualisation.rollout import replay


@needs_ffmpeg
def test_master_matches_replay_frames() -> None:
    frames = ffmpeg.read_frames(classic_master(), 144, 144)
    expected = np.concatenate([c.frames for c in replay(classic_rollout(), bps=16)])
    assert np.array_equal(frames, expected)


@needs_ffmpeg
def test_range_render_equals_master_slice(tmp_path: Path) -> None:
    stats = render_master(classic_rollout(), tmp_path / "part.mkv", start=10, end=30)
    assert stats.features_match
    part = ffmpeg.read_frames(tmp_path / "part.mkv", 144, 144)
    assert np.array_equal(part, ffmpeg.read_frames(classic_master(), 144, 144)[10:31])


@needs_ffmpeg
def test_render_kept_uses_final_tile_size(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    classic_rollout().save(runs / "classic_heur_1.npz")
    row = {
        "id": "c001",
        "keep": "keep",
        "rollout": "classic_heur_1",
        "game": "classic",
        "start": "5",
        "end": "12",
    }
    render_kept(
        [row, {**row, "id": "c002", "keep": "drop"}], "land", runs, tmp_path / "hi"
    )
    info = ffmpeg.probe(tmp_path / "hi" / "c001.mkv")["streams"][0]
    assert (info["width"], info["height"]) == (1008, 1008)  # 112 px per tile = 16 px x7
    assert ffmpeg.count_frames(tmp_path / "hi" / "c001.mkv") == 8
    assert not (tmp_path / "hi" / "c002.mkv").exists()
