"""Tests for clip cutting, cards, the short and long cuts, GIFs and the audio mix."""

from pathlib import Path

import pytest
from _gameplay_run import classic_master, needs_ffmpeg

from wut_murl_craftax.visualisation import cuts, ffmpeg

ROW = {
    "id": "c001",
    "keep": "keep",
    "score": "9",
    "type": "achievement",
    "caption": "COLLECT_WOOD",
    "rollout": "classic_heur_1",
    "game": "classic",
    "policy": "heuristic",
    "label": "scripted agent",
    "start": "10",
    "end": "29",
    "peak": "19",
    "seconds": "2.0",
    "speed": "1",
    "events": "",
}


@needs_ffmpeg
def test_clip_has_freeze_and_layout(tmp_path: Path) -> None:
    out = cuts.cut_clip(ROW, "land", classic_master(), tmp_path)
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (1920, 1080)
    assert ffmpeg.duration(out) == pytest.approx(
        2.0 + 1.0, abs=0.05
    )  # 20 frames + 1 s freeze
    fast = cuts.cut_clip(
        {**ROW, "id": "c002", "type": "fight", "speed": "2"},
        "vert",
        classic_master(),
        tmp_path,
    )
    assert ffmpeg.duration(fast) == pytest.approx(1.0, abs=0.05)  # no freeze for fights
    assert ffmpeg.probe(fast)["streams"][0]["height"] == 1920


def test_order_puts_a_short_classic_warm_up_first() -> None:
    rows = [{**ROW, "id": f"k{i}", "game": "craftax"} for i in range(3)]
    rows += [{**ROW, "id": f"c{i}"} for i in range(4)]
    ids = [r["id"] for r in cuts.order_for_cut(rows, classic_max=2)]
    assert ids == ["c0", "c1", "k0", "k1", "k2"]


@needs_ffmpeg
def test_short_cut_crossfades_and_stays_a_short(tmp_path: Path) -> None:
    rows = [ROW, {**ROW, "id": "c002", "start": "30", "end": "49", "peak": "40"}]
    voice = _tone(tmp_path / "voice.wav", 300, 3, delay=4)
    music = _tone(tmp_path / "music.wav", 1000, 12)
    encodes = {
        "land": cuts.Encode(preset="veryfast"),
        "vert": cuts.Encode(preset="veryfast", voice=voice, music=music),
    }
    for layout in ("land", "vert"):
        clips = tmp_path / layout
        for r in rows:
            cuts.cut_clip(r, layout, classic_master(), clips)
        out = cuts.short_cut(
            rows,
            layout,
            clips,
            tmp_path / f"short_{layout}.mp4",
            cards_dir=tmp_path / "cards",
            enc=encodes[layout],
        )
        # 2 cards x 3 s + 2 clips x 3 s - 3 crossfades x 0.5 s
        assert ffmpeg.duration(out) == pytest.approx(10.5, abs=0.1)
        assert ffmpeg.check_youtube(out).ok


@needs_ffmpeg
def test_long_cut_has_chapters(tmp_path: Path) -> None:
    rows = [
        ROW,
        {
            **ROW,
            "id": "c002",
            "type": "fight",
            "start": "30",
            "end": "49",
            "peak": "40",
        },
    ]
    for r in rows:
        cuts.cut_clip(r, "land", classic_master(), tmp_path / "clips")
    out = cuts.long_cut(
        rows,
        tmp_path / "clips",
        tmp_path / "long.mp4",
        cards_dir=tmp_path / "cards",
        enc=cuts.Encode(preset="veryfast"),
    )
    # intro 4 + Classic warm-up card 3 + 2 clips (3 + 2) + outro 3 = 15 s
    assert ffmpeg.duration(out) == pytest.approx(15.0, abs=0.1)


@needs_ffmpeg
def test_gif_is_clamped_to_max_seconds(tmp_path: Path) -> None:
    out = cuts.gif(
        {**ROW, "end": "60"}, classic_master(), tmp_path / "a.gif", max_seconds=3
    )
    assert ffmpeg.count_frames(out) == 30
    assert ffmpeg.probe(out)["streams"][0]["width"] == 144 * 3


@needs_ffmpeg
def test_grid_holds_shorter_masters_and_labels(tmp_path: Path) -> None:
    out = cuts.grid([classic_master(), classic_master()], tmp_path / "g.mp4", 2, 1, 7.0)
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (1920, 1080)
    assert ffmpeg.duration(out) == pytest.approx(7.0, abs=0.05)  # master is 6.1 s
    assert info["r_frame_rate"] == "60/1"
    assert abs(ffmpeg.count_frames(out) - 420) <= 1  # trim=duration may drop the last
    assert ffmpeg.check_youtube(out).keyframe_spacing == {30}
    assert "-b:a" not in ffmpeg.youtube_args(audio=False)


def _band_level(path: Path, freq: int, start: float, end: float) -> float:
    """RMS level (dB) of a narrow band around ``freq`` between ``start`` and ``end``."""
    log = ffmpeg.run(
        [
            "-ss",
            str(start),
            "-t",
            str(end - start),
            "-i",
            path,
            "-af",
            f"bandpass=f={freq}:width_type=q:w=8,astats=measure_overall=RMS_level",
            "-f",
            "null",
            "-",
        ],
        loglevel="info",
    )
    return float(log.split("RMS level dB:")[-1].split()[0])


def _tone(path: Path, freq: int, seconds: float, delay: float = 0.0) -> Path:
    src = f"sine=frequency={freq}:sample_rate=48000:duration={seconds}"
    ffmpeg.run([
        "-f",
        "lavfi",
        "-i",
        src,
        "-af",
        f"adelay={int(delay * 1000)}:all=1",
        path,
    ])
    return path


@needs_ffmpeg
def test_audio_mix_ducks_music_and_normalises(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    ffmpeg.run([
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=60:d=8",
        *ffmpeg.LOSSLESS_YUV,
        video,
    ])
    voice = _tone(tmp_path / "voice.wav", 300, 3, delay=4)
    music = _tone(tmp_path / "music.wav", 1000, 8)
    out = tmp_path / "mixed.mp4"
    ffmpeg.mux_audio(video, out, voice=voice, music=music)
    audio = next(s for s in ffmpeg.probe(out)["streams"] if s["codec_type"] == "audio")
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == (
        "aac",
        "48000",
        2,
    )
    assert ffmpeg.integrated_loudness(out) == pytest.approx(-14.0, abs=1.5)
    assert not ffmpeg.check_youtube(out).edit_list  # AAC priming must not shift video
    before = _band_level(out, 1000, 1.0, 3.0)
    during = _band_level(out, 1000, 5.0, 7.0)
    assert before - during > 6  # music at least 6 dB lower under the voice
    alone = ffmpeg.mux_audio(video, tmp_path / "voice.mp4", voice=voice)
    assert ffmpeg.integrated_loudness(alone) == pytest.approx(-14.0, abs=1.5)
    silent = tmp_path / "silent.mp4"
    ffmpeg.mux_audio(video, silent)
    assert ffmpeg.duration(silent) == pytest.approx(8.0, abs=0.1)
