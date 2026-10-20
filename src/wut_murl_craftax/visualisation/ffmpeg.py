"""Typed ffmpeg/ffprobe wrappers: argument lists only, never a shell string.

Encode settings are the ones validated in ``knowledge/visualisation/Encoding GIFs and
YouTube videos.md``; this ffmpeg build has no ``drawtext``, so text comes from ``text``.
"""

import json
import logging
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

type Arg = str | Path
"""One command-line argument."""

BT709 = (
    "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams="
    "color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
"""RGB -> limited-range BT.709 4:2:0 with matching tags (swscale defaults to BT.601)."""
BT709_TAGS = (
    *("-colorspace", "bt709", "-color_primaries", "bt709"),
    *("-color_trc", "bt709", "-color_range", "tv"),
)
"""Stream colour tags written next to ``BT709``."""
LOSSLESS_RGB = ("-c:v", "libx264rgb", "-qp", "0", "-preset", "medium")
"""Master codec: decodes bit-exact to the rendered RGB, half the size of FFV1."""
LOSSLESS_YUV = ("-c:v", "libx264", "-qp", "0", "-preset", "veryfast", *BT709_TAGS)
"""Intermediate clips: lossless after the BT.709 conversion."""
SILENT_AUDIO = (
    *("-f", "lavfi", "-i"),
    "anullsrc=channel_layout=stereo:sample_rate=48000",
)
"""Input arguments of a silent stereo 48 kHz track."""
AUDIO_OUT = ("-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2")
"""AAC-LC stereo 48 kHz at the 384 kbps YouTube recommends."""
FASTSTART = ("-movflags", "+faststart")
"""moov atom before mdat."""
LOUDNESS = "I=-14:TP=-1:LRA=11"
"""YouTube-style ``loudnorm`` target (integrated -14 LUFS)."""
DUCK = "sidechaincompress=threshold=0.02:ratio=12:attack=20:release=400"
"""Music under the voice: at least 6 dB lower in the tone test."""


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg or ffprobe is not on ``PATH``."""

    def __init__(self, name: str) -> None:
        """Name the missing tool and how to install it."""
        super().__init__(
            f"{name} not found on PATH: install ffmpeg "
            "(brew install ffmpeg / apt install ffmpeg)"
        )


class FfmpegError(RuntimeError):
    """An ffmpeg/ffprobe call exited non-zero."""

    def __init__(self, tool: str, stderr: bytes) -> None:
        """Keep the tail of stderr in the message."""
        super().__init__(f"{tool} failed: {stderr.decode(errors='replace')[-2000:]}")


def executable(name: str = "ffmpeg") -> str:
    """Path of ``ffmpeg`` or ``ffprobe``, or ``FfmpegNotFoundError``."""
    path = shutil.which(name)
    if path is None:
        raise FfmpegNotFoundError(name)
    return path


def _call(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    logger.debug("run %s", cmd)
    done = subprocess.run(cmd, capture_output=True, check=False)
    if done.returncode:
        raise FfmpegError(Path(cmd[0]).name, done.stderr)
    return done


def run(args: Sequence[Arg], loglevel: str = "error") -> str:
    """Run ``ffmpeg -y`` with ``args``; return stderr (analysis filters log there)."""
    head = [executable(), "-nostdin", "-hide_banner", "-loglevel", loglevel, "-y"]
    return _call([*head, *map(str, args)]).stderr.decode(errors="replace")


def _ffprobe(*args: Arg) -> bytes:
    return _call([executable("ffprobe"), "-v", "error", *map(str, args)]).stdout


def probe(path: Path) -> dict[str, Any]:
    """Streams and format of ``path`` (ffprobe JSON)."""
    out: dict[str, Any] = json.loads(
        _ffprobe("-show_streams", "-show_format", "-of", "json", path)
    )
    return out


def duration(path: Path) -> float:
    """Container duration in seconds."""
    return float(probe(path)["format"]["duration"])


def count_frames(path: Path) -> int:
    """Decoded video frames (exact: decodes the whole stream)."""
    entries = ("-show_entries", "stream=nb_read_frames", "-of", "csv=p=0")
    out = _ffprobe("-count_frames", "-select_streams", "v:0", *entries, path)
    return int(out.decode().strip().strip(","))


def read_frames(
    path: Path, width: int, height: int, vf: str | None = None
) -> npt.NDArray[np.uint8]:
    """Decode all frames (after optional ``vf``) to ``(n, height, width, 3)`` RGB."""
    filters = ["-vf", vf] if vf else []
    raw = _call([
        *(executable(), "-nostdin", "-loglevel", "error", "-i", str(path)),
        *filters,
        *("-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"),
    ]).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


class FrameWriter:
    """Context manager piping RGB frames into one video (lossless master by default)."""

    def __init__(
        self,
        out: Path,
        width: int,
        height: int,
        fps: int = 10,
        codec: Sequence[str] = LOSSLESS_RGB,
    ) -> None:
        """Build the command; ``__enter__`` starts ffmpeg."""
        raw = ("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}")
        self.out = out
        self.cmd = [executable(), "-nostdin", "-y", "-loglevel", "error", *raw]
        self.cmd += ["-r", str(fps), "-i", "-", *codec, str(out)]
        self.shape = (height, width, 3)
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Self:
        """Start ffmpeg reading frames from stdin."""
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self

    def _stdin(self) -> Any:
        if self._proc is None or self._proc.stdin is None:
            msg = "FrameWriter used outside a with block"
            raise RuntimeError(msg)
        return self._proc.stdin

    def write(self, frames: npt.NDArray[np.uint8]) -> None:
        """Append ``(n, height, width, 3)`` frames or one ``(height, width, 3)``."""
        if frames.shape[-3:] != self.shape:
            msg = f"frame shape {frames.shape[-3:]} != {self.shape}"
            raise ValueError(msg)
        self._stdin().write(np.ascontiguousarray(frames).tobytes())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close stdin and wait; raise ``FfmpegError`` if ffmpeg failed."""
        self._stdin().close()
        if self._proc is None:
            msg = "FrameWriter was closed before any frame was written"
            raise RuntimeError(msg)
        err = self._proc.stderr.read() if self._proc.stderr else b""
        if self._proc.wait() and exc_type is None:
            raise FfmpegError("ffmpeg", err)


def youtube_args(
    crf: int = 12, preset: str = "slow", fps: int = 60, audio: bool = True
) -> list[str]:
    """H.264 High, closed GOP of fps/2, 2 B-frames, CABAC, faststart, no edit list.

    ``audio=False`` leaves out the AAC options (video-only outputs such as grids).
    """
    gop = str(fps // 2)
    return [
        *("-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high"),
        *("-bf", "2", "-g", gop, "-keyint_min", gop, "-sc_threshold", "0"),
        *("-flags", "+cgop", "-coder", "cabac", "-pix_fmt", "yuv420p", *BT709_TAGS),
        *(AUDIO_OUT if audio else ()),
        *FASTSTART,
        *("-use_editlist", "0"),
    ]


def gif_filter(
    stream: str = "[g]",
    colors: int = 256,
    stats_mode: str = "full",
    dither: str = "none",
) -> str:
    """Two-pass palette; ``dither=none`` + ``stats_mode=full`` tested best."""
    dz = "bayer:bayer_scale=3" if dither == "bayer" else dither
    gen = f"palettegen=max_colors={colors}:stats_mode={stats_mode}"
    use = f"paletteuse=dither={dz}:diff_mode=rectangle"
    return f"{stream}split[a][b];[a]{gen}[p];[b][p]{use}"


def write_gif(inputs: Sequence[Arg], graph: str, out: Path, **palette: Any) -> Path:
    """Run ``graph`` (ending in ``[g]``) through ``gif_filter`` into a looping GIF."""
    out.parent.mkdir(parents=True, exist_ok=True)
    full = f"{graph};{gif_filter(**palette)}"
    run([*inputs, "-filter_complex", full, "-loop", "0", out])
    return out


def overlay_filter(
    base: str, top: str, out: str, start: float | None = None, end: float | None = None
) -> str:
    """Full-frame ``overlay`` of ``top`` on ``base``, optionally for start..end s."""
    when = f":enable='between(t,{start},{end})'" if start is not None else ""
    return f"{base}{top}overlay=0:0:shortest=1{when}{out}"


def xstack_layout(n: int, cols: int, cell_w: int, cell_h: int) -> str:
    """``xstack`` layout string for ``n`` equal cells, row-major."""
    return "|".join(f"{(i % cols) * cell_w}_{(i // cols) * cell_h}" for i in range(n))


def stack_filter(
    labels: Sequence[str],
    mode: Literal["h", "v", "x"],
    cols: int = 2,
    cell: tuple[int, int] = (0, 0),
    fill: str = "0x121418",
) -> str:
    """``hstack``/``vstack``/``xstack`` of ``labels``; ``x`` needs equal cells."""
    n, joined = len(labels), "".join(labels)
    match mode:
        case "h" | "v":
            return f"{joined}{mode}stack=inputs={n}"
        case "x":
            layout = xstack_layout(n, cols, *cell)
            return f"{joined}xstack=inputs={n}:layout={layout}:fill={fill}"


def _encode_with_silence(
    inputs: Sequence[Arg], video: list[Arg], encode: Sequence[str], out: Path
) -> None:
    audio = f"{sum(a == '-i' for a in inputs)}:a"
    run([*inputs, *SILENT_AUDIO, *video, "-map", audio, "-shortest", *encode, out])


def concat(
    clips: Sequence[Path], out: Path, encode: Sequence[str], fade_out: float = 0.5
) -> Path:
    """Join equal-format clips with the concat demuxer, fade to black, silent audio."""
    total = sum(duration(c) for c in clips)
    vf = f"fade=t=out:st={total - fade_out:.3f}:d={fade_out}"
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "clips.txt"  # entries resolve against the list's folder
        listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        inputs: list[Arg] = ["-f", "concat", "-safe", "0", "-i", listing]
        _encode_with_silence(inputs, ["-vf", vf, "-map", "0:v"], encode, out)
    return out


def xfade(
    clips: Sequence[Path],
    out: Path,
    encode: Sequence[str],
    seconds: float = 0.5,
    fps: int = 60,
) -> Path:
    """Chain ``xfade``: transition k starts at sum(d_0..d_k-1) - k * ``seconds``."""
    inputs: list[Arg] = [arg for c in clips for arg in ("-i", c)]
    graph = [f"[{i}:v]settb=AVTB,fps={fps}[s{i}]" for i in range(len(clips))]
    prev, offset = "[s0]", 0.0
    for i in range(1, len(clips)):
        offset += duration(clips[i - 1]) - seconds
        fade = f"xfade=transition=fade:duration={seconds}:offset={offset:.3f}"
        graph.append(f"{prev}[s{i}]{fade}[x{i}]")
        prev = f"[x{i}]"
    video: list[Arg] = ["-filter_complex", ";".join(graph), "-map", prev]
    _encode_with_silence(inputs, video, encode, out)
    return out


def integrated_loudness(path: Path) -> float:
    """Integrated loudness (LUFS) of the audio of ``path`` (``ebur128``)."""
    args: list[Arg] = ["-i", path, "-map", "0:a", "-af", "ebur128", "-f", "null", "-"]
    return float(run(args, loglevel="info").rsplit(" I:", 1)[1].split()[0])


def _loudnorm_second_pass(path: Path) -> str:
    args: list[Arg] = [
        "-i",
        path,
        "-af",
        f"loudnorm={LOUDNESS}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    log = run(args, loglevel="info")
    m = json.loads(log[log.rindex("{") : log.rindex("}") + 1])
    return (
        f"loudnorm={LOUDNESS}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}:linear=true"
    )


def mux_audio(
    video: Path,
    out: Path,
    voice: Path | None = None,
    music: Path | None = None,
    music_db: float = 0.0,
) -> Path:
    """Replace the audio of ``video``: voice-over over ducked music, two-pass loudnorm.

    Without voice and music the track is silent; ``music_db`` lowers the music bed.
    """
    keep_video = ["-map", "0:v", "-c:v", "copy", *AUDIO_OUT, "-shortest", *FASTSTART]
    keep_video += ["-use_editlist", "0", "-avoid_negative_ts", "make_non_negative"]
    tracks = [p for p in (voice, music) if p is not None]
    if not tracks:
        run([
            "-i",
            video,
            *SILENT_AUDIO,
            *keep_video[:2],
            "-map",
            "1:a",
            *keep_video[2:],
            out,
        ])
        return out
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo,apad"
    graph = [f"[{i + 1}:a]{fmt}[t{i}]" for i in range(len(tracks))]
    if len(tracks) == 2:
        graph += [f"[t1]volume={music_db}dB[bed]", "[t0]asplit[v][sc]"]
        graph += [f"[bed][sc]{DUCK}[duck]", "[v][duck]amix=inputs=2:normalize=0[mix]"]
    else:
        graph.append("[t0]acopy[mix]")
    inputs = [arg for p in (video, *tracks) for arg in ("-i", p)]
    with tempfile.TemporaryDirectory() as tmp:  # pass 1 measures the mix alone
        premix = Path(tmp) / "mix.wav"
        mix = ["-filter_complex", ";".join(graph), "-map", "[mix]"]
        run([*inputs, *mix, "-t", str(duration(video)), premix])
        norm = f"[1:a]{_loudnorm_second_pass(premix)},aresample=48000[a]"
        run([
            "-i",
            video,
            "-i",
            premix,
            "-filter_complex",
            norm,
            *keep_video[:2],
            "-map",
            "[a]",
            *keep_video[2:],
            out,
        ])
    return out


@dataclass(frozen=True, slots=True)
class YoutubeCheck:
    """ffprobe and MP4-box facts against YouTube's upload recommendations."""

    profile: str
    pix_fmt: str
    progressive: bool
    b_frames: int
    bt709: bool
    keyframe_spacing: set[int]
    faststart: bool
    edit_list: bool
    audio: str

    @property
    def ok(self) -> bool:
        """H.264 High 4:2:0, BT.709, 2 B-frames, fixed GOP, faststart, no elst, AAC."""
        return all((
            self.profile == "High",
            self.pix_fmt == "yuv420p",
            self.progressive,
            self.b_frames == 2,
            self.bt709,
            len(self.keyframe_spacing) <= 1,
            self.faststart,
            not self.edit_list,
            self.audio == "aac 48000 2",
        ))


def _boxes(path: Path) -> tuple[list[str], bytes]:
    """Top-level MP4 box order and the ``moov`` payload."""
    order, moov = [], b""
    with path.open("rb") as f:
        while head := f.read(8):
            size, kind = struct.unpack(">I4s", head)
            body = (struct.unpack(">Q", f.read(8))[0] - 16) if size == 1 else size - 8
            order.append(kind.decode(errors="replace"))
            if kind == b"moov":
                moov = f.read(body)
            else:
                f.seek(body, 1)
    return order, moov


def check_youtube(path: Path) -> YoutubeCheck:
    """Inspect ``path`` for the YouTube upload checks (``YoutubeCheck.ok``)."""
    streams = probe(path)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a: dict[str, Any] = next((s for s in streams if s["codec_type"] == "audio"), {})
    keyframes = _ffprobe(
        *("-select_streams", "v", "-skip_frame", "nokey", "-show_entries"),
        *("frame=pts_time", "-of", "csv=p=0", path),
    )
    times = [float(x.strip(",")) for x in keyframes.decode().split() if x.strip(",")]
    num, _, den = v["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    order, moov = _boxes(path)
    tags = (v.get("color_space"), v.get("color_primaries"), v.get("color_transfer"))
    return YoutubeCheck(
        profile=v.get("profile", ""),
        pix_fmt=v.get("pix_fmt", ""),
        progressive=v.get("field_order", "progressive") == "progressive",
        b_frames=int(v.get("has_b_frames", 0)),
        bt709=tags == ("bt709",) * 3,
        keyframe_spacing={round((b - t) * fps) for t, b in pairwise(times)},
        faststart={"moov", "mdat"} <= set(order)
        and order.index("moov") < order.index("mdat"),
        edit_list=b"elst" in moov,
        audio=f"{a.get('codec_name')} {a.get('sample_rate')} {a.get('channels')}",
    )
