"""Cuts from the reviewed CSV: clips, cards, 3-min (16:9, Short) and 12-min cuts, GIFs.

Subcommands of ``python -m wut_murl_craftax.visualisation.cuts``: ``clips``, ``short``,
``long``, ``gif``, ``grid``, ``mux``. Craftax is the main subject, Craftax-Classic a
short warm-up; footage is labelled on screen with its policy (``Rollout.label``).
"""

import argparse
import logging
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from wut_murl_craftax.visualisation import ffmpeg, text
from wut_murl_craftax.visualisation.review import kept, read_rows

logger = logging.getLogger(__name__)

type Row = Mapping[str, str]

ROOT = Path("outputs/visualisation")
"""Default output root (git-ignored)."""
FPS_MASTER, FPS = 10, 60
"""One env step per master frame; output rate (1x = every frame shown 6 times)."""
GAME_WH = {"craftax": (176, 208), "classic": (144, 144)}
"""Frame size of the 16 px masters."""
GAME_NAME = {"craftax": "Craftax", "classic": "Craftax-Classic"}
"""On-screen game names."""
BANNERS = {
    "achievement": "UNLOCKED",
    "floor": "NEW FLOOR",
    "death": "GAME OVER",
    "breakthrough": "BREAKTHROUGH",
    "boss": "BOSS",
    "fight": "FIGHT",
    "close call": "CLOSE CALL",
}
"""Freeze-frame headline per highlight type."""
FREEZE_TYPES = frozenset({"achievement", "floor", "breakthrough", "boss", "death"})
"""Types that hold their peak frame for ``FREEZE`` seconds."""
FREEZE = 1.0
"""Freeze-frame length in seconds."""
SHORT_MAX = 180.0
"""YouTube Shorts limit for square or vertical uploads."""
XFADE = 0.5
"""Crossfade length of the short cut."""
CHAPTERS = (
    ("floor boss", "Going deeper", "floor changes and the boss floor"),
    ("achievement", "Unlocks", "new achievements"),
    ("fight", "Fights", "several enemies at once"),
    ("breakthrough close call", "Breakthroughs", "progress after long droughts"),
    ("death", "Deaths", "how episodes end"),
)
"""Long-cut chapters of Craftax clips: (types, title, subtitle)."""
OUTRO = ("Thanks for watching", "WUT MURL Craftax project")
"""Outro card text."""


@dataclass(frozen=True, slots=True)
class Layout:
    """Canvas, upscale and offset per game, caption box and text sizes."""

    size: tuple[int, int]
    place: Mapping[str, tuple[int, int, int]]
    """game -> (integer scale of the 16 px master, x, y)."""
    panel: tuple[int, int, int]
    """Caption box (x, y, width)."""
    title_px: int
    sub_px: int
    banner_px: int


LAYOUTS = {
    "land": Layout(
        (1920, 1080),
        {"craftax": (5, 520, 20), "classic": (7, 456, 36)},
        (40, 760, 396),
        40,
        24,
        84,
    ),
    "vert": Layout(
        (1080, 1920),
        {"craftax": (6, 12, 336), "classic": (7, 36, 456)},
        (60, 90, 940),
        56,
        30,
        84,
    ),
    "land4k": Layout(
        (3840, 2160),
        {"craftax": (10, 1040, 40), "classic": (14, 912, 72)},
        (80, 1520, 840),
        80,
        48,
        168,
    ),
}
"""``land`` (16:9 1080p, default), ``vert`` (Short, 1080x1920), ``land4k`` (2160p)."""


@dataclass(frozen=True, slots=True)
class Encode:
    """Final-encode settings and optional audio of a cut."""

    crf: int = 12
    preset: str = "slow"
    voice: Path | None = None
    music: Path | None = None


DEFAULT_ENCODE = Encode()
"""CRF 12 slow, silent track."""


def _png(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def captions(row: Row, layout: str, out_dir: Path) -> tuple[Path, Path]:
    """Caption panel and freeze-frame banner PNGs for one clip."""
    lay, game = LAYOUTS[layout], row["game"]
    s, gx, gy = lay.place[game]
    gw, gh = GAME_WH[game]
    who = row.get("label") or row["policy"]
    sub = f"{GAME_NAME[game]} · {who} · steps {row['start']}-{row['end']}"
    sub += f" · {row['speed']}x" if row["speed"] != "1" else ""
    panel = text.caption_panel(
        lay.size, lay.panel, row["caption"], sub, lay.title_px, lay.sub_px
    )
    head = BANNERS.get(row["type"], row["type"].upper())
    center = (gx + gw * s // 2, gy + int(gh * s * 0.3))
    body = row["caption"].split(" after ")[0]
    ban = text.banner(lay.size, center, head, body, lay.banner_px)
    stem = f"{row['id']}_{layout}"
    return (
        _png(panel, out_dir / "caps" / f"{stem}_panel.png"),
        _png(ban, out_dir / "caps" / f"{stem}_banner.png"),
    )


def _clip_graph(row: Row, layout: str, window: bool) -> str:
    """Trim, speed, freeze at the peak, upscale, pad, overlay panel and banner."""
    w, h = LAYOUTS[layout].size
    k, x, y = LAYOUTS[layout].place[row["game"]]
    start, end, peak = (int(row[c]) for c in ("start", "end", "peak"))
    speed = row["speed"]
    if window:  # a render --kept window: already final size, starts at `start`
        start, end, peak, k = 0, end - start, peak - start, 1
    fz = FREEZE if row["type"] in FREEZE_TYPES else 0.0
    a = peak - start + 1  # frames up to and including the peak
    tf = round(a / (FPS_MASTER * float(speed)), 3)
    pts = f"setpts=(PTS-STARTPTS)/{speed}"
    hold = f"tpad=stop_mode=clone:stop_duration={fz}"
    trim = f"[0]trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS"
    if peak < end:
        body = (
            f"{trim},split[x][y];[x]trim=end_frame={a},{pts},{hold}[a];"
            f"[y]trim=start_frame={a},{pts}[b];[a][b]concat=n=2:v=1:a=0"
        )
    else:
        body = f"{trim},{pts},{hold}"
    up = f"scale=iw*{k}:ih*{k}:flags=neighbor,pad={w}:{h}:{x}:{y}:color=0x121418"
    panel = ffmpeg.overlay_filter("[g]", "[1]", "[c]")
    banner = ffmpeg.overlay_filter("[c]", "[2]", "", tf, tf + fz)
    return f"{body},fps={FPS},{up}[g];{panel};{banner},{ffmpeg.BT709}[v]"


def cut_clip(
    row: Row, layout: str, source: Path, out_dir: Path, window: bool = False
) -> Path:
    """One kept row -> lossless 60 fps ``<out_dir>/<id>.mp4``.

    ``window=True``: ``source`` is a ``render --kept`` window at final size.
    """
    panel, ban = captions(row, layout, out_dir)
    out = out_dir / f"{row['id']}.mp4"
    ffmpeg.run([
        *("-i", source, "-loop", "1", "-i", panel, "-loop", "1", "-i", ban),
        *("-filter_complex", _clip_graph(row, layout, window), "-map", "[v]"),
        *("-r", str(FPS), *ffmpeg.LOSSLESS_YUV, out),
    ])
    return out


def cut_kept(
    rows: Iterable[Row],
    layout: str,
    out_dir: Path,
    masters_dir: Path,
    hires_dir: Path | None = None,
) -> list[Path]:
    """Cut every kept row; from ``hires_dir`` windows (``render --kept``) when given."""
    out = []
    for r in kept(rows):
        if hires_dir is None:
            master = masters_dir / f"{r['rollout']}.mkv"
            out.append(cut_clip(r, layout, master, out_dir))
        else:
            out.append(cut_clip(r, layout, hires_dir / f"{r['id']}.mkv", out_dir, True))
    return out


def card(
    layout: str, out: Path, seconds: float, title: str, subtitle: str = ""
) -> Path:
    """Title or chapter card held ``seconds`` at 60 fps, 0.4 s fades from/to black."""
    image = text.title_card(LAYOUTS[layout].size, title, subtitle)
    png = _png(image, out.with_suffix(".png"))
    fades = f"fade=t=in:st=0:d=0.4,fade=t=out:st={seconds - 0.4:.3f}:d=0.4"
    ffmpeg.run([
        *("-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", png),
        *("-vf", f"{fades},{ffmpeg.BT709}", *ffmpeg.LOSSLESS_YUV, out),
    ])
    return out


def order_for_cut(rows: Iterable[Row], classic_max: int) -> list[Row]:
    """At most ``classic_max`` Classic rows first (warm-up), then the Craftax rows."""
    rows = list(rows)
    classic = [r for r in rows if r["game"] == "classic"][:classic_max]
    return classic + [r for r in rows if r["game"] != "classic"]


def _finish(video: Path, out: Path, enc: Encode) -> Path:
    if enc.voice is None and enc.music is None:
        return video.replace(out)
    ffmpeg.mux_audio(video, out, voice=enc.voice, music=enc.music)
    video.unlink()
    return out


def short_cut(
    rows: Iterable[Row],
    layout: str,
    clips_dir: Path,
    out: Path,
    cards_dir: Path,
    classic_max: int = 2,
    max_seconds: float = SHORT_MAX,
    enc: Encode = DEFAULT_ENCODE,
) -> Path:
    """Intro, kept clips (Classic warm-up first) while within ``max_seconds``, outro."""
    title = "Craftax agent highlights"
    intro = card(layout, cards_dir / f"short_intro_{layout}.mp4", 3, title)
    outro = card(layout, cards_dir / f"short_outro_{layout}.mp4", 3, *OUTRO)
    pieces, total = [intro], 6.0 - XFADE
    for r in order_for_cut(kept(rows), classic_max):
        clip = clips_dir / f"{r['id']}.mp4"
        d = ffmpeg.duration(clip) - XFADE
        if total + d > max_seconds:
            break
        pieces.append(clip)
        total += d
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".video.mp4")
    ffmpeg.xfade([*pieces, outro], tmp, ffmpeg.youtube_args(enc.crf, enc.preset), XFADE)
    return _finish(tmp, out, enc)


def long_cut(
    rows: Iterable[Row],
    clips_dir: Path,
    out: Path,
    cards_dir: Path,
    classic_max: int = 10,
    enc: Encode = DEFAULT_ENCODE,
) -> Path:
    """Intro, Classic warm-up chapter, Craftax chapters by type, outro; concatenated."""
    rows = kept(rows)
    title = "Craftax agent highlights"
    pieces = [card("land", cards_dir / "long_intro.mp4", 4, title)]
    classic = [r for r in rows if r["game"] == "classic"][:classic_max]
    groups = [
        ("warm-up", "Warm-up: Craftax-Classic", "the simpler game first", classic)
    ]
    for types, head, sub in CHAPTERS:
        members = [
            r for r in rows if r["game"] != "classic" and r["type"] in types.split()
        ]
        groups.append((head, head, sub, members))
    for slug, head, sub, members in groups:
        if members:
            name = f"long_{slug.replace(' ', '_')}.mp4"
            pieces.append(card("land", cards_dir / name, 3, head, sub))
            pieces += [clips_dir / f"{r['id']}.mp4" for r in members]
    pieces.append(card("land", cards_dir / "long_outro.mp4", 3, *OUTRO))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".video.mp4")
    ffmpeg.concat(pieces, tmp, ffmpeg.youtube_args(enc.crf, enc.preset))
    return _finish(tmp, out, enc)


def gif(
    row: Row,
    master: Path,
    out: Path,
    scale: int = 3,
    max_seconds: float = 20,
    **palette: str | int,
) -> Path:
    """GIF of one candidate from its 16 px master: one frame per step, caption strip."""
    speed = float(row["speed"])
    start = int(row["start"])
    end = min(int(row["end"]), start + int(max_seconds * FPS_MASTER * speed) - 1)
    width = GAME_WH[row["game"]][0] * scale
    with tempfile.TemporaryDirectory() as tmp:
        strip = text.text_strip(row["caption"], width, 34 * scale // 3)
        strip_png = _png(strip, Path(tmp) / "strip.png")
        graph = (
            f"[0]trim=start_frame={start}:end_frame={end + 1},"
            f"setpts=(PTS-STARTPTS)/{row['speed']},fps={FPS_MASTER * speed:g},"
            f"scale=iw*{scale}:ih*{scale}:flags=neighbor[s];[s][1]vstack[g]"
        )
        inputs: list[ffmpeg.Arg] = ["-i", master, "-i", strip_png]
        return ffmpeg.write_gif(inputs, graph, out, **palette)


def grid(
    masters: Sequence[Path], out: Path, cols: int, scale: int, seconds: float
) -> Path:
    """Masters side by side, labelled; shorter episodes hold their last frame."""
    first = ffmpeg.probe(masters[0])["streams"][0]
    cw, ch = int(first["width"]) * scale, int(first["height"]) * scale + 40
    hold = f"tpad=stop_mode=clone:stop=-1,trim=duration={seconds},fps={FPS}"
    up = f"scale=iw*{scale}:ih*{scale}:flags=neighbor"
    with tempfile.TemporaryDirectory() as tmp:
        inputs: list[ffmpeg.Arg] = []
        graph = []
        for i, m in enumerate(masters):
            label = _png(text.label_bar(m.stem, cw), Path(tmp) / f"l{i}.png")
            inputs += ["-i", m, "-loop", "1", "-framerate", str(FPS), "-i", label]
            graph.append(f"[{2 * i}]{hold},{up}[g{i}]")
            graph.append(f"[{2 * i + 1}][g{i}]vstack=shortest=1[c{i}]")
        cells = [f"[c{i}]" for i in range(len(masters))]
        pad = "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x121418"
        stack = ffmpeg.stack_filter(cells, "x", cols, (cw, ch))
        graph.append(f"{stack},{pad},fps={FPS},{ffmpeg.BT709}")
        video = ffmpeg.youtube_args(audio=False)
        ffmpeg.run([*inputs, "-filter_complex", ";".join(graph), *video, out])
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("clips", "short", "long", "gif"):
        s = sub.add_parser(name)
        s.add_argument("csv", type=Path)
        if name == "gif":
            s.add_argument("id")
        s.add_argument("--layout", choices=list(LAYOUTS), default="land")
        s.add_argument("--masters-dir", type=Path, default=ROOT / "masters")
        s.add_argument("--hires-dir", type=Path, help="cut from render --kept windows")
        s.add_argument("--clips-dir", type=Path)
        s.add_argument("--out", type=Path)
        s.add_argument("--voice", type=Path)
        s.add_argument("--music", type=Path)
    g = sub.add_parser("grid")
    g.add_argument("out", type=Path)
    g.add_argument("masters", nargs="+", type=Path)
    g.add_argument("--cols", type=int, default=2)
    g.add_argument("--scale", type=int, default=2)
    g.add_argument("--seconds", type=float, default=60)
    m = sub.add_parser("mux")
    m.add_argument("video", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--voice", type=Path)
    m.add_argument("--music", type=Path)
    return p


def _run(a: argparse.Namespace) -> object:
    if a.cmd == "grid":
        return grid(a.masters, a.out, a.cols, a.scale, a.seconds)
    if a.cmd == "mux":
        return ffmpeg.mux_audio(a.video, a.out, voice=a.voice, music=a.music)
    rows = read_rows(a.csv)
    clips = a.clips_dir or ROOT / f"clips_{a.layout}"
    enc = Encode(voice=a.voice, music=a.music)
    match a.cmd:
        case "clips":
            return cut_kept(rows, a.layout, clips, a.masters_dir, a.hires_dir)
        case "short":
            out = a.out or ROOT / f"out/short_3min_{a.layout}.mp4"
            return short_cut(rows, a.layout, clips, out, ROOT / "cards", enc=enc)
        case "long":
            out = a.out or ROOT / "out/long_12min.mp4"
            return long_cut(rows, clips, out, ROOT / "cards", enc=enc)
        case _:
            row = next(r for r in rows if r["id"] == a.id)
            master = a.masters_dir / f"{row['rollout']}.mkv"
            return gif(row, master, a.out or ROOT / f"gifs/{a.id}.gif")


def main(argv: list[str] | None = None) -> None:
    """Run one cut step (see the module docstring)."""
    result = _run(_parser().parse_args(argv))
    print(f"{len(result)} clips" if isinstance(result, list) else result)
    if isinstance(result, Path) and result.suffix == ".mp4":
        print(ffmpeg.check_youtube(result))


if __name__ == "__main__":
    main()
