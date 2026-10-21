"""Checkpoint comparison grid: one eval seed played by four checkpoints, 2x2, synced.

Every panel starts at step 0 (one game step per frame) and is labelled with its
training steps and fixed-seed eval score; a panel whose episode ended freezes under a
banner. One ffmpeg ``xstack`` call composites lossless per-checkpoint panels.
Usage: ``python -m wut_murl_craftax.visualisation.grid RUN --updates 0,9,33,122``
"""

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from . import ffmpeg, text
from .charts import INK, INK2, SERIES, SURFACE
from .eval_rollouts import EVAL_SEEDS, EvalRun, episode_frames, eval_returns, load_eval

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GridResult:
    """The grid video and the episode length (frames) of each panel."""

    out: Path
    lens: list[int]
    seconds: float


def panel_label(head: str, sub: str, width: int, color: str) -> Image.Image:
    """Translucent two-line label with a colour key matching the curve markers."""
    # TODO(shared): move to text.py (two-line label with a colour key)
    size = max(11, width // 18)
    im = Image.new("RGBA", (width, int(size * 2.9)), (26, 26, 25, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, max(3, width // 54), im.height], fill=color)
    d.text((width // 22, 3), head, font=text.font(size), fill=INK)
    d.text(
        (width // 22, 3 + int(size * 1.3)),
        sub,
        font=text.font(int(size * 0.85)),
        fill=INK2,
    )
    return im


def _panels(
    run: EvalRun, updates: Sequence[int], seed: int, panel: int, fps: int, work: Path
) -> list[int]:
    """Write ``panel_<k>.mkv`` and its label/banner PNGs; return episode lengths."""
    lens = []
    for k, u in enumerate(updates):
        frames = episode_frames(run, u, seed)
        h, w = frames.shape[1:3]
        with ffmpeg.FrameWriter(work / f"panel_{k}.mkv", w, h, fps) as writer:
            writer.write(frames)
        lens.append(len(frames))
        ev = load_eval(run.dir, u)
        score = eval_returns(ev, run.cfg["env"]).mean() / run.max_reward * 100
        head = f"{run.env_steps(u) / 1e6:.1f}M steps"
        sub = f"eval score {score:.0f}% of max ({len(ev['seeds'])} seeds)"
        panel_label(head, sub, panel, SERIES[k % 4]).save(work / f"label_{k}.png")
        over = f"step {len(frames) - 1}"
        size, px = (panel, panel), max(10, panel // 16)
        text.banner(size, (panel // 2, panel // 2), "EPISODE OVER", over, px).save(
            work / f"ended_{k}.png"
        )
    return lens


def build_grid(
    run_dir: Path,
    updates: Sequence[int],
    seed: int = EVAL_SEEDS[0],
    panel: int = 540,
    fps: int = 15,
    max_s: float = 20.0,
    out: Path = Path("outputs/visualisation/grid.mp4"),
) -> GridResult:
    """Composite four checkpoints into a ``2 panel`` square H.264 clip."""
    run = EvalRun.load(run_dir)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lens = _panels(run, updates, seed, panel, fps, work)
    n = min(max(lens), int(max_s * fps))
    dur = n / fps
    inputs: list[ffmpeg.Arg] = []
    for k in range(len(updates)):
        inputs += ["-i", work / f"panel_{k}.mkv"]
    for k in range(len(updates)):
        inputs += ["-i", work / f"label_{k}.png", "-i", work / f"ended_{k}.png"]
    fit = f"scale={panel}:{panel}:force_original_aspect_ratio=decrease:flags=neighbor"
    pad = f"pad={panel}:{panel}:(ow-iw)/2:(oh-ih)/2:color={SURFACE}"
    graph = []
    for k, length in enumerate(lens):
        lab, end = 4 + 2 * k, 5 + 2 * k
        graph.append(
            f"[{k}:v]{fit},{pad},tpad=stop_mode=clone:stop_duration={dur}[s{k}];"
            f"[s{k}][{lab}:v]overlay=0:0[l{k}];"
            f"[l{k}][{end}:v]overlay=0:0:enable='gte(t,{(length - 1) / fps:.3f})'[p{k}]"
        )
    labels = [f"[p{k}]" for k in range(len(updates))]
    stack = ffmpeg.stack_filter(labels, "x", cols=2, cell=(panel, panel), fill=SURFACE)
    ffmpeg.run([
        *inputs,
        "-filter_complex",
        f"{';'.join(graph)};{stack}[v]",
        "-map",
        "[v]",
        "-frames:v",
        str(n),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out,
    ])
    logger.info("grid %s: %d frames in %.1fs", out, n, time.time() - t0)
    return GridResult(out, lens, dur)


def main(argv: list[str] | None = None) -> None:
    """Command line: ``RUN --updates a,b,c,d [--seed S] [--panel 540] [--fps 15]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--updates", required=True, help="four checkpoint updates")
    p.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    p.add_argument("--panel", type=int, default=540, help="px; 540 -> 1080 x 1080")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=Path("outputs/visualisation/grid.mp4"))
    a = p.parse_args(argv)
    updates = [int(u) for u in a.updates.split(",")]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = build_grid(a.run, updates, a.seed, a.panel, a.fps, a.max_s, a.out)
    print(f"grid written to {res.out} ({res.seconds:.2f} s)")


if __name__ == "__main__":
    main()
