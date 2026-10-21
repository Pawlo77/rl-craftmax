"""Learning-process composites: the long-cut montage and the Short (grid over curves).

``sequence`` plays checkpoints one after another beside sequence-synced curves (16:9);
``short`` stacks a checkpoint grid over grid-synced curves (9:16) plus a GIF.
Usage: ``python -m wut_murl_craftax.visualisation.montage RUN --updates 0,9,33,122``
"""

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from . import ffmpeg
from .charts import SERIES, SURFACE
from .curves import render_curves, seq_cursor
from .eval_rollouts import EVAL_SEEDS, EvalRun, episode_frames, eval_returns, load_eval
from .grid import panel_label

logger = logging.getLogger(__name__)


def sequence(
    run_dir: Path,
    updates: Sequence[int],
    seconds: float = 6.0,
    fps: int = 15,
    seed: int = EVAL_SEEDS[0],
    game_px: int = 1080,
    curves_size: tuple[int, int] = (840, 1080),
    out: Path = Path("outputs/visualisation/montage.mp4"),
) -> Path:
    """``seconds`` of each checkpoint's eval episode, labelled, curves on the right."""
    run = EvalRun.load(run_dir)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    n, g = int(seconds * fps), game_px
    t0 = time.time()
    inputs: list[ffmpeg.Arg] = []
    graph = []
    for k, u in enumerate(updates):
        frames = episode_frames(run, u, seed)
        with ffmpeg.FrameWriter(
            work / f"panel_{k}.mkv", frames.shape[2], frames.shape[1], fps
        ) as w:
            w.write(frames)
        ret = eval_returns(load_eval(run.dir, u), run.cfg["env"]).mean()
        score = ret / run.max_reward * 100
        head, sub = (
            f"{run.env_steps(u) / 1e6:.1f}M steps",
            f"eval score {score:.0f}% of max",
        )
        panel_label(head, sub, g // 2, SERIES[k % 4]).save(work / f"label_{k}.png")
        inputs += ["-i", work / f"panel_{k}.mkv", "-i", work / f"label_{k}.png"]
        graph.append(
            f"[{2 * k}:v]scale={g}:{g}:force_original_aspect_ratio=decrease:"
            "flags=neighbor,"
            f"pad={g}:{g}:(ow-iw)/2:(oh-ih)/2:color={SURFACE},"
            f"tpad=stop_mode=clone:stop_duration={seconds},trim=end_frame={n},"
            f"setpts=PTS-STARTPTS[s{k}];[s{k}][{2 * k + 1}:v]overlay=0:0[g{k}]"
        )
    cur, seg = seq_cursor(updates, [n] * len(updates))
    curves = render_curves(
        run.dir, updates, cur, seg, work / "curves.mkv", curves_size, fps
    )
    joined = "".join(f"[g{k}]" for k in range(len(updates)))
    graph.append(f"{joined}concat=n={len(updates)}:v=1:a=0[game]")
    graph.append(f"[game]pad={g}:{curves_size[1]}:0:(oh-ih)/2:color={SURFACE}[gp]")
    graph.append(f"[gp][{2 * len(updates)}:v]hstack=inputs=2,format=yuv420p[v]")
    ffmpeg.run([
        *inputs,
        "-i",
        curves,
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[v]",
        "-frames:v",
        str(n * len(updates)),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        out,
    ])
    logger.info("montage %s in %.1fs", out, time.time() - t0)
    return out


def short(
    grid: Path,
    curves: Path,
    out: Path,
    gif_fps: int | None = 10,
    gif_width: int = 360,
) -> tuple[Path, Path | None]:
    """Stack ``grid`` over ``curves`` (same width and frame count) into MP4 and GIF."""
    n = ffmpeg.count_frames(grid)
    ffmpeg.run([
        "-i",
        grid,
        "-i",
        curves,
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2:shortest=1,format=yuv420p[v]",
        "-map",
        "[v]",
        "-frames:v",
        str(n),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        out,
    ])
    if gif_fps is None:
        return out, None
    gif = out.with_suffix(".gif")
    graph = f"[0:v]fps={gif_fps},scale={gif_width}:-1:flags=lanczos[g]"
    ffmpeg.write_gif(["-i", out], graph, gif, colors=128)
    return out, gif


def main(argv: list[str] | None = None) -> None:
    """Command line: long-cut ``sequence``; ``--short GRID CURVES`` for the Short."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--updates", help="checkpoint updates, comma list")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    p.add_argument("--short", nargs=2, type=Path, metavar=("GRID", "CURVES"))
    p.add_argument(
        "--out", type=Path, default=Path("outputs/visualisation/montage.mp4")
    )
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.short:
        grid, curves = a.short
        mp4, gif = short(grid, curves, a.out)
        print(f"short written to {mp4} and {gif}")
        return
    if a.run is None or a.updates is None:
        p.error("RUN and --updates are required without --short")
    updates = [int(u) for u in a.updates.split(",")]
    out = sequence(a.run, updates, a.seconds, a.fps, a.seed, out=a.out)
    print(f"montage written to {out}")


if __name__ == "__main__":
    main()
