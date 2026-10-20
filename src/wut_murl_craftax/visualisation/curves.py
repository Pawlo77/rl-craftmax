"""Animated training curves: % of max reward, achievements unlocked, success rates.

Each output frame shows the run up to a cursor (an update index). ``grid_cursor``
reveals the curve over a clip that shows all checkpoints at once (``grid``);
``seq_cursor`` jumps to the checkpoint whose gameplay is on screen (``montage``).
Usage: ``python -m wut_murl_craftax.visualisation.curves RUN --updates 0,9 --frames 99``
"""

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..training.records import load_metrics
from . import ffmpeg
from .charts import GRID, INK, MUTED, SERIES, SURFACE, ema, figure, frame, style

logger = logging.getLogger(__name__)

DEFAULT_SIZE = (1080, 840)
"""Width x height under a 1080 x 1080 grid: together a 1080 x 1920 Short."""


def grid_cursor(n_frames: int, n_updates: int, reveal: float = 0.7) -> np.ndarray:
    """Reveal all updates over the first ``reveal`` share of the frames, then hold."""
    n_rev = int(n_frames * reveal)
    sweep = np.linspace(0, n_updates - 1, n_rev)
    hold = np.full(n_frames - n_rev, n_updates - 1)
    return np.concatenate([sweep, hold]).astype(int)


def seq_cursor(
    updates: Sequence[int], seg_frames: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Cursor and segment index per frame when checkpoints play one after another."""
    cur = np.concatenate([
        np.full(n, max(0, u - 1)) for u, n in zip(updates, seg_frames, strict=True)
    ])
    seg = np.concatenate([np.full(n, k) for k, n in enumerate(seg_frames)])
    return cur.astype(int), seg.astype(int)


def render_curves(
    run_dir: Path,
    updates: Sequence[int],
    cursor: np.ndarray,
    segment: np.ndarray | None,
    out: Path,
    size: tuple[int, int] = DEFAULT_SIZE,
    fps: int = 15,
    bars: int = 22,
) -> Path:
    """Render one frame per cursor value to a lossless ``out``.

    Panels: score (main), mean achievements unlocked per episode, and the ``bars``
    achievements with the highest final success rate. Checkpoint markers use the
    grid label colours; with ``segment`` only the checkpoint on screen is marked.
    """
    m = load_metrics(run_dir)
    x, score = m.steps / 1e6, ema(m.score)
    rates = np.stack([ema(r) for r in m.rates.T], 1)
    unlocked = ema(np.nansum(m.rates, 1) / 100)  # expected achievements per episode
    order = np.argsort(-np.nan_to_num(rates[-1]))[:bars]
    fs = size[0] / 60
    fig = figure(*size)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.3, 3.5], hspace=0.55)
    gs.update(left=0.27, right=0.95, top=0.95, bottom=0.07)
    axes = [style(fig.add_subplot(gs[i]), fs * 0.8) for i in range(3)]
    ax, ux, bx = axes
    lines = []
    for a, y, name in (
        (ax, score, f"score, % of max reward ({m.max_reward:.0f})"),
        (ux, unlocked, f"achievements unlocked per episode, of {len(m.names)}"),
    ):
        a.set_xlim(0, x[-1] * 1.03)
        a.set_ylim(0, max(1.0, float(np.nanmax(y))) * 1.15)
        a.set_title(name, loc="left", fontsize=fs, color=INK)
        a.plot(x, y, color=GRID, lw=fs / 6)  # the whole run as a faint track
        lines.append(a.plot([], [], color=INK, lw=fs / 4)[0])
    ux.set_xlabel("environment steps (M)", fontsize=fs * 0.8)
    rows = [max(0, u - 1) for u in updates]
    dots = [
        ax.plot(
            [x[i]],
            [score[i]],
            "o",
            ms=fs,
            color=SERIES[k % 4],
            mec=SURFACE,
            mew=fs / 5,
            visible=False,
            clip_on=False,
        )[0]
        for k, i in enumerate(rows)
    ]
    vlines = [a.axvline(0, color=MUTED, lw=1, ls=":") for a in (ax, ux)]
    hbars = bx.barh(
        np.arange(len(order)), np.zeros(len(order)), height=0.7, color=SERIES[0]
    )
    names = [m.names[i].replace("_", " ") for i in order]
    bx.set_yticks(np.arange(len(order)), names, fontsize=fs * 0.7)
    bx.invert_yaxis()
    bx.set_xlim(0, 100)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("success rate, % of episodes", fontsize=fs * 0.8)
    title = bx.set_title("", loc="left", fontsize=fs, color=INK)
    t0 = time.time()
    with ffmpeg.FrameWriter(out, *size, fps=fps) as writer:
        for f, i in enumerate(cursor):
            lines[0].set_data(x[: i + 1], score[: i + 1])
            lines[1].set_data(x[: i + 1], unlocked[: i + 1])
            for v in vlines:
                v.set_xdata([x[i], x[i]])
            for k, (row, dot) in enumerate(zip(rows, dots, strict=True)):
                dot.set_visible(row <= i if segment is None else k == segment[f])
            for bar, rate in zip(hbars, np.nan_to_num(rates[i, order]), strict=True):
                bar.set_width(rate)
            title.set_text(f"success per achievement at {x[i]:.2f}M steps")
            writer.write(frame(fig))
    logger.info("%d curve frames in %.1fs", len(cursor), time.time() - t0)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: grid sync with ``--frames N``, sequence sync ``--seg-frames``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--updates", required=True, help="checkpoint updates, comma list")
    p.add_argument("--frames", type=int, help="grid sync: frames in the clip")
    p.add_argument("--seg-frames", help="sequence sync: frames per checkpoint")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--size", default="1080x840")
    p.add_argument("--bars", type=int, default=22)
    p.add_argument("--out", type=Path, default=Path("outputs/visualisation/curves.mkv"))
    a = p.parse_args(argv)
    updates = [int(u) for u in a.updates.split(",")]
    if a.seg_frames:
        cur, seg = seq_cursor(updates, [int(n) for n in a.seg_frames.split(",")])
    else:
        cur, seg = grid_cursor(a.frames, len(load_metrics(a.run).steps)), None
    w, h = map(int, a.size.split("x"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_curves(a.run, updates, cur, seg, a.out, (w, h), a.fps, a.bars)
    print(f"curves written to {a.out}")


if __name__ == "__main__":
    main()
