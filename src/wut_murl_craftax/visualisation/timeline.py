"""Achievement timeline: animated heatmap of success per achievement over training.

Rows (22 Classic, 67 Craftax) are sorted by first unlock: the first update in which
any training episode earned the achievement; a ring marks it once the cursor passes.
Colour is the EMA success rate on a square-root scale so 1-5% rates stay visible.
The look is the shared video theme (`charts.Theme`, seaborn); the PNG of the last frame
is written as it is, with every label inside the canvas.
Usage: ``python -m wut_murl_craftax.visualisation.timeline RUN [--logx]``
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from ..training.records import first_unlocks, load_episodes, load_metrics
from . import ffmpeg
from .charts import VIDEO, ema, frame, set_labels, settle, style_axes, themed_figure

logger = logging.getLogger(__name__)


def first_unlock_order(first: np.ndarray) -> np.ndarray:
    """Row order: by first-unlock update, never-unlocked (-1) last, ties by index."""
    key = np.where(first < 0, np.iinfo(np.int64).max, first.astype(np.int64))
    return np.lexsort((np.arange(len(first)), key))


def _cursor(x: np.ndarray, n_anim: int, n_hold: int, logx: bool) -> np.ndarray:
    if logx:
        cur = np.searchsorted(x, np.geomspace(x[0], x[-1], n_anim))
    else:
        cur = np.linspace(0, len(x) - 1, n_anim).astype(int)
    return np.concatenate([cur, np.full(n_hold, len(x) - 1)]).astype(int)


def render_timeline(
    run_dir: Path,
    out: Path,
    seconds: float = 12,
    fps: int = 30,
    hold: float = 2.0,
    size: tuple[int, int] = (1920, 1080),
    logx: bool = False,
) -> Path:
    """Render the sweep and a final ``hold`` to ``out``; last frame also as PNG."""
    m = load_metrics(run_dir)
    first = first_unlocks(load_episodes(run_dir))
    order = first_unlock_order(first)
    rates = np.stack([ema(r, 0.1) for r in m.rates.T])[order]
    first, n_rows = first[order], len(order)
    x = m.steps / 1e6
    step0 = x[1] - x[0] if len(x) > 1 else x[0]
    edges = np.r_[x[0] / 2 if logx else x[0] - step0, x]
    fs = size[1] / (70 if n_rows <= 30 else 120)
    th = VIDEO._replace(
        width=size[0] / VIDEO.dpi, height=size[1] / VIDEO.dpi, font=fs, title=fs * 1.3
    )
    fig = themed_figure(th, 0)
    ax = style_axes(fig.add_subplot(), th, grid="x")
    cmap = LinearSegmentedColormap.from_list("seq", th.seq)
    mesh = ax.pcolormesh(
        edges,
        np.arange(n_rows + 1),
        np.ma.masked_all(rates.shape),
        cmap=cmap,
        norm=PowerNorm(0.5, 0, 100),
        shading="flat",
    )
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(n_rows, 0)
    names = [m.names[i].replace("_", " ") for i in order]
    ax.set_yticks(np.arange(n_rows) + 0.5, names)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    set_labels(ax, th, "environment steps (M)")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(
        "success rate, % of episodes (sqrt scale)", color=th.ink2, fontsize=fs
    )
    bar.ax.tick_params(colors=th.muted, labelsize=fs)
    bar.outline.set_visible(False)
    fx = np.where(first < 0, np.nan, x[np.clip(first - 1, 0, None)])
    rows = np.arange(n_rows) + 0.5
    dots = ax.scatter(
        fx, rows, s=fs * 2.2, facecolors="none", edgecolors=th.ink, linewidths=1.2
    )
    title = ax.set_title("", loc="left", fontsize=fs * 1.3, color=th.ink)
    count = ax.set_title("", loc="right", fontsize=fs * 1.1, color=th.muted)
    title.set_text("Achievements over training - 0.00M steps")
    count.set_text(f"unlocked at least once: {n_rows} / {n_rows}")
    settle(fig)
    fig.set_layout_engine("none")  # the layout is settled; frames only change data
    cursor = _cursor(x, int(seconds * fps), int(hold * fps), logx)
    t0 = time.time()
    with ffmpeg.FrameWriter(out, *size, fps=fps) as writer:
        for i in cursor:
            vis = rates.copy()
            vis[:, i + 1 :] = np.nan
            mesh.set_array(np.ma.masked_invalid(vis).ravel())
            seen = (first >= 0) & (first <= i + 1)
            dots.set_offsets(np.c_[np.where(seen, fx, np.nan), rows])
            title.set_text(f"Achievements over training - {x[i]:.2f}M steps")
            count.set_text(f"unlocked at least once: {seen.sum()} / {n_rows}")
            writer.write(frame(fig))
    fig.savefig(out.with_suffix(".png"), facecolor=fig.get_facecolor(), dpi=th.dpi)
    logger.info("timeline %s: %d frames in %.1fs", out, len(cursor), time.time() - t0)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: ``RUN [--seconds 12] [--fps 30] [--size 1920x1080] [--logx]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--logx", action="store_true")
    p.add_argument(
        "--out", type=Path, default=Path("outputs/visualisation/timeline.mkv")
    )
    a = p.parse_args(argv)
    w, h = map(int, a.size.split("x"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_timeline(a.run, a.out, a.seconds, a.fps, a.hold, (w, h), a.logx)
    print(f"timeline written to {a.out}")


if __name__ == "__main__":
    main()
