"""Matplotlib frames for the learning-process charts: dark theme, colours, smoothing."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SURFACE = "#1a1a19"
"""Chart background (dark surface)."""
INK = "#f0efec"
"""Primary text and highlighted lines."""
INK2 = "#c3c2b7"
"""Secondary text."""
MUTED = "#898781"
"""Axes, tick labels, reference lines."""
GRID = "#2c2c2a"
"""Gridlines and faint tracks."""
SERIES = ("#3987e5", "#d95926", "#199e70", "#c98500")
"""Categorical slots 1-4 for checkpoints, validated for colour-vision deficiency."""
SEQ = ("#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb")
"""One-hue blue ramp for magnitudes, starting at the surface colour."""


def ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average that skips NaN (updates with no finished episode)."""
    out, m = np.empty(len(x)), np.nan
    for i, v in enumerate(x):
        if not np.isnan(v):
            m = v if np.isnan(m) else (1 - alpha) * m + alpha * v
        out[i] = m
    return out


def figure(width: int, height: int) -> Figure:
    """A pyplot-free figure of ``width`` x ``height`` pixels on the dark surface."""
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def style(ax: Axes, size: float = 9) -> Axes:
    """Dark-surface axes: muted ticks and labels, faint grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GRID)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return ax


def frame(fig: Figure) -> np.ndarray:
    """Draw ``fig`` and return its pixels as ``(height, width, 3)`` uint8."""
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        msg = f"expected an Agg canvas, got {type(canvas).__name__}"
        raise TypeError(msg)
    canvas.draw()  # type: ignore[no-untyped-call]  # matplotlib Agg lacks annotations
    rgba = canvas.buffer_rgba()  # type: ignore[no-untyped-call]  # same
    return np.asarray(rgba)[..., :3].copy()
