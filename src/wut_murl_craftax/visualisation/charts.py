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
SURFACE_LIGHT = "#ffffff"
"""Chart background of the report theme."""
INK_LIGHT = "#1a1a19"
"""Primary text and highlighted lines of the report theme."""
INK2_LIGHT = "#4a4944"
"""Secondary text of the report theme."""
MUTED_LIGHT = "#6b6a64"
"""Axes, tick labels and reference lines of the report theme."""
GRID_LIGHT = "#e6e5df"
"""Gridlines and faint tracks of the report theme."""
METHOD_COLORS: dict[str, str] = {
    "ppo": "#d6d5cd",
    "ppo-rnn": "#a3a29b",
    "pqn-rnn": "#706f6a",
    "count-bonus-ppo-rnn": "#3987e5",
    "count-bonus-pqn-rnn": "#86b6ef",
    "sil": "#199e70",
    "smdp-skills": "#c98500",
    "dyna-q": "#d95926",
    "combo": "#a35bd6",
}
"""One colour per registry method on `SURFACE` (video theme): the three controls are
greys, each at least 3:1 against the surface (contracts.md section 6)."""
METHOD_COLORS_LIGHT: dict[str, str] = {
    "ppo": "#2e2e2c",
    "ppo-rnn": "#5c5b57",
    "pqn-rnn": "#8a8983",
    "count-bonus-ppo-rnn": "#1c5cab",
    "count-bonus-pqn-rnn": "#3987e5",
    "sil": "#127552",
    "smdp-skills": "#8f5f00",
    "dyna-q": "#a8411a",
    "combo": "#7a3aa8",
}
"""The same keys on `SURFACE_LIGHT` (report theme)."""


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


SEQUENTIAL_LIGHT = ("#fff7b5", "#bcaf6f", "#7c7b78", "#31446b", "#00204d")
"""Heat-map ramp on `SURFACE_LIGHT`, low to high: sampled from cividis (colour-vision
safe); no value equals a method colour, so a cell is never read as a method."""
SEQUENTIAL_DARK = ("#00204d", "#31446b", "#7c7b78", "#bcaf6f", "#ffea46")
"""The same ramp on `SURFACE` (low values dark)."""
DIVERGING_LIGHT = ("#9a8700", "#f1f0ec", "#00306f")
"""Below, at and above the midpoint on `SURFACE_LIGHT` (probability of improvement)."""
DIVERGING_DARK = ("#d4bd2a", "#2c2c2a", "#2e5fb0")
"""Below, at and above the midpoint on `SURFACE`."""
