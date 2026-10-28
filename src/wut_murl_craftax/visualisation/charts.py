"""Chart colours, the seaborn figure theme and the dark animation frames.

Two jobs. `Theme` is the one visual standard of every static figure: report (light,
80 or 170 mm wide, 300 dpi) and video (dark, 1920 x 1080), built on seaborn
(`rc_params`: ``sns.axes_style`` plus ``sns.plotting_context``) and applied per figure
through ``rc_context`` (`themed_figure`), never as global state. `style_axes`,
`set_labels`, `set_title`, `finish` and `save_png` are what ``analysis.figures``,
``analysis.hard``, ``interpretation.run`` and ``visualisation.timeline`` draw with:
legends sit outside the axes, figures are PNG only. `figure`, `style` and `frame` are
the pixel-sized dark frames of the animated learning-process videos.
"""

import functools
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import matplotlib as mpl
import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import ConstrainedLayoutEngine

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


# ------------------------------------------------------------------------- theme

COLUMN_MM = 80.0
"""One Interspeech column: (170 mm text width - 10 mm gap) / 2."""
MM = 1 / 25.4
WIDTHS = {"column": COLUMN_MM, "double": 170.0}
"""Report figure widths in mm: one column, or two columns."""
SERIES_LIGHT = ("#1c5cab", "#a8411a", "#127552", "#8f5f00")
"""`SERIES` on `SURFACE_LIGHT`: checkpoints and units of one method, report theme."""
FONT = "DejaVu Sans"
"""Bundled with matplotlib, so every machine measures the same text widths."""


class Theme(NamedTuple):
    """Surface, inks, method colours and sizes of one output (report or video)."""

    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    colors: Mapping[str, str]
    seq: Sequence[str]
    """Heat-map ramp, low to high."""
    div: Sequence[str]
    """Diverging ramp: below, at and above the midpoint."""
    width: float
    """Figure width in inches."""
    height: float | None
    """Fixed height in inches (video); None: from the rows (report)."""
    dpi: int
    font: float
    title: float
    pitch_mm: float
    """Row pitch in mm (report); the video fills its frame."""
    lw: float
    ms: float
    suffix: str
    legend_cols: int
    wrap: int
    """Characters per line of a note."""
    wide: bool
    """Double-column report width: panels sit side by side as in the video."""
    series: Sequence[str] = SERIES
    """Categorical colours for units that are not methods (checkpoints, ablations)."""


def report_theme(width: str = "column") -> Theme:
    """The light report theme at one column or the double width."""
    mm = WIDTHS[width]
    return Theme(
        "report", SURFACE_LIGHT, INK_LIGHT, INK2_LIGHT, MUTED_LIGHT, GRID_LIGHT,
        METHOD_COLORS_LIGHT, SEQUENTIAL_LIGHT, DIVERGING_LIGHT, mm * MM, None, 300,
        7.0, 7.5, 3.3, 0.9, 3.0, "_double" if width == "double" else "",
        4 if width == "double" else 2, round(56 * mm / COLUMN_MM), width == "double",
        SERIES_LIGHT,
    )  # fmt: skip


VIDEO = Theme(
    "video", SURFACE, INK, INK2, MUTED, GRID, METHOD_COLORS, SEQUENTIAL_DARK,
    DIVERGING_DARK, 19.2, 10.8, 100, 14.0, 17.0, 0.0, 2.2, 7.0, "_video", 5, 170,
    True, SERIES,
)  # fmt: skip


def theme(name: str, width: str = "column") -> Theme:
    """The theme of ``name``; ``width`` (column or double) sizes the report only.

    Raises:
        ValueError: If ``name`` is not report or video, or ``width`` is unknown.
    """
    if name not in ("report", "video"):
        msg = f"theme must be 'report' or 'video', got {name!r}"
        raise ValueError(msg)
    if width not in WIDTHS:
        msg = f"width must be one of {sorted(WIDTHS)}, got {width!r}"
        raise ValueError(msg)
    return VIDEO if name == "video" else report_theme(width)


def rc_params(th: Theme) -> dict[Any, Any]:
    """The seaborn style and context of ``th`` as matplotlib rc parameters.

    ``sns.axes_style`` gives the base (white grid on paper, dark grid on the video),
    ``sns.plotting_context`` the sizes (paper, talk); the theme's inks, fonts and
    method colours go on top. Pass the result to ``mpl.rc_context``.
    """
    dark = th.name == "video"
    style = sns.axes_style(
        "darkgrid" if dark else "whitegrid",
        rc={
            "axes.facecolor": th.surface, "figure.facecolor": th.surface,
            "savefig.facecolor": th.surface, "axes.edgecolor": th.muted,
            "axes.labelcolor": th.ink2, "text.color": th.ink,
            "xtick.color": th.muted, "ytick.color": th.muted, "grid.color": th.grid,
            "grid.linestyle": "-", "axes.spines.top": False,
            "axes.spines.right": False, "axes.axisbelow": True,
            "font.sans-serif": [FONT], "legend.frameon": False,
            "legend.labelcolor": th.ink2, "legend.title_fontsize": th.font,
            "axes.prop_cycle": cycler(color=list(th.colors.values())),
        },
    )  # fmt: skip
    context = sns.plotting_context(
        "talk" if dark else "paper",
        rc={
            "font.size": th.font, "axes.labelsize": th.font,
            "axes.titlesize": th.title, "xtick.labelsize": th.font,
            "ytick.labelsize": th.font, "legend.fontsize": th.font,
            "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": th.lw,
            "lines.markersize": th.ms, "xtick.major.size": 2.0,
            "ytick.major.size": 2.0, "xtick.major.width": 0.5,
            "ytick.major.width": 0.5, "figure.titlesize": th.title,
        },
    )  # fmt: skip
    return {**style, **context}


_CREATORS = (
    "add_axes", "add_subplot", "legend", "text", "suptitle", "supxlabel", "supylabel",
    "colorbar",
)  # fmt: skip
"""Figure methods that read the rc parameters when they create an artist."""


def _bind(fig: FigureBase, rc: Any) -> None:
    """Make the artist-creating methods of ``fig`` (and its subfigures) use ``rc``."""

    def wrap(plain: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(plain)
        def themed(*args: Any, **kwargs: Any) -> Any:
            with mpl.rc_context(rc):
                return plain(*args, **kwargs)

        return themed

    for name in _CREATORS:
        setattr(fig, name, wrap(getattr(fig, name)))

    plain_sub = fig.subfigures

    @functools.wraps(plain_sub)
    def subfigures(*args: Any, **kwargs: Any) -> Any:
        subs = plain_sub(*args, **kwargs)
        for sub in np.atleast_1d(subs).ravel():
            _bind(sub, rc)
        return subs

    fig.subfigures = subfigures  # type: ignore[method-assign]  # per-instance wrapper


def themed_figure(th: Theme, height_mm: float) -> Figure:
    """A constrained-layout figure on the theme surface (report height: the rows).

    Axes, legends and texts created on it (and on its subfigures) read the theme's
    seaborn rc parameters; the global ``rcParams`` stay untouched.
    """
    rc = rc_params(th)
    fig = Figure(
        figsize=(th.width, th.height or height_mm * MM), dpi=th.dpi,
        facecolor=th.surface, layout="constrained",
    )  # fmt: skip
    FigureCanvasAgg(fig)
    _bind(fig, rc)
    pad = 2 / 72 if th.name == "report" else 8 / 72
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(w_pad=pad, h_pad=pad, hspace=0.04, wspace=0.04)
    return fig


def style_axes(ax: Axes, th: Theme, *, grid: Literal["x", "y", "both"] = "x") -> Axes:
    """Surface, muted ticks and labels, faint grid, left and bottom spines only."""
    ax.set_facecolor(th.surface)
    ax.tick_params(colors=th.muted, labelsize=th.font, length=2, width=0.5)
    sns.despine(ax=ax, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color(th.muted)
        spine.set_linewidth(0.5)
    ax.grid(axis=grid, color=th.grid, lw=0.5)
    ax.set_axisbelow(True)
    return ax


def heatmap(
    ax: Axes,
    th: Theme,
    data: np.ndarray,
    cmap: Any,
    *,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    gid: str | None = None,
) -> Any:
    """Cells of ``data`` with ``sns.heatmap`` (NaN is the colormap's ``bad``).

    No colour bar, tick labels, grid or cell edges: the caller labels the axes with
    ticks at ``i + 0.5`` (the cell centres) and adds its own cell texts.
    """
    with mpl.rc_context(rc_params(th)):
        sns.heatmap(
            data, ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, cbar=False,
            xticklabels=False, yticklabels=False, linewidths=0, mask=np.isnan(data),
        )  # fmt: skip
    ax.grid(False)
    ax.set_aspect("auto")
    mesh = ax.collections[-1]
    mesh.set_gid(gid)
    return mesh


def set_labels(ax: Axes, th: Theme, x: str | None = None, y: str | None = None) -> None:
    """Axis labels in the theme's secondary ink."""
    if x is not None:
        ax.set_xlabel(x, fontsize=th.font, color=th.ink2)
    if y is not None:
        ax.set_ylabel(y, fontsize=th.font, color=th.ink2)


def set_title(
    ax: Axes, th: Theme, text: str, *, narrow: bool = False, width: int | None = None
) -> None:
    """Left-aligned title; ``narrow`` wraps it into a panel with long row labels."""
    width = width or int(th.wrap * (0.45 if narrow else 0.8))
    wrapped = "\n".join(textwrap.wrap(text, width))
    ax.set_title(wrapped, loc="left", fontsize=th.title, color=th.ink)


TITLE_GAP_PX = 6.0
"""Space kept between a wrapped title and the panel or canvas edge to its right."""
FIT_PASSES = 4
"""Layout passes `fit_canvas` may take to bring every artist onto the canvas."""


def _px_width(renderer: Any, text: str, font: Any) -> float:
    return float(renderer.get_text_width_height_descent(text, font, False)[0])


def _wrap_px(renderer: Any, text: str, font: Any, limit: float) -> str:
    """``text`` broken into lines no wider than ``limit`` pixels (words stay whole)."""
    lines: list[str] = []
    for word in text.split():
        if lines and _px_width(renderer, f"{lines[-1]} {word}", font) <= limit:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return "\n".join(lines)


def wrap_titles(fig: Figure) -> bool:
    """Re-wrap panel titles that run into the next panel or off the canvas.

    Titles are left-aligned and the layout engine does not count their width, so a
    long title of a narrow panel would spill sideways. Each is broken at word
    boundaries to the room the panel has: up to the tight box of the nearest axes on
    its right (same rows), else the canvas edge. True if any title changed.
    """
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    width = renderer.width
    axes = [ax for ax in fig.axes if ax.get_visible()]
    changed = False
    for ax in axes:
        title = ax._left_title  # type: ignore[attr-defined]  # loc="left" has no accessor
        text = title.get_text()
        if not text.strip():
            continue
        box = title.get_window_extent(renderer)
        area = ax.get_window_extent(renderer)
        limit = width - TITLE_GAP_PX
        for other in axes:
            near = other.get_tightbbox(renderer)
            if (
                other is not ax
                and near is not None
                and near.x0 >= area.x1
                and near.y1 > area.y0
                and near.y0 < box.y1
            ):
                limit = min(limit, near.x0 - TITLE_GAP_PX)
        if box.x1 <= limit:
            continue
        room = max(limit - box.x0, 1.0)
        wrapped = _wrap_px(
            renderer, " ".join(text.split()), title.get_fontproperties(), room
        )
        if wrapped != text:
            title.set_text(wrapped)
            changed = True
    return changed


def fit_canvas(fig: Figure, margin_px: float = 2.0) -> None:
    """Shrink the layout area until every artist (ticks, legends, notes) is on canvas.

    The constrained layout normally keeps everything inside, but it ignores the width
    of titles and can leave a tick label half a glyph outside; the figure size is
    fixed (80 mm, 170 mm, 1080p), so the rest of the figure gives way, a few pixels.
    """
    engine = fig.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        return
    for _ in range(FIT_PASSES):
        fig.draw_without_rendering()
        box = fig.get_tightbbox(fig.canvas.get_renderer())  # type: ignore[attr-defined]  # Agg canvas
        w, h = fig.get_size_inches()
        pad = margin_px / fig.dpi
        over = (
            max(0.0, pad - box.x0) / w, max(0.0, pad - box.y0) / h,
            max(0.0, box.x1 - (w - pad)) / w, max(0.0, box.y1 - (h - pad)) / h,
        )  # fmt: skip
        if not any(o * w * fig.dpi > 0.5 for o in over[::2]) and not any(
            o * h * fig.dpi > 0.5 for o in over[1::2]
        ):
            return
        left, bottom, rw, rh = engine.get()["rect"]
        engine.set(
            rect=(
                left + over[0], bottom + over[1],
                rw - over[0] - over[2], rh - over[1] - over[3],
            )
        )  # fmt: skip


def settle(fig: Figure) -> None:
    """Lay ``fig`` out for good: titles wrapped to their panels, all of it on canvas."""
    if wrap_titles(fig):
        fig.draw_without_rendering()
    fit_canvas(fig)


def save_png(fig: Figure, th: Theme, out_dir: Path, name: str) -> Figure:
    """Write ``<name><suffix>.png`` into ``out_dir`` (PNG only) and return ``fig``.

    Titles are wrapped to their panels and the layout is fitted to the canvas first
    (`wrap_titles`, `fit_canvas`), so nothing is cropped and the report widths and
    the 1080p frame stay exact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    settle(fig)
    fig.savefig(
        out_dir / f"{name}{th.suffix}.png", facecolor=fig.get_facecolor(), dpi=th.dpi
    )
    return fig


def finish(
    fig: Figure | FigureBase,
    th: Theme,
    handles: Sequence[Any] = (),
    notes: Sequence[str] = (),
    cols: int | None = None,
    loc: Any = "outside upper center",
) -> None:
    """Legend outside the axes (``loc``: above them by default), notes below them."""
    if handles:
        fig.legend(
            handles=list(handles), loc=loc, ncol=cols or th.legend_cols,
            fontsize=th.font, frameon=False, labelcolor=th.ink2,
        )  # fmt: skip
    text = "\n".join(textwrap.fill(n, th.wrap) for n in notes if n)
    if text:
        fig.supxlabel(
            text, x=0.01, ha="left", fontsize=th.font, color=th.muted,
            fontweight="normal",
        )  # fmt: skip


def placeholder(fig_name: str, th: Theme, out_dir: Path, text: str) -> Figure:
    """A figure that only says why there is nothing to draw (still written to disk)."""
    fig = themed_figure(th, 30)
    ax = fig.add_subplot()
    ax.set_axis_off()
    shown = "\n".join(textwrap.wrap(text, th.wrap * 4 // 5))
    ax.text(
        0.5, 0.5, shown, ha="center", va="center", fontsize=th.title, color=th.muted,
        transform=ax.transAxes,
    )  # fmt: skip
    return save_png(fig, th, out_dir, fig_name)
