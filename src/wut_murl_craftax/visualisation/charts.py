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

import contextlib
import csv
import enum
import functools
import logging
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
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
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

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

logger = logging.getLogger(__name__)

COLUMN_MM = 80.0
"""One Interspeech column: (170 mm text width - 10 mm gap) / 2."""
MM = 1 / 25.4
TILE_MM = (COLUMN_MM - 2) / 2
"""Two panels side by side in one column (2 mm gap); only if a panel reads at 39 mm."""
WIDTHS = {"column": COLUMN_MM, "double": 170.0, "tile": TILE_MM}
"""Report figure widths in mm: one column, two columns, half a column.

Figures are included at ``width=\\linewidth`` with no scaling, so these are final."""


class Shape(enum.StrEnum):
    """The height class of a report figure: its width times an aspect, or rows."""

    CURVE = "curve"
    """Curves and scatters: height 0.62 of the width."""
    DOTS = "dots"
    """Short dot or bar panels (a few rows): 0.45 of the width."""
    SQUARE = "square"
    """Heat maps and polar plots: 1.0 of the width."""
    ROWS = "rows"
    """Long category lists: a fixed height per row, at most `MAX_ROWS_MM`."""
    LEGEND = "legend"
    """A figure with no axes (the method legend, a placeholder): sized by its text."""


ASPECT: dict[Shape, float] = {Shape.CURVE: 0.62, Shape.DOTS: 0.45, Shape.SQUARE: 1.0}
"""Height over width of the fixed-aspect classes."""
MAX_ROWS_MM = 80.0
"""A `Shape.ROWS` figure taller than this (about a third of a page) moves to the double
width or is split into panels."""
ROWS_BASE_MM = 14.0
"""x axis, its label and the margins of a `Shape.ROWS` figure, besides its rows."""


def shape_height_mm(
    th: "Theme",
    shape: Shape,
    rows: int | None = None,
    base_mm: float | None = None,
    pitch_mm: float | None = None,
) -> float:
    """Report height in mm of ``shape`` at the theme's width (the video has its own).

    ``CURVE``, ``DOTS`` and ``SQUARE`` are width times `ASPECT`; ``ROWS`` is
    ``rows * pitch_mm + base_mm`` (needs ``rows``; ``pitch_mm`` defaults to the theme's
    row pitch, ``base_mm`` to `ROWS_BASE_MM`, give more for a legend or long rotated
    labels); ``LEGEND`` needs
    ``rows`` as the number of legend rows (4.2 mm each, 4 mm margin).

    Raises:
        ValueError: If ``rows`` is missing for ``ROWS`` or ``LEGEND``.
    """
    if shape in ASPECT:
        return th.width * 25.4 * ASPECT[shape]
    if rows is None:
        msg = f"shape {shape.value!r} needs the number of rows"
        raise ValueError(msg)
    if shape is Shape.LEGEND:
        return rows * 4.2 + 4.0
    pitch = th.pitch_mm if pitch_mm is None else pitch_mm
    return rows * pitch + (ROWS_BASE_MM if base_mm is None else base_mm)


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
        7.0, 7.5, 3.3, 0.9, 3.0, {"double": "_double", "tile": "_tile"}.get(width, ""),
        4 if width == "double" else 1 if width == "tile" else 2,
        round(56 * mm / COLUMN_MM), width == "double",
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


def themed_figure(
    th: Theme,
    height_mm: float | None = None,
    *,
    shape: Shape | None = None,
    rows: int | None = None,
    base_mm: float | None = None,
    pitch_mm: float | None = None,
) -> Figure:
    """A constrained-layout figure on the theme surface.

    The report height is ``height_mm``, or from ``shape`` (`shape_height_mm`: a fixed
    aspect, or ``rows`` rows); the figure remembers its ``shape`` for the audit. The
    video is always 1920 x 1080. Axes, legends and texts created on it (and on its
    subfigures) read the theme's seaborn rc parameters; the global ``rcParams`` stay
    untouched.

    Raises:
        ValueError: If neither ``height_mm`` nor ``shape`` is given.
    """
    if shape is not None and height_mm is None:
        height_mm = shape_height_mm(th, shape, rows, base_mm, pitch_mm)
    if height_mm is None:
        msg = "themed_figure needs a height_mm or a shape"
        raise ValueError(msg)
    rc = rc_params(th)
    fig = Figure(
        figsize=(th.width, th.height or height_mm * MM), dpi=th.dpi,
        facecolor=th.surface, layout="constrained",
    )  # fmt: skip
    fig.shape = shape  # type: ignore[attr-defined]  # read back by the audit
    fig.font_scale = th.font / 7.0  # type: ignore[attr-defined]  # `add_panel` insets
    FigureCanvasAgg(fig)
    _bind(fig, rc)
    fig.theme_rc = rc  # type: ignore[attr-defined]  # read back by `settle`, `write_png`
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
    nbins = 3 if th.name == "report" and not th.wide else 5  # 7 pt labels need room
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    return ax


def category_figure(
    th: Theme, rows: int, extra_mm: float = 0.0, pitch_mm: float | None = None
) -> Figure:
    """A figure for ``rows`` category rows: `Shape.DOTS` if they fit, else `Shape.ROWS`.

    Short lists get the fixed 0.45 aspect (they tile with their neighbours); a longer
    list grows by the row pitch (the theme's, or ``pitch_mm`` where each row holds
    several dodged marks). ``extra_mm`` adds room for a legend or long rotated labels.
    """
    base = ROWS_BASE_MM + extra_mm
    pitch = th.pitch_mm if pitch_mm is None else pitch_mm
    if rows * pitch + base <= th.width * 25.4 * ASPECT[Shape.DOTS]:
        return themed_figure(th, shape=Shape.DOTS)
    return themed_figure(
        th, shape=Shape.ROWS, rows=rows, base_mm=base, pitch_mm=pitch_mm
    )


def label_inset_mm(th: Theme, labels: Sequence[str], pad_mm: float = 3.5) -> float:
    """Left inset in mm that fits the longest of ``labels`` (measured) and a y label.

    Panels meant to tile share one inset (the largest of the group), so their axes
    line up when LaTeX sets them side by side or stacked.
    """
    from matplotlib.textpath import TextPath

    widest = max(
        (TextPath((0, 0), t, size=th.font).get_extents().width for t in labels),
        default=0.0,
    )  # points
    return widest / 72 * 25.4 + pad_mm


def add_panel(
    fig: Figure,
    *,
    left_mm: float,
    bottom_mm: float = 11.0,
    right_mm: float = 4.0,
    top_mm: float = 2.0,
    **kwargs: Any,
) -> Axes:
    """One axes with exact insets in mm from the canvas edges (no layout engine).

    ``left_mm`` holds the category or tick labels and the y label (see
    `label_inset_mm`), ``bottom_mm`` the x tick labels and x label; bottom, right and
    top are given for 7 pt text and grow with the theme's font (the video). Give every
    panel of a tile group the same insets and size: their axes areas are then identical.
    """
    fig.set_layout_engine("none")
    scale = getattr(fig, "font_scale", 1.0)
    bottom_mm, right_mm, top_mm = bottom_mm * scale, right_mm * scale, top_mm * scale
    w, h = fig.get_size_inches() * 25.4
    return fig.add_axes(
        (left_mm / w, bottom_mm / h, 1 - (left_mm + right_mm) / w,
         1 - (bottom_mm + top_mm) / h),
        **kwargs,
    )  # fmt: skip


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


def _left_title(ax: Axes) -> Text | None:
    """The ``loc="left"`` title text of ``ax`` (matplotlib has no accessor for it).

    Found among the axes' children as the text at the axes' top left corner that
    carries the title string; None if the axes has no left title.
    """
    wanted = ax.get_title(loc="left")
    for child in ax.get_children():
        if (
            isinstance(child, Text)
            and child is not ax.title
            and child.get_text() == wanted
            and child.get_position() == (0.0, 1.0)
        ):
            return child
    return None


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
        title = _left_title(ax)
        if title is None or not title.get_text().strip():
            continue
        text = title.get_text()
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


def _overflow_px(fig: Figure, margin_px: float) -> tuple[float, float, float, float]:
    """Pixels by which the artists leave the canvas: left, bottom, right, top."""
    fig.draw_without_rendering()
    box = fig.get_tightbbox(fig.canvas.get_renderer())  # type: ignore[attr-defined]  # Agg canvas
    w, h = fig.get_size_inches() * fig.dpi
    x0, y0, x1, y1 = (
        box.x0 * fig.dpi,
        box.y0 * fig.dpi,
        box.x1 * fig.dpi,
        box.y1 * fig.dpi,
    )
    return (
        max(0.0, margin_px - x0), max(0.0, margin_px - y0),
        max(0.0, x1 - (w - margin_px)), max(0.0, y1 - (h - margin_px)),
    )  # fmt: skip


def _wrap_wide_texts(fig: Figure, margin_px: float) -> bool:
    """Wrap figure-level texts (notes, titles) wider than the canvas; True if any."""
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    limit = fig.get_size_inches()[0] * fig.dpi - 2 * margin_px
    changed = False
    for text in fig.texts:
        if text.get_window_extent(renderer).width <= limit or not text.get_text():
            continue
        lines = [
            _wrap_px(renderer, line, text.get_fontproperties(), limit)
            for line in text.get_text().splitlines()
        ]
        text.set_text("\n".join(lines))
        changed = True
    return changed


def fit_canvas(fig: Figure, margin_px: float = 2.0) -> None:
    """Shrink the layout area until every artist (ticks, legends, notes) is on canvas.

    The constrained layout normally keeps everything inside, but it ignores the width
    of titles and can leave a tick label half a glyph outside; the figure size is
    fixed (80 mm, 170 mm, 1080p), so the rest of the figure gives way, a few pixels.
    After the last pass the result is measured again: a note or title wider than the
    canvas is wrapped (shrinking cannot help it); anything still outside is logged.
    """
    engine = fig.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        return
    w, h = fig.get_size_inches() * fig.dpi
    for attempt in range(2):
        for _ in range(FIT_PASSES):
            over = _overflow_px(fig, margin_px)
            if max(over) <= 0.5:
                return
            left, bottom, rw, rh = engine.get()["rect"]
            engine.set(
                rect=(
                    left + over[0] / w, bottom + over[1] / h,
                    rw - (over[0] + over[2]) / w, rh - (over[1] + over[3]) / h,
                )
            )  # fmt: skip
        if attempt == 0 and not _wrap_wide_texts(fig, margin_px):
            break
    if max(_overflow_px(fig, margin_px)) > 0.5:
        logger.warning("figure still has artists outside its canvas after fitting")


def settle(fig: Figure) -> None:
    """Lay ``fig`` out for good: titles wrapped to their panels, all of it on canvas."""
    with mpl.rc_context(_rc_of(fig)):
        _wrap_wide_texts(fig, 2.0)  # before any layout: a wider text cannot be solved
        if wrap_titles(fig):
            fig.draw_without_rendering()
        fit_canvas(fig)


def _rc_of(fig: Figure) -> dict[Any, Any]:
    """The rc parameters of a themed figure (empty for any other figure)."""
    return dict(getattr(fig, "theme_rc", {}))


def write_png(fig: Figure, path: Path) -> None:
    """`settle` ``fig`` and save it as ``path``, drawing under its theme's rc values.

    Ticks, title and label pads are created lazily while drawing: they must read the
    seaborn values, so the draw happens inside the figure's ``rc_context`` too.
    """
    settle(fig)
    with mpl.rc_context(_rc_of(fig)):
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=fig.dpi)


class Caption(NamedTuple):
    """What the report says about one figure; the image itself carries no sentence."""

    stem: str
    """File stem without the ``_double`` / ``_video`` suffix (``combat_kills``)."""
    title: str
    """A few words that name the figure (the first cell of its row in the notes)."""
    description: str
    """What is drawn and how to read it, with the details the image leaves out."""


_CAPTURES: list[dict[str, Caption]] = []
"""Open `capture_captions` blocks, innermost last."""


@contextlib.contextmanager
def capture_captions() -> Iterator[list[Caption]]:
    """Collect the `Caption` of every figure saved inside the block (one per stem).

    The list fills while the block runs; a stem saved in several variants (column,
    double, video) or twice keeps its last caption. Blocks may nest: each sees the
    figures saved inside it.
    """
    found: dict[str, Caption] = {}
    out: list[Caption] = []
    _CAPTURES.append(found)
    try:
        yield out
    finally:
        _CAPTURES.pop()
        out.extend(found.values())


def register_caption(stem: str, title: str, description: str) -> None:
    """Record the caption of figure ``stem`` in every open `capture_captions` block.

    Raises:
        ValueError: If the title or the description is empty, or a text has a line
            break or a pipe (one table cell each in the notes).
    """
    for text in (title, description):
        if not text.strip() or "\n" in text or "|" in text:
            msg = f"caption of {stem!r} needs one-line text without '|', got {text!r}"
            raise ValueError(msg)
    for found in _CAPTURES:
        found[stem] = Caption(stem, title, description)


def figure_dir(out_dir: Path, budget: str) -> Path:
    """The folder of one benchmark's figures: ``<out_dir>/<budget>`` (``1m``, ``1b``).

    A 1B run adds ``1b/`` beside ``1m/`` without touching it; stems carry no budget.
    """
    return out_dir / budget


CAPTION_COLUMNS = ("stem", "id", "title", "description")
"""Columns of ``captions.csv``, which every figure stage keeps next to its PNGs."""


def write_captions(folder: Path, fid: str, rows: Sequence[Caption]) -> Path:
    """Replace the ``fid`` rows of ``<folder>/captions.csv`` by ``rows``; return it.

    The file lists every figure of the folder (``stem,id,title,description``), the id
    being the figure id of the stage (``F13``). Rows of other ids are kept, so one
    stage rerun never loses the captions of the others.
    """
    path = folder / "captions.csv"
    keep: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["id"] != fid]
    keep += [
        {"stem": r.stem, "id": fid, "title": r.title, "description": r.description}
        for r in rows
    ]
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAPTION_COLUMNS)
        writer.writeheader()
        writer.writerows(keep)
    return path


def save_png(
    fig: Figure,
    th: Theme,
    out_dir: Path,
    name: str,
    *,
    title: str | None = None,
    description: str | None = None,
) -> Figure:
    """Write ``<name><suffix>.png`` into ``out_dir`` (PNG only) and return ``fig``.

    Titles are wrapped to their panels and the layout is fitted to the canvas first
    (`wrap_titles`, `fit_canvas`), so nothing is cropped and the report widths and
    the 1080p frame stay exact. With ``description`` the figure registers its
    `Caption` (``title`` and ``description``, stem ``name``): that is where the text
    that is not on the image goes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_png(fig, out_dir / f"{name}{th.suffix}.png")
    if description is not None:
        register_caption(name, title or name.replace("_", " "), description)
    return fig


def finish(
    fig: Figure | FigureBase,
    th: Theme,
    handles: Sequence[Any] = (),
    notes: Sequence[str] = (),
    cols: int | None = None,
    loc: Any = "outside upper center",
    *,
    notes_in_legend: bool = False,
) -> None:
    """Legend outside the axes (``loc``: above them by default), notes below them.

    ``notes_in_legend`` puts the notes in the legend as its left-aligned title: a
    legend and a note both at the bottom edge are placed over each other by the
    layout engine, one box has no such problem.
    """
    text = "\n".join(textwrap.fill(n, th.wrap) for n in notes if n)
    fixed = isinstance(fig, Figure) and fig.get_layout_engine() is None
    if fixed and str(loc).startswith("outside "):
        loc = str(loc).removeprefix("outside ")  # fixed insets: the caller left room
    if handles:
        legend = fig.legend(
            handles=list(handles), loc=loc, ncol=cols or th.legend_cols,
            fontsize=th.font, frameon=False, labelcolor=th.ink2,
            title=text if notes_in_legend and text else None,
            alignment="left" if notes_in_legend else "center",
        )  # fmt: skip
        if notes_in_legend and text:
            legend.get_title().set_color(th.muted)
            legend.get_title().set_fontsize(th.font)
            text = ""
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
