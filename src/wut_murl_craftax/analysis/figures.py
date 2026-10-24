"""Core comparison figures F1-F5 and F18-F20: curves, intervals, PoI, spread, hardware.

Each ``plot_*`` draws one figure per theme, writes ``<name>.pdf`` and ``.png`` into
``out_dir`` (``_video`` suffix for the video) and returns the `Figure`. ``report`` is
light, 80 mm wide (one Interspeech column), fonts of 7 pt or more; ``video`` is dark,
1920 x 1080. Colours: `charts.METHOD_COLORS`; controls dashed, ablations dotted, hollow.
Numbers come from the `compare` tables and `evaluation.stats`. A budget without runs is
"pending" in the note (R86); laptop timings carry R97's note. Artists carry a ``gid``.

@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures.py]]
"""

import itertools
import logging
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from ..evaluation.hardware import TUNING_PHASES
from ..evaluation.stability import ROBUST, config_scores
from ..evaluation.stats import BOOTSTRAP_N, curve_band, step_grid
from ..visualisation import charts
from . import compare
from .hard import achievement_status
from .load import FINAL_PHASES, Frames, in_phase
from .obstacles import SCORE

logger = logging.getLogger(__name__)

COLUMN_MM = 80.0
"""One Interspeech column: (170 mm text width - 10 mm gap) / 2."""
MM = 1 / 25.4
BUDGETS = ("1m", "1b")
"""The benchmarks of the project; a missing one is pending, never an empty panel."""
PENDING = "Craftax-{} pending"
PENDING_1B = PENDING.format("1B")
BLOCK_TITLES = {compare.MAIN: "main", compare.SKILLS: "uses hand-written skills"}
CONTROLS = ("ppo", "ppo-rnn", "pqn-rnn")
CONTROL_MARKERS = {"ppo": "o", "ppo-rnn": "s", "pqn-rnn": "^"}
"""Told apart by marker as well as by grey (contracts.md section 6)."""
NAMES = {
    "F1": "curves_score",
    "F2": "aggregate",
    "F3": "poi_matrix",
    "F4": "seed_spread",
    "F5": "success_heatmap",
    "F18": "sensitivity",
    "F19": "stability",
    "F20": "hardware",
}
"""Figure id -> file stem (``.pdf`` and ``.png``; ``_video`` is added for the video)."""
WRAP = {"report": 56, "video": 170}
"""Characters per line of a note, by theme."""
STEPS = {"m": 1e6, "b": 1e9}
LIT_LABEL = 34
"""Characters of a literature name shown beside its mark."""
HARDWARE_PANELS: tuple[tuple[str, tuple[tuple[str, str, float], ...]], ...] = (
    (
        "steps per second",
        (("steady_sps", "steady SPS", 1.0), ("overall_sps", "overall SPS", 1.0)),
    ),
    ("wall-clock (h)", (("wall_clock_s", "wall-clock", 1 / 3600),)),
    ("compile time (s)", (("compile_s", "compile time", 1.0),)),
    (
        "peak memory (GB)",
        (
            ("peak_host_mem_mb", "host memory", 1 / 1024),
            ("peak_rss_mb", "process RSS", 1 / 1024),
            ("peak_device_mem_mb", "device memory", 1 / 1024),
        ),
    ),
    (
        "utilisation (%)",
        (("cpu_util_mean", "CPU", 1.0), ("gpu_util_mean", "GPU", 1.0)),
    ),
    ("steps per joule", (("steps_per_joule", "steps per joule", 1.0),)),
)
"""F20 panels: x label, then (T4 column, legend name, factor to the shown unit)."""
DARK_TEXT, LIGHT_TEXT = charts.INK_LIGHT, charts.SURFACE_LIGHT
"""Text on a light and on a dark heat-map cell."""
SERIES_MARKERS = ("o", "D", "^")
"""Marker of the first, second and third series of a hardware panel."""


class Theme(NamedTuple):
    """Surface, inks, method colours and sizes of one output (report or video)."""

    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    colors: Mapping[str, str]
    low: str
    mid: str
    high: str
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


THEMES: dict[str, Theme] = {
    "report": Theme(
        "report", charts.SURFACE_LIGHT, charts.INK_LIGHT, charts.INK2_LIGHT,
        charts.MUTED_LIGHT, charts.GRID_LIGHT, charts.METHOD_COLORS_LIGHT,
        "#a8411a", "#f1f0ec", "#1c5cab",
        COLUMN_MM * MM, None, 300, 7.0, 7.5, 3.3, 0.9, 3.0, "", 1,
    ),
    "video": Theme(
        "video", charts.SURFACE, charts.INK, charts.INK2, charts.MUTED, charts.GRID,
        charts.METHOD_COLORS, "#d95926", "#2c2c2a", "#3987e5",
        19.2, 10.8, 100, 14.0, 17.0, 0.0, 2.2, 7.0, "_video", 5,
    ),
}  # fmt: skip


def _theme(name: str) -> Theme:
    if name not in THEMES:
        msg = f"theme must be one of {sorted(THEMES)}, got {name!r}"
        raise ValueError(msg)
    return THEMES[name]


# ----------------------------------------------------------------------- plumbing


def _color(th: Theme, method: str) -> str:
    if method not in th.colors:
        msg = f"no colour for method {method!r}: add it to charts.METHOD_COLORS"
        raise ValueError(msg)
    return th.colors[method]


def _style_of(th: Theme, method: str, ref: str = compare.TUNED) -> dict[str, Any]:
    """Line style of a unit: controls dashed with a marker, ablations dotted, hollow."""
    color = _color(th, method)
    if ref != compare.TUNED:
        return {
            "color": color, "ls": ":", "marker": "o", "markerfacecolor": th.surface,
            "label": f"{method} ({ref})",
        }  # fmt: skip
    if method in CONTROLS:
        return {
            "color": color, "ls": "--", "marker": CONTROL_MARKERS[method],
            "markerfacecolor": color, "label": method,
        }  # fmt: skip
    return {
        "color": color, "ls": "-", "marker": "o", "markerfacecolor": color,
        "label": method,
    }  # fmt: skip


def _figure(th: Theme, height_mm: float) -> Figure:
    """A constrained-layout figure on the theme surface (report height: the rows)."""
    height = th.height or height_mm * MM
    fig = Figure(
        figsize=(th.width, height), dpi=th.dpi, facecolor=th.surface,
        layout="constrained",
    )  # fmt: skip
    FigureCanvasAgg(fig)
    pad = 2 / 72 if th.name == "report" else 8 / 72
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(w_pad=pad, h_pad=pad, hspace=0.04, wspace=0.04)
    return fig


def _style(ax: Axes, th: Theme, *, grid: Literal["x", "y", "both"] = "x") -> Axes:
    """Surface, muted ticks and labels, faint grid, left and bottom spines only."""
    ax.set_facecolor(th.surface)
    ax.tick_params(colors=th.muted, labelsize=th.font, length=2, width=0.5)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(th.muted)
        spine.set_linewidth(0.5)
    ax.grid(axis=grid, color=th.grid, lw=0.5)
    ax.set_axisbelow(True)
    return ax


def _labels(ax: Axes, th: Theme, x: str | None = None, y: str | None = None) -> None:
    if x is not None:
        ax.set_xlabel(x, fontsize=th.font, color=th.ink2)
    if y is not None:
        ax.set_ylabel(y, fontsize=th.font, color=th.ink2)


def _title(
    ax: Axes, th: Theme, text: str, *, narrow: bool = False, width: int | None = None
) -> None:
    """Left-aligned title; ``narrow`` wraps it into a panel with long row labels."""
    width = width or int(WRAP[th.name] * (0.45 if narrow else 0.8))
    wrapped = "\n".join(textwrap.wrap(text, width))
    ax.set_title(wrapped, loc="left", fontsize=th.title, color=th.ink)


def _budget_name(budget: str) -> str:
    return f"Craftax-{budget.upper()}"


def _panel_title(budget: str, what: str) -> str:
    """``Craftax-1M: what``; a 1B title says indicative (3 seeds, R86)."""
    flag = " (indicative)" if budget not in compare.SIGNIFICANCE_BUDGETS else ""
    return f"{_budget_name(budget)}{flag}: {what}"


def _pending(present: Sequence[str]) -> list[str]:
    """One note per project benchmark without runs."""
    return [PENDING.format(b.upper()) for b in BUDGETS if b not in present]


def _save(fig: Figure, th: Theme, out_dir: Path, name: str) -> Figure:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Type 42 keeps the PDF's text as real fonts, which the venue requires
    with mpl.rc_context({"pdf.fonttype": 42}):
        for ext in ("pdf", "png"):
            fig.savefig(
                out_dir / f"{name}{th.suffix}.{ext}", facecolor=fig.get_facecolor(),
                dpi=th.dpi,
            )  # fmt: skip
    return fig


def _finish(
    fig: Figure | SubFigure,
    th: Theme,
    handles: Sequence[Any] = (),
    notes: Sequence[str] = (),
    cols: int | None = None,
) -> None:
    """Legend above the panels, the wrapped notes below them."""
    if handles:
        fig.legend(
            handles=list(handles), loc="outside upper center",
            ncol=cols or th.legend_cols, fontsize=th.font, frameon=False,
            labelcolor=th.ink2,
        )  # fmt: skip
    text = "\n".join(textwrap.fill(n, WRAP[th.name]) for n in notes if n)
    if text:
        fig.supxlabel(
            text, x=0.01, ha="left", fontsize=th.font, color=th.muted,
            fontweight="normal",
        )  # fmt: skip


def _placeholder(fig_name: str, th: Theme, out_dir: Path, text: str) -> Figure:
    """A figure that only says why there is nothing to draw (still written to disk)."""
    fig = _figure(th, 30)
    ax = fig.add_subplot()
    ax.set_axis_off()
    shown = "\n".join(textwrap.wrap(text, WRAP[th.name] * 4 // 5))
    ax.text(
        0.5, 0.5, shown, ha="center", va="center", fontsize=th.title, color=th.muted,
        transform=ax.transAxes,
    )  # fmt: skip
    return _save(fig, th, out_dir, fig_name)


def _no_data(name: str, th: Theme, out_dir: Path, what: str) -> Figure:
    return _placeholder(name, th, out_dir, f"no data: {what}")


class Row(NamedTuple):
    """One row of an interval-style panel: heading, control, method or ablation."""

    y: int
    label: str
    kind: str
    block: str
    method: str
    ref: str


def _rows(
    members: Sequence[compare.Member],
    ablations: Mapping[tuple[str, str], Sequence[str]] | None = None,
    *,
    headings: bool = True,
) -> list[Row]:
    """Rows top to bottom: per block a heading, its controls and methods, ablations.

    With ``headings=False`` a control appears once (hardware does not depend on the
    block) and blocks are told apart by a separator only.
    """
    rows: list[Row] = []
    seen: set[str] = set()
    block = ""

    def add(label: str, kind: str, blk: str, method: str, ref: str) -> None:
        rows.append(Row(len(rows), label, kind, blk, method, ref))

    for blk, method, role in members:
        if blk != block and headings:
            add(BLOCK_TITLES[blk], "heading", blk, "", "")
        block = blk
        if role == "control" and not headings:
            if method in seen:
                continue
            seen.add(method)
        add(method, role, blk, method, compare.TUNED)
        for ref in (ablations or {}).get((blk, method), ()):
            add(f"{method} ({ref})", "ablation", blk, method, ref)
    return rows


def _apply_rows(ax: Axes, th: Theme, rows: Sequence[Row]) -> None:
    """Row labels on y (headings bold), top to bottom, a rule between blocks."""
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks([r.y for r in rows], [r.label for r in rows])
    ax.tick_params(axis="y", length=0)
    for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(th.ink if row.kind == "heading" else th.ink2)
        tick.set_fontweight("bold" if row.kind == "heading" else "normal")
    ax.grid(axis="y", visible=False)
    for prev, row in itertools.pairwise(rows):
        if row.block != prev.block:
            ax.axhline(row.y - 0.5, color=th.muted, lw=0.5)


def _height_mm(th: Theme, n_rows: int, panels: int, extra: float = 16.0) -> float:
    """Report height: row pitch per row plus a title and x axis per panel."""
    return n_rows * th.pitch_mm + panels * extra + 14.0


def _stack(fig: Figure, th: Theme, ratios: Sequence[float]) -> list[Axes]:
    """One axes per panel: stacked in the report column, side by side in the video."""
    n = len(ratios)
    if th.name == "report":
        axes = fig.subplots(n, 1, squeeze=False, height_ratios=list(ratios))
    else:
        axes = fig.subplots(1, n, squeeze=False)
    return [ax for row in axes for ax in row]


def _budget_members(runs: pd.DataFrame) -> dict[str, list[compare.Member]]:
    """Per budget the table rows (block, method, role) of its tuned runs."""
    return {b: compare.members(r) for b, r in compare.budgets(compare.tuned(runs))}


def _ablation_refs(
    table: pd.DataFrame, budget: str
) -> dict[tuple[str, str], list[str]]:
    """Block and method -> their ablations' ``config_ref`` in table order."""
    out: dict[tuple[str, str], list[str]] = {}
    for r in table[table["benchmark"] == budget].itertuples():
        out.setdefault((r.block, r.method), []).append(r.config_ref)
    return out


def _jitter(n: int, spread: float = 0.2) -> np.ndarray:
    """Fixed vertical offsets that keep neighbouring dots apart without randomness."""
    return (np.arange(n) % 3 - 1) * spread


def _steps_text(v: float, _pos: int | None = None) -> str:
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= scale:
            return f"{v / scale:.3g}{unit}"
    return f"{v:.0f}"


def _budget_steps(budget: str) -> float:
    return float(budget[:-1]) * STEPS[budget[-1]]


def _gid(kind: str, *parts: str) -> str:
    return "/".join((kind, *parts))


def _ref_gid(kind: str, budget: str, row: Row) -> str:
    return _gid(kind, budget, row.block, row.method, row.ref)


# --------------------------------------------------------------------------- F1


def _final_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs of the final phases (an ablation is a final-phase unit)."""
    mask = np.logical_or.reduce([in_phase(runs, p) for p in FINAL_PHASES])
    return runs[mask]


def _unit_curves(
    score: pd.DataFrame, runs: pd.DataFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One ``(env_steps, score)`` pair per run of ``runs`` that logged the curve."""
    mine = score[score["run_id"].isin(runs["run_id"])]
    return [
        (g["env_steps"].to_numpy(dtype=float), g["value"].to_numpy(dtype=float))
        for _, g in mine.groupby("run_id")
    ]


def _draw_curve(ax: Axes, th: Theme, band: pd.DataFrame, row: Row, budget: str) -> None:
    style = _style_of(th, row.method, row.ref)
    x, y = band["env_steps"].to_numpy(), band["estimate"].to_numpy()
    fill = ax.fill_between(
        x, band["lo"], band["hi"], color=style["color"], alpha=0.15, lw=0,
        label="_band",
    )  # fmt: skip
    fill.set_gid(_ref_gid("band", budget, row))
    (line,) = ax.plot(x, y, lw=th.lw, ms=th.ms, markevery=max(1, len(x) // 7), **style)
    line.set_gid(_ref_gid("curve", budget, row))


def _curve_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    rows: Sequence[Row],
    score: pd.DataFrame,
    runs: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> None:
    grid = step_grid(_budget_steps(budget))
    for row in rows:
        units = runs[
            (runs["budget"] == budget)
            & (runs["method"] == row.method)
            & (runs["config_ref"] == row.ref)
        ]
        curves = _unit_curves(score, units)
        if not curves:
            logger.warning("%s %s: no score curve, left out of F1", budget, row.label)
            continue
        band = curve_band(curves, grid, "mean", n_boot, seed)
        _draw_curve(ax, th, band, row, budget)
    _style(ax, th, grid="both")
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_text))
    ax.set_xlim(0, _budget_steps(budget))
    _labels(ax, th, "env steps", "score (% of max)")


def plot_curves_score(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    ablations: bool = False,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F1 ``curves_score``: mean train score with a 95% bootstrap band over seeds.

    One panel per (budget, block); the block's controls dashed in grey, its tuned
    methods solid; the band is `evaluation.stats.curve_band` over the seeds' curves.
    ``ablations=True`` adds each ablation as a dotted curve of its own.
    """
    th, name = _theme(theme), NAMES["F1"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the learning curves")
    members = _budget_members(runs)
    score = frames["curves"]
    score = score[score["metric"] == SCORE[0]]
    panels: list[tuple[str, str, list[Row]]] = []
    for budget, mem in members.items():
        refs = _refs_of(runs, budget, mem) if ablations else {}
        for block in compare.BLOCKS:
            rows = [
                r
                for r in _rows([m for m in mem if m[0] == block], refs)
                if r.kind != "heading"
            ]
            if rows:
                panels.append((budget, block, rows))
    fig = _figure(th, len(panels) * 46 + 24)
    axes = _stack(fig, th, [1.0] * len(panels))
    for ax, (budget, block, rows) in zip(axes, panels, strict=True):
        _curve_panel(ax, th, budget, rows, score, runs, n_boot, seed)
        _title(ax, th, _panel_title(budget, f"{BLOCK_TITLES[block]} block"))
    handles = _unique_handles(axes)
    _finish(fig, th, handles, _pending(list(members)))
    return _save(fig, th, out_dir, name)


def _refs_of(
    runs: pd.DataFrame, budget: str, members: Sequence[compare.Member]
) -> dict[tuple[str, str], list[str]]:
    """Block and method -> the ablations (final-phase non-tuned units) of ``budget``."""
    final = _final_runs(runs)
    units = final[(final["budget"] == budget) & (final["config_ref"] != compare.TUNED)]
    block = {m: b for b, m, _ in members}
    out: dict[tuple[str, str], list[str]] = {}
    for method, ref in sorted(
        set(zip(units["method"], units["config_ref"], strict=True))
    ):
        if method in block:
            out.setdefault((block[method], method), []).append(ref)
    return out


def _unique_handles(axes: Sequence[Axes]) -> list[Line2D]:
    """One legend handle per label over the panels, in first-seen order."""
    seen: dict[str, Line2D] = {}
    for ax in axes:
        for line in ax.get_lines():
            label = str(line.get_label())
            if not label.startswith("_"):
                seen.setdefault(label, line)
    return list(seen.values())


# --------------------------------------------------------------------------- F2


def _marks(budget: str) -> tuple[tuple[str, str, str, float, str], ...]:
    """(stat, CI low, CI high, row offset, marker): IQM and mean at 1M, else median."""
    if budget in compare.SIGNIFICANCE_BUDGETS:
        return (
            ("iqm", "iqm_lo", "iqm_hi", -0.17, "o"),
            ("mean", "mean_lo", "mean_hi", 0.17, "D"),
        )
    return (
        ("median", "median_lo", "median_hi", -0.17, "o"),
        ("mean", "t_lo", "t_hi", 0.17, "D"),
    )


def _interval_row(ax: Axes, th: Theme, budget: str, row: Row, stats: pd.Series) -> None:
    """The CI whisker and point of each statistic of one row (skipped where NaN)."""
    style = _style_of(th, row.method, row.ref)
    for key, lo, hi, dy, marker in _marks(budget):
        x, a, b = stats[key], stats[lo], stats[hi]
        if not np.isfinite(x):
            continue
        if np.isfinite(a) and np.isfinite(b):
            ax.hlines(
                row.y + dy, a, b, colors=style["color"], linestyles=style["ls"],
                linewidth=th.lw, gid=_ref_gid(f"ci-{key}", budget, row),
            )  # fmt: skip
        (point,) = ax.plot(
            [x], [row.y + dy], ls="", marker=marker, ms=th.ms, color=style["color"],
            markerfacecolor=style["markerfacecolor"], markeredgewidth=th.lw,
        )  # fmt: skip
        point.set_gid(_ref_gid(key, budget, row))


def _mark_handles(th: Theme, budgets: Sequence[str], ablations: bool) -> list[Line2D]:
    """Grey proxies for the statistics drawn and for the ablation style."""
    kinds = {"1m": ("IQM", "mean"), "1b": ("median", "mean")}
    names = dict.fromkeys(
        n
        for b in budgets
        for n in kinds["1m" if b in compare.SIGNIFICANCE_BUDGETS else "1b"]
    )
    markers = {"IQM": "o", "median": "o", "mean": "D"}
    handles = [
        Line2D([], [], ls="", marker=markers[n], color=th.ink2, ms=th.ms, label=n)
        for n in names
    ]
    if ablations:
        handles.append(
            Line2D(
                [],
                [],
                ls=":",
                marker="o",
                color=th.ink2,
                markerfacecolor=th.surface,
                ms=th.ms,
                label="ablation (hollow, dotted)",
            )
        )
    return handles


def _lookup(table: pd.DataFrame, key: tuple[str, ...], cols: Sequence[str]) -> Any:
    """The one row of ``table`` whose ``cols`` equal ``key``, None if there is none."""
    mask = np.logical_and.reduce([
        table[c] == k for c, k in zip(cols, key, strict=True)
    ])
    found = table[mask]
    return found.iloc[0] if len(found) else None


def plot_aggregate(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    table: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
    ablations: bool = True,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F2 ``aggregate``: IQM and mean with 95% CIs (1M), median and mean (1B).

    One panel per budget (the 1B panel marked indicative, its mean CI a t interval);
    the main block on top, the skills block under its own heading with the controls
    repeated. Numbers are T1 (`compare.main_table`); with ``ablations`` each ablation
    (`compare.ablation_table`) is a dotted, hollow row under its method, never pooled.
    """
    th, name = _theme(theme), NAMES["F2"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the aggregate scores")
    t1 = (
        table
        if table is not None
        else compare.main_table(frames, n_boot=n_boot, seed=seed)
    )
    if ablations and ablation is None:
        ablation = compare.ablation_table(frames, n_boot=n_boot, seed=seed)
    members = _budget_members(runs)
    layouts = {
        b: _rows(m, _ablation_refs(ablation, b) if ablation is not None else None)
        for b, m in members.items()
    }
    total = sum(len(r) for r in layouts.values())
    fig = _figure(th, _height_mm(th, total, len(layouts)))
    axes = _stack(fig, th, [len(r) for r in layouts.values()])
    for ax, (budget, rows) in zip(axes, layouts.items(), strict=True):
        for row in rows:
            if row.kind == "heading":
                continue
            stats = (
                _lookup(
                    ablation,
                    (budget, row.block, row.method, row.ref),
                    ["benchmark", "block", "method", "config_ref"],
                )
                if row.kind == "ablation" and ablation is not None
                else _lookup(
                    t1,
                    (budget, row.block, row.method),
                    ["benchmark", "block", "method"],
                )
            )
            if stats is not None:
                _interval_row(ax, th, budget, row, stats)
        _style(ax, th)
        _apply_rows(ax, th, rows)
        _labels(ax, th, "final score (% of max)")
        what = "IQM, mean" if budget in compare.SIGNIFICANCE_BUDGETS else "median, mean"
        _title(ax, th, _panel_title(budget, f"{what}, 95% CI"), narrow=True)
    handles = _mark_handles(
        th, list(members), ablation is not None and len(ablation) > 0
    )
    _finish(fig, th, handles, _pending(list(members)))
    return _save(fig, th, out_dir, name)


# --------------------------------------------------------------------------- F3


def _cmap(th: Theme, *colors: str) -> Colormap:
    cmap = LinearSegmentedColormap.from_list("figure", list(colors))
    return cmap.with_extremes(bad=th.surface)


def _ink_on(rgba: Sequence[float]) -> str:
    """Dark or light text, whichever reads on the cell colour."""
    r, g, b = (
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]
    )
    return DARK_TEXT if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.4 else LIGHT_TEXT


def _poi_cells(
    rows: pd.DataFrame, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PoI, CI low and high as square arrays (row beats column), NaN on the diagonal."""
    n = len(names)
    out = [np.full((n, n), np.nan) for _ in range(3)]
    index = {m: i for i, m in enumerate(names)}
    for r in rows.itertuples():
        i, j = index[r.method], index[r.versus]
        out[0][i, j], out[1][i, j], out[2][i, j] = r.poi, r.poi_lo, r.poi_hi
    return out[0], out[1], out[2]


def _poi_panel(
    ax: Axes, th: Theme, rows: pd.DataFrame, names: Sequence[str], block: str
) -> None:
    """One block's matrix: colour by PoI, bold where the CI excludes 0.5."""
    poi, lo, hi = _poi_cells(rows, names)
    cmap = _cmap(th, th.low, th.mid, th.high)
    image = ax.imshow(poi, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    image.set_gid(_gid("poi", "1m", block))
    n = len(names)
    ax.set_xticks(range(n), names, rotation=90)
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if np.isnan(poi[i, j]):
                continue
            excluded = lo[i, j] > 0.5 or hi[i, j] < 0.5
            text = f"{poi[i, j]:.2f}"
            if th.name == "video":  # the report's cells are too narrow for a CI
                text += f"\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=th.font,
                color=_ink_on(cmap(poi[i, j])),
                fontweight="bold" if excluded else "normal",
                gid="poi-cell/bold" if excluded else "poi-cell/plain",
            )  # fmt: skip
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _labels(ax, th, "versus (column)", "method (row)")
    _title(
        ax,
        th,
        f"{_budget_name('1m')}: P(row beats column), {BLOCK_TITLES[block]} block",
        narrow=True,
    )


def plot_poi_matrix(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    table: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F3 ``poi_matrix``: probability of improvement between the methods, per block.

    Cells are `compare.poi_matrix` (1M only: 3 seeds at 1B cannot support it); one
    panel per block with its three controls, no cross-block cell and no stars: a cell
    is bold when its 95% CI excludes 0.5 (the video prints the interval).
    """
    th, name = _theme(theme), NAMES["F3"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the PoI matrix")
    members = _budget_members(runs).get("1m")
    if members is None:
        text = f"{PENDING.format('1M')}: the PoI matrix needs the 10-seed 1M runs"
        return _placeholder(name, th, out_dir, text)
    t = (
        table
        if table is not None
        else compare.poi_matrix(frames, n_boot=n_boot, seed=seed)
    )
    blocks = [b for b in compare.BLOCKS if b in set(t["block"])]
    if not blocks:
        return _no_data(name, th, out_dir, "no pairs in the PoI table")
    names = {b: [m for blk, m, _ in members if blk == b] for b in blocks}
    fig = _figure(th, sum(len(v) for v in names.values()) * 9 + 50 * len(blocks))
    axes = _stack(fig, th, [len(names[b]) for b in blocks])
    for ax, block in zip(axes, blocks, strict=True):
        rows = t[(t["benchmark"] == "1m") & (t["block"] == block)]
        _poi_panel(ax, th, rows, names[block], block)
    note = (
        "Cell: P(row score > column score) over seed pairs, blue above 0.5; bold: 95% "
        "bootstrap CI excludes 0.5. A method is never set against the other block."
    )
    _finish(fig, th, notes=[note])
    return _save(fig, th, out_dir, name)


# --------------------------------------------------------------------------- F4


def _literature_rows(literature: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Published rows of the like-for-like training-episode block at ``budget``."""
    if literature.empty:
        return literature
    keep = (
        (literature["source"] == "literature")
        & (literature["score_kind"] == "training")
        & (literature["benchmark"] == budget)
    )
    return literature[keep]


def _short(name: str) -> str:
    return name if len(name) <= LIT_LABEL else name[: LIT_LABEL - 3] + "..."


def _seed_dots(ax: Axes, th: Theme, budget: str, row: Row, scores: np.ndarray) -> None:
    """Per-seed final scores of a row (jittered) and a bar at their mean."""
    color = _color(th, row.method)
    dots = ax.scatter(
        scores, row.y + _jitter(len(scores)), s=th.ms**2, color=color,
        marker=CONTROL_MARKERS.get(row.method, "o"), zorder=3,
    )  # fmt: skip
    dots.set_gid(_gid("seeds", budget, row.block, row.method))
    (mean,) = ax.plot(
        [scores.mean()] * 2, [row.y - 0.4, row.y + 0.4], color=th.ink, lw=th.lw,
    )  # fmt: skip
    mean.set_gid(_gid("seed-mean", budget, row.block, row.method))


def _literature_marks(
    ax: Axes, th: Theme, budget: str, lit: pd.DataFrame, first_row: int
) -> list[Row]:
    """A diamond per published score (filled: our protocol), whiskers for sd or s.e."""
    rows = [Row(first_row, "published", "heading", "lit", "", "")]
    for k, r in enumerate(lit.itertuples()):
        y = first_row + 1 + k
        rows.append(Row(y, _short(str(r.name)), "lit", "lit", "", ""))
        same = r.protocol_match == "same"
        if r.spread_kind != "none" and np.isfinite(r.spread):
            ax.hlines(
                y,
                r.score_pct - r.spread,
                r.score_pct + r.spread,
                colors=th.ink2,
                lw=th.lw,
            )
        mark = ax.scatter(
            [r.score_pct], [y], marker="D", s=th.ms**2,
            facecolors=th.ink2 if same else "none", edgecolors=th.ink2, zorder=3,
        )  # fmt: skip
        mark.set_gid(_gid("lit", budget, str(k)))
    return rows


def plot_seed_spread(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    literature: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F4 ``seed_spread``: every seed's final score, with published values as marks.

    Per budget panel: a dot per seed of each tuned method (controls repeated under the
    skills block), a bar at the mean, and below them the published rows of the
    like-for-like block (T2: A at 1B, D at 1M) as diamonds, filled when their protocol
    is ours, hollow when it differs or is unstated, whiskers for a stated sd or s.e.
    """
    th, name = _theme(theme), NAMES["F4"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the seed spread")
    members = _budget_members(runs)
    if literature is None:
        literature = compare.literature_table(frames, n_boot=n_boot, seed=seed)
    tuned = compare.tuned(runs)
    panels = {
        b: (_rows(m), _literature_rows(literature, b)) for b, m in members.items()
    }
    total = sum(len(r) + len(lit) + (len(lit) > 0) for r, lit in panels.values())
    fig = _figure(th, _height_mm(th, total, len(panels)))
    axes = _stack(
        fig, th, [len(r) + len(lit) + (len(lit) > 0) for r, lit in panels.values()]
    )
    for ax, (budget, (rows, lit)) in zip(axes, panels.items(), strict=True):
        mine = tuned[tuned["budget"] == budget]
        for row in rows:
            if row.kind != "heading":
                scores = mine.loc[mine["method"] == row.method, "final_score"]
                _seed_dots(ax, th, budget, row, scores.to_numpy(dtype=float))
        if len(lit):
            rows = [*rows, *_literature_marks(ax, th, budget, lit, len(rows))]
        _style(ax, th)
        _apply_rows(ax, th, rows)
        _labels(ax, th, "final score (% of max)")
        _title(ax, th, _panel_title(budget, "final score per seed"), narrow=True)
    note = (
        "Dot: one seed; bar: mean. Diamond: published score (filled: same protocol, "
        "hollow: different or unstated); whisker: published sd or s.e."
    )
    _finish(fig, th, notes=[note, *_pending(list(members))])
    return _save(fig, th, out_dir, name)


# --------------------------------------------------------------------------- F5


def _tier_spans(status: pd.DataFrame) -> list[tuple[str, int, int]]:
    """(tier, first row, last row) of consecutive achievements of one tier."""
    spans: list[tuple[str, int, int]] = []
    for i, tier in enumerate(status["tier"]):
        if spans and spans[-1][0] == tier:
            spans[-1] = (tier, spans[-1][1], i)
        else:
            spans.append((str(tier), i, i))
    return spans


def _floor_labels(ax: Axes, th: Theme, floors: Sequence[int]) -> None:
    """Right-hand ``F<n>`` labels at the middle of each run of one home floor."""
    runs: list[tuple[int, int, int]] = []
    for i, floor in enumerate(floors):
        if runs and runs[-1][0] == floor:
            runs[-1] = (floor, runs[-1][1], i)
        else:
            runs.append((floor, i, i))
    gap = 2.0
    ticks, labels, last = [], [], -gap
    for floor, a, b in runs:
        mid = (a + b) / 2
        if mid - last >= gap:
            ticks.append(mid)
            labels.append(f"F{floor}")
            last = mid
    side = ax.secondary_yaxis("right")
    side.set_yticks(ticks, labels)
    side.tick_params(colors=th.muted, labelsize=th.font, length=0)
    side.spines["right"].set_visible(False)


def _heat_panel(
    ax: Axes, th: Theme, budget: str, block: str, names: Sequence[str],
    status: pd.DataFrame, *, first: bool,
) -> None:  # fmt: skip
    """One block's heat map: achievements down, methods across, final success rate."""
    per_method = {m: g.reset_index(drop=True) for m, g in status.groupby("method")}
    ref = next(iter(per_method.values()))
    n = len(ref)
    cells = np.column_stack([
        per_method[m]["final_rate"].to_numpy(dtype=float)
        if m in per_method
        else np.full(n, np.nan)
        for m in names
    ])
    image = ax.imshow(
        cells, cmap=_cmap(th, th.mid, th.high), vmin=0.0, vmax=1.0, aspect="auto",
        interpolation="nearest",
    )  # fmt: skip
    image.set_gid(_gid("heat", budget, block))
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    spans = _tier_spans(ref)
    for _, _, last in spans[:-1]:
        ax.axhline(last + 0.5, color=th.ink2, lw=0.6)
    if first:
        ax.set_yticks([(a + b) / 2 for _, a, b in spans], [t for t, _, _ in spans])
        for tick in ax.get_yticklabels():
            tick.set_gid("heat/tier")
    else:
        ax.set_yticks([])
    _floor_labels(ax, th, [int(f) for f in ref["floor"]])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _labels(ax, th, "tuned method", "achievement tier" if first else None)
    _title(ax, th, _panel_title(budget, block), width=14 if th.name == "report" else 40)


def plot_success_heatmap(
    frames: Frames, out_dir: Path, theme: str = "report", *, mode: str = "stochastic"
) -> Figure:
    """F5 ``success_heatmap``: share of final eval episodes with each achievement.

    Methods (tuned, the block's three controls first) across, the achievements down in
    tier then floor order (`hard.achievement_status`: ``final_rate``, the mean over
    seeds of each seed's share of ``mode`` episodes), tier names on the left, home
    floors on the right. One block per column, one row per budget.
    """
    th, name = _theme(theme), NAMES["F5"]
    runs, eps = frames["runs"], frames["eval_episodes"]
    if runs.empty or not any(c.startswith("ach/") for c in eps.columns):
        return _no_data(
            name, th, out_dir, "no achievement columns in the eval episodes"
        )
    members = _budget_members(runs)
    tuned = compare.tuned(runs)
    status = {
        b: achievement_status(
            eps[eps["run_id"].isin(tuned.loc[tuned["budget"] == b, "run_id"])],
            mode=mode,
        )
        for b in members
    }
    rows = max((v["achievement"].nunique() for v in status.values()), default=0)
    fig = _figure(th, (1.6 * rows + 60) * len(members))
    grid = fig.subplots(len(members), len(compare.BLOCKS), squeeze=False)
    for ax_row, (budget, mem) in zip(grid, members.items(), strict=True):
        for col, (ax, block) in enumerate(zip(ax_row, compare.BLOCKS, strict=True)):
            names = [m for b, m, _ in mem if b == block]
            if not names or status[budget].empty:
                ax.set_visible(False)
                continue
            _heat_panel(ax, th, budget, block, names, status[budget], first=col == 0)
    note = (
        "Cell: mean over seeds of the share of final stochastic eval episodes with the "
        "achievement (light: none, dark blue: all). Right: home floor."
    )
    _finish(fig, th, notes=[note, *_pending(list(members))])
    return _save(fig, th, out_dir, name)


# -------------------------------------------------------------------------- F18

SENSITIVE_ALPHA = 0.3
"""Opacity of a bar whose ``S_h`` is below the noise floor."""
NAME_CHARS = 18
"""Characters of a hyperparameter name shown on a bar's row."""


def _sens_rows(sub: pd.DataFrame) -> list[Row]:
    """One row per method of one tuning phase, controls first (T7 order)."""
    return _rows(
        list(zip(sub["block"], sub["method"], sub["role"], strict=True)), headings=False
    )


def _bar_rank(
    ax: Axes, th: Theme, phase: str, row: Row, t7: pd.Series, rank: int
) -> None:
    """The ``rank``-th most sensitive hyperparameter's ``S_h`` bar (faint if noise)."""
    name, s = t7[f"top{rank}"], t7[f"top{rank}_s"]
    if not name or not np.isfinite(s):
        return
    color = _color(th, row.method)
    weak = bool(t7[f"top{rank}_insensitive"])
    (bar,) = ax.barh(
        2 * row.y + rank - 1, s, height=0.7, label=name,
        facecolor=to_rgba(color, SENSITIVE_ALPHA if weak else 1.0), edgecolor=color,
        linewidth=0.6,
    )  # fmt: skip
    bar.set_gid(_gid("sens", phase, row.method, str(rank)))
    bar.set_label(name)  # the container's label does not reach the rectangle


def _sens_axes(
    ax_rc: Axes, ax_s: Axes, th: Theme, phase: str, rows: Sequence[Row],
    t7: pd.DataFrame, runs: pd.DataFrame,
) -> None:  # fmt: skip
    """Config-score strip (left) and sensitivity bars with the noise floor (right)."""
    names: list[str] = []
    for row in rows:
        mine = runs[(runs["method"] == row.method) & in_phase(runs, phase)]
        r_c = config_scores(mine)["r_c"].to_numpy()
        base = 2 * row.y
        dots = ax_rc.scatter(
            r_c, base + 0.5 + _jitter(len(r_c), 0.25), s=th.ms**2,
            color=_color(th, row.method), zorder=3,
        )  # fmt: skip
        dots.set_gid(_gid("rc", phase, row.method))
        stats = _lookup(t7, (phase, row.method), ["phase", "method"])
        if stats is None:
            continue
        for rank in (1, 2):
            _bar_rank(ax_s, th, phase, row, stats, rank)
            names += [str(stats[f"top{rank}"])[:NAME_CHARS] or "(none varies)"]
        (floor,) = ax_s.plot(
            [stats["noise_floor"]] * 2, [base - 0.4, base + 1.4], color=th.ink,
            lw=th.lw, ls=":",
        )  # fmt: skip
        floor.set_gid(_gid("noise", phase, row.method))
    n = 2 * len(rows)
    for ax in (ax_rc, ax_s):
        _style(ax, th)
        ax.set_ylim(n - 0.5, -0.7)
        ax.tick_params(axis="y", length=0)
    ax_rc.set_yticks([2 * r.y + 0.5 for r in rows], [r.method for r in rows])
    ax_s.set_yticks(range(n), names[:n] if len(names) >= n else [""] * n)
    ax_rc.axvline(ROBUST, color=th.muted, lw=0.5)
    ax_rc.set_xlim(0, 1.05)
    ax_s.set_xlim(0, 1.05)
    _labels(ax_rc, th, "config score r_c")
    _labels(ax_s, th, "sensitivity S_h")


def plot_sensitivity(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    table: pd.DataFrame | None = None,
) -> Figure:
    """F18 ``sensitivity``: config scores and hyperparameter sensitivity at 1M.

    ``frames`` hold the tuning phases (as for `compare.sensitivity_table`). Per phase
    (coarse, fine): left, one dot per config, its score ``r_c`` over the best config
    of the method (`stability.config_scores`, the vertical line is the robust 0.9);
    right, the two most sensitive hyperparameters' ``S_h`` (T7), faint when below the
    method's noise floor (dotted line).
    """
    th, name = _theme(theme), NAMES["F18"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no tuning runs")
    t7 = table if table is not None else compare.sensitivity_table(frames)
    t7 = t7[t7["benchmark"] == "1m"]
    phases = sorted(TUNING_PHASES)
    layouts = {
        p: _sens_rows(t7[t7["phase"] == p].drop_duplicates("method")) for p in phases
    }
    fig = _figure(
        th, sum(2 * max(len(r), 1) for r in layouts.values()) * th.pitch_mm + 70
    )
    grid = fig.add_gridspec(
        len(phases), 2, height_ratios=[2 * max(len(r), 1) for r in layouts.values()],
        width_ratios=[1.0, 1.3],
    )  # fmt: skip
    for i, (phase, rows) in enumerate(layouts.items()):
        if not rows:
            ax = fig.add_subplot(grid[i, :])
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, f"{phase}: no runs", ha="center", va="center",
                fontsize=th.title, color=th.muted, transform=ax.transAxes,
            )  # fmt: skip
            continue
        ax_rc, ax_s = fig.add_subplot(grid[i, 0]), fig.add_subplot(grid[i, 1])
        _sens_axes(ax_rc, ax_s, th, phase, rows, t7, runs)
        _title(ax_rc, th, f"{_budget_name('1m')}: {phase}")
    handles = [
        Line2D([], [], color=th.ink, ls=":", lw=th.lw, label="noise floor"),
        Patch(
            facecolor=to_rgba(th.ink2, SENSITIVE_ALPHA), edgecolor=th.ink2,
            label="S_h below noise floor",
        ),
    ]  # fmt: skip
    note = "Dots: one config of the grid. Bars: the two most sensitive hyperparameters."
    _finish(fig, th, handles, [note], cols=2)
    return _save(fig, th, out_dir, name)


# -------------------------------------------------------------------------- F19


def _stability_row(
    ax_dd: Axes, ax_bar: Axes, th: Theme, budget: str, row: Row,
    per_run: pd.DataFrame, t3: pd.DataFrame,
) -> None:  # fmt: skip
    """Per-run max drawdowns (hollow: a drop, cross: collapsed) and T3's counts."""
    color = _color(th, row.method)
    mine = per_run[
        (per_run["benchmark"] == budget) & (per_run["method"] == row.method)
    ].dropna(subset=["max_drawdown"])
    x = mine["max_drawdown"].to_numpy(dtype=float)
    dots = ax_dd.scatter(
        x, row.y + _jitter(len(x)), s=th.ms**2, edgecolors=color, linewidths=th.lw,
        facecolors=[th.surface if d else color for d in mine["drops"]], zorder=3,
    )  # fmt: skip
    dots.set_gid(_gid("dd", budget, row.block, row.method))
    gone = mine[mine["collapsed"]]
    if len(gone):
        cross = ax_dd.scatter(
            gone["max_drawdown"], [row.y] * len(gone), marker="x", s=th.ms**2 * 3,
            color=th.ink, zorder=4,
        )  # fmt: skip
        cross.set_gid(_gid("collapsed", budget, row.block, row.method))
    stats = _lookup(
        t3, (budget, row.block, row.method), ["benchmark", "block", "method"]
    )
    if stats is None:
        return
    if np.isfinite(stats["max_drawdown_median"]):
        (tick,) = ax_dd.plot(
            [stats["max_drawdown_median"]], [row.y], marker="|", ls="", color=th.ink,
            ms=th.ms * 3, markeredgewidth=th.lw * 1.5,
        )  # fmt: skip
        tick.set_gid(_gid("dd-median", budget, row.block, row.method))
    for key, dy, alpha in (("drops", -0.17, 0.45), ("collapse", 0.17, 1.0)):
        share = stats[f"runs_with_{key}"] / stats["seeds"]
        (bar,) = ax_bar.barh(row.y + dy, share, height=0.3, color=to_rgba(color, alpha))
        bar.set_gid(_gid(key, budget, row.block, row.method))


def plot_stability(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    per_run: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
) -> Figure:
    """F19 ``stability``: max drawdown per run, drops and collapses per method.

    Per budget: left, one dot per tuned run at its deepest drawdown of the smoothed
    score curve (`compare.run_stability`; hollow: the run had a drop, a cross: it
    collapsed; the bar marks the median, T3); right, the share of the method's runs
    with a drop (light) and with a collapse (T3).
    """
    th, name = _theme(theme), NAMES["F19"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the stability figure")
    per_run = per_run if per_run is not None else compare.run_stability(frames)
    t3 = table if table is not None else compare.stability_table(frames)
    members = _budget_members(runs)
    layouts = {b: _rows(m) for b, m in members.items()}
    fig = _figure(th, _height_mm(th, sum(map(len, layouts.values())), len(layouts)))
    grid = fig.subplots(
        len(layouts), 2, squeeze=False, sharey="row", width_ratios=[1.6, 1.0],
        height_ratios=[len(r) for r in layouts.values()],
    )  # fmt: skip
    for (ax_dd, ax_bar), (budget, rows) in zip(grid, layouts.items(), strict=True):
        for row in rows:
            if row.kind != "heading":
                _stability_row(ax_dd, ax_bar, th, budget, row, per_run, t3)
        for ax in (ax_dd, ax_bar):
            _style(ax, th)
        _apply_rows(ax_dd, th, rows)
        ax_dd.set_xlim(0, 1.02)
        ax_bar.set_xlim(0, 1.0)
        _labels(ax_dd, th, "max drawdown")
        _labels(ax_bar, th, "share of runs")
        _title(
            ax_dd, th, _panel_title(budget, "stability of the score curve"), narrow=True
        )
    handles = [
        Line2D(
            [], [], ls="", marker="o", color=th.ink2, markerfacecolor=th.surface,
            ms=th.ms, label="run with a drop",
        ),
        Line2D([], [], ls="", marker="x", color=th.ink, ms=th.ms, label="collapsed"),
        Line2D([], [], ls="", marker="|", color=th.ink, ms=th.ms * 2, label="median"),
        Patch(facecolor=to_rgba(th.ink2, 0.45), label="share with a drop"),
        Patch(facecolor=th.ink2, label="share collapsed"),
    ]  # fmt: skip
    note = (
        "Drawdown: fall from the running max of the smoothed score; no dot below "
        "the run's drawdown floor (R92)."
    )
    _finish(fig, th, handles, [note, *_pending(list(members))], cols=2)
    return _save(fig, th, out_dir, name)


# -------------------------------------------------------------------------- F20


def _hardware_panel(
    ax: Axes, th: Theme, profile: str, rows: Sequence[Row], t4: pd.DataFrame,
    series: Sequence[tuple[str, str, float]],
) -> bool:  # fmt: skip
    """The medians of one metric's series per method row; False if none is measured."""
    measured = False
    for k, (col, _, factor) in enumerate(series):
        for row in rows:
            hit = _lookup(t4, (row.block, row.method), ["block", "method"])
            if hit is None or not np.isfinite(hit[col]):
                continue
            measured = True
            mark = ax.scatter(
                [hit[col] * factor], [row.y], marker=SERIES_MARKERS[k], s=th.ms**2,
                color=_color(th, row.method), zorder=3,
            )  # fmt: skip
            mark.set_gid(_gid("hw", profile, col, row.method))
    if not measured:
        ax.text(
            0.5, 0.5, "n.m.", ha="center", va="center", fontsize=th.font,
            color=th.muted, transform=ax.transAxes,
        )  # fmt: skip
    return measured


def _hardware_profile(
    sub: SubFigure, th: Theme, profile: str, t4: pd.DataFrame
) -> bool:
    """The six metric panels of one profile; True if any panel had nothing measured."""
    rows = _rows(
        list(zip(t4["block"], t4["method"], t4["role"], strict=True)), headings=False
    )
    axes = sub.subplots(3, 2, squeeze=False, sharey=True)
    missing = False
    for ax, (label, series) in zip(axes.flat, HARDWARE_PANELS, strict=True):
        measured = _hardware_panel(ax, th, profile, rows, t4, series)
        missing |= not measured
        _style(ax, th)
        _labels(ax, th, label)
    _apply_rows(axes[0, 0], th, rows)
    _title(axes[0, 0], th, f"{profile}")
    return missing


def plot_hardware(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    budget: str = "1m",
    table: pd.DataFrame | None = None,
) -> Figure:
    """F20 ``hardware``: SPS, wall-clock, compile time, memory, utilisation, efficiency.

    Medians over each method's tuned runs at ``budget`` (T4, `compare.hardware_table`),
    one block of six panels per profile (profiles are never compared). Not measured is
    ``n.m.``. Laptop timings carry the R97 note (two concurrent workers, context only).
    """
    th, name = _theme(theme), NAMES["F20"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the hardware figure")
    t4 = table if table is not None else compare.hardware_table(frames)
    t4 = t4[t4["benchmark"] == budget]
    if t4.empty:
        return _placeholder(name, th, out_dir, PENDING.format(budget.upper()))
    profiles = dict(iter(t4.groupby("profile")))
    rows_of = {
        p: len(
            _rows(
                list(zip(g["block"], g["method"], g["role"], strict=True)),
                headings=False,
            )
        )
        for p, g in profiles.items()
    }
    fig = _figure(th, sum(3 * n * th.pitch_mm + 3 * 14 for n in rows_of.values()) + 30)
    heights = [
        3 * n * th.pitch_mm + 3 * 14 + (n if th.pitch_mm == 0 else 0)
        for n in rows_of.values()
    ]
    grid = fig.subfigures(len(profiles), 1, height_ratios=heights)
    subs: list[SubFigure] = [grid] if isinstance(grid, SubFigure) else list(grid)
    missing = False
    for sub, (profile, g) in zip(subs, profiles.items(), strict=True):
        missing |= _hardware_profile(sub, th, str(profile), g)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=SERIES_MARKERS[k],
            color=th.ink2,
            ms=th.ms,
            label=label,
        )
        for _, series in HARDWARE_PANELS
        if len(series) > 1
        for k, (_, label, _) in enumerate(series)
    ]
    notes = [f"Median over the tuned runs at {_budget_name(budget)}."]
    notes += [
        f"{p} timings: {n}."
        for p, g in profiles.items()
        for n in sorted(set(g["note"]) - {""})
    ]
    if missing:
        notes.append("n.m.: not measured on this profile.")
    notes += _pending(sorted(set(runs["budget"])))
    _finish(fig, th, handles, notes, cols=3)
    return _save(fig, th, out_dir, name)


FIGURES: dict[str, Callable[..., Figure]] = {
    "F1": plot_curves_score,
    "F2": plot_aggregate,
    "F3": plot_poi_matrix,
    "F4": plot_seed_spread,
    "F5": plot_success_heatmap,
    "F18": plot_sensitivity,
    "F19": plot_stability,
    "F20": plot_hardware,
}
"""Figure id -> its function, in the plan's order (the results CLI loops over it)."""
