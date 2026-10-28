"""Core comparison figures F1-F5 and F18-F20: curves, intervals, PoI, spread, hardware.

Each ``plot_*`` draws one figure per theme, writes ``<name>.png`` into ``out_dir``
(``_video`` suffix for the video) and returns the `Figure`. ``report`` is light, 80 mm
wide (one Interspeech column) at 300 dpi, fonts of 7 pt or more; ``video`` is dark,
1920 x 1080. Look: `charts.Theme`; colours `charts.METHOD_COLORS`; controls dashed,
ablations dotted, hollow; legends outside the axes; nothing cropped. Numbers: `compare`
and `evaluation.stats`; a budget without runs is "pending" (R86); artists carry a gid.

@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures.py]]
"""

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
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

BUDGETS = ("1m", "1b")
"""The benchmarks of the project; a missing one is pending, never an empty panel."""
PENDING = "Craftax-{} pending"
PENDING_1B = PENDING.format("1B")
BLOCK_TITLES = {compare.MAIN: "main", compare.SKILLS: "uses hand-written skills"}
BLOCK_HEADINGS = {compare.MAIN: "main", compare.SKILLS: "skills"}
"""Row headings of the two blocks (the caption says what the skills block is)."""
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
    "LEG": "legend_methods",
}
"""Figure id -> file stem (``.png``; ``_double`` and ``_video`` mark the variants)."""
STEPS = {"m": 1e6, "b": 1e9}
LIT_LABEL = 26
"""Characters of a literature name shown beside its mark."""
HARDWARE_PANELS: tuple[tuple[str, str, tuple[tuple[str, str, float], ...]], ...] = (
    (
        "sps",
        "steps per second",
        (("steady_sps", "steady SPS", 1.0), ("overall_sps", "overall SPS", 1.0)),
    ),
    ("wall_clock", "wall-clock (h)", (("wall_clock_s", "wall-clock", 1 / 3600),)),
    ("compile", "compile time (s)", (("compile_s", "compile time", 1.0),)),
    (
        "memory",
        "peak memory (GB)",
        (
            ("peak_host_mem_mb", "host", 1 / 1024),
            ("peak_rss_mb", "RSS", 1 / 1024),
            ("peak_device_mem_mb", "device", 1 / 1024),
        ),
    ),
    (
        "utilisation",
        "utilisation (%)",
        (("cpu_util_mean", "CPU", 1.0), ("gpu_util_mean", "GPU", 1.0)),
    ),
    ("efficiency", "steps per joule", (("steps_per_joule", "steps per joule", 1.0),)),
)
"""F20 files: stem part, x label, then (T4 column, legend name, factor to the unit)."""
DARK_TEXT, LIGHT_TEXT = charts.INK_LIGHT, charts.SURFACE_LIGHT
"""Text on a light and on a dark heat-map cell."""
ABLATION_KEY = ["benchmark", "block", "method", "config_ref"]
"""Key columns of T1 (the first three) and of the ablation table."""
SERIES_MARKERS = ("o", "D", "^")
ROWS_BASE = charts.ROWS_BASE_MM
"""Marker of the first, second and third series of a hardware panel."""


Theme = charts.Theme
_theme = charts.theme
_report = charts.report_theme
VIDEO = charts.VIDEO
COLUMN_MM, MM, WIDTHS = charts.COLUMN_MM, charts.MM, charts.WIDTHS


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


# The figure plumbing lives in `charts` (one theme for every producer); these names
# are what `figures_game` and the other figure modules import.
_figure = charts.themed_figure
_style = charts.style_axes
_labels = charts.set_labels
_title = charts.set_title


def _budget_name(budget: str) -> str:
    return f"Craftax-{budget.upper()}"


def _panel_title(budget: str, what: str) -> str:
    """``Craftax-1M: what``; a 1B title says indicative (3 seeds, R86)."""
    flag = " (indicative)" if budget not in compare.SIGNIFICANCE_BUDGETS else ""
    return f"{_budget_name(budget)}{flag}: {what}"


def _pending(present: Sequence[str]) -> list[str]:
    """One note per project benchmark without runs."""
    return [PENDING.format(b.upper()) for b in BUDGETS if b not in present]


_save = charts.save_png
_finish = charts.finish
_placeholder = charts.placeholder


def _no_data(name: str, th: Theme, out_dir: Path, what: str) -> Figure:
    return _placeholder(name, th, out_dir, f"no data: {what}")


Figs = dict[str, Figure]
"""What a figure function returns: file stem -> figure (one panel per file)."""
Shape = charts.Shape
DOUBLE_STEMS: tuple[str, ...] = ()
"""``fnmatch`` patterns of the stems that also get a 170 mm variant: a one-panel figure
fits the column, so only a figure that needs the width lists itself."""
DOUBLE_ONLY: tuple[str, ...] = ("order_*",)
"""Stems drawn at 170 mm and as video only (their node names do not fit a column);
every other report figure is one column wide. The video is always drawn."""
COLUMN_STEMS: tuple[str, ...] = ("order_agreement",)
"""Stems that match `DOUBLE_ONLY` but fit the column."""
PROFILE_NAMES = {
    "laptop": "Apple Silicon laptop, 48 GB (CPU actors, Metal updates)",
    "cuda": "CUDA GPU machine",
}
"""Hardware profile -> the machine it names, for the captions."""


def wants(th: Theme, stem: str) -> bool:
    """Whether ``stem`` is drawn in this variant.

    The video always; a column figure unless it is `DOUBLE_ONLY`; a ``_double`` file
    only for a `DOUBLE_ONLY` or `DOUBLE_STEMS` figure (others fit the column).
    """
    if th.name == "video":
        return True
    only = any(fnmatch(stem, p) for p in DOUBLE_ONLY) and stem not in COLUMN_STEMS
    if th.suffix == "_double":
        return only or any(fnmatch(stem, p) for p in DOUBLE_STEMS)
    return not only


def _single_budget(runs: pd.DataFrame) -> str:
    """The one benchmark of ``runs``: a figure set shows one benchmark.

    Raises:
        ValueError: If ``runs`` hold several or no benchmark (`results.draw` cuts them).
    """
    budgets = sorted(set(runs["budget"]))
    if len(budgets) != 1:
        msg = f"draw one benchmark at a time (results.draw cuts), got {budgets}"
        raise ValueError(msg)
    return str(budgets[0])


def _panel(
    th: Theme,
    shape: charts.Shape,
    rows: int | None = None,
    base_mm: float | None = None,
) -> tuple[Figure, Axes]:
    """A figure of the size class ``shape`` with its one axes."""
    fig = _figure(th, shape=shape, rows=rows, base_mm=base_mm)
    return fig, fig.add_subplot()


def _category_panel(th: Theme, rows: int, extra_mm: float = 0.0) -> tuple[Figure, Axes]:
    """A figure for ``rows`` category rows (DOTS if they fit, else ROWS), its axes."""
    fig = charts.category_figure(th, rows, extra_mm)
    return fig, fig.add_subplot()


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
            add(BLOCK_HEADINGS[blk], "heading", blk, "", "")
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


def _height_mm(
    th: Theme, counts: Sequence[int], *, stacked: bool | None = None
) -> float:
    """Report height: row pitch per row plus a title and x axis per stacked panel."""
    stacked = not th.wide if stacked is None else stacked
    rows, panels = (sum(counts), len(counts)) if stacked else (max(counts), 1)
    return rows * th.pitch_mm + panels * 16.0 + 14.0


def _stack(fig: Figure, th: Theme, ratios: Sequence[float]) -> list[Axes]:
    """One axes per panel: stacked in the report column, side by side when wide."""
    n = len(ratios)
    if not th.wide:
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
    width: str = "column",
    ablations: bool = False,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F1 ``curves_score_<block>``: mean train score with a 95% bootstrap band.

    One file per block (``main``, ``skills``), the same size and y range so they stack:
    the block's controls dashed in grey, its tuned methods solid; the band is
    `evaluation.stats.curve_band` over the seeds' curves. ``ablations=True`` adds each
    ablation as a dotted curve of its own.
    """
    th, name = _theme(theme, width), NAMES["F1"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the learning curves")}
    budget = _single_budget(runs)
    mem = _budget_members(runs)[budget]
    score = frames["curves"]
    score = score[score["metric"] == SCORE[0]]
    refs = _refs_of(runs, budget, mem) if ablations else {}
    panels: dict[str, tuple[Figure, Axes, str]] = {}
    for block in compare.BLOCKS:
        stem = f"{name}_{block}"
        rows = _rows([m for m in mem if m[0] == block], refs, headings=False)
        if rows and wants(th, stem):
            fig, ax = _panel(th, Shape.CURVE)
            _curve_panel(ax, th, budget, rows, score, runs, n_boot, seed)
            panels[stem] = (fig, ax, block)
    low = min((ax.get_ylim()[0] for _, ax, _ in panels.values()), default=0.0)
    high = max((ax.get_ylim()[1] for _, ax, _ in panels.values()), default=1.0)
    for stem, (fig, ax, block) in panels.items():
        ax.set_ylim(low, high)
        what = (
            "controls (dashed) and tuned methods"
            if block == compare.MAIN
            else ("controls (dashed) and the methods that use hand-written skills")
        )
        _save(
            fig, th, out_dir, stem,
            title=f"Training score, {BLOCK_TITLES[block]} block",
            description=(
                f"Mean training score (% of the maximum) over environment steps: the "
                f"{what}. Band: 95% bootstrap CI over seeds; both blocks share axes."
            ),
        )  # fmt: skip
    return {stem: fig for stem, (fig, _, _) in panels.items()}


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


def _lookup(table: pd.DataFrame, key: tuple[str, ...], cols: Sequence[str]) -> Any:
    """The one row of ``table`` whose ``cols`` equal ``key``, None if there is none."""
    mask = np.logical_and.reduce([
        table[c] == k for c, k in zip(cols, key, strict=True)
    ])
    found = table[mask]
    return found.iloc[0] if len(found) else None


def _interval_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    rows: Sequence[Row],
    t1: pd.DataFrame,
    ablation: pd.DataFrame | None,
) -> None:
    """The interval rows of one budget (T1 for units, the ablation table for those)."""
    for row in rows:
        if row.kind == "heading":
            continue
        if row.kind == "ablation" and ablation is not None:
            key, cols = (budget, row.block, row.method, row.ref), ABLATION_KEY
            stats = _lookup(ablation, key, cols)
        else:
            stats = _lookup(t1, (budget, row.block, row.method), ABLATION_KEY[:3])
        if stats is not None:
            _interval_row(ax, th, budget, row, stats)
    _style(ax, th)
    _apply_rows(ax, th, rows)
    _labels(ax, th, "final score (% of max)")


def plot_aggregate(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    table: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
    ablations: bool = True,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F2 ``aggregate``: IQM and mean with 95% CIs (1M), median and mean (1B).

    The main block on top, the skills block under its own heading with the controls
    repeated. Numbers are T1 (`compare.main_table`); with ``ablations`` each ablation
    (`compare.ablation_table`) is a dotted, hollow row under its method, never pooled.
    """
    th, name = _theme(theme, width), NAMES["F2"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the aggregate scores")}
    if not wants(th, name):
        return {}
    budget = _single_budget(runs)
    t1 = (
        table
        if table is not None
        else compare.main_table(frames, n_boot=n_boot, seed=seed)
    )
    if ablations and ablation is None:
        ablation = compare.ablation_table(frames, n_boot=n_boot, seed=seed)
    rows = _rows(
        _budget_members(runs)[budget],
        _ablation_refs(ablation, budget) if ablation is not None else None,
    )
    fig, ax = _category_panel(th, len(rows))
    _interval_panel(ax, th, budget, rows, t1, ablation)
    stats = "IQM (circle) and mean (diamond)"
    if budget not in compare.SIGNIFICANCE_BUDGETS:
        stats = "Median (circle) and mean (diamond, t interval)"
    _save(
        fig, th, out_dir, name, title="Final score per method",
        description=(
            f"Final training score (% of the maximum) of each method: {stats} over "
            "seeds with 95% bootstrap CIs (one whisker each). Main block on top, the "
            "methods that use "
            "hand-written skills below with the controls repeated; each ablation is a "
            "dotted, hollow row under its method and is never pooled with it."
        ),
    )  # fmt: skip
    return {name: fig}


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
    cmap = _cmap(th, *th.div)
    charts.heatmap(ax, th, poi, cmap, vmin=0.0, vmax=1.0, gid=_gid("poi", "1m", block))
    n = len(names)
    ax.set_xticks(np.arange(n) + 0.5, names, rotation=90)
    ax.set_yticks(np.arange(n) + 0.5, names)
    for i in range(n):
        for j in range(n):
            if np.isnan(poi[i, j]):
                continue
            excluded = lo[i, j] > 0.5 or hi[i, j] < 0.5
            text = f"{poi[i, j]:.2f}"
            if th.name == "video":  # the report's cells are too narrow for a CI
                text += f"\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]"
            ax.text(
                j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=th.font,
                color=_ink_on(cmap(poi[i, j])),
                fontweight="bold" if excluded else "normal",
                gid="poi-cell/bold" if excluded else "poi-cell/plain",
            )  # fmt: skip
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_poi_matrix(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    table: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F3 ``poi_matrix_<block>``: probability of improvement between the methods.

    Cells are `compare.poi_matrix` (1M only: 3 seeds at 1B cannot support it); one
    square file per block with its three controls, no cross-block cell and no stars: a
    cell is bold when its 95% CI excludes 0.5 (the video prints the interval).
    """
    th, name = _theme(theme, width), NAMES["F3"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the PoI matrix")}
    members = _budget_members(runs).get("1m")
    if members is None:
        text = "no PoI matrix: it needs the 10-seed 1M runs"
        return {name: _placeholder(name, th, out_dir, text)}
    t = (
        table
        if table is not None
        else compare.poi_matrix(frames, n_boot=n_boot, seed=seed)
    )
    blocks = [b for b in compare.BLOCKS if b in set(t["block"])]
    if not blocks:
        return {name: _no_data(name, th, out_dir, "no pairs in the PoI table")}
    figs: Figs = {}
    for block in blocks:
        stem = f"{name}_{block}"
        if not wants(th, stem):
            continue
        names = [m for blk, m, _ in members if blk == block]
        fig, ax = _panel(th, Shape.SQUARE)
        rows = t[(t["benchmark"] == "1m") & (t["block"] == block)]
        _poi_panel(ax, th, rows, names, block)
        _save(
            fig, th, out_dir, stem,
            title=f"Probability of improvement, {BLOCK_TITLES[block]} block",
            description=(
                "P(row score > column score) over seed pairs, blue above 0.5; bold: "
                "the 95% bootstrap CI excludes 0.5. Rows and columns are the block's "
                "controls and methods; a method is never set against the other block."
            ),
        )  # fmt: skip
        figs[stem] = fig
    return figs


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


def _literature_marks(ax: Axes, th: Theme, budget: str, lit: pd.DataFrame) -> list[Row]:
    """A diamond per published score (filled: our protocol), whiskers for sd or s.e."""
    rows = []
    for k, r in enumerate(lit.itertuples()):
        rows.append(Row(k, _short(str(r.name)), "lit", "lit", "", ""))
        same = r.protocol_match == "same"
        if r.spread_kind != "none" and np.isfinite(r.spread):
            ax.hlines(
                k, r.score_pct - r.spread, r.score_pct + r.spread, colors=th.ink2,
                lw=th.lw,
            )  # fmt: skip
        mark = ax.scatter(
            [r.score_pct], [k], marker="D", s=th.ms**2,
            facecolors=th.ink2 if same else "none", edgecolors=th.ink2, zorder=3,
        )  # fmt: skip
        mark.set_gid(_gid("lit", budget, str(k)))
    return rows


def _score_top(scores: Sequence[float], lit: pd.DataFrame) -> float:
    """The largest value any seed dot or published whisker reaches."""
    spread = lit["spread"].where(lit["spread_kind"] != "none").fillna(0.0)
    top = [*scores, *(lit["score_pct"] + spread).tolist()]
    return float(max(top, default=1.0))


def plot_seed_spread(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    literature: pd.DataFrame | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F4 ``seed_spread_<block>`` and ``seed_spread_published``: final score per seed.

    A dot per seed of each tuned method of the block (the controls repeated), a bar at
    the mean; ``seed_spread_published`` holds the published rows of the like-for-like
    block (T2: A at 1B, D at 1M) as diamonds, filled when their protocol is ours,
    hollow when it differs or is unstated, whiskers for a stated sd or s.e. The files
    share the x range and the left inset, so stacked they read as one axis.
    """
    th, name = _theme(theme, width), NAMES["F4"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the seed spread")}
    budget = _single_budget(runs)
    mem = _budget_members(runs)[budget]
    if literature is None:
        literature = compare.literature_table(frames, n_boot=n_boot, seed=seed)
    mine = compare.tuned(runs)
    lit = _literature_rows(literature, budget)
    groups = {
        f"{name}_{b}": _rows([m for m in mem if m[0] == b], headings=False)
        for b in compare.BLOCKS
    }
    groups = {k: v for k, v in groups.items() if v}
    top = _score_top(mine["final_score"].dropna().tolist(), lit)
    labels = [r.label for rows in groups.values() for r in rows]
    labels += [_short(str(n)) for n in lit["name"]]
    inset = charts.label_inset_mm(th, labels)
    figs: Figs = {}
    plans: list[tuple[str, list[Row] | None]] = [*groups.items()]
    if len(lit):
        plans.append((f"{name}_published", None))
    for stem, rows in plans:
        if not wants(th, stem):
            continue
        n = len(rows) if rows is not None else len(lit)
        fig = _figure(th, shape=Shape.ROWS, rows=n)
        ax = charts.add_panel(fig, left_mm=inset)
        _style(ax, th)
        if rows is not None:
            for row in rows:
                scores = mine.loc[mine["method"] == row.method, "final_score"]
                _seed_dots(ax, th, budget, row, scores.to_numpy(dtype=float))
        else:
            rows = _literature_marks(ax, th, budget, lit)
        _apply_rows(ax, th, rows)
        ax.set_xlim(left=0, right=top * 1.05)
        _labels(ax, th, "final score (% of max)")
        if stem.endswith("published"):
            title = "Published scores"
            text = (
                "Published scores of the like-for-like block on the same axis as the "
                "seed-spread files. Diamond: filled when the protocol is ours, hollow "
                "when it differs or is unstated; whisker: the published sd or s.e."
            )
        else:
            block = stem.removeprefix(f"{name}_")
            title = f"Final score per seed, {BLOCK_TITLES[block]} block"
            text = (
                "Final training score (% of the maximum) of each seed (dot, jittered) "
                "and its mean (bar) for the block's controls and tuned methods; the "
                "x range and left inset are shared with the other seed-spread files."
            )
        _save(fig, th, out_dir, stem, title=title, description=text)
        figs[stem] = fig
    return figs


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


def _rate_text(value: float) -> str:
    """``.35`` or ``1.0``: a rate in a narrow cell (zero cells stay blank)."""
    text = f"{value:.2f}"
    return "1.0" if text == "1.00" else text.replace("0.", ".", 1)


HEAT_MIN_RATE = 0.01
"""An achievement some method unlocks in under this share of episodes is not a row."""


def _heat_panel(
    ax: Axes, th: Theme, budget: str, block: str, names: Sequence[str],
    status: pd.DataFrame, rows: Sequence[str],
) -> None:  # fmt: skip
    """One block's heat map: ``rows`` achievements down, methods across, final rate."""
    sub = status[status["achievement"].isin(rows)]
    per_method = {m: g.set_index("achievement") for m, g in sub.groupby("method")}
    cells = np.column_stack([
        per_method[m]["final_rate"].reindex(rows).to_numpy(dtype=float)
        if m in per_method
        else np.full(len(rows), np.nan)
        for m in names
    ])
    cmap = _cmap(th, *th.seq)
    charts.heatmap(
        ax, th, cells, cmap, vmin=0.0, vmax=1.0, gid=_gid("heat", budget, block)
    )
    ax.set_xticks(np.arange(len(names)) + 0.5, names, rotation=90)
    ax.set_yticks(np.arange(len(rows)) + 0.5, list(rows))
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    for i, j in zip(*np.where(cells >= 0.005), strict=True):
        ax.text(
            j + 0.5, i + 0.5, _rate_text(cells[i, j]), ha="center", va="center",
            fontsize=th.font, color=_ink_on(cmap(cells[i, j])),
        )  # fmt: skip
    tier = per_method[next(iter(per_method))].reindex(rows)["tier"].tolist()
    for i in range(1, len(rows)):
        if tier[i] != tier[i - 1]:
            ax.axhline(i, color=th.ink2, lw=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_success_heatmap(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    mode: str = "stochastic",
) -> Figs:
    """F5 ``success_heatmap_<block>``: share of final eval episodes per achievement.

    Methods (tuned, the block's three controls first) across, the achievements down in
    tier then floor order (`hard.achievement_status`: ``final_rate``, the mean over
    seeds of each seed's share of ``mode`` episodes); only achievements some method
    reaches in `HEAT_MIN_RATE` of its episodes are rows, the caption names the rest.
    Both blocks share the rows, so the files line up.
    """
    th, name = _theme(theme, width), NAMES["F5"]
    runs, eps = frames["runs"], frames["eval_episodes"]
    if runs.empty or not any(c.startswith("ach/") for c in eps.columns):
        text = "no achievement columns in the eval episodes"
        return {name: _no_data(name, th, out_dir, text)}
    budget = _single_budget(runs)
    mem = _budget_members(runs)[budget]
    tuned = compare.tuned(runs)
    status = achievement_status(eps[eps["run_id"].isin(tuned["run_id"])], mode=mode)
    if status.empty:
        return {name: _no_data(name, th, out_dir, "no eval episodes")}
    best = status.groupby("achievement", sort=False)["final_rate"].max()
    ordered = list(dict.fromkeys(status["achievement"]))
    rows = [a for a in ordered if best[a] >= HEAT_MIN_RATE]
    folded = [a for a in ordered if best[a] < HEAT_MIN_RATE]
    rows = rows or ordered[:1]
    all_names = [m for _, m, _ in mem]
    left = charts.label_inset_mm(th, rows)
    bottom = charts.label_inset_mm(th, all_names, pad_mm=2.0)
    figs: Figs = {}
    for block in compare.BLOCKS:
        stem = f"{name}_{block}"
        names = [m for b, m, _ in mem if b == block]
        if not names or not wants(th, stem):
            continue
        fig = _figure(th, shape=Shape.ROWS, rows=len(rows), base_mm=bottom + 3.0)
        ax = charts.add_panel(fig, left_mm=left, bottom_mm=bottom, top_mm=1.5)
        _heat_panel(ax, th, budget, block, names, status, rows)
        _save(
            fig, th, out_dir, stem,
            title=f"Achievements reached, {BLOCK_TITLES[block]} block",
            description=(
                f"Share of the final stochastic eval episodes in which a tuned method "
                f"unlocks each achievement (mean over seeds; light: none, dark: all; "
                f"blank: under half a percent); lines separate the tiers. "
                f"Achievements reached by under {HEAT_MIN_RATE:.0%} of the episodes of "
                f"every method are not drawn ({len(folded)}: {', '.join(folded)})."
            ),
        )  # fmt: skip
        figs[stem] = fig
    return figs


# -------------------------------------------------------------------------- F18

SENSITIVE_ALPHA = 0.5
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
        linewidth=0.9,
    )  # fmt: skip
    bar.set_gid(_gid("sens", phase, row.method, str(rank)))
    bar.set_label(name)  # the container's label does not reach the rectangle


def _scores_panel(
    ax: Axes, th: Theme, phase: str, rows: Sequence[Row], runs: pd.DataFrame
) -> None:
    """One dot per config, its score ``r_c`` over the method's best (a row a method)."""
    for row in rows:
        mine = runs[(runs["method"] == row.method) & in_phase(runs, phase)]
        r_c = config_scores(mine)["r_c"].to_numpy()
        dots = ax.scatter(
            r_c, row.y + _jitter(len(r_c), 0.25), s=th.ms**2,
            color=_color(th, row.method), zorder=3,
        )  # fmt: skip
        dots.set_gid(_gid("rc", phase, row.method))
    _style(ax, th)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([r.y for r in rows], [r.method for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.axvline(ROBUST, color=th.muted, lw=0.5)
    ax.set_xlim(0, 1.05)
    _labels(ax, th, "config score r_c")


def _bars_panel(
    ax: Axes, th: Theme, phase: str, rows: Sequence[Row], t7: pd.DataFrame
) -> None:
    """The two most sensitive hyperparameters of each method: ``S_h``, noise floor."""
    names = [""] * (2 * len(rows))
    for row in rows:
        stats = _lookup(t7, (phase, row.method), ["phase", "method"])
        if stats is None:
            msg = f"no T7 row for {row.method} in {phase}"
            raise ValueError(msg)
        base = 2 * row.y
        for rank in (1, 2):
            _bar_rank(ax, th, phase, row, stats, rank)
            names[base + rank - 1] = (
                str(stats[f"top{rank}"])[:NAME_CHARS] or "(none varies)"
            )
        (floor,) = ax.plot(
            [stats["noise_floor"]] * 2, [base - 0.4, base + 1.4], color=th.ink,
            lw=th.lw, ls=":",
        )  # fmt: skip
        floor.set_gid(_gid("noise", phase, row.method))
    _style(ax, th)
    ax.set_ylim(2 * len(rows) - 0.5, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(range(2 * len(rows)), names)
    ax.set_xlim(0, 1.05)
    _labels(ax, th, "sensitivity S_h")


def plot_sensitivity(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    table: pd.DataFrame | None = None,
) -> Figs:
    """F18 ``config_scores_<phase>`` and ``sensitivity_<phase>`` (``coarse``, ``fine``).

    ``frames`` hold the tuning phases (as for `compare.sensitivity_table`). Per phase,
    ``config_scores``: one dot per config, its score ``r_c`` over the best config of
    the method (`stability.config_scores`, the vertical line is the robust 0.9);
    ``sensitivity``: the two most sensitive hyperparameters' ``S_h`` (T7), faint when
    below the method's noise floor (dotted line).
    """
    th, name = _theme(theme, width), NAMES["F18"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no tuning runs")}
    t7 = table if table is not None else compare.sensitivity_table(frames)
    t7 = t7[t7["benchmark"] == _single_budget(runs)]
    figs: Figs = {}
    for phase in sorted(TUNING_PHASES):
        rows = _sens_rows(t7[t7["phase"] == phase].drop_duplicates("method"))
        short = phase.removeprefix("tune-")
        if not rows:
            continue
        scores, bars = f"config_scores_{short}", f"sensitivity_{short}"
        if wants(th, scores):
            fig, ax = _category_panel(th, len(rows))
            _scores_panel(ax, th, phase, rows, runs)
            _save(
                fig, th, out_dir, scores, title=f"Config scores, {short} tuning",
                description=(
                    f"Every config of the {short} tuning grid as a dot: its score r_c "
                    "relative to the best config of the method (1 is the best); the "
                    "vertical line at 0.9 marks the robust configs."
                ),
            )  # fmt: skip
            figs[scores] = fig
        if wants(th, bars):
            fig, ax = _category_panel(th, 2 * len(rows), extra_mm=4.0)
            _bars_panel(ax, th, phase, rows, t7)
            handles = [
                Line2D([], [], color=th.ink, ls=":", lw=th.lw, label="noise floor"),
                Patch(
                    facecolor=to_rgba(th.ink2, SENSITIVE_ALPHA), edgecolor=th.ink2,
                    label="S_h below noise floor",
                ),
            ]  # fmt: skip
            _finish(fig, th, handles, cols=2)
            _save(
                fig, th, out_dir, bars, title=f"Hyperparameter sensitivity, {short}",
                description=(
                    f"Sensitivity S_h of the two most sensitive hyperparameters of "
                    f"each method in the {short} tuning grid; a faint bar is below "
                    "the method's noise floor (dotted line)."
                ),
            )  # fmt: skip
            figs[bars] = fig
    return figs


# -------------------------------------------------------------------------- F19


def _drawdown_row(
    ax: Axes, th: Theme, budget: str, row: Row, per_run: pd.DataFrame, t3: pd.DataFrame
) -> None:
    """Per-run max drawdowns (hollow: a drop, cross: collapsed) and T3's median."""
    color = _color(th, row.method)
    mine = per_run[
        (per_run["benchmark"] == budget) & (per_run["method"] == row.method)
    ].dropna(subset=["max_drawdown"])
    x = mine["max_drawdown"].to_numpy(dtype=float)
    dots = ax.scatter(
        x, row.y + _jitter(len(x)), s=th.ms**2, edgecolors=color, linewidths=th.lw,
        facecolors=[th.surface if d else color for d in mine["drops"]], zorder=3,
    )  # fmt: skip
    dots.set_gid(_gid("dd", budget, row.block, row.method))
    gone = mine[mine["collapsed"]]
    if len(gone):
        cross = ax.scatter(
            gone["max_drawdown"], [row.y] * len(gone), marker="x", s=th.ms**2 * 3,
            color=th.ink, zorder=4,
        )  # fmt: skip
        cross.set_gid(_gid("collapsed", budget, row.block, row.method))
    stats = _lookup(
        t3, (budget, row.block, row.method), ["benchmark", "block", "method"]
    )
    if stats is not None and np.isfinite(stats["max_drawdown_median"]):
        (tick,) = ax.plot(
            [stats["max_drawdown_median"]], [row.y], marker="|", ls="", color=th.ink,
            ms=th.ms * 3, markeredgewidth=th.lw * 1.5,
        )  # fmt: skip
        tick.set_gid(_gid("dd-median", budget, row.block, row.method))


def _events_row(ax: Axes, th: Theme, budget: str, row: Row, t3: pd.DataFrame) -> None:
    """T3's share of the method's runs with a drop (light) and a collapse (solid)."""
    color = _color(th, row.method)
    stats = _lookup(
        t3, (budget, row.block, row.method), ["benchmark", "block", "method"]
    )
    if stats is None or not stats["curve_runs"]:  # T3 counts runs that have a curve
        return
    for key, dy, alpha in (("drops", -0.17, 0.45), ("collapse", 0.17, 1.0)):
        share = stats[f"runs_with_{key}"] / stats["curve_runs"]
        (bar,) = ax.barh(row.y + dy, share, height=0.3, color=to_rgba(color, alpha))
        bar.set_gid(_gid(key, budget, row.block, row.method))


def plot_stability(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    per_run: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
) -> Figs:
    """F19 ``stability_drawdown`` and ``stability_events``: score-curve stability.

    Drawdown: one dot per tuned run at its deepest drawdown of the smoothed score curve
    (`compare.run_stability`; hollow: the run had a drop, a cross: it collapsed; the
    bar marks the median, T3). Events: the share of the method's runs with a curve (T3
    `curve_runs`) that had a drop (light) and a collapse. The two files have one size
    and one left inset, so they line up.
    """
    th, name = _theme(theme, width), NAMES["F19"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the stability figure")}
    budget = _single_budget(runs)
    per_run = per_run if per_run is not None else compare.run_stability(frames)
    t3 = table if table is not None else compare.stability_table(frames)
    rows = _rows(_budget_members(runs)[budget])
    inset = charts.label_inset_mm(th, [r.label for r in rows])
    plans: dict[str, tuple[list[Any], str, str]] = {
        "stability_drawdown": (
            [
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    color=th.ink2,
                    markerfacecolor=th.surface,
                    ms=th.ms,
                    label="run with a drop",
                ),
                Line2D(
                    [], [], ls="", marker="x", color=th.ink, ms=th.ms, label="collapsed"
                ),
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="|",
                    color=th.ink,
                    ms=th.ms * 2,
                    label="median",
                ),
            ],
            "max drawdown",
            (
                "Deepest drawdown (fall from the running maximum of the smoothed "
                "score curve, as a share of it) of each tuned run, one dot per seed. "
                "Hollow dot: the run had a drop of at least 20% for 2% of the budget; "
                "cross: it collapsed and did not recover; bar: the method's median. "
                "No dot below the run's drawdown floor (R92)."
            ),
        ),
        "stability_events": (
            [
                Patch(facecolor=to_rgba(th.ink2, 0.45), label="share with a drop"),
                Patch(facecolor=th.ink2, label="share collapsed"),
            ],
            "share of runs",
            (
                "Share of each method's runs with a score curve that had a drop "
                "(light bar) and that collapsed (solid bar); the same rows and size "
                "as the drawdown file."
            ),
        ),
    }
    figs: Figs = {}
    for stem, (handles, xlabel, text) in plans.items():
        if not wants(th, stem):
            continue
        fig = _figure(th, shape=Shape.ROWS, rows=len(rows), base_mm=ROWS_BASE + 5.0)
        ax = charts.add_panel(fig, left_mm=inset, top_mm=6.5)
        for row in rows:
            if row.kind != "heading":
                if stem == "stability_drawdown":
                    _drawdown_row(ax, th, budget, row, per_run, t3)
                else:
                    _events_row(ax, th, budget, row, t3)
        _style(ax, th)
        _apply_rows(ax, th, rows)
        ax.set_xlim(0, 1.02 if stem == "stability_drawdown" else 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        _labels(ax, th, xlabel)
        _finish(fig, th, handles, cols=3)
        title = "Drawdown per run" if "drawdown" in stem else "Drops and collapses"
        _save(fig, th, out_dir, stem, title=title, description=text)
        figs[stem] = fig
    return figs


# -------------------------------------------------------------------------- F20


def _hardware_panel(
    ax: Axes, th: Theme, profile: str, rows: Sequence[Row], t4: pd.DataFrame,
    series: Sequence[tuple[str, str, float]],
) -> None:  # fmt: skip
    """The medians of one metric's series per method row."""
    for k, (col, _, factor) in enumerate(series):
        for row in rows:
            hit = _lookup(t4, (row.block, row.method), ["block", "method"])
            if hit is None or not np.isfinite(hit[col]):
                continue
            mark = ax.scatter(
                [hit[col] * factor], [row.y], marker=SERIES_MARKERS[k], s=th.ms**2,
                color=_color(th, row.method), zorder=3,
            )  # fmt: skip
            mark.set_gid(_gid("hw", profile, col, row.method))


def _measured(t4: pd.DataFrame, series: Sequence[tuple[str, str, float]]) -> bool:
    """True if any method has a value in one of the series' columns."""
    return any(c in t4 and t4[c].notna().any() for c, _, _ in series)


def plot_hardware(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    budget: str = "1m",
    table: pd.DataFrame | None = None,
) -> Figs:
    """F20 ``hardware_<metric>``: SPS, wall-clock, compile time, memory, utilisation.

    One file per metric (steps per joule only if some method measured it), medians over
    each method's tuned runs at ``budget`` (T4, `compare.hardware_table`); with several
    profiles each gets its own files (``hardware_<metric>_<profile>``): profiles are
    never compared. A metric nobody measured is not drawn, the caption says so.
    """
    th, name = _theme(theme, width), NAMES["F20"]
    runs = frames["runs"]
    if runs.empty:
        return {name: _no_data(name, th, out_dir, "no runs for the hardware figure")}
    t4 = table if table is not None else compare.hardware_table(frames)
    t4 = t4[t4["benchmark"] == budget]
    if t4.empty:
        return {
            name: _placeholder(
                name, th, out_dir, "no hardware table for this benchmark"
            )
        }
    profiles = dict(iter(t4.groupby("profile")))
    figs: Figs = {}
    for profile, g in profiles.items():
        rows = _rows(
            list(zip(g["block"], g["method"], g["role"], strict=True)), headings=False
        )
        machine = PROFILE_NAMES.get(str(profile), str(profile))
        notes = sorted(set(g["note"]) - {""})
        skipped = [x for _, x, series in HARDWARE_PANELS if not _measured(g, series)]
        for key, xlabel, series in HARDWARE_PANELS:
            stem = f"{name}_{key}" + (f"_{profile}" if len(profiles) > 1 else "")
            if not _measured(g, series) or not wants(th, stem):
                continue
            multi = len(series) > 1
            fig, ax = _category_panel(th, len(rows), extra_mm=4.0 if multi else 0.0)
            _hardware_panel(ax, th, str(profile), rows, g, series)
            _style(ax, th)
            _apply_rows(ax, th, rows)
            _labels(ax, th, xlabel)
            if multi:
                handles = [
                    Line2D(
                        [], [], ls="", marker=SERIES_MARKERS[k], color=th.ink2,
                        ms=th.ms, label=label,
                    )
                    for k, (_, label, _) in enumerate(series)
                ]  # fmt: skip
                _finish(fig, th, handles, cols=len(series))
            text = (
                f"{xlabel.capitalize()}: median over each method's tuned runs on the "
                f"{machine}; a dot per method, controls once. Profiles are never "
                "compared."
            )
            if notes:
                text += f" Timing note: {'; '.join(notes)}."
            if skipped:
                text += f" Not measured, so not drawn: {', '.join(skipped)}."
            _save(fig, th, out_dir, stem, title=xlabel.capitalize(), description=text)
            figs[stem] = fig
    return figs


LEGEND_DESCRIPTION = (
    "Colour, line style and marker of every method; the other figures carry no method "
    "legend. Controls are dashed with their own marker, tuned methods solid, an "
    "ablation of a method dotted with a hollow marker in the method's colour."
)


def method_handles(
    th: Theme, names: Sequence[str], *, ablation: bool = False
) -> list[Line2D]:
    """One legend entry per method in its colour, line style and marker.

    ``ablation`` adds a grey entry for the dotted, hollow ablation style.
    """
    handles = []
    for m in names:
        s = _style_of(th, m)
        handles.append(
            Line2D(
                [], [], color=s["color"], ls=s["ls"], marker=s["marker"], ms=th.ms,
                lw=max(th.lw, 1.0), label=m,
            )
        )  # fmt: skip
    if ablation:
        handles.append(
            Line2D(
                [], [], color=th.ink2, ls=":", marker="o", markerfacecolor=th.surface,
                ms=th.ms, lw=max(th.lw, 1.0), label="ablation",
            )
        )  # fmt: skip
    return handles


def plot_legend_methods(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
) -> Figure:
    """``legend_methods``: the one legend of every method figure (no axes).

    Every method of ``frames`` (controls first, then each block's methods) with its
    colour, line style and marker; an ablation entry when the runs hold ablations.
    The other figures drop their method legends and point here.
    """
    th, name = _theme(theme, width), NAMES["LEG"]
    runs = frames["runs"]
    if runs.empty:
        return _no_data(name, th, out_dir, "no runs for the method legend")
    names = list(
        dict.fromkeys(
            m
            for _, r in compare.budgets(compare.tuned(runs))
            for _, m, _ in compare.members(r)
        )
    )
    ablation = bool((runs["config_ref"] != compare.TUNED).any())
    handles = method_handles(th, names, ablation=ablation)
    rows = -(-len(handles) // th.legend_cols)
    fig = _figure(th, shape=charts.Shape.LEGEND, rows=rows)
    fig.legend(
        handles=handles, loc="center", ncol=th.legend_cols, fontsize=th.font,
        frameon=False, labelcolor=th.ink2,
    )  # fmt: skip
    return _save(
        fig, th, out_dir, name, title="Method legend", description=LEGEND_DESCRIPTION
    )


FIGURES: dict[str, Callable[..., Any]] = {
    "LEG": plot_legend_methods,
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
