"""In-game figures F6-F17: race, order graphs, funnel, deaths, splits, radars.

Each ``plot_*`` follows `analysis.figures` (report 80 mm or ``double``, dark ``video``,
files ``<stem>[_double][_video]``, `METHOD_COLORS`; a missing benchmark is pending).
Per method: the median over seeds of each seed's episode median (a share: its mean) of
the final stochastic eval episodes, with an `evaluation.stats` bootstrap interval. The
race is `compare.race_table` (``upto`` only masks later unlocks); a family never logged
reads "not logged" (R78), never 0. Artists carry ``<kind>/<budget>/<method>[/<key>]``.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures_game.py]]
"""

import itertools
import textwrap
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from ..evaluation.milestones import (
    ADVANCED_ACHIEVEMENTS,
    FLOORS,
    RADAR_AXES,
    TECH_LADDER,
)
from ..evaluation.stats import (
    BOOTSTRAP_N,
    Interval,
    curve_band,
    step_grid,
    stratified_bootstrap_ci,
)
from . import compare
from .figures import (
    BLOCK_TITLES,
    PENDING,
    Theme,
    _budget_name,
    _budget_steps,
    _cmap,
    _color,
    _figure,
    _finish,
    _gid,
    _ink_on,
    _labels,
    _no_data,
    _panel_title,
    _pending,
    _placeholder,
    _save,
    _stack,
    _style,
    _style_of,
    _theme,
    _title,
)
from .hard import ACHIEVEMENTS, TIERS, _cells, death_causes, stall_ccdf, stall_stats
from .hard import stall_table as _stall_table
from .load import Frames, final_checkpoint
from .obstacles import ALIASES

GAME_NAMES = {
    "F6": "race_steps",
    "F6W": "race_wall",
    "F7": "order_agreement",
    "F7G": "order_{method}",
    "F8": "funnel",
    "F8T": "funnel_training",
    "F9": "deaths",
    "F9S": "stalls",
    "F10": "splits",
    "F11": "coverage",
    "F12": "survival",
    "F13": "combat",
    "F14": "actions",
    "F15": "resources",
    "F16": "advanced",
    "F17": "radar_{budget}",
}
"""Figure id -> file stem; ``{method}`` and ``{budget}`` are filled per file."""
NOT_LOGGED = "not logged"
"""What a figure says of a family a run never logged (R78)."""
NOT_RECORDED = "not recorded"
"""What a figure says of an eval column no replay filled (NaN, never 0)."""
MILESTONES = (
    *(t.lower() for t in TECH_LADDER),
    *(e.lower() for _, e in FLOORS if e),
)
"""Speedrun-split rows: the tech ladder, then the ``enter_*`` achievements."""
STATS = ("food", "drink", "energy")
ORDER_SHARE, ORDER_P, ORDER_PAIRS = 0.05, 0.9, 20
"""Order graph: a node is unlocked in this share of episodes; an edge needs this
P(a before b) over at least this many episodes with both (metrics plan)."""
PERMUTATIONS = 10_000
"""Permutations of the Kendall tau-b test (metrics plan)."""
TAU_ALPHA = 0.05
SHORT_AXES = 4
"""Letters of a radar axis name in a narrow report panel."""
SPLIT_GAP = 2.0
"""Rows apart (in method offsets) of the speedrun boxes of two milestones."""
DODGE = 0.62
"""Total vertical spread of the methods inside one category row."""
ROW_MM = 1.45
"""Millimetres per method inside one category row of a dot panel."""
LEGEND_MM = 24.0
"""Notes and margins of one report figure; the legend is added by `_legend_mm`."""
LEGEND_ROW_MM = 3.6
"""Millimetres per row of a legend or of the notes."""
PANEL_MM = 16.0
"""Title and x axis allowance of one report panel."""
BOX_ALPHA = 0.4
HEAT_TICK_MM = 22.0
"""Method labels under a heat map."""


class Ctx(NamedTuple):
    """One benchmark's tuned runs and their final stochastic eval episodes."""

    budget: str
    names: list[str]
    """Methods drawn: the controls first, then each block's methods (combo last)."""
    runs: pd.DataFrame
    eps: pd.DataFrame
    curves: pd.DataFrame
    n_boot: int = BOOTSTRAP_N
    seed: int = 0


Drawer = Callable[[Axes, Theme, Ctx, list[str]], None]


class Panel(NamedTuple):
    """One panel of a figure, drawn once per benchmark."""

    title: str
    draw: Drawer
    mm: Callable[[Ctx], float]
    """Report height in millimetres, title and axis included."""
    narrow: bool = True
    """Wrap the title for row labels beside it."""


# ----------------------------------------------------------------------- plumbing


def _contexts(frames: Frames, n_boot: int = BOOTSTRAP_N, seed: int = 0) -> list[Ctx]:
    """One `Ctx` per benchmark with tuned runs, 1M before 1B."""
    out = []
    episodes, curves = frames["eval_episodes"], frames["curves"]
    for budget, runs in compare.budgets(compare.tuned(frames["runs"])):
        names = list(dict.fromkeys(m for _, m, _ in compare.members(runs)))
        mine = episodes[
            episodes["run_id"].isin(runs["run_id"]) & (episodes["mode"] == "stochastic")
        ]
        final = final_checkpoint(mine) if len(mine) else mine
        keep = curves[curves["run_id"].isin(runs["run_id"])]
        out.append(Ctx(budget, names, runs, final, keep, n_boot, seed))
    return out


def _method_handles(th: Theme, names: Sequence[str]) -> list[Line2D]:
    """One legend entry per method in its colour and line style."""
    handles = []
    for m in names:
        s = _style_of(th, m)
        handles.append(
            Line2D(
                [],
                [],
                color=s["color"],
                ls=s["ls"],
                marker=s["marker"],
                ms=th.ms,
                lw=max(th.lw, 1.0),
                label=m,
            )
        )
    return handles


def _grid(fig: Figure, th: Theme, mm: Sequence[Sequence[float]]) -> list[list[Axes]]:
    """Axes ``[row][col]``: stacked in the report column, a grid when wide."""
    rows, cols = len(mm), len(mm[0])
    if th.wide:
        grid = fig.subplots(
            rows, cols, squeeze=False, height_ratios=[max(r) for r in mm]
        )
        return [list(r) for r in grid]
    flat = fig.subplots(
        rows * cols, 1, squeeze=False, height_ratios=list(itertools.chain(*mm))
    )
    axes = [ax for r in flat for ax in r]
    return [axes[i * cols : (i + 1) * cols] for i in range(rows)]


def _legend_mm(th: Theme, entries: int) -> float:
    """Report height of the notes and of a legend of ``entries`` methods."""
    return LEGEND_MM + ceil(entries / th.legend_cols) * LEGEND_ROW_MM


def _height(th: Theme, mm: Sequence[Sequence[float]], entries: int) -> float:
    """Report figure height: every panel stacked, or the tallest of each row."""
    rows = [max(r) for r in mm] if th.wide else [sum(r) for r in mm]
    return sum(rows) + _legend_mm(th, entries)


def _unique(notes: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(n for n in notes if n))


def _draw(
    name: str,
    frames: Frames,
    out_dir: Path,
    theme: str,
    width: str,
    panels: Sequence[Panel],
    *,
    notes: Sequence[str] = (),
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """The figure of ``panels`` per benchmark, written as ``name`` in ``out_dir``."""
    th = _theme(theme, width)
    ctxs = _contexts(frames, n_boot, seed)
    if not ctxs:
        return _no_data(name, th, out_dir, "no tuned runs for the in-game figures")
    mm = [[p.mm(c) for p in panels] for c in ctxs]
    names = list(dict.fromkeys(m for c in ctxs for m in c.names))
    fig = _figure(th, _height(th, mm, len(names)))
    axes = _grid(fig, th, mm)
    extra: list[str] = []
    for row, ctx in zip(axes, ctxs, strict=True):
        for ax, panel in zip(row, panels, strict=True):
            panel.draw(ax, th, ctx, extra)
            _title(ax, th, _panel_title(ctx.budget, panel.title), narrow=panel.narrow)
    pending = _pending([c.budget for c in ctxs])
    _finish(fig, th, _method_handles(th, names), [*notes, *_unique(extra), *pending])
    return _save(fig, th, out_dir, name)


def _empty(ax: Axes, th: Theme, text: str) -> None:
    """Say why a panel has nothing to draw."""
    ax.set_axis_off()
    shown = "\n".join(textwrap.wrap(text, max(12, th.wrap // 2)))
    ax.text(
        0.5,
        0.5,
        shown,
        ha="center",
        va="center",
        fontsize=th.font,
        color=th.muted,
        transform=ax.transAxes,
    )


def _column(frame: pd.DataFrame, column: str) -> pd.Series:
    """``column`` of ``frame``, NaN throughout when the export has no such column."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)


def _per_run(ctx: Ctx, values: pd.Series, how: str = "median") -> dict[str, np.ndarray]:
    """Method -> one value per seed: the ``how`` of its episodes (NaN-skipping)."""
    frame = pd.DataFrame({
        "method": ctx.eps["method"],
        "run": ctx.eps["run_id"],
        "v": values,
    })
    per = frame.groupby(["method", "run"])["v"].agg(how)
    return {str(m): g.to_numpy(dtype=float) for m, g in per.groupby(level=0)}


def _interval(values: np.ndarray, how: str, n_boot: int, seed: int) -> Interval | None:
    """Median (or mean) over seeds with its bootstrap interval; None without a seed."""
    v = values[np.isfinite(values)]
    if not v.size:
        return None
    stat = np.mean if how == "mean" else np.median
    return stratified_bootstrap_ci(
        {"seeds": v.tolist()}, lambda d: float(stat(d["seeds"])), n_boot, seed
    )


def _dot_mm(ctx: Ctx, rows: int) -> float:
    return rows * (len(ctx.names) * ROW_MM + 2.0) + PANEL_MM


def _dodge(n: int) -> np.ndarray:
    return np.linspace(-DODGE / 2, DODGE / 2, n) if n > 1 else np.zeros(1)


class Cats(NamedTuple):
    """Rows of a dot panel: keys, labels and the per-seed values of (method, key)."""

    keys: Sequence[str]
    labels: Sequence[str]
    values: Callable[[str, str], np.ndarray]
    unlogged: Sequence[str] = ()
    """Methods that never logged one of the categories (said in a note, R78)."""


def _dot_panel(
    ax: Axes,
    th: Theme,
    ctx: Ctx,
    kind: str,
    cats: Cats,
    xlabel: str,
    notes: list[str],
    *,
    how: str = "median",
    log: bool = False,
    missing: str = NOT_RECORDED,
) -> None:
    """Per category row, each method's estimate with its bootstrap interval."""
    offsets = _dodge(len(ctx.names))
    seen: set[str] = set()
    for j, method in enumerate(ctx.names):
        style = _style_of(th, method)
        for i, key in enumerate(cats.keys):
            est = _interval(cats.values(method, key), how, ctx.n_boot, ctx.seed)
            if est is None:
                continue
            seen.add(method)
            y = i + offsets[j]
            gid = _gid("ci", ctx.budget, method, key)
            ax.hlines(
                y, est.lo, est.hi, colors=style["color"], linewidth=th.lw, gid=gid
            )
            (dot,) = ax.plot(
                [est.estimate],
                [y],
                ls="",
                marker=style["marker"],
                ms=th.ms,
                color=style["color"],
            )
            dot.set_gid(_gid(kind, ctx.budget, method, key))
    if cats.unlogged:
        notes.append(
            f"{_budget_name(ctx.budget)}: {', '.join(cats.unlogged)} "
            f"training entropy {NOT_LOGGED} (R78)."
        )
    absent = [m for m in ctx.names if m not in seen]
    if not seen:
        _empty(ax, th, f"{missing}: {', '.join(absent)}")
        return
    if absent:
        notes.append(f"{_budget_name(ctx.budget)}: {', '.join(absent)} {missing}.")
    _style(ax, th)
    ax.set_yticks(range(len(cats.keys)), list(cats.labels))
    ax.set_ylim(len(cats.keys) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_num_text))
    _labels(ax, th, xlabel)


def _num_text(v: float, _pos: int | None = None) -> str:
    return f"{v:.3g}"


def _steps_label(v: float, _pos: int | None = None) -> str:
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= scale:
            return f"{v / scale:.3g}{unit}"
    return f"{v:.0f}"


def _heat(
    ax: Axes,
    th: Theme,
    ctx: Ctx,
    kind: str,
    labels: Sequence[str],
    matrix: np.ndarray,
    *,
    annotate: bool = False,
    vmax: float | None = None,
) -> None:
    """Rows ``labels`` down, ``ctx.names`` across; NaN cells are the surface."""
    top = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    norm = Normalize(0.0, vmax if vmax is not None else (top or 1.0))
    cmap = _cmap(th, *th.seq)
    image = ax.imshow(
        matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
    )
    image.set_gid(_gid("heat", ctx.budget, kind))
    ax.set_xticks(range(len(ctx.names)), ctx.names, rotation=90)
    ax.set_yticks(range(len(labels)), list(labels))
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    if annotate:
        for i, j in zip(*np.where(np.isfinite(matrix)), strict=True):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=th.font,
                color=_ink_on(cmap(norm(matrix[i, j]))),
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _mean_by_method(ctx: Ctx, columns: Sequence[str]) -> np.ndarray:
    """Rows ``columns``, columns ``ctx.names``: mean over seeds of the episode means."""
    out = np.full((len(columns), len(ctx.names)), np.nan)
    for i, col in enumerate(columns):
        per = _per_run(ctx, _column(ctx.eps, col), "mean")
        for j, m in enumerate(ctx.names):
            if m in per and np.isfinite(per[m]).any():
                out[i, j] = float(np.nanmean(per[m]))
    return out


def _per_seed_of(
    ctx: Ctx, column: str, how: str = "median"
) -> Callable[[str, str], np.ndarray]:
    """A `Cats.values` over one column per category: ``column`` format with the key."""

    def values(method: str, key: str) -> np.ndarray:
        found = _per_run(ctx, _column(ctx.eps, column.format(key)), how)
        return found.get(method, np.empty(0))

    return values


# --------------------------------------------------------------------------- F6


def _race_rows(table: pd.DataFrame) -> list[str]:
    """Achievements some method is eligible for, by tier, floor and game order."""
    names = table.loc[table["eligible"].astype(bool), "achievement"].unique()
    value = {n: i for i, n in enumerate(ACHIEVEMENTS)}

    def key(n: str) -> tuple[int, int, int]:
        info = ACHIEVEMENTS.get(n)
        if info is None:  # a name outside the game list goes last, never raises
            return (len(TIERS), 0, len(value))
        return (TIERS.index(info.tier), info.floor, value[n])

    return sorted(map(str, names), key=key)


def _masked(frames: Frames, upto: int) -> Frames:
    """``frames`` with every first unlock at env step ``upto`` or later set to +inf.

    The run stays in the race (its ``n`` is unchanged): it has only not unlocked yet.
    """
    firsts = compare.first_unlocks(frames["curves"], frames["runs"])
    late = firsts[firsts["steps"] >= upto]
    keys = set(zip(late["run_id"], late["achievement"], strict=True))
    curves = frames["curves"].copy()
    metric = curves["metric"]
    ach = metric.str.split("/", n=1).str[1]
    family = metric.str.startswith(("first_unlock/", "first_unlock_wall/"))
    hit = [
        bool(f) and (r, a) in keys
        for f, r, a in zip(family, curves["run_id"], ach, strict=True)
    ]
    curves.loc[hit, "value"] = np.inf
    return {**frames, "curves": curves}


def _spread(
    firsts: pd.DataFrame, runs: pd.DataFrame, column: str, upto: int | None
) -> dict[tuple[str, str], tuple[float, float]]:
    """(method, achievement) -> 25th and 75th percentile over the logging runs.

    A run that never unlocked it (or not before ``upto``) counts as +inf, as in the
    race medians.
    """
    mine = firsts[firsts["run_id"].isin(runs["run_id"])].copy()
    if upto is not None:
        mine.loc[mine["steps"] >= upto, column] = np.inf
    wide = mine.pivot_table(
        index="run_id", columns="achievement", values=column, aggfunc="first"
    ).fillna(np.inf)
    method = runs.set_index("run_id")["method"].reindex(wide.index)
    out = {}
    for m, g in wide.groupby(method):
        for a in wide.columns:
            lo, hi = np.percentile(g[a].to_numpy(float), [25, 75], method="nearest")
            out[(str(m), str(a))] = (float(lo), float(hi))
    return out


def _race_axes(ax: Axes, th: Theme, rows: Sequence[str], clock: str) -> None:
    _style(ax, th)
    ax.set_xscale("log")
    steps = clock == "steps"
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_label if steps else _num_text))
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    unit = "env steps" if steps else "wall-clock s"
    _labels(ax, th, f"median first unlock ({unit}, log)")


def _race_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    table: pd.DataFrame,
    rows: Sequence[str],
    spread: Mapping[tuple[str, str], tuple[float, float]],
    clock: str,
    limits: tuple[float, float],
) -> None:
    """Median first unlock per achievement and method, IQR whiskers, winner ringed."""
    col, win = f"median_{clock}", f"winner_{clock}"
    index = {a: i for i, a in enumerate(rows)}
    names = list(dict.fromkeys(table["method"]))
    offsets = dict(zip(names, _dodge(len(names)), strict=True))
    for r in table.itertuples():
        x = float(getattr(r, col))
        if r.achievement not in index or not r.eligible or not np.isfinite(x):
            continue
        y = index[r.achievement] + offsets[r.method]
        style = _style_of(th, r.method)
        lo, hi = spread.get((r.method, r.achievement), (x, x))
        ax.hlines(
            y,
            lo,
            min(hi, limits[1]),
            colors=style["color"],
            linewidth=th.lw,
            gid=_gid("iqr", budget, r.method, r.achievement),
        )
        (dot,) = ax.plot(
            [x], [y], ls="", marker=style["marker"], ms=th.ms, color=style["color"]
        )
        dot.set_gid(_gid("race", budget, r.method, r.achievement))
        if getattr(r, win):
            (ring,) = ax.plot(
                [x],
                [y],
                ls="",
                marker="o",
                color=style["color"],
                ms=th.ms * 2.2,
                markerfacecolor="none",
                markeredgecolor=style["color"],
                markeredgewidth=th.lw,
            )
            ring.set_gid(_gid("winner", budget, r.method, r.achievement))
    _race_axes(ax, th, rows, clock)
    ax.set_xlim(*limits)


def _race_notes(
    table: pd.DataFrame, runs: pd.DataFrame, budget: str, clock: str
) -> list[str]:
    """Methods without the family, methods with fewer logging seeds, the R97 note."""
    notes = []
    logged = table.groupby("method")["n"].first()
    count = runs["method"].value_counts()
    absent = sorted(m for m in count.index if m not in logged.index)
    if absent:
        notes.append(f"{', '.join(absent)}: first unlocks {NOT_LOGGED} (R78).")
    short = [f"{m} {int(n)} of {count[m]}" for m, n in logged.items() if n < count[m]]
    if short:
        notes.append(f"Seeds that logged the race: {', '.join(short)}.")
    if clock == "wall":
        walls = sorted({w for w in table["wall_note"].dropna() if w})
        notes += [f"{_budget_name(budget)} wall-clock: {w}." for w in walls]
    return notes


def _race_limits(full: pd.DataFrame, budget: str, clock: str) -> tuple[float, float]:
    """X limits of a race panel: fixed by the full race, so frames line up."""
    mine = full[(full["benchmark"] == budget) & full["eligible"].astype(bool)]
    v = mine[f"median_{clock}"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    low = float(v.min()) / 2 if v.size else 1.0
    high = (
        _budget_steps(budget)
        if clock == "steps"
        else float(v.max()) * 2
        if v.size
        else 10.0
    )
    return low, high


def _race_heights(th: Theme, sizes: Mapping[tuple[str, str], int]) -> list[float]:
    """Report height per stacked panel, or per benchmark row when wide."""
    if not th.wide:
        return [n * th.pitch_mm + PANEL_MM for n in sizes.values()]
    rows: dict[str, int] = {}
    for (budget, _block), n in sizes.items():
        rows[budget] = max(rows.get(budget, 0), n)
    return [n * th.pitch_mm + PANEL_MM for n in rows.values()]


def _race_layout(
    fig: Figure, th: Theme, sizes: Mapping[tuple[str, str], int], heights: list[float]
) -> list[Axes]:
    """One axes per (budget, block): stacked, or a benchmark x block grid when wide."""
    if not th.wide:
        return _stack(fig, th, [float(n) for n in sizes.values()])
    budgets = list(dict.fromkeys(b for b, _ in sizes))
    grid = fig.subplots(
        len(budgets), len(compare.BLOCKS), squeeze=False, height_ratios=heights
    )
    axes = [grid[budgets.index(b), compare.BLOCKS.index(k)] for b, k in sizes]
    for ax in grid.ravel():
        ax.set_visible(any(ax is a for a in axes))
    return axes


def race(
    frames: Frames,
    out_dir: Path | None,
    theme: str = "report",
    *,
    width: str = "column",
    clock: str = "steps",
    upto: int | None = None,
) -> Figure:
    """F6 ``race_steps`` / ``race_wall``: median first unlock of each achievement.

    Achievements down (tier, floor, game order; those some method is eligible for),
    the median over seeds on a log x axis per method with 25th-75th percentile
    whiskers, the winner ringed; one panel per (budget, block), a method racing only
    inside its block. Medians, eligibility and winners are `compare.race_table`.
    ``upto`` (env steps) draws only the first unlocks before that step, on the axes of
    the full race (the frames of the race GIF); ``out_dir=None`` writes no file.

    Raises:
        ValueError: If ``clock`` is neither ``steps`` nor ``wall``.
    """
    if clock not in ("steps", "wall"):
        msg = f"clock must be 'steps' or 'wall', got {clock!r}"
        raise ValueError(msg)
    th, name = _theme(theme, width), f"race_{clock}"
    tuned = compare.tuned(frames["runs"])
    full = compare.race_table(frames) if len(tuned) else pd.DataFrame()
    if full.empty:
        if out_dir is None:
            msg = "no first-unlock curves for the race"
            raise ValueError(msg)
        return _no_data(name, th, out_dir, "no first-unlock curves for the race")
    shown = full if upto is None else compare.race_table(_masked(frames, upto))
    firsts = compare.first_unlocks(frames["curves"], tuned)
    panels = [
        (b, blk)
        for b, _ in compare.budgets(tuned)
        for blk in compare.BLOCKS
        if ((full["benchmark"] == b) & (full["block"] == blk)).any()
    ]

    def of(table: pd.DataFrame, b: str, blk: str) -> pd.DataFrame:
        return table[(table["benchmark"] == b) & (table["block"] == blk)]

    rows = {p: _race_rows(of(full, *p)) for p in panels}
    sizes = {p: max(len(r), 4) for p, r in rows.items()}
    heights = _race_heights(th, sizes)
    fig = _figure(th, sum(heights) + _legend_mm(th, 9))
    axes = _race_layout(fig, th, sizes, heights)
    notes: list[str] = []
    for ax, (budget, block) in zip(axes, panels, strict=True):
        mine = tuned[tuned["budget"] == budget]
        spread = _spread(firsts, mine, clock, upto)
        _race_panel(
            ax,
            th,
            budget,
            of(shown, budget, block),
            rows[budget, block],
            spread,
            clock,
            _race_limits(full, budget, clock),
        )
        what = f"{BLOCK_TITLES[block]} block first unlocks"
        if upto is not None:
            what += f" up to env step {upto:,}"
        _title(ax, th, _panel_title(budget, what), narrow=True)
        in_block = [m for b, m, _ in compare.members(mine) if b == block]
        notes += _race_notes(
            of(full, budget, block),
            mine[mine["method"].isin(in_block)],
            budget,
            clock,
        )
    names = list(dict.fromkeys(full["method"]))
    pending = _pending([b for b, _ in compare.budgets(tuned)])
    _finish(fig, th, _method_handles(th, names), [*_unique(notes), *pending])
    return _save(fig, th, out_dir, name) if out_dir is not None else fig


def plot_race(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
) -> Figure:
    """F6: ``race_wall`` and ``race_steps`` files; returns the env-steps figure."""
    race(frames, out_dir, theme, width=width, clock="wall")
    return race(frames, out_dir, theme, width=width, clock="steps")


# --------------------------------------------------------------------------- F7


def transitive_reduction[T](edges: set[tuple[T, T]]) -> set[tuple[T, T]]:
    """``edges`` without each ``a -> c`` that a path ``a -> .. -> c`` implies."""
    out: dict[T, set[T]] = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)

    def reaches(start: T, goal: T, skip: tuple[T, T]) -> bool:
        seen, todo = {start}, [start]
        while todo:
            node = todo.pop()
            for nxt in out.get(node, ()):
                if (node, nxt) == skip or nxt in seen:
                    continue
                if nxt == goal:
                    return True
                seen.add(nxt)
                todo.append(nxt)
        return False

    return {(a, b) for a, b in edges if not reaches(a, b, (a, b))}


def _unlock_matrix(eps: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Achievement names (game order) and the unlock steps, NaN where never."""
    cols = [c for c in eps.columns if c.startswith("unlock_step/")]
    names = sorted((c.removeprefix("unlock_step/") for c in cols), key=_game_index)
    steps = eps[[f"unlock_step/{n}" for n in names]].to_numpy(dtype=float)
    return names, np.where(steps >= 0, steps, np.nan)


def _game_index(name: str) -> int:
    order = list(ACHIEVEMENTS)
    return order.index(name) if name in order else len(order)


def _mean_ranks(steps: np.ndarray) -> np.ndarray:
    """Per achievement the mean normalised rank ``(rank - 1) / (n_unlocked - 1)``.

    The rank is the achievement's place among the episode's unlocks by step, a tie
    going to the lower game index; an episode with one unlock has no rank.
    """
    total = np.zeros(steps.shape[1])
    count = np.zeros(steps.shape[1])
    for row in steps:
        got = np.flatnonzero(~np.isnan(row))
        if got.size < 2:
            continue
        order = got[np.lexsort((got, row[got]))]
        total[order] += np.arange(got.size) / (got.size - 1)
        count[order] += 1
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def order_graph(eps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes and edges of the achievement order graph of one method's eval episodes.

    Nodes are the achievements unlocked in at least `ORDER_SHARE` of the episodes, with
    ``rho`` their mean normalised rank. ``P(a before b)`` is the share of the episodes
    with both where ``a`` is unlocked at a strictly earlier step; an edge needs
    ``P >= ORDER_P`` over at least `ORDER_PAIRS` such episodes and is kept after
    transitive reduction. Columns: nodes ``achievement tier floor share rho``; edges
    ``source target p pairs``.
    """
    names, steps = _unlock_matrix(eps)
    nodes_cols = ["achievement", "tier", "floor", "share", "rho"]
    edges_cols = ["source", "target", "p", "pairs"]
    if not names:
        return pd.DataFrame(columns=nodes_cols), pd.DataFrame(columns=edges_cols)
    have = ~np.isnan(steps)
    rho = _mean_ranks(steps)
    share = have.mean(axis=0)
    keep = [i for i in range(len(names)) if share[i] >= ORDER_SHARE]
    both = have[:, keep, None] & have[:, None, keep]
    before = (steps[:, keep, None] < steps[:, None, keep]) & both
    pairs = both.sum(axis=0)
    prob = before.sum(axis=0) / np.maximum(pairs, 1)
    found = {
        (a, b)
        for a, b in itertools.permutations(range(len(keep)), 2)
        if pairs[a, b] >= ORDER_PAIRS and prob[a, b] >= ORDER_P
    }
    nodes = pd.DataFrame(
        [
            {
                "achievement": names[i],
                "tier": ACHIEVEMENTS[names[i]].tier if names[i] in ACHIEVEMENTS else "",
                "floor": ACHIEVEMENTS[names[i]].floor
                if names[i] in ACHIEVEMENTS
                else -1,
                "share": float(share[i]),
                "rho": float(rho[i]),
            }
            for i in keep
        ],
        columns=nodes_cols,
    )
    kept = transitive_reduction(found)
    edges = pd.DataFrame(
        [
            {
                "source": names[keep[a]],
                "target": names[keep[b]],
                "p": float(prob[a, b]),
                "pairs": int(pairs[a, b]),
            }
            for a, b in sorted(kept)
        ],
        columns=edges_cols,
    )
    return nodes, edges


def _tau_b(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign matrices of ``x`` and ``y`` (pairwise comparisons), as float32."""
    return (
        np.sign(x[:, None] - x[None, :]).astype(np.float32),
        np.sign(y[:, None] - y[None, :]).astype(np.float32),
    )


def kendall_permutation(
    x: np.ndarray, y: np.ndarray, n_perm: int = PERMUTATIONS, seed: int = 0
) -> tuple[float, float]:
    """Kendall tau-b of ``x`` and ``y`` with a two-sided permutation p-value.

    ``p = (1 + #{|tau*| >= |tau|}) / (1 + n_perm)`` over permutations of ``y``;
    ``(nan, nan)`` below three points or when one side is constant.
    """
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    sx, sy = _tau_b(np.asarray(x, float), np.asarray(y, float))
    scale = float(np.sqrt((sx**2).sum() * (sy**2).sum()))
    if scale == 0:
        return float("nan"), float("nan")
    tau = float((sx * sy).sum() / scale)
    rng = np.random.default_rng(seed)
    hits, chunk = 0, 256
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        perm = np.argsort(rng.random((m, n)), axis=1)
        shuffled = sy[perm[:, :, None], perm[:, None, :]]
        stat = (sx[None] * shuffled).sum(axis=(1, 2)) / scale
        hits += int((np.abs(stat) >= abs(tau) - 1e-9).sum())
    return tau, (1 + hits) / (1 + n_perm)


AGREEMENT_COLUMNS = [
    "benchmark",
    "method",
    "other",
    "tau",
    "p",
    "n_shared",
    "jaccard",
]


def order_agreement(
    frames: Frames, *, n_perm: int = PERMUTATIONS, seed: int = 0
) -> pd.DataFrame:
    """Kendall tau-b and edge Jaccard between the order graphs of every method pair.

    Per benchmark and ordered pair of methods (a pair appears twice, once each way):
    ``tau`` over the mean normalised rank of the shared nodes with its permutation
    ``p`` (`PERMUTATIONS`), ``jaccard`` of the two edge sets (NaN when both are empty).
    """
    rows = []
    for ctx in _contexts(frames):
        graphs = {m: order_graph(ctx.eps[ctx.eps["method"] == m]) for m in ctx.names}
        for a, b in itertools.permutations(ctx.names, 2):
            (na, ea), (nb, eb) = graphs[a], graphs[b]
            shared = na.merge(nb, on="achievement", suffixes=("_a", "_b"))
            tau, p = kendall_permutation(
                shared["rho_a"].to_numpy(), shared["rho_b"].to_numpy(), n_perm, seed
            )
            sa = set(zip(ea["source"], ea["target"], strict=True))
            sb = set(zip(eb["source"], eb["target"], strict=True))
            union = len(sa | sb)
            rows.append({
                "benchmark": ctx.budget,
                "method": a,
                "other": b,
                "tau": tau,
                "p": p,
                "n_shared": len(shared),
                "jaccard": len(sa & sb) / union if union else float("nan"),
            })
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)


def _order_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    method: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    labels: bool,
) -> None:
    """Nodes by mean rank (x) and tier (y), the reduced edges as arrows."""
    color = _color(th, method)
    pos = {
        r.achievement: (r.rho, TIERS.index(r.tier) if r.tier in TIERS else len(TIERS))
        for r in nodes.itertuples()
    }
    for e in edges.itertuples():
        arrow = ax.annotate(
            "",
            color=color,
            xy=pos[e.target],
            xytext=pos[e.source],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": th.lw * 0.6,
                "shrinkA": 3,
                "shrinkB": 3,
            },
        )
        arrow.set_gid(_gid("edge", budget, method, f"{e.source}>{e.target}"))
    for name, (x, y) in pos.items():
        (dot,) = ax.plot([x], [y], ls="", marker="o", ms=th.ms, color=color)
        dot.set_gid(_gid("node", budget, method, name))
        if labels:
            ax.annotate(
                name,
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=th.font,
                color=th.ink2,
            )
    _style(ax, th, grid="both")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(len(TIERS) - 0.5, -0.5)
    ax.set_yticks(range(len(TIERS)), [t.replace("_", " ") for t in TIERS])
    ax.tick_params(axis="y", length=0)
    _labels(ax, th, "mean normalised rank (0 first, 1 last)", "tier")


def plot_order_graph(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    method: str,
) -> Figure:
    """F7 ``order_<method>``: the achievement order graph of one method per benchmark.

    Nodes: achievements unlocked in 5% of the final stochastic eval episodes, x their
    mean normalised rank, y their tier; arrows: ``P(a before b) >= 0.9`` over 20
    co-occurring episodes after transitive reduction (`order_graph`). Node names are
    printed in the double-width and video versions.
    """
    th, name = _theme(theme, width), f"order_{method}"
    ctxs = [c for c in _contexts(frames) if method in c.names]
    if not ctxs:
        return _no_data(name, th, out_dir, f"no tuned runs of {method}")
    fig = _figure(th, len(ctxs) * (4 * th.pitch_mm + 38) + _legend_mm(th, 1))
    axes = _stack(fig, th, [1.0] * len(ctxs))
    notes = []
    for ax, ctx in zip(axes, ctxs, strict=True):
        nodes, edges = order_graph(ctx.eps[ctx.eps["method"] == method])
        _order_panel(ax, th, ctx.budget, method, nodes, edges, labels=th.wide)
        _title(ax, th, _panel_title(ctx.budget, f"{method} order graph"))
        if edges.empty:
            notes.append(
                f"{_budget_name(ctx.budget)}: no edge reaches P >= {ORDER_P} over "
                f"{ORDER_PAIRS} episodes."
            )
    notes.append("Arrow: a is unlocked before b in at least 90% of episodes with both.")
    if not th.wide:
        notes.append("Node names: double-width and video figures.")
    pending = _pending([c.budget for c in ctxs])
    _finish(fig, th, _method_handles(th, [method]), [*notes, *pending])
    return _save(fig, th, out_dir, name)


def plot_order_graphs(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> dict[str, Figure]:
    """`plot_order_graph` for every method of ``frames``."""
    methods = list(dict.fromkeys(frames["runs"]["method"]))
    return {
        m: plot_order_graph(frames, out_dir, theme, width=width, method=m)
        for m in methods
    }


def plot_order_agreement(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_perm: int = PERMUTATIONS,
    seed: int = 0,
) -> Figure:
    """F7 ``order_agreement``: Kendall tau-b between the methods' unlock orders.

    One matrix per benchmark (`order_agreement`); a cell is bold when its permutation
    p-value is below 0.05, blank when fewer than three nodes are shared.
    """
    th, name = _theme(theme, width), GAME_NAMES["F7"]
    table = order_agreement(frames, n_perm=n_perm, seed=seed)
    ctxs = _contexts(frames)
    if table.empty:
        return _no_data(name, th, out_dir, "no order graphs (need unlock steps)")
    fig = _figure(th, sum(len(c.names) * 7 + 32 for c in ctxs) + _legend_mm(th, 0))
    axes = _stack(fig, th, [float(len(c.names)) for c in ctxs])
    cmap = _cmap(th, *th.div)
    for ax, ctx in zip(axes, ctxs, strict=True):
        n = len(ctx.names)
        index = {m: i for i, m in enumerate(ctx.names)}
        cells = np.full((n, n), np.nan)
        sig = np.zeros((n, n), bool)
        for r in table[table["benchmark"] == ctx.budget].itertuples():
            cells[index[r.method], index[r.other]] = r.tau
            sig[index[r.method], index[r.other]] = r.p < TAU_ALPHA
        image = ax.imshow(cells, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        image.set_gid(_gid("tau", ctx.budget, "all"))
        for i, j in zip(*np.where(np.isfinite(cells)), strict=True):
            ax.text(
                j,
                i,
                f"{cells[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=th.font,
                color=_ink_on(cmap((cells[i, j] + 1) / 2)),
                fontweight="bold" if sig[i, j] else "normal",
                gid="tau-cell/bold" if sig[i, j] else "tau-cell/plain",
            )
        ax.set_xticks(range(n), ctx.names, rotation=90)
        ax.set_yticks(range(n), ctx.names)
        ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _labels(ax, th, "method", "method")
        _title(ax, th, _panel_title(ctx.budget, "Kendall tau-b of the unlock order"))
    note = (
        f"Mean normalised rank of the shared order-graph nodes; bold: permutation "
        f"p < {TAU_ALPHA} ({n_perm:,} permutations). Descriptive: no cell ranks "
        "methods."
    )
    _finish(fig, th, notes=[note, *_pending([c.budget for c in ctxs])])
    return _save(fig, th, out_dir, name)


# --------------------------------------------------------------------------- F8

FLOOR_NAMES = tuple(name for name, _ in FLOORS)
LADDER_NAMES = tuple(t.lower() for t in TECH_LADDER)
FUNNEL_COLUMNS = ["method", "kind", "step", "share"]


def _funnel_flags(eps: pd.DataFrame, kind: str, step: int) -> pd.Series:
    """Whether each episode reached ladder step (``tech``) or floor ``step``."""
    column = "tech_depth" if kind == "tech" else "max_floor"
    return (_column(eps, column) >= step).where(_column(eps, column).notna())


def funnel_shares(eps: pd.DataFrame) -> pd.DataFrame:
    """Share of episodes with tech depth >= k (``tech``, k = 1..10) and max floor >= f.

    One row per (method, kind, step): the mean over seeds of each seed's share. The
    shares of a kind never increase with the step.
    """
    rows = []
    for method, g in eps.groupby("method", sort=False):
        for kind, steps in (
            ("tech", range(1, len(LADDER_NAMES) + 1)),
            ("floor", range(len(FLOOR_NAMES))),
        ):
            for step in steps:
                share = _funnel_flags(g, kind, step).groupby(g["run_id"]).mean()
                rows.append({
                    "method": method,
                    "kind": kind,
                    "step": step,
                    "share": float(share.mean()),
                })
    return pd.DataFrame(rows, columns=FUNNEL_COLUMNS)


def _funnel_cats(ctx: Ctx, kind: str) -> Cats:
    steps = (
        range(1, len(LADDER_NAMES) + 1) if kind == "tech" else range(len(FLOOR_NAMES))
    )
    labels = LADDER_NAMES if kind == "tech" else FLOOR_NAMES

    def values(method: str, key: str) -> np.ndarray:
        flags = _funnel_flags(ctx.eps, kind, int(key))
        return _per_run(ctx, flags, "mean").get(method, np.empty(0))

    return Cats([str(s) for s in steps], list(labels), values)


def _funnel_panel(kind: str) -> Callable[[Axes, Theme, Ctx, list[str]], None]:
    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        _dot_panel(
            ax,
            th,
            ctx,
            f"funnel-{kind}",
            _funnel_cats(ctx, kind),
            "share of episodes",
            notes,
            how="mean",
        )
        ax.set_xlim(-0.02, 1.02)

    return draw


def plot_funnel(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F8 ``funnel``: progress at the final checkpoint, tech ladder and floors.

    Per method the share of eval episodes with tech depth at least k on the ladder
    wood to diamond pickaxe, and with max floor at least f; mean over seeds with a
    95% bootstrap interval, so a row never exceeds the one above it.
    """
    panels = [
        Panel(
            "tech ladder reached",
            _funnel_panel("tech"),
            lambda c: _dot_mm(c, len(LADDER_NAMES)),
        ),
        Panel(
            "deepest floor reached",
            _funnel_panel("floor"),
            lambda c: _dot_mm(c, len(FLOOR_NAMES)),
        ),
    ]
    return _draw(
        GAME_NAMES["F8"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


def _run_series(
    curves: pd.DataFrame, runs: pd.Index, metrics: Sequence[str], how: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run -> (env steps, series): ``sum`` / 100 of all ``metrics``, or their mean."""
    sub = curves[curves["run_id"].isin(runs) & curves["metric"].isin(metrics)]
    if sub.empty:
        return {}
    wide = sub.pivot_table(
        index=["run_id", "env_steps"], columns="metric", values="value"
    )
    if how == "sum":
        if set(metrics) - set(wide.columns):
            return {}
        total = wide[list(metrics)].dropna().sum(axis=1) / 100.0
    else:
        total = wide.mean(axis=1)
    return {
        str(r): (g.index.get_level_values(1).to_numpy(float), g.to_numpy(float))
        for r, g in total.groupby(level=0)
    }


def _series_panel(
    kind: str,
    metrics: Callable[[Ctx], Sequence[str]],
    how: str,
    ylabel: str,
) -> Drawer:
    """A mean curve with a bootstrap band per method from ``metrics`` of its runs."""

    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        grid = step_grid(_budget_steps(ctx.budget))
        missing = []
        for method in ctx.names:
            ids = pd.Index(ctx.runs.loc[ctx.runs["method"] == method, "run_id"])
            series = _run_series(ctx.curves, ids, metrics(ctx), how)
            if not series:
                missing.append(method)
                continue
            band = curve_band(list(series.values()), grid, "mean", ctx.n_boot, ctx.seed)
            style = _style_of(th, method)
            x = band["env_steps"].to_numpy()
            fill = ax.fill_between(
                x, band["lo"], band["hi"], color=style["color"], alpha=0.15, lw=0
            )
            fill.set_gid(_gid("band", ctx.budget, method))
            (line,) = ax.plot(
                x,
                band["estimate"],
                lw=th.lw,
                ls=style["ls"],
                color=style["color"],
                marker=style["marker"],
                ms=th.ms,
                markevery=max(1, len(x) // 7),
            )
            line.set_gid(_gid(kind, ctx.budget, method))
        if missing:
            notes.append(
                f"{_budget_name(ctx.budget)}: {', '.join(missing)} {NOT_LOGGED} (R78)."
            )
        if len(missing) == len(ctx.names):
            _empty(ax, th, f"{NOT_LOGGED} for every method")
            return
        _style(ax, th, grid="both")
        ax.xaxis.set_major_formatter(FuncFormatter(_steps_label))
        ax.set_xlim(0, _budget_steps(ctx.budget))
        _labels(ax, th, "env steps", ylabel)

    return draw


def plot_funnel_training(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F8 ``funnel_training``: how far the funnel reaches during training.

    The mean tech depth (sum over k of the share ``train/funnel/depth_k``) and the mean
    deepest floor (sum over f >= 1 of ``train/funnel/floor_f``), mean over seeds with
    a bootstrap band; the reused PPO and PPO-RNN runs did not log them (R78).
    """
    depth = [f"train/funnel/depth_{k}" for k in range(1, len(LADDER_NAMES) + 1)]
    floor = [f"train/funnel/floor_{f}" for f in range(1, len(FLOOR_NAMES))]
    panels = [
        Panel(
            "mean tech depth in training",
            _series_panel("depth", lambda _c: depth, "sum", "mean tech depth"),
            lambda _c: 40.0 + PANEL_MM,
            narrow=False,
        ),
        Panel(
            "mean deepest floor in training",
            _series_panel("floor", lambda _c: floor, "sum", "mean deepest floor"),
            lambda _c: 40.0 + PANEL_MM,
            narrow=False,
        ),
    ]
    return _draw(
        GAME_NAMES["F8T"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# --------------------------------------------------------------------------- F9

CAUSE_SLOTS = 4
"""Causes drawn apart in a death bar (a ramp step each); the rest are ``other``."""
OTHER = "other"


def _death_shares(ctx: Ctx) -> pd.DataFrame:
    """Method x cause shares of deaths at the final checkpoint (rows: methods)."""
    table = death_causes(ctx.eps)
    final = table[table["scope"] == "final"]
    final = final[final["method"].isin(ctx.names)]
    return final.pivot_table(
        index="method", columns="cause", values="share", aggfunc="first"
    )


def _death_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Stacked cause-of-death shares per method; the largest causes get a ramp step."""
    shares = _death_shares(ctx)
    if shares.empty:
        _empty(ax, th, "no eval episodes")
        return
    shares = shares[[c for c in shares.columns if not c.startswith("(")]].fillna(0.0)
    order = list(shares.sum().sort_values(ascending=False, kind="stable").index)
    top = order[:CAUSE_SLOTS]
    groups = [*top, OTHER] if len(order) > CAUSE_SLOTS else top
    for y, method in enumerate(ctx.names):
        ax.barh(
            y,
            0.025,
            left=-0.04,
            color=_color(th, method),
            height=0.7,
            gid=_gid("swatch", ctx.budget, method),
        )
        if method not in shares.index or shares.loc[method].sum() == 0:
            ax.text(
                0.0, y, "(no deaths)", va="center", fontsize=th.font, color=th.muted
            )
            continue
        left = 0.0
        for k, cause in enumerate(groups):
            value = (
                float(shares.loc[method, order[CAUSE_SLOTS:]].sum())
                if cause == OTHER
                else float(shares.loc[method].get(cause, 0.0))
            )
            ax.barh(
                y,
                value,
                left=left,
                color=th.seq[k],
                height=0.7,
                edgecolor=th.surface,
                linewidth=0.4,
                gid=_gid("death", ctx.budget, method, cause),
            )
            left += value
    _style(ax, th)
    ax.set_xlim(-0.05, 1.0)
    ax.set_yticks(range(len(ctx.names)), ctx.names)
    ax.set_ylim(len(ctx.names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor=th.seq[k], label=c.replace("_", " "))
        for k, c in enumerate(groups)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28 if th.name == "report" else -0.2),
        ncol=min(len(groups), 3 if not th.wide else 5),
        fontsize=th.font,
        frameon=False,
        labelcolor=th.ink2,
    )
    _labels(ax, th, "share of deaths")


def plot_deaths(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F9 ``deaths``: cause-of-death shares per method (`hard.death_causes`).

    Stacked bars at the final checkpoint's stochastic eval episodes; the four largest
    causes of the benchmark have a ramp step each, the rest are ``other``; a method
    that never died says so. A thin bar in the method colour marks each row.
    """
    panel = Panel(
        "cause of death",
        _death_panel,
        lambda c: len(c.names) * 5.5 + PANEL_MM + 14,
    )
    return _draw(
        GAME_NAMES["F9"],
        frames,
        out_dir,
        theme,
        width,
        [panel],
        n_boot=n_boot,
        seed=seed,
    )


def _stall_ccdf_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    """Share of stalls at least that long (hard.stall_ccdf), one line per method."""
    table = _stall_table(ctx.eps, "stochastic")
    ccdf = stall_ccdf(table[table["scope"] == "final"]) if len(table) else table
    drawn = set()
    for method in ctx.names:
        mine = ccdf[ccdf["method"] == method] if len(ccdf) else ccdf
        if mine.empty:
            continue
        style = _style_of(th, method)
        (line,) = ax.step(
            mine["length"],
            mine["ccdf"],
            where="post",
            lw=th.lw,
            ls=style["ls"],
            color=style["color"],
        )
        line.set_gid(_gid("ccdf", ctx.budget, method))
        drawn.add(method)
    absent = [m for m in ctx.names if m not in drawn]
    if not drawn:
        _empty(ax, th, "no stalls in the eval episodes")
        return
    if absent:
        notes.append(f"{_budget_name(ctx.budget)}: {', '.join(absent)} had no stall.")
    _style(ax, th, grid="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_num_text))
    ax.yaxis.set_major_formatter(FuncFormatter(_num_text))
    _labels(ax, th, "stall length (steps)", "share of stalls at least this long")


def _escape_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    """Share of stalls that ended in an unlock (hard.stall_stats), bar per method."""
    stats = stall_stats(ctx.eps, "stochastic", n=ctx.n_boot)
    rate = (
        stats.set_index("method")["escape_rate"]
        if len(stats)
        else pd.Series(dtype=float)
    )
    drawn = []
    for y, method in enumerate(ctx.names):
        value = float(rate.get(method, np.nan))
        if not np.isfinite(value):
            continue
        ax.barh(
            y,
            value,
            color=_color(th, method),
            height=0.7,
            gid=_gid("escape", ctx.budget, method),
        )
        drawn.append(method)
    if not drawn:
        _empty(ax, th, "no stalls in the eval episodes")
        return
    absent = [m for m in ctx.names if m not in drawn]
    if absent:
        notes.append(f"{_budget_name(ctx.budget)}: {', '.join(absent)} had no stall.")
    _style(ax, th)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(ctx.names)), ctx.names)
    ax.set_ylim(len(ctx.names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    _labels(ax, th, "escape rate (stalls ending in an unlock)")


def plot_stalls(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F9 ``stalls``: stall-length CCDF and escape rate per method (`hard`)."""
    panels = [
        Panel(
            "stall lengths", _stall_ccdf_panel, lambda _c: 42.0 + PANEL_MM, narrow=False
        ),
        Panel(
            "escape rate",
            _escape_panel,
            lambda c: len(c.names) * 4.0 + PANEL_MM,
        ),
    ]
    return _draw(
        GAME_NAMES["F9S"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# -------------------------------------------------------------------------- F10

SPLIT_COLUMNS = ["method", "milestone", "n", "share", "q1", "median", "q3"]


def split_table(eps: pd.DataFrame) -> pd.DataFrame:
    """Per method and milestone: episodes, share reaching it, quartiles of its steps.

    Over all the method's episodes; ``split/<milestone>`` is -1 where never reached
    and those episodes are left out of the quartiles (NaN with none).
    """
    rows = []
    for method, g in eps.groupby("method", sort=False):
        for m in MILESTONES:
            col = f"split/{m}"
            if col not in g.columns:
                continue
            v = g[col].to_numpy(float)
            hit = v[v >= 0]
            q1, med, q3 = (
                np.percentile(hit, [25, 50, 75]) if hit.size else (np.nan,) * 3
            )
            rows.append({
                "method": method,
                "milestone": m,
                "n": int(hit.size),
                "share": float(hit.size / len(v)),
                "q1": q1,
                "median": med,
                "q3": q3,
            })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def _split_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Box plots of the steps to each milestone per method, share reaching as text."""
    table = split_table(ctx.eps)
    rows = [m for m in MILESTONES if (table["milestone"] == m).any()]
    if table.empty or not rows:
        _empty(ax, th, f"{NOT_RECORDED}: no split columns")
        return
    offsets = dict(zip(ctx.names, _dodge(len(ctx.names)), strict=True))
    for r in table.itertuples():
        if not r.n:
            continue
        y = rows.index(r.milestone) + offsets[r.method]
        stats = [
            {
                "med": r.median,
                "q1": r.q1,
                "q3": r.q3,
                "whislo": r.q1,
                "whishi": r.q3,
                "fliers": [],
            }
        ]
        parts = ax.bxp(
            stats,
            positions=[y],
            widths=DODGE / len(ctx.names) * 0.9,
            orientation="horizontal",
            patch_artist=True,
            showfliers=False,
            showcaps=False,
            boxprops={
                "facecolor": _color(th, r.method),
                "edgecolor": _color(th, r.method),
            },
            medianprops={"color": th.ink, "linewidth": th.lw},
            whiskerprops={"color": _color(th, r.method)},
            manage_ticks=False,
        )
        parts["boxes"][0].set_alpha(BOX_ALPHA + 0.3)
        parts["boxes"][0].set_gid(_gid("split", ctx.budget, r.method, r.milestone))
    for i, m in enumerate(rows):
        share = table[table["milestone"] == m]["share"]
        ax.annotate(
            f"{share.min():.0%}-{share.max():.0%}",
            xy=(1.0, i),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=th.font,
            color=th.muted,
            annotation_clip=False,
        ).set_gid(_gid("share", ctx.budget, "all", m))
    _style(ax, th)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_label))
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    _labels(ax, th, "steps to the first unlock (log); right: share reaching it")


def plot_splits(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F10 ``splits``: speedrun splits, box plots of steps to each milestone.

    Rows: the tech ladder then the ``enter_*`` achievements; a box per method spans the
    quartiles of the episodes that reached it, a line marks the median; the text at
    the right is the lowest and highest share of a method's episodes reaching it.
    """
    panel = Panel(
        "speedrun splits",
        _split_panel,
        lambda c: len(MILESTONES) * (len(c.names) * 0.9 + 2.0) + PANEL_MM,
    )
    return _draw(
        GAME_NAMES["F10"],
        frames,
        out_dir,
        theme,
        width,
        [panel],
        n_boot=n_boot,
        seed=seed,
    )


# ----------------------------------------------------------------- F11, F12, F14


def _dots(
    kind: str,
    cats: Callable[[Ctx], Cats],
    xlabel: str,
    *,
    how: str = "median",
    log: bool = False,
    missing: str = NOT_RECORDED,
) -> Drawer:
    """A dot panel of ``cats(ctx)`` as a panel drawer."""

    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        _dot_panel(
            ax,
            th,
            ctx,
            kind,
            cats(ctx),
            xlabel,
            notes,
            how=how,
            log=log,
            missing=missing,
        )

    return draw


def _coverage_cats(ctx: Ctx) -> Cats:
    """Floors (``coverage/<floor name>``) in game order, then any unknown name."""
    found = [
        c.removeprefix("coverage/")
        for c in ctx.eps.columns
        if c.startswith("coverage/")
    ]
    rank = {n: i for i, n in enumerate(FLOOR_NAMES)}
    keys = sorted(found, key=lambda n: (rank.get(n, len(rank)), n))
    return Cats(keys, keys, _per_seed_of(ctx, "coverage/{}"))


def _tiles_cats(ctx: Ctx) -> Cats:
    return Cats(
        ["new_tiles_per_100"],
        ["new tiles per 100 steps"],
        _per_seed_of(ctx, "{}"),
    )


def plot_coverage(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F11 ``coverage``: share of each floor seen, and new tiles per 100 steps.

    Per method the median over seeds of each seed's median episode (floors never
    visited are skipped), with a bootstrap interval over seeds; the new-tile rate is
    the episode mean the replay stores, not a series over the episode.
    """
    panels = [
        Panel(
            "exploration coverage per floor",
            _dots("coverage", _coverage_cats, "share of the floor seen"),
            lambda c: _dot_mm(
                c, max(1, sum(x.startswith("coverage/") for x in c.eps.columns))
            ),
        ),
        Panel(
            "new tiles",
            _dots("tiles", _tiles_cats, "new tiles per 100 steps"),
            lambda c: _dot_mm(c, 1),
        ),
    ]
    return _draw(
        GAME_NAMES["F11"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


def _share_cats(ctx: Ctx) -> Cats:
    keys = [f"{kind}_share/{s}" for s in STATS for kind in ("low", "zero")]
    labels = [f"{s} {kind}" for s in STATS for kind in ("low", "zero")]
    return Cats(keys, labels, _per_seed_of(ctx, "{}"))


def _level_cats(ctx: Ctx) -> Cats:
    pairs = (("food", "eat"), ("drink", "drink"), ("energy", "sleep"))
    keys = [f"survival/level/{s}" for s, _ in pairs]
    labels = [f"{s} when you {act}" for s, act in pairs]
    return Cats(keys, labels, _per_seed_of(ctx, "{}"))


def plot_survival(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F12 ``survival``: time low or empty, and the stat level at eat, drink, sleep.

    The shares of steps with a stat low and at zero, and the mean stat value in the
    state before each eat, drink and sleep start (replay only: ``not recorded`` where
    the export has no trace); median over seeds with a bootstrap interval.
    """
    panels = [
        Panel(
            "steps with a stat low or zero",
            _dots("share", _share_cats, "share of steps"),
            lambda c: _dot_mm(c, 2 * len(STATS)),
        ),
        Panel(
            "stat level at the action",
            _dots("level", _level_cats, "mean stat value"),
            lambda c: _dot_mm(c, len(STATS)),
        ),
    ]
    return _draw(
        GAME_NAMES["F12"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# -------------------------------------------------------------------------- F13


def _kill_columns(eps: pd.DataFrame) -> list[str]:
    """``kills/<creature>`` columns, the passive kills last (never summed with them)."""
    cols = [c for c in eps.columns if c.startswith("kills/") and c != "kills/passive"]
    return [*cols, "kills/passive"] if "kills/passive" in eps.columns else cols


def _kills_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Heat map of mean kills per episode: creature types down, methods across."""
    cols = _kill_columns(ctx.eps)
    matrix = _mean_by_method(ctx, cols)
    if not cols or not np.isfinite(matrix).any():
        _empty(ax, th, f"{NOT_RECORDED}: no kill columns")
        return
    _heat(
        ax,
        th,
        ctx,
        "kills",
        [c.removeprefix("kills/") for c in cols],
        matrix,
        annotate=len(cols) <= 12,
    )
    _labels(ax, th, None, "creature (passive last)")


def _damage_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Damage dealt against damage taken per episode, a dot per method."""
    dealt = _per_run(ctx, _column(ctx.eps, "damage_dealt"), "mean")
    taken = _per_run(ctx, _column(ctx.eps, "damage_taken"), "mean")
    drawn = []
    for method in ctx.names:
        if method not in dealt or method not in taken:
            continue
        x, y = float(np.nanmean(taken[method])), float(np.nanmean(dealt[method]))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        style = _style_of(th, method)
        (dot,) = ax.plot(
            [x],
            [y],
            ls="",
            marker=style["marker"],
            ms=th.ms * 1.3,
            color=style["color"],
        )
        dot.set_gid(_gid("damage", ctx.budget, method))
        drawn.append(method)
    if not drawn:
        _empty(ax, th, f"{NOT_RECORDED}: no damage columns")
        return
    _style(ax, th, grid="both")
    _labels(ax, th, "damage taken per episode", "damage dealt per episode")


def _fight_cats(ctx: Ctx) -> Cats:
    counts = (
        ctx.eps["fights"].map(lambda c: len(_cells(c))) if "fights" in ctx.eps else None
    )
    series = counts if counts is not None else pd.Series(np.nan, index=ctx.eps.index)

    def values(method: str, _key: str) -> np.ndarray:
        return _per_run(ctx, series.astype(float), "mean").get(method, np.empty(0))

    return Cats(["fights"], ["fights per episode"], values)


def plot_combat(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F13 ``combat``: kills per creature type, damage dealt against taken, fights.

    Kills are the mean per episode over seeds (hostile types, then passive kills
    apart); damage is the episode mean per method; fights are the episode mean of the
    stored fight records, with a bootstrap interval over seeds.
    """
    panels = [
        Panel(
            "kills per episode",
            _kills_panel,
            lambda c: (len(_kill_columns(c.eps)) + 1) * 2.9 + PANEL_MM + HEAT_TICK_MM,
            narrow=False,
        ),
        Panel("damage", _damage_panel, lambda _c: 42.0 + PANEL_MM, narrow=False),
        Panel(
            "fights",
            _dots("fights", _fight_cats, "fights per episode"),
            lambda c: _dot_mm(c, 1),
        ),
    ]
    return _draw(
        GAME_NAMES["F13"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# -------------------------------------------------------------------------- F14


def _action_columns(eps: pd.DataFrame) -> list[str]:
    return [c for c in eps.columns if c.startswith("action_count/")]


def _actions_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Heat map of the share of steps per action (not forced), methods across."""
    cols = _action_columns(ctx.eps)
    if not cols:
        _empty(ax, th, f"{NOT_RECORDED}: no action counts")
        return
    counts = ctx.eps[cols].to_numpy(float)
    shares = pd.DataFrame(
        counts / np.maximum(counts.sum(axis=1, keepdims=True), 1),
        columns=cols,
        index=ctx.eps.index,
    )
    shown = ctx._replace(eps=ctx.eps.assign(**shares.to_dict("series")))
    matrix = _mean_by_method(shown, cols)
    _heat(
        ax, th, ctx, "actions", [c.removeprefix("action_count/") for c in cols], matrix
    )
    _labels(ax, th, None, "action")


def _last_by_method(ctx: Ctx, metric: str) -> dict[str, np.ndarray]:
    """Method -> the last logged value of ``metric`` in each of its runs."""
    sub = ctx.curves[ctx.curves["metric"] == metric].sort_values("env_steps")
    if sub.empty:
        return {}
    last = sub.groupby("run_id")["value"].last()
    method = ctx.runs.set_index("run_id")["method"].reindex(last.index)
    return {str(m): g.to_numpy(float) for m, g in last.groupby(method)}


def _entropy_cats(ctx: Ctx) -> Cats:
    metric = ALIASES["action entropy"][0]
    training = _last_by_method(ctx, metric)
    evaluation = _per_run(ctx, _column(ctx.eps, "action_entropy"), "median")

    def values(method: str, key: str) -> np.ndarray:
        src = evaluation if key == "eval" else training
        return src.get(method, np.empty(0))

    unlogged = [m for m in ctx.names if m not in training]
    return Cats(["eval", "train"], ["eval episodes", "training"], values, unlogged)


def _wasted_cats(ctx: Ctx) -> Cats:
    return Cats(["wasted_share"], ["wasted share"], _per_seed_of(ctx, "{}"))


def plot_actions(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F14 ``actions``: action usage heat map, entropy and wasted share per method.

    Heat map: the mean over seeds of the share of non-forced steps per action. Entropy
    (nats) is the eval-episode action entropy and the last logged training action
    entropy (``not logged`` for the reused PPO and PPO-RNN runs, R78); the wasted share
    is the eval episodes' median.
    """
    panels = [
        Panel(
            "share of steps per action",
            _actions_panel,
            lambda c: (
                max(1, len(_action_columns(c.eps))) * 2.9 + PANEL_MM + HEAT_TICK_MM
            ),
            narrow=False,
        ),
        Panel(
            "action entropy",
            _dots("entropy", _entropy_cats, "entropy (nats)", missing=NOT_LOGGED),
            lambda c: _dot_mm(c, 2),
        ),
        Panel(
            "wasted share",
            _dots("wasted", _wasted_cats, "share of steps wasted"),
            lambda c: _dot_mm(c, 1),
        ),
    ]
    return _draw(
        GAME_NAMES["F14"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# ------------------------------------------------------------------------ F15


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
"""Marker of the first, second ... resource in the gathered-against-used panel."""


def _resource_names(eps: pd.DataFrame) -> list[str]:
    return [
        c.removeprefix("gathered/") for c in eps.columns if c.startswith("gathered/")
    ]


def _use_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Mean gathered against mean used per resource (marker) and method (colour)."""
    names = _resource_names(ctx.eps)
    drawn, top = 0, 0.0
    for k, res in enumerate(names):
        got = _per_run(ctx, _column(ctx.eps, f"gathered/{res}"), "mean")
        used = _per_run(ctx, _column(ctx.eps, f"used/{res}"), "mean")
        for method in ctx.names:
            if method not in got or method not in used:
                continue
            x, y = float(np.nanmean(got[method])), float(np.nanmean(used[method]))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            (dot,) = ax.plot(
                [x],
                [y],
                ls="",
                marker=MARKERS[k % len(MARKERS)],
                ms=th.ms,
                color=_color(th, method),
            )
            dot.set_gid(_gid("use", ctx.budget, method, res))
            drawn += 1
            top = max(top, x, y)
    if not drawn:
        _empty(ax, th, f"{NOT_RECORDED}: no replay trace")
        return
    ax.plot([0, top], [0, top], color=th.muted, lw=0.5, ls="--", gid="diagonal")
    _style(ax, th, grid="both")
    _labels(ax, th, "gathered per episode", "used per episode")
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=MARKERS[k % len(MARKERS)],
            color=th.ink2,
            ms=th.ms,
            label=r,
        )
        for k, r in enumerate(names)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=th.font,
        frameon=False,
        labelcolor=th.ink2,
    )


def _time_cats(ctx: Ctx) -> Cats:
    steps = _column(ctx.eps, "steps_per_achievement")
    w2d = _column(ctx.eps, "wood_to_diamond")
    w2d = w2d.where(w2d >= 0)

    def values(method: str, key: str) -> np.ndarray:
        series = steps if key == "steps" else w2d
        return _per_run(ctx, series.astype(float), "median").get(method, np.empty(0))

    return Cats(["steps", "w2d"], ["steps per achievement", "wood to diamond"], values)


def _diamond_note(ctx: Ctx) -> str:
    w2d = _column(ctx.eps, "wood_to_diamond")
    share = (w2d >= 0).groupby(ctx.eps["method"]).mean()
    text = ", ".join(f"{m} {share[m]:.0%}" for m in ctx.names if m in share.index)
    return f"{_budget_name(ctx.budget)} episodes reaching diamond: {text}."


def plot_resources(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F15 ``resources``: gathered against used, steps per achievement, wood to diamond.

    The first panel is the replay's inventory trace (``not recorded`` without it);
    the time panels are medians of seed medians, wood to diamond over the episodes
    that reached diamond (their share is in the note).
    """
    panels = [
        Panel(
            "gathered and used", _use_panel, lambda _c: 46.0 + PANEL_MM, narrow=False
        ),
        Panel(
            "time to progress",
            _dots("time", _time_cats, "steps (log)", log=True),
            lambda c: _dot_mm(c, 2),
        ),
    ]
    notes = [
        _diamond_note(c)
        for c in _contexts(frames)
        if "wood_to_diamond" in c.eps.columns
    ]
    return _draw(
        GAME_NAMES["F15"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        notes=notes,
        n_boot=n_boot,
        seed=seed,
    )


# ------------------------------------------------------------------------ F16


def _advanced_metrics(_ctx: Ctx) -> list[str]:
    return [f"train/ach/{a.lower()}" for a in ADVANCED_ACHIEVEMENTS]


def _advanced_cats(ctx: Ctx) -> Cats:
    keys = _advanced_metrics(ctx)
    last = {k: _last_by_method(ctx, k) for k in keys}

    def values(method: str, key: str) -> np.ndarray:
        return last[key].get(method, np.empty(0))

    return Cats(keys, [a.lower() for a in ADVANCED_ACHIEVEMENTS], values)


def plot_advanced(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F16 ``advanced``: adoption of the advanced mechanics over training.

    From the ``train/ach/*`` rates (% of episodes) of the nine advanced achievements:
    the mean rate over them per method with a bootstrap band, and each one's final
    rate. The reused PPO and PPO-RNN runs did not log them (R78).
    """
    panels = [
        Panel(
            "mean advanced-achievement rate",
            _series_panel("advanced", _advanced_metrics, "mean", "% of episodes"),
            lambda _c: 40.0 + PANEL_MM,
            narrow=False,
        ),
        Panel(
            "final rate per achievement",
            _dots("adv", _advanced_cats, "% of episodes", missing=NOT_LOGGED),
            lambda c: _dot_mm(c, len(ADVANCED_ACHIEVEMENTS)),
        ),
    ]
    return _draw(
        GAME_NAMES["F16"],
        frames,
        out_dir,
        theme,
        width,
        panels,
        n_boot=n_boot,
        seed=seed,
    )


# ------------------------------------------------------------------------ F17

RADAR_COLUMNS = ["benchmark", "method", "axis", "raw", "scaled"]


def _radar_raw(eps: pd.DataFrame) -> pd.DataFrame:
    """The eight radar axes per run: the mean over its episodes (metrics plan).

    Explorer: mean coverage of visited floors; Survivor: 1 - the largest of the three
    stats' zero shares (a lower bound of the share of steps with any stat zero, the
    export keeps them apart); Fighter: hostile kills per 1000 steps; Crafter: tech
    depth; Diver: max floor; Efficient: achievements per 1000 steps; Mage: share of
    the advanced mechanics used; Precise: 1 - wasted share. NaN where a column is
    missing.
    """
    length = _column(eps, "length").replace(0, np.nan)
    cover = [c for c in eps.columns if c.startswith("coverage/")]
    zero = [c for c in eps.columns if c.startswith("zero_share/")]
    kills = [c for c in eps.columns if c.startswith("kills/") and c != "kills/passive"]
    adv = [c for c in eps.columns if c.startswith("adv/")]
    nan = pd.Series(np.nan, index=eps.index)
    axes = {
        "explorer": eps[cover].mean(axis=1) if cover else nan,
        "survivor": 1 - eps[zero].max(axis=1) if zero else nan,
        "fighter": eps[kills].sum(axis=1) / length * 1000 if kills else nan,
        "crafter": _column(eps, "tech_depth").astype(float),
        "diver": _column(eps, "max_floor").astype(float),
        "efficient": _column(eps, "n_achievements") / length * 1000,
        "mage": (eps[adv] > 0).mean(axis=1) if adv else nan,
        "precise": 1 - _column(eps, "wasted_share"),
    }
    return (
        pd
        .DataFrame(axes)[list(RADAR_AXES)]
        .groupby([eps["method"].to_numpy(), eps["run_id"].to_numpy()])
        .mean()
    )


def radar_table(frames: Frames) -> pd.DataFrame:
    """Raw and min-max scaled radar axes per benchmark and method.

    ``raw`` is the mean over a method's seeds of each seed's episode mean;
    ``scaled`` is min-max over all the methods and controls of the benchmark (0.5 where
    every method is equal, NaN where an axis is not recorded).
    """
    rows: list[dict[str, Any]] = []
    for ctx in _contexts(frames):
        per_run = _radar_raw(ctx.eps)
        raw = per_run.groupby(level=0).mean().reindex(ctx.names)
        lo, hi = raw.min(), raw.max()
        span = (hi - lo).replace(0, np.nan)
        scaled = ((raw - lo) / span).fillna(0.5).where(raw.notna())
        rows.extend(
            {
                "benchmark": ctx.budget,
                "method": method,
                "axis": axis,
                "raw": float(raw.loc[method, axis]),
                "scaled": float(scaled.loc[method, axis]),
            }
            for method, axis in itertools.product(ctx.names, RADAR_AXES)
        )
    return pd.DataFrame(rows, columns=RADAR_COLUMNS)


def _radar_axes(ax: Axes, th: Theme, labels: Sequence[str]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.set_facecolor(th.surface)
    ax.set_xticks(angles, list(labels))
    ax.set_yticks([0.5, 1.0], ["", ""])
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", colors=th.ink2, labelsize=th.font, pad=1)
    ax.grid(color=th.grid, lw=0.5)
    ax.spines["polar"].set_color(th.muted)
    ax.spines["polar"].set_linewidth(0.5)


def _radar_cell(
    ax: Axes, th: Theme, budget: str, method: str, values: np.ndarray, labels: list[str]
) -> None:
    """One method's polygon on shared axes; an axis not recorded is drawn at 0."""
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    shown = np.nan_to_num(values)
    x, y = np.r_[angles, angles[:1]], np.r_[shown, shown[:1]]
    style = _style_of(th, method)
    _radar_axes(ax, th, labels)
    (line,) = ax.plot(x, y, color=style["color"], lw=th.lw, ls=style["ls"])
    line.set_gid(_gid("radar", budget, method))
    (fill,) = ax.fill(x, y, color=style["color"], alpha=0.2, lw=0)
    fill.set_gid(_gid("radar-fill", budget, method))
    ax.set_title(method, loc="left", fontsize=th.title, color=th.ink)


def plot_radar(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    budget: str = "1m",
) -> Figure:
    """F17 ``radar_<budget>``: one playstyle radar per method on shared axes.

    Eight axes (`RADAR_AXES`) from the final stochastic eval episodes, min-max scaled
    over all methods and controls of the benchmark (`radar_table`); narrow report
    panels print the first four letters of an axis name.
    """
    th, name = _theme(theme, width), f"radar_{budget}"
    present = [c.budget for c in _contexts(frames)]
    if budget not in present:
        return _placeholder(name, th, out_dir, PENDING.format(budget.upper()))
    ctx = next(c for c in _contexts(frames) if c.budget == budget)
    table = radar_table(frames)
    table = table[table["benchmark"] == budget]
    narrow = th.name == "report" and not th.wide
    cols = 2 if narrow else 3
    rows = ceil(len(ctx.names) / cols)
    fig = _figure(th, rows * (th.width * 25.4 / cols * 0.9) + _legend_mm(th, 0))
    grid = fig.subplots(rows, cols, squeeze=False, subplot_kw={"projection": "polar"})
    labels = [a[:SHORT_AXES].title() if narrow else a for a in RADAR_AXES]
    notes = []
    for k, ax in enumerate(grid.ravel()):
        if k >= len(ctx.names):
            ax.set_visible(False)
            continue
        method = ctx.names[k]
        mine = table[table["method"] == method].set_index("axis")["scaled"]
        values = mine.reindex(RADAR_AXES).to_numpy(float)
        if np.isnan(values).any():
            lost = ", ".join(np.array(RADAR_AXES)[np.isnan(values)])
            notes.append(f"{method}: {NOT_RECORDED} on {lost}.")
        _radar_cell(ax, th, budget, method, values, labels)
    if narrow:
        names = ", ".join(f"{a[:SHORT_AXES].title()} {a}" for a in RADAR_AXES)
        notes.append(f"Axes: {names}.")
    notes.append(
        "Axes min-max scaled over the methods of the benchmark (0.5: all equal)."
    )
    _finish(fig, th, notes=[*_unique(notes), *_pending(present)])
    return _save(fig, th, out_dir, name)


def plot_radars(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> dict[str, Figure]:
    """`plot_radar` for every benchmark with runs."""
    return {
        c.budget: plot_radar(frames, out_dir, theme, width=width, budget=c.budget)
        for c in _contexts(frames)
    }


GAME_FIGURES: dict[str, Callable[..., Figure]] = {
    "F6": plot_race,
    "F7": plot_order_agreement,
    "F7G": plot_order_graph,
    "F8": plot_funnel,
    "F8T": plot_funnel_training,
    "F9": plot_deaths,
    "F9S": plot_stalls,
    "F10": plot_splits,
    "F11": plot_coverage,
    "F12": plot_survival,
    "F13": plot_combat,
    "F14": plot_actions,
    "F15": plot_resources,
    "F16": plot_advanced,
    "F17": plot_radar,
}
"""Figure id -> its function, in the plan's order (the results CLI loops over it)."""
