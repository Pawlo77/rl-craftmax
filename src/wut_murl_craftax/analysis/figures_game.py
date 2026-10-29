"""In-game figures F6-F17: race, order graphs, funnel, deaths, splits, radars.

Each ``plot_*`` follows `analysis.figures` (report at the stem's one width, dark
``video``, files ``<stem>[_video]``, `METHOD_COLORS`; a missing benchmark is pending).
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
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, NullFormatter

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
from ..visualisation import charts
from . import compare
from .figures import (
    BLOCK_TITLES,
    Figs,
    Shape,
    Theme,
    _budget_steps,
    _cmap,
    _color,
    _figure,
    _gid,
    _ink_on,
    _labels,
    _no_data,
    _save,
    _single_budget,
    _style,
    _style_of,
    _theme,
    method_handles,
    wants,
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
    "F17": "radar_{method}",
}
"""Figure id -> file stem or the prefix of its stems (one panel per file, such as
``funnel_tech``, ``race_steps_main``); ``{method}`` is filled per file."""
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
BOX_ALPHA = 0.4
DODGE_MM = 1.3
"""Millimetres per method inside one category row of a dot panel."""
GAP_MM = 1.5
"""Millimetres between two category rows of a dot panel."""
MAX_DOT_ROWS = 5
"""Category rows a dot panel holds (each row has a mark per method): a longer list is a
profile along x, a fixed-aspect figure."""
SPLIT_ROWS = 6
"""Most milestones of the speedrun splits (the earliest on the ladder)."""
HEAT_ROWS = 12
"""Most rows of a heat map (a creature or action list): the rest go to the caption."""
KILL_MIN = 0.05
"""Mean kills per episode below which a creature is not a row (the caption names it)."""


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


class FigPanel(NamedTuple):
    """One figure file of a family: its stem, caption and how it is drawn and sized."""

    stem: str
    title: str
    """Short name of the figure (the caption's first cell)."""
    caption: str
    """What is drawn and how to read it; the drawer's notes are appended."""
    draw: Drawer
    size: Callable[[Theme, Ctx], Figure]
    """The report figure of the right size class (the video is always 1080p)."""


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
    return method_handles(th, names)


def _ctx(frames: Frames, n_boot: int = BOOTSTRAP_N, seed: int = 0) -> Ctx | None:
    """The one benchmark's `Ctx`, None without tuned runs.

    Raises:
        ValueError: If ``frames`` hold several benchmarks (`results.draw` cuts them).
    """
    ctxs = _contexts(frames, n_boot, seed)
    if len(ctxs) > 1:
        msg = f"draw one benchmark at a time, got {[c.budget for c in ctxs]}"
        raise ValueError(msg)
    return ctxs[0] if ctxs else None


def _unique(notes: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(n for n in notes if n))


def _draw(
    frames: Frames,
    out_dir: Path,
    theme: str,
    width: str,
    panels: Sequence[FigPanel],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """One file per panel of the family, each with its caption (notes folded in)."""
    th = _theme(theme, width)
    ctx = _ctx(frames, n_boot, seed)
    if ctx is None:
        text = "no tuned runs for the in-game figures"
        return {panels[0].stem: _no_data(panels[0].stem, th, out_dir, text)}
    figs: Figs = {}
    for panel in panels:
        if not wants(th, panel.stem):
            continue
        fig = panel.size(th, ctx)
        ax = fig.add_subplot()
        notes: list[str] = []
        panel.draw(ax, th, ctx, notes)
        text = " ".join([panel.caption, *_unique(notes)])
        _save(fig, th, out_dir, panel.stem, title=panel.title, description=text)
        figs[panel.stem] = fig
    return figs


def _curve_size(th: Theme, _ctx: Ctx) -> Figure:
    return _figure(th, shape=Shape.CURVE)


def _square_size(th: Theme, _ctx: Ctx) -> Figure:
    return _figure(th, shape=Shape.SQUARE)


def _dot_pitch(ctx: Ctx) -> float:
    """Millimetres of one category row of a dot panel: a dodged mark per method."""
    return len(ctx.names) * DODGE_MM + GAP_MM


def _dots_size(
    rows: Callable[[Ctx], int], extra_mm: float = 0.0
) -> Callable[[Theme, Ctx], Figure]:
    """Size of a dot panel of ``rows(ctx)`` category rows, each holding every method."""

    def size(th: Theme, ctx: Ctx) -> Figure:
        n = max(1, rows(ctx))
        if n > MAX_DOT_ROWS:  # a long list is a profile along x (`_profile_panel`)
            return _figure(th, shape=Shape.CURVE)
        return charts.category_figure(th, n, extra_mm, _dot_pitch(ctx))

    return size


def _bars_size(extra_mm: float = 0.0) -> Callable[[Theme, Ctx], Figure]:
    """Size of a panel with one row per method."""

    def size(th: Theme, ctx: Ctx) -> Figure:
        return charts.category_figure(th, len(ctx.names), extra_mm)

    return size


def _heat_size(rows: Callable[[Ctx], int]) -> Callable[[Theme, Ctx], Figure]:
    """Size of a heat map: one row per label and room for the rotated method names."""

    def size(th: Theme, ctx: Ctx) -> Figure:
        extra = charts.label_inset_mm(th, ctx.names, pad_mm=0.0)
        return _figure(
            th, shape=Shape.ROWS, rows=max(1, rows(ctx)),
            base_mm=charts.ROWS_BASE_MM + extra,
        )  # fmt: skip

    return size


def _legend_below(ax: Axes, th: Theme, handles: Sequence[Any], cols: int) -> None:
    """A panel's own legend under its x label, never over the data."""
    ax.legend(
        handles=handles, loc="upper center", ncol=cols, fontsize=th.font,
        bbox_to_anchor=(0.5, -0.28 if th.name == "report" else -0.2), frameon=False,
        labelcolor=th.ink2,
    )  # fmt: skip


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
        notes.append(f"{', '.join(cats.unlogged)} training entropy {NOT_LOGGED} (R78).")
    absent = [m for m in ctx.names if m not in seen]
    if not seen:
        _empty(ax, th, f"{missing}: {', '.join(absent)}")
        return
    if absent:
        notes.append(f"{', '.join(absent)} {missing}.")
    _style(ax, th)
    if len(cats.keys) == 1:  # the x label names it: a lone row label would repeat it
        ax.set_yticks([])
    else:
        ax.set_yticks(range(len(cats.keys)), list(cats.labels))
    ax.set_ylim(len(cats.keys) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_num_text))
    _labels(ax, th, xlabel)


def _profile_panel(
    ax: Axes,
    th: Theme,
    ctx: Ctx,
    kind: str,
    cats: Cats,
    ylabel: str,
    notes: list[str],
    *,
    how: str = "mean",
    log: bool = False,
) -> None:
    """Per ordered category, each method's estimate with its bootstrap interval, joined.

    The transposed dot panel for a long ordered list (a funnel): categories along x,
    one line per method, a short vertical interval at each point.
    """
    offsets = np.linspace(-0.25, 0.25, len(ctx.names)) if len(ctx.names) > 1 else [0.0]
    seen: set[str] = set()
    for j, method in enumerate(ctx.names):
        style = _style_of(th, method)
        xs, ys = [], []
        for i, key in enumerate(cats.keys):
            est = _interval(cats.values(method, key), how, ctx.n_boot, ctx.seed)
            if est is None:
                continue
            x = i + offsets[j]
            ax.vlines(
                x, est.lo, est.hi, colors=style["color"], linewidth=th.lw * 0.6,
                gid=_gid("ci", ctx.budget, method, key),
            )  # fmt: skip
            xs.append(x)
            ys.append(est.estimate)
        if not xs:
            continue
        seen.add(method)
        (line,) = ax.plot(
            xs, ys, color=style["color"], ls=style["ls"], lw=th.lw * 0.8,
            marker=style["marker"], ms=th.ms * 0.8,
        )  # fmt: skip
        line.set_gid(_gid(kind, ctx.budget, method))
    absent = [m for m in ctx.names if m not in seen]
    if not seen:
        _empty(ax, th, f"{NOT_RECORDED}: {', '.join(absent)}")
        return
    if absent:
        notes.append(f"{', '.join(absent)} {NOT_RECORDED}.")
    _style(ax, th, grid="both")
    ax.set_xticks(range(len(cats.keys)), list(cats.labels), rotation=90)
    ax.set_xlim(-0.6, len(cats.keys) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_num_text))
    _labels(ax, th, None, ylabel)


def _num_text(v: float, _pos: int | None = None) -> str:
    return f"{v:.3g}"


def _steps_label(v: float, _pos: int | None = None) -> str:
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= scale:
            return f"{v / scale:.3g}{unit}"
    return f"{v:.0f}"


def _cell_number(value: float, narrow: bool) -> str:
    """``0.72``, or ``.72`` (and ``1.0``) where nine columns share an 80 mm panel."""
    text = f"{value:.2f}"
    if not narrow:
        return text
    return "1.0" if text == "1.00" else text.replace("0.", ".", 1)


def _compact(value: float) -> str:
    """A number for a narrow cell: ``12``, ``1.5``, ``.35`` (never wider than four)."""
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}".replace("0.", ".", 1)


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
    solo: int | None = None,
) -> None:
    """Rows ``labels`` down, ``ctx.names`` across; NaN cells are the surface.

    ``solo`` is a row with its own colour scale (a quantity of another size, like the
    passive kills); cells carry their value when ``annotate``.
    """
    cmap = _cmap(th, *th.seq)

    def norm_of(part: np.ndarray, cap: float | None = None) -> Normalize:
        top = float(np.nanmax(part)) if np.isfinite(part).any() else 1.0
        return Normalize(0.0, cap if cap is not None else (top or 1.0))

    if solo is None:
        parts = [(matrix, norm_of(matrix, vmax))]
    else:
        main = matrix.copy()
        main[solo] = np.nan
        alone = np.full_like(matrix, np.nan)
        alone[solo] = matrix[solo]
        parts = [(main, norm_of(main)), (alone, norm_of(alone))]
    for k, (part, norm) in enumerate(parts):
        gid = _gid("heat", ctx.budget, kind if k == 0 else f"{kind}-solo")
        charts.heatmap(ax, th, part, cmap, norm=norm, gid=gid)
        if annotate:
            for i, j in zip(*np.where(np.isfinite(part)), strict=True):
                ax.text(
                    j + 0.5, i + 0.5, _compact(part[i, j]), ha="center", va="center",
                    fontsize=th.font, color=_ink_on(cmap(norm(part[i, j]))),
                )  # fmt: skip
    ax.set_xticks(np.arange(len(ctx.names)) + 0.5, ctx.names, rotation=90)
    if len(labels) == 1:  # a lone row label is an axis label
        ax.set_yticks([])
        _labels(ax, th, None, labels[0])
    else:
        ax.set_yticks(np.arange(len(labels)) + 0.5, list(labels))
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
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
    curves = frames["curves"]
    family = curves["metric"].str.startswith(("first_unlock/", "first_unlock_wall/"))
    unlock = curves[family].copy()  # only these rows are masked, never the 12M others
    firsts = compare.first_unlocks(unlock, frames["runs"])
    late = firsts[firsts["steps"] >= upto]
    keys = pd.MultiIndex.from_frame(late[["run_id", "achievement"]])
    ach = unlock["metric"].str.split("/", n=1).str[1]
    hit = pd.MultiIndex.from_arrays([unlock["run_id"], ach]).isin(keys)
    unlock.loc[hit, "value"] = np.inf
    return {**frames, "curves": pd.concat([curves[~family], unlock])}


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


def _race_notes(table: pd.DataFrame, runs: pd.DataFrame, clock: str) -> list[str]:
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
    return notes + _wall_notes(table, clock)


def _ineligible_note(table: pd.DataFrame, rows: Sequence[str], where: str) -> list[str]:
    """Cells reached by fewer than half of a method's seeds: not drawn, said here."""
    cells = table[table["achievement"].isin(rows)]
    bad = cells[~cells["eligible"].astype(bool)]
    if bad.empty:
        return []
    text = (
        f"{where}: {len(bad)} of {len(cells)} method-achievement cells were reached "
        "by fewer than half of the seeds (k < n/2) and are not drawn."
    )
    return [text]


def _wall_notes(table: pd.DataFrame, clock: str) -> list[str]:
    notes: list[str] = []
    if clock == "wall":
        walls = sorted({w for w in table["wall_note"].dropna() if w})
        notes += [f"Wall-clock: {w}." for w in walls]
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


def race(
    frames: Frames,
    out_dir: Path | None,
    theme: str = "report",
    *,
    width: str = "column",
    clock: str = "steps",
    upto: int | None = None,
) -> Figs:
    """F6 ``race_<clock>_<block>``: median first unlock of each achievement.

    Achievements down (tier, floor, game order; those some method is eligible for, the
    same rows in both blocks), the median over seeds on a log x axis per method with
    25th-75th percentile whiskers, the winner ringed; one file per block, a method
    racing only inside its block. Medians, eligibility and winners are
    `compare.race_table`. ``upto`` (env steps) draws only the first unlocks before that
    step, on the axes of the full race (the frames of the race GIF); ``out_dir=None``
    writes no file.

    Raises:
        ValueError: If ``clock`` is neither ``steps`` nor ``wall``, or ``out_dir`` is
            None and there is nothing to race.
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
        return {
            name: _no_data(name, th, out_dir, "no first-unlock curves for the race")
        }
    budget = _single_budget(tuned)
    shown = full if upto is None else compare.race_table(_masked(frames, upto))
    firsts = compare.first_unlocks(frames["curves"], tuned)

    def of(table: pd.DataFrame, blk: str) -> pd.DataFrame:
        return table[(table["benchmark"] == budget) & (table["block"] == blk)]

    blocks = [b for b in compare.BLOCKS if len(of(full, b))]
    rows = _race_rows(full[full["benchmark"] == budget])  # shared: the files tile
    limits = _race_limits(full, budget, clock)
    left = charts.label_inset_mm(th, rows)
    unit = "environment steps" if clock == "steps" else "wall-clock seconds"
    figs: Figs = {}
    for block in blocks:
        stem = f"{name}_{block}"
        if not wants(th, stem):
            continue
        fig = _figure(th, shape=Shape.ROWS, rows=max(len(rows), 4))
        ax = charts.add_panel(fig, left_mm=left)
        in_block = [m for b, m, _ in compare.members(tuned) if b == block]
        mine = tuned[tuned["method"].isin(in_block)]
        spread = _spread(firsts, tuned, clock, upto)
        _race_panel(ax, th, budget, of(shown, block), rows, spread, clock, limits)
        notes = [
            *_race_notes(of(full, block), mine, clock),
            *_ineligible_note(of(full, block), rows, f"{BLOCK_TITLES[block]} block"),
        ]
        what = f"{BLOCK_TITLES[block]} block first unlocks"
        if upto is not None:
            what += f" up to env step {upto:,}"
        if out_dir is not None:
            text = (
                f"Median first unlock ({unit}, log axis) of each achievement in the "
                f"{BLOCK_TITLES[block]} block: dot = median over seeds, line = 25th "
                "to 75th percentile, ring = the fastest method; a seed that never "
                f"unlocked it counts as infinitely late. {' '.join(_unique(notes))}"
            )
            _save(fig, th, out_dir, stem, title=what.capitalize(), description=text)
        figs[stem] = fig
    return figs


def plot_race(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
) -> Figs:
    """F6: the ``race_wall_*`` and ``race_steps_*`` files (every stem, both clocks)."""
    wall = race(frames, out_dir, theme, width=width, clock="wall")
    return {**wall, **race(frames, out_dir, theme, width=width, clock="steps")}


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

    Per benchmark and ordered pair of methods (a pair appears twice, once each way;
    the cells of two non-control methods of different blocks are NaN, never compared):
    ``tau`` over the mean normalised rank of the shared nodes with its permutation
    ``p`` (`PERMUTATIONS`), ``jaccard`` of the two edge sets (NaN when both are empty).
    """
    rows = []
    for ctx in _contexts(frames):
        graphs = {m: order_graph(ctx.eps[ctx.eps["method"] == m]) for m in ctx.names}
        block = {m: b for b, m, role in compare.members(ctx.runs) if role == "method"}
        for a, b in itertools.permutations(ctx.names, 2):
            (na, ea), (nb, eb) = graphs[a], graphs[b]
            shared = na.merge(nb, on="achievement", suffixes=("_a", "_b"))
            if a in block and b in block and block[a] != block[b]:
                tau, p = float("nan"), float("nan")  # never across blocks
                rows.append({
                    "benchmark": ctx.budget, "method": a, "other": b, "tau": tau,
                    "p": p, "n_shared": 0, "jaccard": float("nan"),
                })  # fmt: skip
                continue
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


LABEL_BASE_PT = 7.0
"""Gap between a node and the lowest label above it (clear of dots and arrows)."""
TIER_PT = 33.0
"""Least height of one tier row of an order graph, in points."""


def _node_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, int]]:
    return {
        r.achievement: (r.rho, TIERS.index(r.tier) if r.tier in TIERS else len(TIERS))
        for r in nodes.itertuples()
    }


def _text_pt(name: str, th: Theme) -> float:
    """Width in points of ``name`` at the theme's font size (DejaVu Sans, measured)."""
    return float(TextPath((0, 0), name, size=th.font).get_extents().width)


def _label_levels(
    pos: Mapping[str, tuple[float, float]], th: Theme, columns: int = 1
) -> dict[str, int]:
    """A stacking level per node name: labels of one tier never overlap or cross.

    Right to left, a label takes one level above every label to its right that its
    (estimated, 0.62 em a character) width reaches, so the vertical leader of a label
    never passes through another label.
    """
    share = (
        0.7 / columns - 0.1
    )  # axes width in figure widths (labels, gaps eat the rest)
    per_pt = 1.1 / (th.width * 72 * share)  # x data units per point of axes width
    levels: dict[str, int] = {}
    for name, (x, y) in sorted(pos.items(), key=lambda kv: (kv[1][1], -kv[1][0])):
        reach = x + (1.5 * _text_pt(name, th) + 8) * per_pt
        above = [
            levels[o] for o, (ox, oy) in pos.items() if o in levels and oy == y
            and x < ox < reach
        ]  # fmt: skip
        levels[name] = 1 + max(above) if above else 0
    return levels


def _label_pt(th: Theme, levels: Mapping[str, int]) -> float:
    """Height above a node that its labels need, in points."""
    return LABEL_BASE_PT + (max(levels.values(), default=0) + 1) * th.font * 1.7


def _tier_pt(th: Theme, need_pt: float, rows: int) -> float:
    """One tier row in points: the label stack of the busiest tier fits in it."""
    if th.height is not None:
        return th.height * 72 * 0.8 / (rows + 1.5)
    return max(TIER_PT, need_pt + 12, 35.0 / rows)


def _tier_names(tiers: Sequence[int]) -> list[str]:
    return [TIERS[t].replace("_", " ") if t < len(TIERS) else "other" for t in tiers]


def _order_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    method: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    labels: bool,
    tiers: Sequence[int],
    columns: int = 1,
) -> None:
    """Nodes by mean rank (x) and tier (y, only the ``tiers`` drawn), arrows after."""
    color = _color(th, method)
    row = {t: i for i, t in enumerate(tiers)}
    pos = {k: (x, row[y]) for k, (x, y) in _node_positions(nodes).items()}
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
    need = 0.0
    if labels:
        levels = _label_levels(pos, th, columns)
        need = _label_pt(th, levels)
        step = th.font * 1.7
        for name, (x, y) in pos.items():
            ax.annotate(
                name,
                (x, y),
                xytext=(0, LABEL_BASE_PT + levels[name] * step),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=th.font,
                color=th.ink2,
                arrowprops={
                    "arrowstyle": "-", "color": th.muted, "lw": 0.4,
                    "relpos": (0, 0), "shrinkA": 0, "shrinkB": 2.5,
                },
            )  # fmt: skip
    _style(ax, th, grid="both")
    ax.set_xlim(-0.05, 1.05)
    height = _tier_pt(th, need, len(tiers))
    ax.set_ylim(len(tiers) - 0.5, -max(0.5, (need + 4) / height))
    if len(tiers) == 1:  # the caption names the one tier: a lone label would repeat it
        ax.set_yticks([])
    else:
        ax.set_yticks(range(len(tiers)), _tier_names(tiers))
    ax.tick_params(axis="y", length=0)
    _labels(ax, th, "mean normalised rank (0 first, 1 last)")


def plot_order_graph(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    method: str,
) -> Figs:
    """F7 ``order_<method>``: the achievement order graph of one method.

    Nodes: achievements unlocked in 5% of the final stochastic eval episodes, x their
    mean normalised rank, y their tier (only tiers with a node are drawn, the caption
    names the others); arrows: ``P(a before b) >= 0.9`` over 20 co-occurring episodes
    after transitive reduction (`order_graph`). Node names are printed, so the figure
    is drawn at the double width and as video only (`charts.DOUBLE_STEMS`).
    """
    th, name = _theme(theme, width), f"order_{method}"
    ctx = _ctx(frames)
    if ctx is None or method not in ctx.names:
        return {name: _no_data(name, th, out_dir, f"no tuned runs of {method}")}
    if not wants(th, name):
        return {}
    nodes, edges = order_graph(ctx.eps[ctx.eps["method"] == method])
    pos = _node_positions(nodes)
    tiers = sorted({y for _, y in pos.values()}) or [0]  # no node: the first tier
    need = _label_pt(th, _label_levels(pos, th)) if th.wide else 0.0
    row = _tier_pt(th, need, len(tiers))
    axes_mm = row * (len(tiers) - 0.5 + max(0.5, (need + 4) / row)) / 72 * 25.4
    fig = _figure(th, height_mm=axes_mm + 12.0, shape=Shape.ROWS)
    ax = fig.add_subplot()
    _order_panel(ax, th, ctx.budget, method, nodes, edges, labels=th.wide, tiers=tiers)
    notes = []
    if edges.empty:
        notes.append(f"No edge reaches P >= {ORDER_P} over {ORDER_PAIRS} episodes.")
    if missing := [t for t in range(len(TIERS)) if t not in tiers]:
        notes.append(f"No node in the tiers: {', '.join(_tier_names(missing))}.")
    if len(tiers) == 1:
        notes.append(f"All nodes are in the tier {_tier_names(tiers)[0]}.")
    text = (
        f"Order graph of {method}: achievements unlocked in at least 5% of its final "
        "stochastic eval episodes, placed by their mean normalised rank among an "
        "episode's unlocks (0 first, 1 last) and by tier. Arrow: a is unlocked before "
        f"b in at least {ORDER_P:.0%} of the episodes with both ({ORDER_PAIRS} or "
        f"more), after transitive reduction. {' '.join(notes)}"
    )
    _save(fig, th, out_dir, name, title=f"Order graph, {method}", description=text)
    return {name: fig}


def plot_order_graphs(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figs:
    """`plot_order_graph` for every method of ``frames``."""
    methods = list(dict.fromkeys(frames["runs"]["method"]))
    figs: Figs = {}
    for m in methods:
        figs |= plot_order_graph(frames, out_dir, theme, width=width, method=m)
    return figs


def plot_order_agreement(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_perm: int = PERMUTATIONS,
    seed: int = 0,
) -> Figs:
    """F7 ``order_agreement``: Kendall tau-b between the methods' unlock orders.

    One matrix (`order_agreement`); a cell is bold when its permutation p-value is
    below 0.05, blank when fewer than three nodes are shared.
    """
    th, name = _theme(theme, width), GAME_NAMES["F7"]
    table = order_agreement(frames, n_perm=n_perm, seed=seed)
    ctx = _ctx(frames)
    if table.empty or ctx is None:
        return {
            name: _no_data(name, th, out_dir, "no order graphs (need unlock steps)")
        }
    if not wants(th, name):
        return {}
    fig = _figure(th, shape=Shape.SQUARE)
    ax = fig.add_subplot()
    n = len(ctx.names)
    index = {m: i for i, m in enumerate(ctx.names)}
    cells = np.full((n, n), np.nan)
    sig = np.zeros((n, n), bool)
    for r in table[table["benchmark"] == ctx.budget].itertuples():
        cells[index[r.method], index[r.other]] = r.tau
        sig[index[r.method], index[r.other]] = r.p < TAU_ALPHA
    cmap = _cmap(th, *th.div)
    charts.heatmap(
        ax, th, cells, cmap, vmin=-1, vmax=1, gid=_gid("tau", ctx.budget, "all")
    )
    for i, j in zip(*np.where(np.isfinite(cells)), strict=True):
        ax.text(
            j + 0.5, i + 0.5, _cell_number(cells[i, j], n > 6 and not th.wide),
            ha="center", va="center", fontsize=th.font,
            color=_ink_on(cmap((cells[i, j] + 1) / 2)),
            fontweight="bold" if sig[i, j] else "normal",
            gid="tau-cell/bold" if sig[i, j] else "tau-cell/plain",
        )  # fmt: skip
    ax.set_xticks(np.arange(n) + 0.5, ctx.names, rotation=90)
    ax.set_yticks(np.arange(n) + 0.5, ctx.names)
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    text = (
        "Kendall tau-b between the unlock orders of two methods: the mean normalised "
        "rank of the order-graph nodes they share. Bold: permutation p < "
        f"{TAU_ALPHA} ({n_perm:,} permutations). Blank: methods of different blocks "
        "are not compared (controls are, in both)."
    )
    _save(fig, th, out_dir, name, title="Order agreement", description=text)
    return {name: fig}


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


def _funnel_profile(
    ax: Axes, th: Theme, ctx: Ctx, kind: str, cats: Cats, notes: list[str]
) -> None:
    _profile_panel(
        ax, th, ctx, f"funnel-{kind}", cats, "share of episodes", notes, how="mean"
    )
    ax.set_ylim(-0.02, 1.02)


def _funnel_dots(kind: str) -> Callable[[Axes, Theme, Ctx, list[str]], None]:
    """A funnel with a few steps as a dot panel (the tech ladder is a profile)."""

    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        shown, tail = _trim_zero_tail(ctx, _funnel_cats(ctx, kind))
        if tail:
            notes.append(f"No episode reached {', '.join(tail)}.")
        if len(shown.keys) > MAX_DOT_ROWS:
            _funnel_profile(ax, th, ctx, kind, shown, notes)
            return
        _dot_panel(
            ax, th, ctx, f"funnel-{kind}", shown, "share of episodes", notes, how="mean"
        )
        ax.set_xlim(-0.02, 1.02)

    return draw


def _funnel_panel(kind: str) -> Callable[[Axes, Theme, Ctx, list[str]], None]:
    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        shown, tail = _trim_zero_tail(ctx, _funnel_cats(ctx, kind))
        if tail:
            notes.append(f"No episode reached {', '.join(tail)}.")
        _profile_panel(
            ax, th, ctx, f"funnel-{kind}", shown, "share of episodes", notes, how="mean"
        )
        ax.set_ylim(-0.02, 1.02)

    return draw


def plot_funnel(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F8 ``funnel_tech`` and ``funnel_floor``: progress at the final checkpoint.

    Per method the share of eval episodes with tech depth at least k on the ladder
    wood to diamond pickaxe, and with max floor at least f; mean over seeds with a
    95% bootstrap interval, so a step never exceeds the one before.
    """
    panels = [
        FigPanel(
            "funnel_tech", "Tech ladder reached",
            "Share of the final stochastic eval episodes that reach each step of the "
            "tech ladder, wood to diamond pickaxe: mean over seeds, vertical line = "
            "95% bootstrap interval.",
            _funnel_panel("tech"), _curve_size,
        ),
        FigPanel(
            "funnel_floor", "Deepest floor reached",
            "Share of the final stochastic eval episodes that reach each floor: mean "
            "over seeds, vertical line = 95% bootstrap interval.",
            _funnel_dots("floor"),
            _dots_size(
                lambda c: len(_trim_zero_tail(c, _funnel_cats(c, "floor"))[0].keys)
            ),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


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
            notes.append(f"{', '.join(missing)} {NOT_LOGGED} (R78).")
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
) -> Figs:
    """F8 ``funnel_training_depth`` and ``funnel_training_floor``: funnel in training.

    The mean tech depth (sum over k of the share ``train/funnel/depth_k``) and the mean
    deepest floor (sum over f >= 1 of ``train/funnel/floor_f``), mean over seeds with
    a bootstrap band; the reused PPO and PPO-RNN runs did not log them (R78).
    """
    depth = [f"train/funnel/depth_{k}" for k in range(1, len(LADDER_NAMES) + 1)]
    floor = [f"train/funnel/floor_{f}" for f in range(1, len(FLOOR_NAMES))]
    panels = [
        FigPanel(
            "funnel_training_depth", "Mean tech depth in training",
            "Mean tech-ladder depth over training (sum over the ladder steps of the "
            "share of training episodes that reached them); band: 95% bootstrap CI.",
            _series_panel("depth", lambda _c: depth, "sum", "mean tech depth"),
            _curve_size,
        ),
        FigPanel(
            "funnel_training_floor", "Mean deepest floor in training",
            "Mean deepest floor over training (sum over floors 1 and deeper of the "
            "share of training episodes that reached them); band: 95% bootstrap CI.",
            _series_panel("floor", lambda _c: floor, "sum", "mean deepest floor"),
            _curve_size,
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- F9

CAUSE_SLOTS = 4
"""Causes drawn apart in a death bar (a ramp step each); the rest are ``other``."""
OTHER = "other"
NEW_TILES_NOTE = (
    "New tiles: the episode mean stored by the replay, not a series over the episode."
)


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
    _legend_below(ax, th, handles, min(len(groups), 3 if not th.wide else 5))
    _labels(ax, th, "share of deaths")


def plot_deaths(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F9 ``deaths``: cause-of-death shares per method (`hard.death_causes`).

    Stacked bars at the final checkpoint's stochastic eval episodes; the four largest
    causes of the benchmark have a ramp step each, the rest are ``other``; a method
    that never died says so. A thin bar in the method colour marks each row.
    """
    panel = FigPanel(
        "deaths", "Cause of death",
        "Share of the deaths of each method by cause, in the final stochastic eval "
        "episodes; the four largest causes have their own colour, the rest are other.",
        _death_panel, _bars_size(extra_mm=6.0),
    )  # fmt: skip
    return _draw(frames, out_dir, theme, width, [panel], n_boot=n_boot, seed=seed)


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
        notes.append(f"{', '.join(absent)} had no stall.")
    _style(ax, th, grid="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_num_text))
    ax.yaxis.set_major_formatter(FuncFormatter(_num_text))
    for axis in (ax.xaxis, ax.yaxis):  # minor labels of a narrow range run together
        axis.set_minor_formatter(NullFormatter())
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
        notes.append(f"{', '.join(absent)} had no stall.")
    _style(ax, th)
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(ctx.names)), ctx.names)
    ax.set_ylim(len(ctx.names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    _labels(ax, th, "escape rate")


def plot_stalls(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F9 ``stalls_ccdf`` and ``stalls_escape``: stall lengths and escape rate."""
    panels = [
        FigPanel(
            "stalls_ccdf", "Stall lengths",
            "Share of the stalls (steps with no progress) that last at least a given "
            "number of steps, per method; both axes logarithmic.",
            _stall_ccdf_panel, _curve_size,
        ),
        FigPanel(
            "stalls_escape", "Escape rate",
            "Share of the stalls that ended in an unlock, per method.",
            _escape_panel, _bars_size(),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


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


def _split_rows(table: pd.DataFrame) -> list[str]:
    """The first `SPLIT_ROWS` milestones some method reached (the rest: a note)."""
    reached = set(table.loc[table["n"] > 0, "milestone"])
    return [m for m in MILESTONES if m in reached][:SPLIT_ROWS]


def _split_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    """Box plots of the steps to each milestone per method, share reaching as text."""
    table = split_table(ctx.eps)
    rows = _split_rows(table)
    if table.empty or not rows:
        _empty(ax, th, f"{NOT_RECORDED}: no split columns")
        return
    reached = set(table.loc[table["n"] > 0, "milestone"])
    missed = [
        m for m in MILESTONES if m not in reached and (table["milestone"] == m).any()
    ]
    later = [m for m in MILESTONES if m in reached and m not in rows]
    if missed:
        _notes.append(f"no episode reached {', '.join(missed)}.")
    if later:
        _notes.append(
            f"Not drawn (only the first {SPLIT_ROWS} are): {', '.join(later)}."
        )
    offsets = dict(zip(ctx.names, _dodge(len(ctx.names)), strict=True))
    for r in table.itertuples():
        if not r.n or r.milestone not in rows:
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
    ax.xaxis.set_minor_formatter(NullFormatter())  # minor labels run together
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    _labels(ax, th, "steps to the first unlock (log)")


def plot_splits(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F10 ``splits``: speedrun splits, box plots of steps to each milestone.

    Rows: the tech ladder then the ``enter_*`` achievements; a box per method spans the
    quartiles of the episodes that reached it, a line marks the median; the text at
    the right is the lowest and highest share of a method's episodes reaching it.
    """

    def rows(c: Ctx) -> int:
        return max(1, len(_split_rows(split_table(c.eps))))

    def size(th: Theme, ctx: Ctx) -> Figure:
        pitch = len(ctx.names) * 0.9 + 2.0
        return charts.category_figure(th, rows(ctx), 0.0, pitch)

    panel = FigPanel(
        "splits", "Speedrun splits",
        "Steps to the first unlock of each milestone (log axis), per method: box = "
        "quartiles of the episodes that reached it, line = median. The text at the "
        "right is the lowest and highest share of a method's episodes that reached it.",
        _split_panel, size,
    )  # fmt: skip
    return _draw(frames, out_dir, theme, width, [panel], n_boot=n_boot, seed=seed)


# ----------------------------------------------------------------- F11, F12, F14


def _with_data(ctx: Ctx, cats: Cats) -> tuple[Cats, list[str]]:
    """``cats`` without the rows no method has a value in, and those rows' labels."""
    keep = [
        i
        for i, key in enumerate(cats.keys)
        if any(np.isfinite(cats.values(m, key)).any() for m in ctx.names)
    ]
    dropped = [str(cats.labels[i]) for i in range(len(cats.keys)) if i not in keep]
    return (
        cats._replace(
            keys=[cats.keys[i] for i in keep], labels=[cats.labels[i] for i in keep]
        ),
        dropped,
    )


def _trim_zero_tail(ctx: Ctx, cats: Cats) -> tuple[Cats, list[str]]:
    """``cats`` up to one row past the last row any method is above zero in.

    A funnel or a list of rare mechanics ends where nothing is left: the zero rows
    after that point are not drawn (returned as labels, said in a note).
    """

    def mean(m: str, k: str) -> float:
        v = cats.values(m, k)
        return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0

    top = [max(mean(m, k) for m in ctx.names) for k in cats.keys]
    last = max((i for i, v in enumerate(top) if v > 0), default=-1)
    n = min(len(cats.keys), last + 2)
    return (
        cats._replace(keys=list(cats.keys[:n]), labels=list(cats.labels[:n])),
        [str(x) for x in cats.labels[n:]],
    )


def _dots(
    kind: str,
    cats: Callable[[Ctx], Cats],
    xlabel: str,
    *,
    how: str = "median",
    log: bool = False,
    missing: str = NOT_RECORDED,
    trim: bool = False,
) -> Drawer:
    """A dot panel of ``cats(ctx)`` as a panel drawer (``trim``: no zero tail)."""

    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        shown, dropped = _with_data(ctx, cats(ctx))
        if trim:
            shown, tail = _trim_zero_tail(ctx, shown)
            if tail:
                notes.append(f"nothing above zero in {', '.join(tail)}.")
        if dropped and shown.keys:  # rows without any value are not drawn, but said
            notes.append(f"no value for {', '.join(dropped)}.")
        if len(shown.keys) > MAX_DOT_ROWS:
            _profile_panel(ax, th, ctx, kind, shown, xlabel, notes, how=how, log=log)
            return
        _dot_panel(
            ax,
            th,
            ctx,
            kind,
            shown,
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
) -> Figs:
    """F11 ``coverage`` and ``new_tiles``: share of each floor seen, new tiles per 100.

    Per method the median over seeds of each seed's median episode (floors never
    visited are skipped), with a bootstrap interval over seeds; the new-tile rate is
    the episode mean the replay stores, not a series over the episode.
    """
    panels = [
        FigPanel(
            "coverage", "Exploration coverage per floor",
            "Share of each floor that an episode sees: median over seeds of each "
            "seed's median episode, line = 95% bootstrap interval over seeds.",
            _dots("coverage", _coverage_cats, "share of the floor seen"),
            _dots_size(lambda c: len(_with_data(c, _coverage_cats(c))[0].keys)),
        ),
        FigPanel(
            "new_tiles", "New tiles",
            "Tiles of the map an episode sees for the first time per 100 steps: "
            f"median over seeds, line = 95% bootstrap interval. {NEW_TILES_NOTE}",
            _dots("tiles", _tiles_cats, "new tiles per 100 steps"),
            _dots_size(lambda _c: 1),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


def _share_cats(ctx: Ctx, kind: str) -> Cats:
    keys = [f"{kind}_share/{s}" for s in STATS]
    return Cats(keys, list(STATS), _per_seed_of(ctx, "{}"))


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
) -> Figs:
    """F12 ``survival_low``, ``survival_zero``, ``survival_level``.

    The shares of steps with a stat low and at zero, and the mean stat value in the
    state before each eat, drink and sleep start (replay only: ``not recorded`` where
    the export has no trace); median over seeds with a bootstrap interval.
    """
    panels = [
        FigPanel(
            "survival_low", "Steps with a stat low",
            "Share of the steps in which food, drink or energy is low: median over "
            "seeds of each seed's median episode, line = 95% bootstrap interval.",
            _dots("share", lambda c: _share_cats(c, "low"), "share of steps"),
            _dots_size(lambda _c: len(STATS)),
        ),
        FigPanel(
            "survival_zero", "Steps with a stat at zero",
            "Share of the steps in which food, drink or energy is zero: median over "
            "seeds of each seed's median episode, line = 95% bootstrap interval.",
            _dots("zero", lambda c: _share_cats(c, "zero"), "share of steps"),
            _dots_size(lambda _c: len(STATS)),
        ),
        FigPanel(
            "survival_level", "Stat level at the action",
            "Mean value of the stat in the state before each eat, drink and sleep: "
            "median over seeds, line = 95% bootstrap interval.",
            _dots("level", _level_cats, "mean stat value"),
            _dots_size(lambda _c: len(STATS)),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


# -------------------------------------------------------------------------- F13


def _kill_columns(eps: pd.DataFrame) -> list[str]:
    """``kills/<creature>`` columns, the passive kills last (never summed with them)."""
    cols = [c for c in eps.columns if c.startswith("kills/") and c != "kills/passive"]
    return [*cols, "kills/passive"] if "kills/passive" in eps.columns else cols


def _kills_view(
    ctx: Ctx,
) -> tuple[list[str], np.ndarray, int | None, list[str]]:
    """Creature rows some method killed, their matrix, the passive row, the rest."""
    cols = _kill_columns(ctx.eps)
    matrix = _mean_by_method(ctx, cols)
    names = [c.removeprefix("kills/") for c in cols]
    top = [float(np.nanmax(row)) if np.isfinite(row).any() else 0.0 for row in matrix]
    best = sorted(range(len(cols)), key=lambda i: -top[i])[:HEAT_ROWS]
    seen = sorted(i for i in best if top[i] >= KILL_MIN)  # game order, passive last
    never = [names[i] for i in range(len(cols)) if i not in seen]
    labels = [names[i] for i in seen]
    solo = labels.index("passive") if "passive" in labels else None
    return labels, matrix[seen], solo, never


def _kills_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    """Heat map of mean kills per episode: creature types down, methods across."""
    labels, matrix, solo, never = _kills_view(ctx)
    if not labels:
        _empty(ax, th, f"{NOT_RECORDED}: no kill columns")
        return
    if never:
        notes.append(
            f"Not drawn (killed under {KILL_MIN} times per episode by every method): "
            f"{', '.join(never)}."
        )
    _heat(ax, th, ctx, "kills", labels, matrix, annotate=True, solo=solo)


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
            [x], [y], ls="", marker=style["marker"], ms=th.ms * 1.3,
            color=style["color"],
        )  # fmt: skip
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
) -> Figs:
    """F13 ``combat_kills``, ``combat_damage`` and ``combat_fights``.

    Kills are the mean per episode over seeds (the creature types some method killed,
    passive kills on their own colour scale, the others named in the caption); damage
    is the episode mean per method; fights are the episode mean of the stored fight
    records, with a bootstrap interval over seeds.
    """
    panels = [
        FigPanel(
            "combat_kills", "Kills per episode",
            "Mean kills per episode of each creature type, per method (mean over "
            "seeds). The last row, passive creatures, has its own colour scale.",
            _kills_panel, _heat_size(lambda c: len(_kills_view(c)[0])),
        ),
        FigPanel(
            "combat_damage", "Damage dealt and taken",
            "Damage dealt against damage taken per episode, one dot per method "
            "(episode mean over seeds).",
            _damage_panel, _curve_size,
        ),
        FigPanel(
            "combat_fights", "Fights per episode",
            "Fights per episode, from the stored fight records: mean over seeds, line "
            "= 95% bootstrap interval.",
            _dots("fights", _fight_cats, "fights per episode"),
            _dots_size(lambda _c: 1),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


# -------------------------------------------------------------------------- F14


def _action_columns(eps: pd.DataFrame) -> list[str]:
    return [c for c in eps.columns if c.startswith("action_count/")]


def _actions_view(ctx: Ctx) -> tuple[list[str], np.ndarray, list[str]]:
    """Actions some method used, their share-of-steps matrix, the unused ones."""
    cols = _action_columns(ctx.eps)
    if not cols:
        return [], np.empty((0, len(ctx.names))), []
    counts = ctx.eps[cols].to_numpy(float)
    shares = pd.DataFrame(
        counts / np.maximum(counts.sum(axis=1, keepdims=True), 1),
        columns=cols,
        index=ctx.eps.index,
    )
    shown = ctx._replace(eps=ctx.eps.assign(**shares.to_dict("series")))
    matrix = _mean_by_method(shown, cols)
    names = [c.removeprefix("action_count/") for c in cols]
    top = [float(np.nanmax(row)) if np.isfinite(row).any() else 0.0 for row in matrix]
    best = sorted(range(len(cols)), key=lambda i: -top[i])[:HEAT_ROWS]
    used = sorted(i for i in best if top[i] > 0)
    return (
        [names[i] for i in used],
        matrix[used],
        [n for i, n in enumerate(names) if i not in used],
    )


def _actions_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    """Heat map of the share of steps per action (not forced), methods across."""
    labels, matrix, unused = _actions_view(ctx)
    if not labels:
        _empty(ax, th, f"{NOT_RECORDED}: no action counts")
        return
    if unused:
        notes.append(
            f"Only the {HEAT_ROWS} most used actions are drawn; not drawn: "
            f"{', '.join(unused)}."
        )
    _heat(ax, th, ctx, "actions", labels, matrix, annotate=True)


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
) -> Figs:
    """F14 ``actions_share``, ``actions_entropy`` and ``actions_wasted``.

    Heat map: the mean over seeds of the share of non-forced steps per action (only the
    actions some method takes, the others named in the caption). Entropy (nats) is the
    eval-episode action entropy and the last logged training action entropy (``not
    logged`` for the reused PPO and PPO-RNN runs, R78); the wasted share is the eval
    episodes' median.
    """
    panels = [
        FigPanel(
            "actions_share", "Share of steps per action",
            "Share of the non-forced steps spent on each action, per method (mean over "
            "seeds).",
            _actions_panel, _heat_size(lambda c: len(_actions_view(c)[0])),
        ),
        FigPanel(
            "actions_entropy", "Action entropy",
            "Entropy of the action distribution in nats, in the eval episodes and at "
            "the end of training: median over seeds, line = 95% bootstrap interval.",
            _dots("entropy", _entropy_cats, "entropy (nats)", missing=NOT_LOGGED),
            _dots_size(lambda _c: 2),
        ),
        FigPanel(
            "actions_wasted", "Wasted steps",
            "Share of the steps that change nothing (a wasted action): median over "
            "seeds of the eval episodes, line = 95% bootstrap interval.",
            _dots("wasted", _wasted_cats, "share of steps wasted"),
            _dots_size(lambda _c: 1),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


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
    _legend_below(ax, th, handles, min(len(names), 3 if not th.wide else 5))


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
    return f"Episodes reaching diamond: {text}."


def plot_resources(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figs:
    """F15 ``resources_use`` and ``resources_time``.

    Gathered against used per resource (the replay's inventory trace, ``not recorded``
    without it); the time panel is medians of seed medians, wood to diamond over the
    episodes that reached diamond (their share goes to the caption).
    """
    ctx = _ctx(frames, n_boot, seed)
    diamond = _diamond_note(ctx) if ctx and "wood_to_diamond" in ctx.eps.columns else ""
    panels = [
        FigPanel(
            "resources_use", "Gathered and used",
            "Mean amount gathered against mean amount used per episode, one marker per "
            "resource and one colour per method; dashed: used = gathered.",
            _use_panel, lambda th, _c: _figure(th, shape=Shape.CURVE),
        ),
        FigPanel(
            "resources_time", "Time to progress",
            "Steps per achievement and steps from wood to diamond (log axis): median "
            f"over seeds of each seed's median episode. {diamond}",
            _dots("time", _time_cats, "steps (log)", log=True),
            _dots_size(lambda _c: 2),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


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
) -> Figs:
    """F16 ``advanced_mean`` and ``advanced_final``: adoption of the advanced mechanics.

    From the ``train/ach/*`` rates (% of episodes) of the nine advanced achievements:
    the mean rate over them per method with a bootstrap band, and each one's final
    rate. The reused PPO and PPO-RNN runs did not log them (R78).
    """
    panels = [
        FigPanel(
            "advanced_mean", "Advanced achievements over training",
            "Mean training rate (% of episodes) of the nine advanced achievements, per "
            "method; band: 95% bootstrap CI over seeds.",
            _series_panel("advanced", _advanced_metrics, "mean", "% of episodes"),
            _curve_size,
        ),
        FigPanel(
            "advanced_final", "Advanced achievements at the end",
            "Final training rate (% of episodes) of each advanced achievement: mean "
            "over seeds, line = 95% bootstrap interval.",
            _dots(
                "adv", _advanced_cats, "% of episodes", missing=NOT_LOGGED, trim=True
            ),
            _dots_size(
                lambda c: len(
                    _trim_zero_tail(c, _with_data(c, _advanced_cats(c))[0])[0].keys
                )
            ),
        ),
    ]  # fmt: skip
    return _draw(frames, out_dir, theme, width, panels, n_boot=n_boot, seed=seed)


# ------------------------------------------------------------------------ F17

RADAR_COLUMNS = ["benchmark", "method", "axis", "raw", "scaled"]
SURVIVOR_COLUMN = "any_zero_share"
"""The export column of the Survivor axis; older exports lack it (`_radar_raw`)."""


def _radar_raw(eps: pd.DataFrame) -> pd.DataFrame:
    """The eight radar axes per run: the mean over its episodes (metrics plan).

    Explorer: mean coverage of visited floors; Survivor: 1 - ``any_zero_share``, the
    share of steps with any stat zero (an export without that column gets 1 - the
    largest of the three stats' zero shares, an upper bound of survival); Fighter:
    hostile kills per 1000 steps; Crafter: tech depth; Diver: max floor; Efficient:
    achievements per 1000 steps; Mage: share of the advanced mechanics used; Precise:
    1 - wasted share. NaN where a column is missing.
    """
    length = _column(eps, "length").replace(0, np.nan)
    cover = [c for c in eps.columns if c.startswith("coverage/")]
    zero = [c for c in eps.columns if c.startswith("zero_share/")]
    kills = [c for c in eps.columns if c.startswith("kills/") and c != "kills/passive"]
    adv = [c for c in eps.columns if c.startswith("adv/")]
    nan = pd.Series(np.nan, index=eps.index)
    if SURVIVOR_COLUMN in eps.columns:
        survivor = 1 - eps[SURVIVOR_COLUMN]
    else:
        survivor = 1 - eps[zero].max(axis=1) if zero else nan
    axes = {
        "explorer": eps[cover].mean(axis=1) if cover else nan,
        "survivor": survivor,
        "fighter": eps[kills].sum(axis=1) / length * 1000 if kills else nan,
        "crafter": _column(eps, "tech_depth").astype(float),
        "diver": _column(eps, "max_floor").astype(float),
        "efficient": _column(eps, "n_achievements") / length * 1000,
        "mage": eps[adv].gt(0).where(eps[adv].notna()).mean(axis=1) if adv else nan,
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
    """One method's polygon on shared axes; a missing axis leaves a gap, never a 0."""
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    x, y = np.r_[angles, angles[:1]], np.r_[values, values[:1]]
    style = _style_of(th, method)
    _radar_axes(ax, th, labels)
    (line,) = ax.plot(x, y, color=style["color"], lw=th.lw, ls=style["ls"])
    line.set_gid(_gid("radar", budget, method))
    (fill,) = ax.fill(
        x,
        np.nan_to_num(y),
        color=style["color"],
        alpha=0.0 if np.isnan(y).any() else 0.2,
        lw=0,
    )  # an incomplete polygon is not filled
    fill.set_gid(_gid("radar-fill", budget, method))
    ax.set_title(method, loc="left", fontsize=th.title, color=th.ink)


def _survivor_text(eps: pd.DataFrame) -> str:
    """The caption's definition of the Survivor axis, as `_radar_raw` computes it."""
    if SURVIVOR_COLUMN in eps.columns:
        return "one minus the share of steps with any stat at zero."
    return "one minus the largest single-stat zero share, an upper bound of survival."


def plot_radar(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
) -> Figs:
    """F17 ``radar_<method>``: the playstyle radar of each method on shared axes.

    Eight axes (`RADAR_AXES`) from the final stochastic eval episodes, min-max scaled
    over all methods and controls of the benchmark (`radar_table`); one square file per
    method, titled with its name.
    """
    th = _theme(theme, width)
    ctx = _ctx(frames)
    if ctx is None:
        return {"radar": _no_data("radar", th, out_dir, "no tuned runs for the radar")}
    table = radar_table(frames)
    figs: Figs = {}
    for method in ctx.names:
        stem = f"radar_{method}"
        if not wants(th, stem):
            continue
        mine = table[table["method"] == method].set_index("axis")["scaled"]
        values = mine.reindex(RADAR_AXES).to_numpy(float)
        notes = []
        if np.isnan(values).any():
            lost = ", ".join(np.array(RADAR_AXES)[np.isnan(values)])
            notes.append(f"{NOT_RECORDED} on {lost}.")
        fig = _figure(th, shape=Shape.SQUARE)
        ax = fig.add_subplot(projection="polar")
        _radar_cell(ax, th, ctx.budget, method, values, list(RADAR_AXES))
        text = (
            f"Playstyle radar of {method}: eight axes from the final stochastic eval "
            "episodes, each min-max scaled over the methods of the benchmark (0.5: all "
            f"equal). Survivor is {_survivor_text(ctx.eps)} {' '.join(notes)}"
        )
        _save(fig, th, out_dir, stem, title=f"Radar, {method}", description=text)
        figs[stem] = fig
    return figs


GAME_FIGURES: dict[str, Callable[..., Any]] = {
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
