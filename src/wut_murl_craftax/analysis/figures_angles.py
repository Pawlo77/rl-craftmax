"""Analysis-angle figures F21-F28: questions the core figures leave open.

Each ``plot_*`` follows `analysis.figures` (report 80 mm or ``double``, dark ``video``,
files ``<stem>[_double][_video]``, `METHOD_COLORS`; a missing benchmark is pending).
Eval figures read the tuned runs' checkpoints in both modes; intervals and bands are
`evaluation.stats` bootstraps over seeds. F24 and F25 draw `compare.ablation_table` and
`compare.tuning_optimism` (R101, R102: fresh seeds only). F27 and F28 read the finished
policy folders of the distillation (R106) and name the methods still pending. Heat maps
are seaborn on the theme's ramp. Artists carry ``<kind>/<budget>/<method>[/<key>]``.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures_angles.py]]
"""

import json
import logging
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats as sps

from ..evaluation.hardware import reused_from_tuning
from ..evaluation.stats import BOOTSTRAP_N, curve_band, stratified_bootstrap_ci
from . import compare
from .figures import (
    Theme,
    _budget_name,
    _cmap,
    _figure,
    _finish,
    _gid,
    _labels,
    _no_data,
    _panel_title,
    _pending,
    _save,
    _steps_text,
    _style,
    _style_of,
    _theme,
    _title,
)
from .figures_game import (
    LEGEND_MM,
    LEGEND_ROW_MM,
    PANEL_MM,
    Cats,
    Ctx,
    Panel,
    _contexts,
    _dot_mm,
    _dot_panel,
    _empty,
    _grid,
    _height,
    _method_handles,
    _unique,
)
from .load import Frames
from .results import INTERPRETATION

logger = logging.getLogger(__name__)

NAMES = {
    "F21": "greedy_gap",
    "F22": "gatekeepers",
    "F23": "seed_fate",
    "F24": "ablation_effects",
    "F25": "tuning_optimism",
    "F26": "episode_spread",
    "F27": "distil_score",
    "F28": "distil_features",
}
"""Figure id -> file stem."""
GAP_COLUMNS = [
    "budget",
    "method",
    "run_id",
    "update",
    "env_steps",
    "stochastic",
    "greedy",
    "gap",
]
"""`greedy_gap`: per tuned run and checkpoint the mean episode score of each mode."""
LIFT_COLUMNS = ["method", "run_id", "achievement", "at", "with_a", "without_a", "lift"]
"""`gatekeepers`: per run and achievement, the step ``at`` (the method's median unlock
step), the episodes alive then with and without it, and the lift in later unlocks."""
FATE_COLUMNS = ["budget", "method", "update", "env_steps", "seeds", "rho"]
"""`seed_fate`: Spearman of the seeds' checkpoint scores and final scores."""
GATE_SHARE = 0.05
"""Gatekeeper rows: achievements unlocked in this share of episodes to 1 minus it."""
GATE_MIN = 10
"""Alive episodes with and without the achievement a method needs for a dot."""
LENGTH_BINS = 8
"""Equal-count episode-length bins of the benchmark (F26 right panel)."""
LENGTH_MIN = 10
"""Episodes of a method a length bin needs to be drawn."""
FEATURE_ROWS = 12
"""Inputs of the importance heat map: the largest per-method means."""
MIN_WITHIN = 5
"""Policies a method needs for its own score-fidelity Spearman (F27)."""
POLICY_SUMMARY = "complexity.json"
"""`interpretation.run.SUMMARY`, written last in a finished policy folder (that module
is not imported here: it loads JAX)."""
REPORT_LEGEND_COLS = 3
"""Legend columns in the 80 mm report column; wide variants keep the theme's."""
LEGEND_RC: dict[Any, Any] = {
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.9,
}
"""Tighter legend spacing, so `REPORT_LEGEND_COLS` method names fit in 80 mm."""
CELL_MM = 3.6
"""Height of one heat-map row in the report."""
SCATTER_MM = 62.0
"""Report height of a square scatter panel, title and axis included."""


# ----------------------------------------------------------------------- plumbing


def _draw(
    name: str,
    frames: Frames,
    out_dir: Path,
    theme: str,
    width: str,
    panels: Sequence[Panel],
    *,
    notes: Sequence[str] = (),
    extra: Sequence[Callable[[Theme], Any]] = (),
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """``panels`` per benchmark with the method legend plus ``extra`` entries."""
    th = _theme(theme, width)
    ctxs = _contexts(frames, n_boot, seed)
    if not ctxs:
        return _no_data(name, th, out_dir, "no tuned runs")
    mm = [[p.mm(c) for p in panels] for c in ctxs]
    names = list(dict.fromkeys(m for c in ctxs for m in c.names))
    # extra entries first: the two long count-bonus names then share a legend column
    handles = [*(h(th) for h in extra), *_method_handles(th, names)]
    fig = _figure(th, _height(th, mm, 0) + _legend_rows(th, len(handles)))
    added: list[str] = []
    for row, ctx in zip(_grid(fig, th, mm), ctxs, strict=True):
        for ax, panel in zip(row, panels, strict=True):
            panel.draw(ax, th, ctx, added)
            _title(ax, th, _panel_title(ctx.budget, panel.title), narrow=panel.narrow)
    pending = _pending([c.budget for c in ctxs])
    _close(fig, th, handles, [*notes, *_unique(added), *pending])
    return _save(fig, th, out_dir, name)


def _legend_cols(th: Theme) -> int:
    return th.legend_cols if th.wide else REPORT_LEGEND_COLS


def _legend_rows(th: Theme, entries: int) -> float:
    """Report height of a legend of ``entries`` in `_legend_cols` columns."""
    return ceil(entries / _legend_cols(th)) * LEGEND_ROW_MM


def _close(
    fig: Figure, th: Theme, handles: Sequence[Any], notes: Sequence[str]
) -> None:
    """`figures._finish` with the legend in `_legend_cols` columns, fewer if too wide.

    A legend of one entry is dropped and named in the notes instead (figstyle's rule).

    Long labels (F25 carries its optimism in them) can make the columns wider than
    the figure, so the legend is measured and redrawn with one column less until fits.
    """
    if len(handles) == 1:  # one entry: a note says what is drawn, no legend box
        notes = [f"Shown: {handles[0].get_label()}.", *notes]
        handles = ()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    for cols in range(_legend_cols(th), 0, -1):
        for old in list(fig.legends):
            old.remove()
        with mpl.rc_context(LEGEND_RC):
            _finish(fig, th, handles, notes, cols=cols)
        if not fig.legends or fig.legends[0].get_window_extent(renderer).width <= (
            fig.bbox.width
        ):
            return


def _budget_axes(
    th: Theme, heights: Sequence[float], entries: int
) -> tuple[Figure, list[Axes]]:
    """A figure, one axes per benchmark: stacked in the column, side by side wide."""
    rows = max(heights) if th.wide else sum(heights)
    total = rows + LEGEND_MM + _legend_rows(th, entries)
    fig = _figure(th, total)
    if th.wide:
        return fig, list(fig.subplots(1, len(heights), squeeze=False)[0])
    grid = fig.subplots(len(heights), 1, squeeze=False, height_ratios=list(heights))
    return fig, list(grid[:, 0])


def _log_steps(ax: Axes, th: Theme, ylabel: str) -> None:
    _style(ax, th, grid="both")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_text))
    _labels(ax, th, "env steps (log)", ylabel)


def _line(
    ax: Axes, th: Theme, x: Any, y: Any, method: str, gid: str, alpha: float = 1.0
) -> None:
    s = _style_of(th, method)
    (line,) = ax.plot(
        x,
        y,
        color=s["color"],
        ls=s["ls"],
        marker=s["marker"],
        ms=th.ms * 0.8,
        markerfacecolor=s["markerfacecolor"],
        lw=th.lw,
        alpha=alpha,
    )
    line.set_gid(gid)


def _cell_text(value: float, fmt: str) -> str:
    """``value`` in ``fmt``; empty for NaN, unsigned when it rounds to zero."""
    if pd.isna(value):
        return ""
    text = f"{value:{fmt}}"
    return text.lstrip("-") if float(text) == 0 else text


def _heatmap(
    ax: Axes, th: Theme, data: pd.DataFrame, gid: str, label: str, fmt: str
) -> None:
    """``data`` as a seaborn heat map on the theme ramp; NaN cells stay the surface.

    Wide variants print each cell with ``fmt``; a value that rounds to zero prints
    unsigned (a tiny negative importance is noise, not "-0.00").
    """
    shown = data.map(lambda v: _cell_text(v, fmt))
    sns.heatmap(
        data,
        ax=ax,
        cmap=_cmap(th, *th.seq),
        vmin=0.0,
        mask=data.isna(),
        annot=shown.to_numpy() if th.wide else False,
        fmt="",
        annot_kws={"fontsize": th.font},
        linewidths=0.4,
        linecolor=th.surface,
        cbar_kws={"label": label, "shrink": 0.8, "pad": 0.02},
    )
    ax.collections[0].set_gid(gid)
    ax.tick_params(colors=th.muted, labelsize=th.font, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(colors=th.muted, labelsize=th.font, length=2)
        cbar.ax.yaxis.label.set(color=th.ink2, fontsize=th.font)
        cbar.outline.set_visible(False)


# --------------------------------------------------------------------------- F21


def _mode_means(frames: Frames) -> pd.DataFrame:
    """Per tuned run and checkpoint the mean episode score of each eval mode."""
    runs = compare.tuned(frames["runs"])
    eps = frames["eval_episodes"]
    eps = eps[eps["run_id"].isin(runs["run_id"])]
    wide = eps.pivot_table(
        index=["run_id", "update", "env_steps"],
        columns="mode",
        values="score_pct",
        aggfunc="mean",
    ).reset_index()
    meta = runs.set_index("run_id")[["budget", "method"]]
    return wide.join(meta, on="run_id")


def greedy_gap(frames: Frames) -> pd.DataFrame:
    """Greedy minus stochastic mean eval score per tuned run and checkpoint."""
    wide = _mode_means(frames)
    for mode in ("stochastic", "greedy"):
        if mode not in wide.columns:
            wide[mode] = np.nan
    wide["gap"] = wide["greedy"] - wide["stochastic"]
    return wide[GAP_COLUMNS].sort_values(["budget", "method", "run_id", "update"])


def _gap_panel(table: pd.DataFrame) -> Callable[[Axes, Theme, Ctx, list[str]], None]:
    def draw(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
        mine = table[(table["budget"] == ctx.budget) & (table["env_steps"] > 0)]
        mine = mine.dropna(subset=["gap"])
        if mine.empty:
            _empty(ax, th, "no greedy and stochastic eval pairs")
            return
        ax.axhline(0.0, color=th.muted, lw=0.6)
        for method in ctx.names:
            runs = mine[mine["method"] == method]
            if runs.empty:
                continue
            grid = np.sort(runs["env_steps"].unique()).astype(float)
            curves = [
                (g["env_steps"].to_numpy(float), g["gap"].to_numpy(float))
                for _, g in runs.groupby("run_id")
            ]
            band = curve_band(curves, grid, "mean", ctx.n_boot, ctx.seed)
            color = _style_of(th, method)["color"]
            fill = ax.fill_between(
                grid, band["lo"], band["hi"], color=color, alpha=0.12, lw=0
            )
            fill.set_gid(_gid("band", ctx.budget, method))
            _line(
                ax, th, grid, band["estimate"], method, _gid("gap", ctx.budget, method)
            )
        _log_steps(ax, th, "greedy minus stochastic score (pp)")

    return draw


def plot_greedy_gap(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F21 ``greedy_gap``: does the argmax policy track the sampled one in training?

    Per method the mean over seeds of greedy minus stochastic eval score (`greedy_gap`,
    the same 64 eval seeds in both modes) at each checkpoint, with a 95% bootstrap
    band; below 0 the argmax policy scores less than the sampled one.
    """
    table = greedy_gap(frames)
    panel = Panel(
        "does the argmax policy keep up with the sampled one?",
        _gap_panel(table),
        lambda _c: 46.0 + PANEL_MM,
        narrow=False,
    )
    notes = [
        (
            "Mean over seeds of each seed's greedy minus stochastic mean eval score "
            "(percentage points) with a 95% bootstrap band. Value-based methods sample "
            "epsilon-greedy, so their gap closes partly by construction as epsilon "
            "decays; the policy-gradient gap compares the argmax with the policy's "
            "own distribution."
        ),
    ]
    return _draw(
        NAMES["F21"],
        frames,
        out_dir,
        theme,
        width,
        [panel],
        notes=notes,
        n_boot=n_boot,
        seed=seed,
    )


# --------------------------------------------------------------------------- F22


def _achievements(eps: pd.DataFrame) -> list[str]:
    return [
        c.removeprefix("ach/")
        for c in eps.columns
        if c.startswith("ach/") and f"unlock_step/{c[4:]}" in eps.columns
    ]


def _later(got: np.ndarray, steps: np.ndarray, skip: int, at: float) -> np.ndarray:
    """Per episode the unlocks other than column ``skip`` later than step ``at``."""
    later = got & (steps > at)
    later[:, skip] = False
    return later.sum(axis=1).astype(float)


def gatekeepers(eps: pd.DataFrame) -> pd.DataFrame:
    """Per run and achievement A: the lift in later unlocks among episodes alive at A.

    ``at`` is the method's median unlock step of A. Among the run's episodes still
    alive then (``length > at``), the lift is the mean number of other achievements
    unlocked after ``at`` in episodes that unlock A minus that in episodes that never
    do; NaN when one group is empty. Survival to ``at`` is so held equal.
    """
    names = _achievements(eps)
    if not names or eps.empty:
        return pd.DataFrame(columns=LIFT_COLUMNS)
    rows = []
    for method, mine in eps.groupby("method", sort=False):
        got = mine[[f"ach/{n}" for n in names]].to_numpy(bool)
        steps = mine[[f"unlock_step/{n}" for n in names]].to_numpy(float)
        length = mine["length"].to_numpy(float)
        runs = mine["run_id"].to_numpy()
        for i, name in enumerate(names):
            if not got[:, i].any():
                continue
            at = float(np.median(steps[got[:, i], i]))
            alive = length > at
            later, has = _later(got, steps, i, at), got[:, i]
            for run in np.unique(runs):
                here = alive & (runs == run)
                yes, no = later[here & has], later[here & ~has]
                lift = yes.mean() - no.mean() if yes.size and no.size else np.nan
                rows.append((method, run, name, at, yes.size, no.size, float(lift)))
    return pd.DataFrame(rows, columns=LIFT_COLUMNS)


def _gate_cats(ctx: Ctx) -> tuple[Cats, list[str]]:
    """Achievements of mid share, the largest mean lift over methods first.

    Returns:
        The rows, and the mid-share achievements no method has a dot for (not drawn).
    """
    table = gatekeepers(ctx.eps)
    names = _achievements(ctx.eps)
    share = {n: float(ctx.eps[f"ach/{n}"].mean()) for n in names}
    keep = [n for n, s in share.items() if GATE_SHARE <= s <= 1 - GATE_SHARE]
    sums = table.groupby(["method", "achievement"])[["with_a", "without_a"]].sum()
    enough = sums.min(axis=1) >= GATE_MIN
    means = table.groupby(["method", "achievement"])["lift"].mean()[enough]
    order = means.groupby(level="achievement").mean().to_dict()
    keys = sorted((n for n in keep if n in order), key=lambda n: -order[n])

    def values(method: str, key: str) -> np.ndarray:
        if not enough.get((method, key), False):
            return np.empty(0)
        mine = table[(table["method"] == method) & (table["achievement"] == key)]
        return np.asarray(mine["lift"], dtype=float)

    labels = [k.replace("_", " ") for k in keys]
    return Cats(keys, labels, values), [n for n in keep if n not in order]


def _gate_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    cats, empty = _gate_cats(ctx)
    if empty:  # rows no method can fill are not drawn, only named
        notes.append(
            f"{_budget_name(ctx.budget)}: no method has {GATE_MIN} alive episodes with "
            f"and without {', '.join(n.replace('_', ' ') for n in empty)}."
        )
    if not cats.keys:
        _empty(ax, th, "no achievement in the middle share of episodes")
        return
    ax.axvline(0.0, color=th.muted, lw=0.6)
    _dot_panel(
        ax,
        th,
        ctx,
        "gate",
        cats,
        "extra later unlocks with it\n(episodes alive at its median step)",
        notes,
        how="mean",
        missing=f"has under {GATE_MIN} alive episodes with or without each",
    )


def plot_gatekeepers(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F22 ``gatekeepers``: which achievements open up the rest of the run?

    Rows: achievements unlocked in 5% to 95% of the benchmark's final stochastic eval
    episodes, the largest mean lift first. A dot is the mean over seeds of the lift of
    `gatekeepers` (episodes alive at the method's median unlock step of the row, with
    the achievement against without it) with a bootstrap interval; a method needs
    `GATE_MIN` alive episodes in each group.
    """
    panel = Panel(
        "which achievements open up the rest of the run?",
        _gate_panel,
        lambda c: _dot_mm(c, max(1, len(_gate_cats(c)[0].keys))),
    )
    notes = [
        (
            "Final stochastic eval episodes. For each achievement only the episodes "
            "still alive at the method's median step of unlocking it count; a dot is "
            "the mean over seeds of the later unlocks of the episodes that unlock it "
            "minus those of the episodes that never do, with a 95% bootstrap interval. "
            "Survival is held equal, skill is not: an association, not an effect."
        ),
    ]
    return _draw(
        NAMES["F22"],
        frames,
        out_dir,
        theme,
        width,
        [panel],
        notes=notes,
        n_boot=n_boot,
        seed=seed,
    )


# --------------------------------------------------------------------------- F23


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho; NaN under 3 pairs or for a constant side."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def seed_fate(frames: Frames) -> pd.DataFrame:
    """Per method and checkpoint: Spearman of seed checkpoint and final scores.

    The checkpoint score is the run's mean stochastic eval score; the final score is the
    comparison's ``final_score``, so the last checkpoint is not 1 by construction.
    """
    wide = _mode_means(frames)
    if "stochastic" not in wide.columns:
        return pd.DataFrame(columns=FATE_COLUMNS)
    final = compare.tuned(frames["runs"]).set_index("run_id")["final_score"]
    wide = wide.join(final, on="run_id")
    rows = [
        {
            "budget": b,
            "method": m,
            "update": u,
            "env_steps": float(g["env_steps"].median()),
            "seeds": len(g),
            "rho": _spearman(
                g["stochastic"].to_numpy(float), g["final_score"].to_numpy(float)
            ),
        }
        for (b, m, u), g in wide.groupby(["budget", "method", "update"])
    ]
    return pd.DataFrame(rows, columns=FATE_COLUMNS)


def critical_rho(seeds: int, alpha: float = 0.05) -> float:
    """|rho| under which a two-sided test at ``alpha`` finds no association."""
    t = float(sps.t.ppf(1 - alpha / 2, seeds - 2))
    return t / float(np.sqrt(seeds - 2 + t**2))


def _fate_panel(table: pd.DataFrame) -> Callable[[Axes, Theme, Ctx, list[str]], None]:
    def draw(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
        mine = table[(table["budget"] == ctx.budget) & (table["env_steps"] > 0)]
        mine = mine.dropna(subset=["rho"])
        if mine.empty:
            _empty(ax, th, "no checkpoint with three seeds and distinct scores")
            return
        seeds = int(mine["seeds"].median())
        crit = critical_rho(seeds) if seeds > 3 else 1.0
        null = ax.axhspan(-crit, crit, color=th.grid, alpha=0.6, lw=0)
        null.set_gid(_gid("null", ctx.budget))
        for method in ctx.names:
            g = mine[mine["method"] == method].sort_values("update")
            if len(g):
                _line(
                    ax,
                    th,
                    g["env_steps"],
                    g["rho"],
                    method,
                    _gid("fate", ctx.budget, method),
                    alpha=0.5,
                )
        pooled = mine.groupby("update").agg(
            x=("env_steps", "median"), y=("rho", "mean")
        )
        (line,) = ax.plot(pooled["x"], pooled["y"], color=th.ink, lw=th.lw * 2.2)
        line.set_gid(_gid("fate", ctx.budget, "pooled"))
        ax.set_ylim(-1.05, 1.05)
        _log_steps(ax, th, "Spearman rho with the final score")
        notes.append(
            f"{_budget_name(ctx.budget)}: grey band |rho| < {crit:.2f}, not "
            f"significant at 5% with {seeds} seeds."
        )

    return draw


def _pooled_handle(th: Theme) -> Line2D:
    return Line2D([], [], color=th.ink, lw=th.lw * 2.2, label="method mean")


def plot_seed_fate(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """F23 ``seed_fate``: when is a seed's fate decided?

    Per method the Spearman correlation between its seeds' mean stochastic eval score
    at each checkpoint and their final score (`seed_fate`), the mean over methods in
    bold; the grey band is the 5% critical value for the seed count.
    """
    panel = Panel(
        "when is a seed's fate decided?",
        _fate_panel(seed_fate(frames)),
        lambda _c: 46.0 + PANEL_MM,
        narrow=False,
    )
    notes = [
        (
            "Spearman rho between the seeds' mean stochastic eval score at a "
            "checkpoint and their final score (the training-episode score, so the last "
            "checkpoint also measures eval against training agreement): high early "
            "means a run could be stopped early. Early checkpoints have many tied "
            "scores, which pull rho towards 0; the method mean is no test."
        ),
    ]
    return _draw(
        NAMES["F23"],
        frames,
        out_dir,
        theme,
        width,
        [panel],
        notes=notes,
        extra=[_pooled_handle],
    )


# --------------------------------------------------------------------------- F24


def _effect_label(r: Any) -> str:
    head = f"{r.method} ({r.config_ref})"
    if np.isfinite(r.poi_tuned):
        return f"{head}\nPoI {r.poi_tuned:.2f}, p {r.p_holm:.2g}"
    return head


def _effect_panel(
    ax: Axes, th: Theme, budget: str, rows: pd.DataFrame, runs: pd.DataFrame
) -> None:
    ax.axvline(0.0, color=th.muted, lw=0.6)
    tuned = compare.tuned(runs).groupby("method")["final_score"].mean()
    fresh = compare.tuned(runs)[~compare.tuned(runs)["phases"].map(reused_from_tuning)]
    counts = compare.tuned(runs).groupby("method").size()
    fresh_n, fresh_mean = (
        fresh.groupby("method").size(),
        fresh.groupby("method")["final_score"].mean(),
    )
    rng = np.random.default_rng(0)  # jitter only; it moves no number
    for y, r in enumerate(rows.itertuples()):
        s = _style_of(th, r.method, r.config_ref)
        key = (budget, r.method, r.config_ref)
        mine = runs[(runs["method"] == r.method) & (runs["config_ref"] == r.config_ref)]
        diffs = mine["final_score"].to_numpy(float) - tuned.get(r.method, np.nan)
        dots = ax.scatter(
            diffs,
            y + rng.uniform(-0.22, 0.22, len(diffs)),
            s=(th.ms * 0.8) ** 2,
            color=s["color"],
            alpha=0.35,
            lw=0,
        )
        dots.set_gid(_gid("seeds", *key))
        ci = ax.hlines(y, r.diff_lo, r.diff_hi, colors=s["color"], lw=th.lw * 1.6)
        ci.set_gid(_gid("ci", *key))
        (dot,) = ax.plot(
            [r.diff],
            [y],
            ls="",
            marker="o",
            ms=th.ms * 1.3,
            color=s["color"],
            markerfacecolor=th.surface,
            markeredgewidth=th.lw * 1.4,
        )
        dot.set_gid(_gid("effect", *key))
        n = int(fresh_n.get(r.method, 0))
        if compare.MIN_FRESH <= n < counts.get(r.method, 0):  # some seeds were reused
            (cross,) = ax.plot(
                [mine["final_score"].mean() - fresh_mean[r.method]],
                [y],
                ls="",
                marker="x",
                ms=th.ms * 1.2,
                mew=th.lw * 1.4,
                color=s["color"],
            )
            cross.set_gid(_gid("fresh", *key))
    _style(ax, th)
    ax.set_yticks(range(len(rows)), [_effect_label(r) for r in rows.itertuples()])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    blocks = rows["block"].to_numpy()
    for edge in np.flatnonzero(blocks[1:] != blocks[:-1]):
        ax.axhline(edge + 0.5, color=th.muted, lw=0.5)
    _labels(ax, th, "ablation minus tuned\nfinal score (pp)")


def plot_ablation_effects(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
    table: pd.DataFrame | None = None,
) -> Figure:
    """F24 ``ablation_effects``: which component carries each method?

    A forest of `compare.ablation_table`: per ablation the difference of means to its
    tuned config with the 95% bootstrap CI (Welch t at 1B), each ablation seed minus
    the tuned mean as a faint dot, and at 1M the PoI and Holm-adjusted p in the label.
    """
    th = _theme(theme, width)
    table = (
        compare.ablation_table(frames, n_boot=n_boot, seed=seed)
        if table is None
        else table
    )
    present = [b for b, _ in compare.budgets(frames["runs"])]
    shown = [b for b in present if (table["benchmark"] == b).any()]
    if not shown:
        return _no_data(NAMES["F24"], th, out_dir, "no ablation runs")
    sizes = [int((table["benchmark"] == b).sum()) for b in shown]
    fig, axes = _budget_axes(th, [n * th.pitch_mm * 2.2 + PANEL_MM for n in sizes], 2)
    for ax, budget in zip(axes, shown, strict=True):
        rows = table[table["benchmark"] == budget].reset_index(drop=True)
        runs = frames["runs"][frames["runs"]["budget"] == budget]
        _effect_panel(ax, th, budget, rows, runs)
        what = "which component carries each method?"
        _title(ax, th, _panel_title(budget, what), narrow=True)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=th.ms * 1.3,
            color=th.ink2,
            markerfacecolor=th.surface,
            label="vs all tuned seeds",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="x",
            ms=th.ms * 1.2,
            color=th.ink2,
            label="vs fresh tuned seeds",
        ),
    ]
    notes = [
        (
            "Hollow dot: ablation minus tuned mean final score with its 95% bootstrap "
            "CI (Welch t at 1B); faint dots: each ablation seed minus the tuned mean. "
            "PoI: P(ablation beats tuned); p: Mann-Whitney, Holm-adjusted per block "
            "(1M only). Cross: the difference to the tuned seeds not reused from "
            "tuning; the reused ones were selected, so the table's baseline is "
            "optimistic (R102)."
        ),
        *_pending(present),
    ]
    _close(fig, th, handles, notes)
    return _save(fig, th, out_dir, NAMES["F24"])


# --------------------------------------------------------------------------- F25


def _fresh_scores(runs: pd.DataFrame, budget: str, method: str) -> np.ndarray:
    mine = compare.tuned(runs)
    mine = mine[(mine["budget"] == budget) & (mine["method"] == method)]
    fresh = mine.loc[~mine["phases"].map(reused_from_tuning), "final_score"]
    return np.asarray(fresh, dtype=float)


def _optimism_panel(
    ax: Axes,
    th: Theme,
    budget: str,
    rows: pd.DataFrame,
    frames: Frames,
    n_boot: int,
    seed: int,
) -> None:
    values = []
    for r in rows.itertuples():
        s = _style_of(th, r.method)
        fresh = _fresh_scores(frames["runs"], budget, r.method)
        ci = stratified_bootstrap_ci(
            {"seeds": fresh.tolist()},
            lambda d: float(np.mean(d["seeds"])),
            n_boot,
            seed,
        )
        bar = ax.vlines(r.tuning_mean, ci.lo, ci.hi, colors=s["color"], lw=th.lw * 1.4)
        bar.set_gid(_gid("ci", budget, r.method))
        (dot,) = ax.plot(
            [r.tuning_mean],
            [r.fresh_mean],
            ls="",
            marker=s["marker"],
            ms=th.ms * 1.3,
            color=s["color"],
            markerfacecolor=s["markerfacecolor"],
        )
        dot.set_gid(_gid("optimism", budget, r.method))
        values += [r.tuning_mean, ci.lo, ci.hi]
    lo, hi = min(values), max(values)
    pad = 0.08 * (hi - lo or 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axline((lo, lo), slope=1.0, color=th.muted, lw=0.7, ls="--")
    ax.set_aspect("equal", adjustable="box")
    _style(ax, th, grid="both")
    _labels(ax, th, "tuning mean of the chosen config", "fresh-seed final mean")


def plot_tuning_optimism(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
    table: pd.DataFrame | None = None,
) -> Figure:
    """F25 ``tuning_optimism``: did tuning flatter the configs it chose?

    Per tuned unit with a computable optimism (`compare.tuning_optimism`; R101, R102:
    fresh final seeds only), its tuning mean against its fresh final mean with a 95%
    bootstrap CI over the fresh seeds; below the diagonal the tuning score was too high.
    The units without one are listed with the table's reason.
    """
    th = _theme(theme, width)
    table = compare.tuning_optimism(frames) if table is None else table
    table = table.drop_duplicates(["benchmark", "method"])  # controls: both blocks
    ok = table[table["optimism"].notna()]
    present = [b for b, _ in compare.budgets(frames["runs"])]
    shown = [b for b in present if (ok["benchmark"] == b).any()]
    if not shown:
        return _no_data(NAMES["F25"], th, out_dir, "no tuned config with fresh seeds")
    names = list(dict.fromkeys(ok["method"]))
    fig, axes = _budget_axes(th, [SCATTER_MM] * len(shown), len(names))
    for ax, budget in zip(axes, shown, strict=True):
        rows = ok[ok["benchmark"] == budget]
        _optimism_panel(ax, th, budget, rows, frames, n_boot, seed)
        _title(ax, th, _panel_title(budget, "did tuning flatter the chosen configs?"))
    na = table[table["optimism"].isna()]
    reasons = [
        f"{_budget_name(b)} {r.removeprefix('n/a: ')}: {', '.join(g['method'])}."
        for (b, r), g in na.groupby(["benchmark", "optimism_note"])
    ]
    notes = [
        (
            "Dot: mean of the fresh final seeds (not reused from tuning, R102) with a "
            "95% bootstrap CI; legend: the tuning optimism, tuning minus fresh mean "
            "(R101). The tuning mean has 3 seeds and no interval; with seed noise "
            "alone it would still regress towards the fresh mean."
        ),
        *reasons,
        *_pending(present),
    ]
    handles = _method_handles(th, names)
    gaps = ok.groupby("method")["optimism"].agg(list)
    for h in handles:
        shown_gaps = ", ".join(f"{g:+.2f}" for g in gaps[h.get_label()])
        h.set_label(f"{h.get_label()} ({shown_gaps})")
    _close(fig, th, handles, notes)
    return _save(fig, th, out_dir, NAMES["F25"])


# --------------------------------------------------------------------------- F26


def _count_shares(ctx: Ctx) -> pd.DataFrame:
    """Methods down, achievements per episode across: the share of the episodes."""
    counts = ctx.eps["n_achievements"].astype(int)
    top = int(counts.max()) if len(counts) else 0
    table = pd.crosstab(ctx.eps["method"], counts, normalize="index")
    table = table.reindex(columns=range(top + 1), fill_value=0.0)
    return table.reindex(index=[m for m in ctx.names if m in table.index])


def _spread_panel(ax: Axes, th: Theme, ctx: Ctx, _notes: list[str]) -> None:
    if ctx.eps.empty:
        _empty(ax, th, "no final eval episodes")
        return
    shares = _count_shares(ctx)
    _heatmap(
        ax,
        th,
        shares * 100,
        _gid("heat", ctx.budget, "achievements"),
        "% of episodes",
        ".0f",
    )
    means = ctx.eps.groupby("method")["n_achievements"].mean()
    for i, method in enumerate(shares.index):
        if method in means:
            # low in the row, under the number a wide variant prints in the cell
            (dot,) = ax.plot(
                [means[method] + 0.5],
                [i + 0.72],
                ls="",
                marker="^",
                ms=th.ms * 1.1,
                color=th.ink,
                mec=th.surface,
                mew=0.6,
            )
            dot.set_gid(_gid("mean", ctx.budget, method))
    ax.set_yticks(np.arange(len(shares)) + 0.5, list(shares.index), rotation=0)
    ax.tick_params(axis="x", rotation=0)
    if not th.wide:  # 13 count labels do not fit the column side by side
        for tick in ax.get_xticklabels()[1::2]:
            tick.set_visible(False)
    _labels(ax, th, "achievements per episode")


def _length_panel(ax: Axes, th: Theme, ctx: Ctx, notes: list[str]) -> None:
    eps = ctx.eps
    if eps.empty:
        _empty(ax, th, "no final eval episodes")
        return
    edges = np.unique(np.quantile(eps["length"], np.linspace(0, 1, LENGTH_BINS + 1)))
    bins = pd.cut(eps["length"], edges, include_lowest=True)
    stats = (
        eps
        .assign(bin=bins)
        .groupby(["method", "bin"], observed=True)
        .agg(x=("length", "median"), y=("n_achievements", "mean"), n=("length", "size"))
    )
    for method in ctx.names:
        if method not in stats.index.get_level_values(0):
            continue
        g = stats.loc[method]
        g = g[g["n"] >= LENGTH_MIN]
        if len(g):
            _line(ax, th, g["x"], g["y"], method, _gid("length", ctx.budget, method))
    _style(ax, th, grid="both")
    _labels(ax, th, "episode length (steps, bin median)", "achievements per episode")
    notes.append(
        f"{_budget_name(ctx.budget)}: lines over {LENGTH_BINS} equal-count length "
        f"bins; a bin needs {LENGTH_MIN} episodes of the method."
    )


def plot_episode_spread(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """F26 ``episode_spread``: how uneven are a policy's episodes, and does length pay?

    A heat map of the share of each method's final stochastic eval episodes by
    achievements unlocked (the episode return is that count plus a small health term),
    a triangle at its mean; then the mean achievements per episode over equal-count
    episode-length bins.
    """
    panels = [
        Panel(
            "how uneven are a policy's episodes?",
            _spread_panel,
            lambda c: len(c.names) * CELL_MM + PANEL_MM + 6.0,
        ),
        Panel(
            "does surviving longer buy achievements?",
            _length_panel,
            lambda _c: 46.0 + PANEL_MM,
            narrow=False,
        ),
    ]
    notes = [
        (
            "Final stochastic eval episodes of the tuned runs, all seeds pooled; heat "
            "map triangle: the method's mean achievements per episode. Length and "
            "achievements are associated, not causal: eating, drinking and sleeping "
            "also keep an episode alive; the last length bin is open-ended."
        ),
    ]
    return _draw(NAMES["F26"], frames, out_dir, theme, width, panels, notes=notes)


# ------------------------------------------------------------------ F27 and F28


def distilled(root: Path, budget: str, variant: str = "base") -> pd.DataFrame:
    """Finished tuned policies under ``root/<budget>``: score and 256-leaf fidelity.

    A policy is finished once its `POLICY_SUMMARY` exists (written last); a torn one
    (a live distillation rewriting it) is left out with a warning.
    """
    rows = []
    for path in sorted((root / budget).glob(f"*/*/{POLICY_SUMMARY}")):
        try:
            s = json.loads(path.read_text())
        except ValueError:
            logger.warning("%s is torn (being written?): left out", path)
            continue
        if s.get("config_ref", compare.TUNED) != compare.TUNED:
            continue
        fit = s.get("fits", {}).get(variant) or {}
        rows.append({
            "method": s["method"],
            "run_id": s["run_id"],
            "seed": s["seed"],
            "score": float(s["score"]),
            "fidelity_256": fit.get("fidelity_256", np.nan),
            "folder": path.parent,
        })
    cols = ["method", "run_id", "seed", "score", "fidelity_256", "folder"]
    return pd.DataFrame(rows, columns=cols)


def importance_means(policies: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    """Inputs down, methods across: mean over seeds of the 256-leaf importance."""
    parts = []
    for r in policies.itertuples():
        path = Path(r.folder) / f"importance_{variant}.csv"
        if path.is_file():
            parts.append(pd.read_csv(path).assign(method=r.method, run_id=r.run_id))
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    return rows.pivot_table(
        index="feature", columns="method", values="importance", aggfunc="mean"
    )


def _distil_pending(
    frames: Frames, root: Path
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Per benchmark its finished policies, and one note per benchmark short of some."""
    found, notes = {}, []
    for budget, runs in compare.budgets(compare.tuned(frames["runs"])):
        policies = distilled(root, budget)
        policies = policies[policies["run_id"].isin(runs["run_id"])]
        have = policies.groupby("method").size()
        want = runs.groupby("method").size()
        short = [
            f"{m} ({int(have.get(m, 0))} of {n})"
            for m, n in want.items()
            if have.get(m, 0) < n
        ]
        if short:
            notes.append(
                f"{_budget_name(budget)} distillation pending: {', '.join(short)}."
            )
        if len(policies):
            found[budget] = policies
    return found, notes


def _score_panel(
    ax: Axes, th: Theme, budget: str, policies: pd.DataFrame, names: Sequence[str]
) -> str:
    """Score against fidelity per policy; returns the Spearman line of the title."""
    for method in names:
        g = policies[policies["method"] == method]
        if g.empty:
            continue
        s = _style_of(th, method)
        dots = ax.scatter(
            g["fidelity_256"],
            g["score"],
            s=(th.ms * 1.1) ** 2,
            marker=s["marker"],
            color=s["color"],
            edgecolors=th.surface,
            linewidths=0.3,
        )
        dots.set_gid(_gid("distil", budget, method))
    ok = policies.dropna(subset=["fidelity_256", "score"])
    _style(ax, th, grid="both")
    _labels(ax, th, "fidelity at 256 leaves (top-1)", "final score (%)")
    if len(ok) < 3:
        return "Spearman needs 3 policies"
    res = sps.spearmanr(ok["fidelity_256"], ok["score"])
    within = within_rho(ok)
    shown = f"{within:.2f}" if np.isfinite(within) else "n/a"
    return (
        f"pooled Spearman rho {res.statistic:.2f} (p {res.pvalue:.2g}, {len(ok)} "
        f"policies), within methods {shown}"
    )


def within_rho(policies: pd.DataFrame) -> float:
    """Mean over methods with `MIN_WITHIN` policies of their score-fidelity rho."""
    rhos = [
        _spearman(g["fidelity_256"].to_numpy(float), g["score"].to_numpy(float))
        for _, g in policies.groupby("method")
        if len(g) >= MIN_WITHIN
    ]
    rhos = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(rhos)) if rhos else float("nan")


def _distil_figure(
    name: str,
    frames: Frames,
    out_dir: Path,
    th: Theme,
    root: Path,
    draw: Callable[[Axes, str, pd.DataFrame], str],
    handles: Callable[[dict[str, pd.DataFrame]], list[Any]],
    panel_mm: float,
    notes: Sequence[str],
) -> Figure:
    """One panel per benchmark with finished policies, pending ones in the notes."""
    found, pending = _distil_pending(frames, root)
    present = [b for b, _ in compare.budgets(frames["runs"])]
    if not found:
        return _no_data(name, th, out_dir, "distillation pending, no policy finished")
    entries = handles(found)
    fig, axes = _budget_axes(th, [panel_mm] * len(found), len(entries))
    for ax, (budget, policies) in zip(axes, found.items(), strict=True):
        _title(ax, th, _panel_title(budget, draw(ax, budget, policies)))
    _close(fig, th, entries, [*notes, *pending, *_pending(present)])
    return _save(fig, th, out_dir, name)


def _order(frames: Frames) -> list[str]:
    """Methods in table order: the controls, then each block's methods."""
    tuned = compare.tuned(frames["runs"])
    return list(dict.fromkeys(m for _, m, _ in compare.members(tuned)))


def plot_distil_score(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    interpretation: Path = INTERPRETATION,
) -> Figure:
    """F27 ``distil_score``: do the policies that distil well also score well?

    One dot per finished tuned policy of the distillation (``interpretation``): its
    final score against the test fidelity of its 256-leaf tree (base fit), with R106's
    Spearman over all policies in the title; methods short of policies are pending.
    """
    th, order = _theme(theme, width), _order(frames)

    def draw(ax: Axes, budget: str, policies: pd.DataFrame) -> str:
        spearman = _score_panel(ax, th, budget, policies, order)
        return f"do policies that distil well score well? {spearman}"

    def handles(found: dict[str, pd.DataFrame]) -> list[Any]:
        have = {m for p in found.values() for m in p["method"]}
        return _method_handles(th, [m for m in order if m in have])

    notes = [
        (
            "One dot per distilled policy: base fit, test states of its final "
            "checkpoint. The pooled rho (R106) mostly compares methods; within "
            f"methods: the mean rho over methods with {MIN_WITHIN} or more policies."
        ),
    ]
    return _distil_figure(
        NAMES["F27"],
        frames,
        out_dir,
        th,
        interpretation,
        draw,
        handles,
        SCATTER_MM,
        notes,
    )


def plot_distil_features(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    interpretation: Path = INTERPRETATION,
) -> Figure:
    """F28 ``distil_features``: which inputs do the distilled trees rely on?

    The mean over a method's finished policies of each input's permutation importance
    for its 256-leaf tree (base fit); the `FEATURE_ROWS` inputs with the largest
    per-method mean, a blank cell where the input is not one of that method's.
    """
    th, order = _theme(theme, width), _order(frames)
    title = "which inputs do the distilled trees rely on?"

    def draw(ax: Axes, budget: str, policies: pd.DataFrame) -> str:
        means = importance_means(policies)
        if means.empty:
            _empty(ax, th, "no importance file yet")
            return title
        top = means.max(axis=1).sort_values(ascending=False).index[:FEATURE_ROWS]
        data = means.loc[top, [m for m in order if m in means.columns]]
        data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
        _heatmap(
            ax,
            th,
            data,
            _gid("heat", budget, "importance"),
            "permutation importance",
            ".2f",
        )
        ax.tick_params(axis="x", rotation=90)
        return title

    notes = [
        (
            "Permutation importance: drop in the 256-leaf tree's test fidelity when "
            "the input is shuffled, mean over a method's distilled seeds (base fit); "
            "blank: not an input of that method. Methods differ in their inputs "
            "(memory proxies, skill flags), and correlated inputs split importance, so "
            "compare within a column before across."
        ),
    ]
    rows_mm = FEATURE_ROWS * CELL_MM + PANEL_MM + 18.0  # rotated method names below
    return _distil_figure(
        NAMES["F28"],
        frames,
        out_dir,
        th,
        interpretation,
        draw,
        lambda _f: [],
        rows_mm,
        notes,
    )


ANGLE_FIGURES: dict[str, Callable[..., Figure]] = {
    "F21": plot_greedy_gap,
    "F22": plot_gatekeepers,
    "F23": plot_seed_fate,
    "F24": plot_ablation_effects,
    "F25": plot_tuning_optimism,
    "F26": plot_episode_spread,
    "F27": plot_distil_score,
    "F28": plot_distil_features,
}
"""Figure id -> its function (the results CLI loops over it)."""
