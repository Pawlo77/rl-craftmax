"""Analysis-angle figures F21-F28: questions the core figures leave open.

Each ``plot_*`` follows `analysis.figures` (report 80 mm or ``double``, dark ``video``,
``<stem>[_double][_video].png``, one panel per file, no title or text on the image) and
draws ONE benchmark (the results CLI cuts the frames per benchmark). Question, reading
and numbers go to the caption each figure registers; method colours are in
``legend_methods`` (F24 keeps a hollow-dot against cross legend). Intervals are
`evaluation.stats` bootstraps; F24 and F25 draw `compare` tables; F27 and F28 read the
finished distillation policies (R106). Artist gid ``<kind>/<budget>/<method>[/<key>]``.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures_angles.py]]
"""

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats as sps

from ..evaluation.hardware import reused_from_tuning
from ..evaluation.stats import BOOTSTRAP_N, curve_band, stratified_bootstrap_ci
from ..visualisation.charts import MAX_ROWS_MM, ROWS_BASE_MM, Shape, shape_height_mm
from . import compare
from .figures import (
    Theme,
    _cmap,
    _figure,
    _finish,
    _gid,
    _labels,
    _no_data,
    _save,
    _style,
    _style_of,
    _theme,
)
from .figures_game import Ctx, _contexts, _interval
from .latex import INTERPRETATION
from .load import Frames

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
"""Figure id -> file stem (F26 also writes `LENGTH_STEM`)."""
LENGTH_STEM = "achievements_vs_length"
"""Second file of F26: achievements per episode over episode-length bins."""
STEMS = {fid: [stem] for fid, stem in NAMES.items()} | {
    "F26": [NAMES["F26"], LENGTH_STEM]
}
"""Figure id -> every file stem it writes."""
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
"""`gatekeepers`: per run and achievement, the landmark step ``at`` (the method's median
unlock step), the episodes alive then that hold it and do not yet, and the lift."""
FATE_COLUMNS = ["budget", "method", "update", "env_steps", "seeds", "rho"]
"""`seed_fate`: Spearman of the seeds' checkpoint scores and final scores."""
GATE_SHARE = 0.05
"""Gatekeeper rows: achievements unlocked in this share of episodes to 1 minus it."""
GATE_MIN = 10
"""Alive episodes with and without the achievement a method needs for a dot."""
GATE_METHOD_MM = 1.05
"""Row height per method of the gatekeeper dots (a dot is 0.9 mm)."""
GATE_GAP_MM = 1.4
"""Space between two achievement rows of the gatekeeper figure."""
GATE_SPREAD = 0.8
"""Share of a row the dodged methods use."""
ABLATION_PITCH = 1.5
"""Row pitch of the ablation forest in `Theme.pitch_mm` (a dot, a cross, seed dots)."""
LENGTH_BINS = 8
"""Equal-count episode-length bins of the benchmark (`LENGTH_STEM`)."""
LENGTH_MIN = 10
"""Episodes of a method a length bin needs to be drawn."""
FEATURE_ROWS = 12
"""Inputs of the importance heat map: the largest per-method means."""
MIN_WITHIN = 5
"""Policies a method needs for its own score-fidelity Spearman (F27)."""
POLICY_SUMMARY = "complexity.json"
"""`interpretation.run.SUMMARY`, written last in a finished policy folder (that module
is not imported here: it loads JAX)."""
BAR_FRACTION = 0.05
"""Share of the canvas width a heat map's colour bar takes."""


# ----------------------------------------------------------------------- plumbing


def _context(frames: Frames, n_boot: int = BOOTSTRAP_N, seed: int = 0) -> Ctx | None:
    """The benchmark ``frames`` hold (None without tuned runs).

    Raises:
        ValueError: If the frames hold several benchmarks: a figure is drawn per
            benchmark (the results CLI cuts them), its file stem has no budget.
    """
    ctxs = _contexts(frames, n_boot, seed)
    if len(ctxs) > 1:
        msg = f"draw one benchmark at a time, got {[c.budget for c in ctxs]}"
        raise ValueError(msg)
    return ctxs[0] if ctxs else None


def _describe(*parts: str) -> str:
    """One line of caption from sentences (the notes table has a cell per figure)."""
    return " ".join(" ".join(p.split()) for p in parts if p)


def _panel(
    th: Theme, shape: Shape, height_mm: float | None = None
) -> tuple[Figure, Axes]:
    """A figure of the size class ``shape`` with its one axes."""
    fig = _figure(th, height_mm, shape=shape)
    return fig, fig.add_subplot()


def _millions(v: float, _pos: int | None = None) -> str:
    """Ticks in millions of steps, plain numbers (the axis label carries the unit)."""
    return f"{v / 1e6:.3g}"


def _log_steps(ax: Axes, th: Theme, ylabel: str) -> None:
    _style(ax, th, grid="both")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _labels(ax, th, "env steps (M, log)", ylabel)


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
        cbar_kws={"label": label, "shrink": 0.8, "pad": 0.02, "fraction": BAR_FRACTION},
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


class Rows(NamedTuple):
    """The rows of a dot figure: keys and labels, the seeds' values of (method, key)."""

    keys: list[str]
    labels: list[str]
    values: Callable[[str, str], np.ndarray]


def _dot_rows(
    ax: Axes, th: Theme, ctx: Ctx, kind: str, rows: Rows, xlabel: str
) -> set[str]:
    """Per row each method's mean over seeds with its bootstrap interval, dodged.

    Returns:
        The methods that have a dot in some row.
    """
    offsets = np.linspace(-GATE_SPREAD / 2, GATE_SPREAD / 2, max(len(ctx.names), 2))
    seen: set[str] = set()
    for j, method in enumerate(ctx.names):
        style = _style_of(th, method)
        for i, key in enumerate(rows.keys):
            est = _interval(rows.values(method, key), "mean", ctx.n_boot, ctx.seed)
            if est is None:
                continue
            seen.add(method)
            y = i + offsets[j]
            ax.hlines(
                y,
                est.lo,
                est.hi,
                colors=style["color"],
                linewidth=th.lw,
                gid=_gid("ci", ctx.budget, method, key),
            )
            (dot,) = ax.plot(
                [est.estimate],
                [y],
                ls="",
                marker=style["marker"],
                ms=th.ms * 0.8,
                color=style["color"],
            )
            dot.set_gid(_gid(kind, ctx.budget, method, key))
    _style(ax, th)
    ax.set_ylim(len(rows.keys) - 0.5, -0.5)
    # one row: its name goes to the caption, a lone tick label would repeat the axis
    ax.set_yticks(range(len(rows.keys)), rows.labels if len(rows.keys) > 1 else [])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    _labels(ax, th, xlabel)
    return seen


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
    th, name = _theme(theme, width), NAMES["F21"]
    ctx, table = _context(frames, n_boot, seed), greedy_gap(frames)
    mine = table[table["env_steps"] > 0].dropna(subset=["gap"])
    if ctx is None or mine.empty:
        return _no_data(name, th, out_dir, "no greedy and stochastic eval pairs")
    fig, ax = _panel(th, Shape.CURVE)
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
        fill = ax.fill_between(
            grid,
            band["lo"],
            band["hi"],
            color=_style_of(th, method)["color"],
            alpha=0.12,
            lw=0,
        )
        fill.set_gid(_gid("band", ctx.budget, method))
        _line(ax, th, grid, band["estimate"], method, _gid("gap", ctx.budget, method))
    _log_steps(ax, th, "greedy minus stochastic (pp)")
    description = _describe(
        "Does the argmax policy keep up with the sampled one?",
        "Mean over seeds of each seed's greedy minus stochastic mean eval score, in",
        "percentage points, at each checkpoint (env steps in millions, log axis),",
        "with a 95% bootstrap band; below 0 the argmax policy scores less than the",
        "sampled one.",
        "Value-based methods sample epsilon-greedy, so their gap closes partly by",
        "construction as epsilon decays; the policy-gradient gap compares the argmax",
        "with the policy's own distribution. Colours: method legend.",
    )
    return _save(fig, th, out_dir, name, title="Greedy gap", description=description)


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
    """Per run and achievement A: a landmark comparison of the unlocks after A's step.

    The landmark ``at`` is the method's median unlock step of A. Among the run's
    episodes still alive then (``length > at``), the lift is the mean number of other
    achievements unlocked after ``at`` in episodes that already hold A (unlocked at or
    before ``at``) minus that in episodes that do not yet; NaN when a group is empty.
    Grouping by "A ever" instead would put late unlockers, which survived longer by
    construction, in the A group (immortal-time bias: a lift of about 1 in a world
    where A does nothing).
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
            later = _later(got, steps, i, at)
            has = got[:, i] & (steps[:, i] <= at)  # held at the landmark
            for run in np.unique(runs):
                here = alive & (runs == run)
                yes, no = later[here & has], later[here & ~has]
                lift = yes.mean() - no.mean() if yes.size and no.size else np.nan
                rows.append((method, run, name, at, yes.size, no.size, float(lift)))
    return pd.DataFrame(rows, columns=LIFT_COLUMNS)


class Gates(NamedTuple):
    """The rows of F22 and the mid-share achievements that are not rows."""

    rows: Rows
    cut: list[str]
    """Achievements with a dot that the row cap leaves out (smaller mean lift)."""
    empty: list[str]
    """Achievements no method has `GATE_MIN` alive episodes with and without."""
    mean: dict[str, float]
    """Mean lift over methods of every achievement that has dots."""


def _gate_limit(methods: int) -> int:
    """Rows that fit `MAX_ROWS_MM` (a third of a page) with ``methods`` dodged dots."""
    row_mm = methods * GATE_METHOD_MM + GATE_GAP_MM
    return max(1, int((MAX_ROWS_MM - ROWS_BASE_MM) // row_mm))


def _gate_rows(ctx: Ctx) -> Gates:
    """Mid-share achievements: the `_gate_limit` largest mean lifts, highest first."""
    table = gatekeepers(ctx.eps)
    names = _achievements(ctx.eps)
    share = {n: float(ctx.eps[f"ach/{n}"].mean()) for n in names}
    keep = [n for n, s in share.items() if GATE_SHARE <= s <= 1 - GATE_SHARE]
    sums = table.groupby(["method", "achievement"])[["with_a", "without_a"]].sum()
    enough = sums.min(axis=1) >= GATE_MIN
    means = table.groupby(["method", "achievement"])["lift"].mean()[enough]
    order = means.groupby(level="achievement").mean().to_dict()
    ranked = sorted((n for n in keep if n in order), key=lambda n: -abs(order[n]))
    limit = _gate_limit(len(ctx.names))
    keys = sorted(ranked[:limit], key=lambda n: -order[n])
    cut = sorted(ranked[limit:], key=lambda n: -order[n])

    def values(method: str, key: str) -> np.ndarray:
        if not enough.get((method, key), False):
            return np.empty(0)
        mine = table[(table["method"] == method) & (table["achievement"] == key)]
        return np.asarray(mine["lift"], dtype=float)

    rows = Rows(keys, [k.replace("_", " ") for k in keys], values)
    return Gates(rows, cut, [n for n in keep if n not in order], order)


def _signed(names: Sequence[str], mean: dict[str, float]) -> str:
    return ", ".join(f"{n.replace('_', ' ')} {mean[n]:+.2f}" for n in names)


def plot_gatekeepers(
    frames: Frames,
    out_dir: Path,
    theme: str = "report",
    *,
    width: str = "column",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Figure:
    """F22 ``gatekeepers``: does an early unlock open up the rest of the run?

    Rows: achievements unlocked in 5% to 95% of the benchmark's final stochastic eval
    episodes, the largest mean lifts first (at most what fits a third of a page; the
    rest are named in the caption). A dot is the mean over seeds of the landmark lift of
    `gatekeepers` (episodes alive at the method's median unlock step of the row that
    hold the achievement against those that do not yet) with a bootstrap interval; a
    method needs `GATE_MIN` alive episodes in each group.
    """
    th, name = _theme(theme, width), NAMES["F22"]
    ctx = _context(frames, n_boot, seed)
    if ctx is None or ctx.eps.empty:
        return _no_data(name, th, out_dir, "no final eval episodes")
    gates = _gate_rows(ctx)
    keys = gates.rows.keys
    note = ""
    if gates.empty:  # rows no method can fill are not drawn, only named
        note = (
            f"No method has {GATE_MIN} alive episodes with and without "
            f"{', '.join(n.replace('_', ' ') for n in gates.empty)}."
        )
    if not keys:
        return _no_data(
            name, th, out_dir, note or "no achievement with a lift to compare"
        )
    row_mm = len(ctx.names) * GATE_METHOD_MM + GATE_GAP_MM
    fig, ax = _panel(th, Shape.ROWS, len(keys) * row_mm + ROWS_BASE_MM)
    ax.axvline(0.0, color=th.muted, lw=0.6)
    seen = _dot_rows(ax, th, ctx, "gate", gates.rows, "extra later unlocks if held")
    absent = [m for m in ctx.names if m not in seen]
    description = _describe(
        "Does an early unlock open up the rest of the run?",
        "Final stochastic eval episodes. Landmark at the method's median step of",
        "unlocking each achievement: among the episodes alive then, a dot is the",
        "mean over seeds of the later unlocks of those that already hold it minus",
        "those that do not yet, with a 95% bootstrap interval. Near 0 means the early",
        "unlock buys nothing later; an association, not an effect. Rows: achievements",
        f"unlocked in {GATE_SHARE:.0%} to {1 - GATE_SHARE:.0%} of episodes, largest",
        "mean lift first",
        f"(mean over methods: {_signed(keys, gates.mean)}).",
        f"Not drawn, smaller mean lift: {_signed(gates.cut, gates.mean)}."
        * bool(gates.cut),
        note,
        f"No dots (under {GATE_MIN} alive episodes with or without): "
        f"{', '.join(absent)}." * bool(absent),
        f"The single row is {gates.rows.labels[0]}." * (len(keys) == 1),
    )
    return _save(
        fig, th, out_dir, name, title="Early unlock lift", description=description
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


def plot_seed_fate(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """F23 ``seed_fate``: when is a seed's fate decided?

    Per method the Spearman correlation between its seeds' mean stochastic eval score
    at each checkpoint and their final score (`seed_fate`), the mean over methods in
    bold; the grey band is the 5% critical value for the seed count (caption).
    """
    th, name = _theme(theme, width), NAMES["F23"]
    ctx, table = _context(frames), seed_fate(frames)
    mine = table[table["env_steps"] > 0].dropna(subset=["rho"])
    if ctx is not None:
        mine = mine[mine["budget"] == ctx.budget]
    if ctx is None or mine.empty:
        return _no_data(name, th, out_dir, "no checkpoint with three seeds")
    seeds = int(mine["seeds"].median())
    crit = critical_rho(seeds) if seeds > 3 else 1.0
    fig, ax = _panel(th, Shape.CURVE)
    null = ax.axhspan(-crit, crit, color=th.grid, alpha=0.6, lw=0)
    null.set_gid(_gid("null", ctx.budget))
    for method in ctx.names:
        g = mine[mine["method"] == method].sort_values("update")
        if len(g):
            _line(
                ax, th, g["env_steps"], g["rho"], method,
                _gid("fate", ctx.budget, method), alpha=0.5,
            )  # fmt: skip
    pooled = mine.groupby("update").agg(x=("env_steps", "median"), y=("rho", "mean"))
    (line,) = ax.plot(pooled["x"], pooled["y"], color=th.ink, lw=th.lw * 2.2)
    line.set_gid(_gid("fate", ctx.budget, "pooled"))
    ax.set_ylim(-1.05, 1.05)
    _log_steps(ax, th, "Spearman rho with final score")
    description = _describe(
        "When is a seed's fate decided?",
        "Spearman rho between the seeds' mean stochastic eval score at a checkpoint",
        "and their final score (the training-episode score, so the last checkpoint",
        "also measures eval against training agreement): high early means a run could",
        "be stopped early. Thin lines: methods; bold black line: mean over methods;",
        f"grey band: rho from -{crit:.2f} to {crit:.2f}, not significant at 5% with",
        f"{seeds} seeds.",
        "Early checkpoints have many tied scores, which pull rho towards 0; the method",
        "mean is no test.",
    )
    return _save(fig, th, out_dir, name, title="Seed fate", description=description)


# --------------------------------------------------------------------------- F24


def _effect_stats(r: Any) -> str:
    """``name PoI x, p y`` of an ablation row (1M only, else empty)."""
    if not np.isfinite(r.poi_tuned):
        return ""
    return f"{r.method} ({r.config_ref}) PoI {r.poi_tuned:.2f}, p {r.p_holm:.2g}"


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
    # one row: its name goes to the caption, a lone tick label would repeat the axis
    names = [f"{r.method} ({r.config_ref})" for r in rows.itertuples()]
    ax.set_yticks(range(len(rows)), names if len(rows) > 1 else [])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    blocks = rows["block"].to_numpy()
    for edge in np.flatnonzero(blocks[1:] != blocks[:-1]):
        ax.axhline(edge + 0.5, color=th.muted, lw=0.5)
    _labels(ax, th, "ablation minus tuned (pp)")


def _marker_legend(th: Theme) -> list[Line2D]:
    """Hollow dot against cross: which baseline the difference is taken from."""
    return [
        Line2D(
            [], [], ls="", marker="o", ms=th.ms * 1.3, color=th.ink2,
            markerfacecolor=th.surface, label="vs all tuned seeds",
        ),
        Line2D(
            [], [], ls="", marker="x", ms=th.ms * 1.2, color=th.ink2,
            label="vs fresh tuned seeds",
        ),
    ]  # fmt: skip


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
    the tuned mean as a faint dot, and a cross for the difference to the fresh tuned
    seeds. The PoI and Holm-adjusted p (1M only) go to the caption.
    """
    th, name = _theme(theme, width), NAMES["F24"]
    table = (
        compare.ablation_table(frames, n_boot=n_boot, seed=seed)
        if table is None
        else table
    )
    ctx = _context(frames, n_boot, seed)
    if ctx is None or table.empty:
        return _no_data(name, th, out_dir, "no ablation runs")
    rows = table[table["benchmark"] == ctx.budget].reset_index(drop=True)
    if rows.empty:
        return _no_data(name, th, out_dir, "no ablation runs")
    legend_mm = shape_height_mm(th, Shape.LEGEND, 1)
    pitch = min(
        th.pitch_mm * ABLATION_PITCH,
        (MAX_ROWS_MM - ROWS_BASE_MM - legend_mm) / len(rows),
    )
    fig, ax = _panel(th, Shape.ROWS, len(rows) * pitch + ROWS_BASE_MM + legend_mm)
    runs = frames["runs"][frames["runs"]["budget"] == ctx.budget]
    _effect_panel(ax, th, ctx.budget, rows, runs)
    _finish(fig, th, _marker_legend(th), cols=2)
    stats = "; ".join(s for r in rows.itertuples() if (s := _effect_stats(r)))
    description = _describe(
        "Which component carries each method?",
        "Hollow dot: ablation minus tuned mean final score with its 95% bootstrap CI;",
        "faint dots: each ablation seed minus the tuned mean. Cross:",
        "the difference to the tuned seeds not reused from tuning; the reused ones",
        "were selected, so the baseline of the dots is optimistic (R102).",
        "PoI: P(ablation beats tuned); p: Mann-Whitney, Holm-adjusted per block:",
        f"{stats}." * bool(stats),
        f"The single row is {rows['method'][0]} ({rows['config_ref'][0]})."
        * (len(rows) == 1),
    )
    return _save(
        fig, th, out_dir, name, title="Ablation effects", description=description
    )


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
    The optimism of each method and the units without one (with the table's reason)
    are in the caption.
    """
    th, name = _theme(theme, width), NAMES["F25"]
    table = compare.tuning_optimism(frames) if table is None else table
    table = table.drop_duplicates(["benchmark", "method"])  # controls: both blocks
    ctx = _context(frames, n_boot, seed)
    table = table if ctx is None else table[table["benchmark"] == ctx.budget]
    ok = table[table["optimism"].notna()]
    if ctx is None or ok.empty:
        return _no_data(name, th, out_dir, "no tuned config with fresh seeds")
    fig, ax = _panel(th, Shape.CURVE)
    _optimism_panel(ax, th, ctx.budget, ok, frames, n_boot, seed)
    gaps = ", ".join(f"{r.method} {r.optimism:+.2f}" for r in ok.itertuples())
    na = table[table["optimism"].isna()]
    reasons = " ".join(
        f"{r.removeprefix('n/a: ')}: {', '.join(g['method'])}."
        for r, g in na.groupby("optimism_note")
    )
    description = _describe(
        "Did tuning flatter the chosen configs?",
        "Dot: mean of the fresh final seeds (not reused from tuning, R102) with a 95%",
        "bootstrap CI against the tuning mean of the chosen config; below the dashed",
        "diagonal the tuning score was too high. Optimism, tuning minus fresh mean",
        f"(R101): {gaps}. The tuning mean has 3 seeds and no interval; with seed noise",
        "alone it would still regress towards the fresh mean.",
        reasons,
    )
    return _save(
        fig, th, out_dir, name, title="Tuning optimism", description=description
    )


# --------------------------------------------------------------------------- F26


def _count_shares(ctx: Ctx) -> pd.DataFrame:
    """Methods down, achievements per episode across: the share of the episodes."""
    counts = ctx.eps["n_achievements"].astype(int)
    top = int(counts.max()) if len(counts) else 0
    table = pd.crosstab(ctx.eps["method"], counts, normalize="index")
    table = table.reindex(columns=range(top + 1), fill_value=0.0)
    return table.reindex(index=[m for m in ctx.names if m in table.index])


def plot_episode_spread(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """F26 ``episode_spread``: how uneven are a policy's episodes?

    A heat map of the share of each method's final stochastic eval episodes by
    achievements unlocked (the episode return is that count plus a small health term),
    a triangle at its mean. Also writes `LENGTH_STEM` (`plot_achievements_vs_length`).

    Returns:
        The heat map's figure.
    """
    th, name = _theme(theme, width), NAMES["F26"]
    ctx = _context(frames)
    if ctx is None or ctx.eps.empty:
        fig = _no_data(name, th, out_dir, "no final eval episodes")
        plot_achievements_vs_length(frames, out_dir, theme, width=width)
        return fig
    shares = _count_shares(ctx)
    fig, ax = _panel(th, Shape.SQUARE)
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
    description = _describe(
        "How uneven are a policy's episodes?",
        "Share of each method's final stochastic eval episodes (tuned runs, all seeds",
        "pooled) by the achievements they unlocked; the triangle marks the method's",
        "mean achievements per episode. The episode return is that count plus a small",
        "health term.",
    )
    _save(fig, th, out_dir, name, title="Episode spread", description=description)
    plot_achievements_vs_length(frames, out_dir, theme, width=width)
    return fig


def plot_achievements_vs_length(
    frames: Frames, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """`LENGTH_STEM`: does surviving longer buy achievements?

    The mean achievements per episode over `LENGTH_BINS` equal-count episode-length
    bins of the benchmark (a bin needs `LENGTH_MIN` episodes of the method), at the
    bin's median length, one line per method.
    """
    th = _theme(theme, width)
    ctx = _context(frames)
    if ctx is None or ctx.eps.empty:
        return _no_data(LENGTH_STEM, th, out_dir, "no final eval episodes")
    eps = ctx.eps
    edges = np.unique(np.quantile(eps["length"], np.linspace(0, 1, LENGTH_BINS + 1)))
    bins = pd.cut(eps["length"], edges, include_lowest=True)
    stats = (
        eps
        .assign(bin=bins)
        .groupby(["method", "bin"], observed=True)
        .agg(x=("length", "median"), y=("n_achievements", "mean"), n=("length", "size"))
    )
    fig, ax = _panel(th, Shape.CURVE)
    for method in ctx.names:
        if method not in stats.index.get_level_values(0):
            continue
        g = stats.loc[method]
        g = g[g["n"] >= LENGTH_MIN]
        if len(g):
            _line(ax, th, g["x"], g["y"], method, _gid("length", ctx.budget, method))
    _style(ax, th, grid="both")
    _labels(ax, th, "episode length (steps, bin median)", "achievements per episode")
    description = _describe(
        "Does surviving longer buy achievements?",
        "Mean achievements per episode of each method over "
        f"{LENGTH_BINS} equal-count bins of the episode length (final stochastic",
        f"eval episodes, all seeds pooled; a bin needs {LENGTH_MIN} episodes of the",
        "method),",
        "at the bin's median length. Length and achievements are associated, not",
        "causal: eating, drinking and sleeping also keep an episode alive; the last",
        "length bin is open-ended.",
    )
    return _save(
        fig, th, out_dir, LENGTH_STEM, title="Achievements by length",
        description=description,
    )  # fmt: skip


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


def within_rho(policies: pd.DataFrame) -> float:
    """Mean over methods with `MIN_WITHIN` policies of their score-fidelity rho."""
    rhos = [
        _spearman(g["fidelity_256"].to_numpy(float), g["score"].to_numpy(float))
        for _, g in policies.groupby("method")
        if len(g) >= MIN_WITHIN
    ]
    rhos = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(rhos)) if rhos else float("nan")


def _order(frames: Frames) -> list[str]:
    """Methods in table order: the controls, then each block's methods."""
    tuned = compare.tuned(frames["runs"])
    return list(dict.fromkeys(m for _, m, _ in compare.members(tuned)))


def _distil_pending(frames: Frames, root: Path) -> tuple[Ctx | None, pd.DataFrame, str]:
    """The benchmark, its finished policies and a note on the methods short of some."""
    ctx = _context(frames)
    if ctx is None:
        return None, pd.DataFrame(), ""
    policies = distilled(root, ctx.budget)
    policies = policies[policies["run_id"].isin(ctx.runs["run_id"])]
    have = policies.groupby("method").size()
    want = ctx.runs.groupby("method").size()
    short = [
        f"{m} ({int(have.get(m, 0))} of {n})"
        for m, n in want.items()
        if have.get(m, 0) < n
    ]
    return ctx, policies, f"Distillation pending: {', '.join(short)}." * bool(short)


def _score_dots(
    ax: Axes, th: Theme, budget: str, policies: pd.DataFrame, names: Sequence[str]
) -> str:
    """Score against fidelity per policy; returns the Spearman line of the caption."""
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
        return "Spearman needs 3 policies."
    res = sps.spearmanr(ok["fidelity_256"], ok["score"])
    within = within_rho(ok)
    shown = f"{within:.2f}" if np.isfinite(within) else "n/a"
    return (
        f"Pooled Spearman rho {res.statistic:.2f} (p {res.pvalue:.2g}, {len(ok)} "
        f"policies), within methods {shown}."
    )


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
    final score against the test fidelity of its 256-leaf tree (base fit). R106's
    Spearman over all policies, the within-method mean and the methods short of
    policies are in the caption.
    """
    th, name = _theme(theme, width), NAMES["F27"]
    ctx, policies, pending = _distil_pending(frames, interpretation)
    if ctx is None or policies.empty:
        return _no_data(name, th, out_dir, "distillation pending, no policy finished")
    fig, ax = _panel(th, Shape.CURVE)
    spearman = _score_dots(ax, th, ctx.budget, policies, _order(frames))
    description = _describe(
        "Do policies that distil well score well?",
        "One dot per distilled policy: its final score against the test top-1",
        "fidelity of its 256-leaf tree (base fit, test states of its final",
        f"checkpoint). {spearman} The pooled rho (R106) mostly compares methods;",
        f"within methods is the mean rho over methods with {MIN_WITHIN} or more",
        "policies. The p value treats the policies as independent, though seeds",
        "cluster by method.",
        pending,
    )
    return _save(
        fig, th, out_dir, name, title="Distillation score", description=description
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
    th, name = _theme(theme, width), NAMES["F28"]
    ctx, policies, pending = _distil_pending(frames, interpretation)
    means = importance_means(policies) if ctx is not None else pd.DataFrame()
    if ctx is None or means.empty:
        return _no_data(name, th, out_dir, "distillation pending, no importance yet")
    order = _order(frames)
    top = means.max(axis=1).sort_values(ascending=False).index[:FEATURE_ROWS]
    data = means.loc[top, [m for m in order if m in means.columns]]
    data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = _panel(th, Shape.SQUARE)
    _heatmap(
        ax,
        th,
        data,
        _gid("heat", ctx.budget, "importance"),
        "permutation importance",
        ".2f",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    description = _describe(
        "Which inputs do the distilled trees rely on?",
        "Permutation importance: drop in the 256-leaf tree's test fidelity when the",
        "input is shuffled, mean over a method's distilled seeds (base fit); the",
        f"{FEATURE_ROWS} inputs with the largest per-method mean; blank: not an input",
        "of that method. Methods differ in their inputs (memory proxies, skill flags),",
        "and correlated inputs split importance, so compare within a column before",
        "across.",
        pending,
    )
    return _save(
        fig, th, out_dir, name, title="Distillation inputs", description=description
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
"""Figure id -> its function (the results CLI loops over it); F26 writes two files."""
