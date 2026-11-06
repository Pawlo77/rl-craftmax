"""The 1M vs 1B figures S1-S3: where each method stands at the two budgets.

Each ``plot_*`` draws one panel without text on the image (question, reading and
numbers go to the caption it registers) from the scale tables of `analysis.scale` and
the speed table T8. A dot is a method in its colour of ``legend_methods``; the lines
are 95% bootstrap CIs of the two budgets, whose seeds are independent. 1B has 3 seeds,
so the intervals are the claim, not the dots. Artist gid ``scale/<kind>/<method>``.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_figures_scale.py]]
"""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..visualisation import charts
from .figures import Theme, _figure, _gid, _labels, _save, _style, _style_of, _theme
from .figures_angles import _describe

STEMS = ("scale_scores", "scale_advantage", "scale_speed")
"""One report image each, at the single-column width."""
Tables = Mapping[str, pd.DataFrame]


def _plain_log(ax: Axes, name: str, values: pd.Series) -> None:
    """Log axis with a few plain ticks (1, 2, 5 times a power of 10) over ``values``."""
    lo, hi = float(values.min()) / 1.3, float(values.max()) * 1.3
    ticks = [
        m * 10.0**e for e in range(-3, 4) for m in (1, 2, 5) if lo <= m * 10.0**e <= hi
    ]
    axis = ax.xaxis if name == "x" else ax.yaxis
    (ax.set_xlim if name == "x" else ax.set_ylim)(lo, hi)
    axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _dot(ax: Axes, th: Theme, method: str, kind: str, x: float, y: float) -> None:
    s = _style_of(th, method)
    (dot,) = ax.plot(
        [x], [y], ls="", marker=s["marker"], ms=th.ms * 1.3, color=s["color"],
        markerfacecolor=s["markerfacecolor"],
    )  # fmt: skip
    dot.set_gid(_gid("scale", kind, method))


def _bars(
    ax: Axes,
    th: Theme,
    method: str,
    x: float,
    y: float,
    xci: tuple[float, float],
    yci: tuple[float, float],
) -> None:
    c = _style_of(th, method)["color"]
    ax.hlines(y, xci[0], xci[1], colors=c, lw=th.lw * 0.8)
    ax.vlines(x, yci[0], yci[1], colors=c, lw=th.lw * 0.8)


def plot_scores(
    tables: Tables, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """S1 ``scale_scores``: 1M against 1B mean per method, log-log."""
    th = _theme(theme, width)
    t = tables["scale_scaling"]
    fig = _figure(th, shape=charts.Shape.CURVE)
    ax = fig.add_subplot()
    for r in t.itertuples():
        _dot(ax, th, r.method, "score", r.mean_1m, r.mean_1b)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax, "x", t["mean_1m"])
    _plain_log(ax, "y", t["mean_1b"])
    _style(ax, th, grid="both")
    _labels(ax, th, "mean score at 1M (%)", "mean score at 1B (%)")
    ratios = ", ".join(f"{r.method} x{np.exp(r.log_ratio):.1f}" for r in t.itertuples())
    return _save(
        fig, th, out_dir, "scale_scores", title="Scores at 1M and 1B",
        description=_describe(
            "Does the ranking carry over from 1M to 1B?",
            "Mean final training score of every method at the two budgets (log axes);",
            "a monotone rising cloud keeps the order. Gain from 1M to 1B:",
            f"{ratios}.",
            "1B has 3 seeds, so the order of close methods is not resolved.",
        ),
    )  # fmt: skip


def plot_advantage(
    tables: Tables, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """S2 ``scale_advantage``: gain over the base control at 1M against at 1B."""
    th = _theme(theme, width)
    t = tables["scale_advantage"]
    fig = _figure(th, shape=charts.Shape.CURVE)
    ax = fig.add_subplot()
    for r in t.itertuples():
        _bars(ax, th, r.method, r.gain_1m, r.gain_1b,
              (r.gain_1m_lo, r.gain_1m_hi), (r.gain_1b_lo, r.gain_1b_hi))  # fmt: skip
        _dot(ax, th, r.method, "advantage", r.gain_1m, r.gain_1b)
    lim = [min(t["gain_1m_lo"].min(), t["gain_1b_lo"].min()),
           max(t["gain_1m_hi"].max(), t["gain_1b_hi"].max())]  # fmt: skip
    ax.axline((0, 0), slope=1.0, color=th.muted, lw=0.7, ls="--")
    ax.axhline(0, color=th.muted, lw=0.5)
    ax.axvline(0, color=th.muted, lw=0.5)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    _style(ax, th, grid="both")
    _labels(
        ax, th, "gain over the control at 1M (%)", "gain over the control at 1B (%)"
    )
    kept = ", ".join(
        f"{r.method} {r.change:+.1f}"
        for r in t.itertuples()
        if r.change_lo > 0 or r.change_hi < 0
    )
    return _save(
        fig, th, out_dir, "scale_advantage", title="Advantage at 1M and 1B",
        description=_describe(
            "Does a method's advantage over its base control survive at 1B?",
            "Gain in percent over the base control at each budget, 95% bootstrap CIs;",
            "on the dashed diagonal the advantage is unchanged, below it the method",
            "loses ground at 1B. Changes whose interval excludes 0 (points):",
            f"{kept or 'none'}.",
        ),
    )  # fmt: skip


def plot_speed(
    tables: Tables, out_dir: Path, theme: str = "report", *, width: str = "column"
) -> Figure:
    """S3 ``scale_speed``: budget share at 90% of the final score, 1M against 1B."""
    th = _theme(theme, width)
    t = tables["speed"].drop_duplicates(["benchmark", "method"])
    wide = t.pivot(index="method", columns="benchmark", values="to_90").dropna()
    fig = _figure(th, shape=charts.Shape.CURVE)
    ax = fig.add_subplot()
    for m, r in wide.iterrows():
        _dot(ax, th, str(m), "speed", r["1m"], r["1b"])
    ax.axline((0, 0), slope=1.0, color=th.muted, lw=0.7, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, th, grid="both")
    _labels(ax, th, "budget share to 90% at 1M", "budget share to 90% at 1B")
    later = ", ".join(str(m) for m, r in wide.iterrows() if r["1b"] > r["1m"])
    return _save(
        fig, th, out_dir, "scale_speed", title="Learning speed at 1M and 1B",
        description=_describe(
            "Does a method learn at the same pace relative to its budget?",
            "Share of the budget at which the smoothed training score reaches 90%",
            "of its final value, median over seeds (T8); below the dashed diagonal the",
            "method reaches it earlier at 1B. Later at 1B:", f"{later or 'none'}.",
        ),
    )  # fmt: skip


FIGURES: dict[str, Callable[..., Figure]] = {
    "scale_scores": plot_scores,
    "scale_advantage": plot_advantage,
    "scale_speed": plot_speed,
}
"""Stem -> function."""


def draw(tables: Tables, out_dir: Path) -> list[Path]:
    """Draw every figure in the report and video variants; write ``captions.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with charts.capture_captions() as captions:
        for fn in FIGURES.values():
            for theme, width in (("report", "column"), ("video", "column")):
                fn(tables, out_dir, theme, width=width)
    if captions:
        charts.write_captions(out_dir, "S", captions)
    return sorted(out_dir.glob("*.png"))
