"""Seed statistics of the protocol: point estimates, intervals, tests, bands."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sps

TRIM = 0.25
"""Share cut from each end for the IQM, as ``scipy.stats.trim_mean(x, 0.25)``."""
CI_LEVEL = 0.95
"""Coverage of every interval (percentile bootstrap and t)."""
BOOTSTRAP_N = 10_000
"""Bootstrap resamples of the protocol."""
ALPHA = 0.05
"""Significance level of the Holm-corrected Mann-Whitney tests."""
GRID_POINTS = 200
"""Points of the common env-step grid of learning curves."""
HEAD_SHARE = 0.05
"""Leading share of the budget whose grid points are log-spaced."""
HEAD_SPAN = 100.0
"""The log-spaced head runs from ``HEAD_SHARE * budget / HEAD_SPAN`` to its end."""
CHUNK = 1000
"""Bootstrap resamples of a band processed at once (bounds the working memory)."""


class Interval(NamedTuple):
    """A point estimate with its confidence interval."""

    estimate: float
    lo: float
    hi: float


def iqm(scores: Sequence[float] | np.ndarray) -> float:
    """Interquartile mean: the mean after cutting ``int(0.25 n)`` scores off each end.

    Raises:
        ValueError: Without a score.
    """
    if len(scores) == 0:
        msg = "the IQM needs at least one score"
        raise ValueError(msg)
    x = np.sort(np.asarray(scores, dtype=float))
    cut = int(TRIM * len(x))
    return float(x[cut : len(x) - cut].mean())


def probability_of_improvement(x: Sequence[float], y: Sequence[float]) -> float:
    """P(X > Y) over all seed pairs, a tie counting one half."""
    a = np.asarray(x, dtype=float)[:, None]
    b = np.asarray(y, dtype=float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def mean_se(scores: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error over seeds (NaN error for a single score).

    Raises:
        ValueError: Without a score.
    """
    x = np.asarray(scores, dtype=float)
    if not x.size:
        msg = "the mean needs at least one score"
        raise ValueError(msg)
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return float(x.mean()), se


def stratified_bootstrap_ci(
    scores_by_stratum: Mapping[str, Sequence[float]],
    statistic: Callable[[dict[str, np.ndarray]], float],
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval of ``statistic`` over seeds resampled per stratum.

    Each stratum (method, benchmark) is resampled with replacement on its own and
    keeps its size, so strata never mix. ``statistic`` receives the resampled
    arrays under the stratum keys, in the keys' order; the estimate is its value on
    the original scores.

    Raises:
        ValueError: A stratum has no score.
    """
    data = {k: np.asarray(v, dtype=float) for k, v in scores_by_stratum.items()}
    if any(not v.size for v in data.values()):
        msg = "a stratum is empty"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    draws = {k: rng.integers(0, v.size, size=(n, v.size)) for k, v in data.items()}
    boot = np.array([
        statistic({k: v[draws[k][i]] for k, v in data.items()}) for i in range(n)
    ])
    lo, hi = np.percentile(boot, [50 * (1 - CI_LEVEL), 50 * (1 + CI_LEVEL)])
    return Interval(float(statistic(data)), float(lo), float(hi))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order, capped at 1."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = (p.size - np.arange(p.size)) * p[order]
    adjusted = np.empty(p.size)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted.tolist()


def mann_whitney_holm(
    comparisons: Mapping[Hashable, tuple[Sequence[float], Sequence[float]]],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Two-sided Mann-Whitney U per pair, Holm-corrected over the whole mapping.

    Returns:
        One row per comparison key with ``u`` (of the first sample), ``p``,
        ``p_holm`` and ``reject`` (``p_holm < alpha``). The mapping is one Holm
        family: pass the pairs of one table together, a block of its own apart.
    """
    tests = [
        sps.mannwhitneyu(x, y, alternative="two-sided") for x, y in comparisons.values()
    ]
    p = [float(t.pvalue) for t in tests]
    p_holm = holm(p)
    return pd.DataFrame(
        {
            "u": [float(t.statistic) for t in tests],
            "p": p,
            "p_holm": p_holm,
            "reject": [q < alpha for q in p_holm],
        },
        index=list(comparisons),
    )


def t_ci(scores: Sequence[float], level: float = CI_LEVEL) -> Interval:
    """t-based interval of the mean (the 1B report: a 3-value bootstrap is too coarse).

    Raises:
        ValueError: With fewer than two scores.
    """
    if len(scores) < 2:
        msg = "the t interval needs at least two scores"
        raise ValueError(msg)
    mean, se = mean_se(scores)
    half = float(sps.t.ppf(0.5 + level / 2, len(scores) - 1)) * se
    return Interval(mean, mean - half, mean + half)


def seed_spread(scores: Sequence[float]) -> dict[str, float]:
    """Sd, IQR, range and coefficient of variation (NaN at mean 0) of final scores."""
    x = np.asarray(scores, dtype=float)
    mean = x.mean()
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "sd": sd,
        "iqr": float(q3 - q1),
        "range": float(x.max() - x.min()),
        "cv": float(sd / mean) if mean else float("nan"),
    }


def step_grid(budget: float, n: int = GRID_POINTS) -> np.ndarray:
    """Common env-step grid: a quarter of the points log-spaced over the first 5%.

    The log head starts at 0.05% of the budget; the rest is evenly spaced.
    """
    n_head = n // 4
    head_end = HEAD_SHARE * budget
    head = np.geomspace(head_end / HEAD_SPAN, head_end, n_head)
    tail = np.linspace(head_end, budget, n - n_head + 1)[1:]
    return np.concatenate([head, tail])


def _on_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation; NaN before the first step and past the last logging gap."""
    out: np.ndarray = np.interp(grid, steps, values, left=np.nan)
    gap = np.diff(steps).max() if steps.size > 1 else 0.0
    out[grid > steps[-1] + gap] = np.nan
    return out


def curve_band(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    statistic: str = "mean",
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean (or IQM) learning curve over seeds with a 95% percentile bootstrap band.

    Args:
        curves: One ``(env_steps, values)`` pair per seed.
        grid: Env steps to evaluate on (see ``step_grid``).
        statistic: ``"mean"`` or ``"iqm"``, the robust variant.
        n: Bootstrap resamples; whole curves are resampled, seeds being the unit.
        seed: Bootstrap seed.

    Returns:
        Columns ``env_steps``, ``estimate``, ``lo``, ``hi``; NaN at grid points some
        seed has no data for.

    Raises:
        ValueError: Without a curve or with an unknown statistic.
    """
    if statistic not in {"mean", "iqm"}:
        msg = f"statistic must be 'mean' or 'iqm', got {statistic!r}"
        raise ValueError(msg)
    if not curves:
        msg = "a band needs at least one curve"
        raise ValueError(msg)
    grid = np.asarray(grid, dtype=float)
    matrix = np.stack([_on_grid(np.asarray(s), np.asarray(v), grid) for s, v in curves])
    seeds = len(curves)
    cut = int(TRIM * seeds) if statistic == "iqm" else 0

    def reduce(x: np.ndarray) -> np.ndarray:
        """Mean over the seed axis after the IQM cut (none for the mean)."""
        return np.sort(x, axis=-2)[..., cut : seeds - cut, :].mean(axis=-2)

    ok = np.isfinite(matrix).all(axis=0)
    out = np.full((3, grid.size), np.nan)
    data = matrix[:, ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, seeds, size=(n, seeds))
    boot = np.concatenate([
        reduce(data[idx[i : i + CHUNK]]) for i in range(0, n, CHUNK)
    ])
    lo, hi = np.percentile(boot, [50 * (1 - CI_LEVEL), 50 * (1 + CI_LEVEL)], axis=0)
    out[:, ok] = [reduce(data), lo, hi]
    return pd.DataFrame({
        "env_steps": grid,
        "estimate": out[0],
        "lo": out[1],
        "hi": out[2],
    })
