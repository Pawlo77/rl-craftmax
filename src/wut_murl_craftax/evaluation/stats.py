"""Seed statistics of the statistics protocol: IQM and probability of improvement.

The evaluation plan's statistics task adds the bootstrap intervals and tests; the
combination pair (``evaluation.select_pair``) needs these two point estimates first.
"""

from collections.abc import Sequence

import numpy as np

TRIM = 0.25
"""Share cut from each end for the IQM, as ``scipy.stats.trim_mean(x, 0.25)``."""


def iqm(scores: Sequence[float]) -> float:
    """Interquartile mean: the mean after cutting ``int(0.25 n)`` scores off each end.

    Raises:
        ValueError: Without a score.
    """
    if not scores:
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
