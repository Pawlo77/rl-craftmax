"""Learning stability of one run (drawdowns, drops, collapse) and tuning sensitivity."""

import json
import math
from typing import Any

import numpy as np
import pandas as pd

WINDOW_FRAC = 0.01
"""Rolling-median window of the smoothed curve, as a share of the updates."""
MIN_MAX = 1.0
"""Drawdowns are defined where the running max is at least this score (percent)."""
DROP_DEPTH = 0.2
"""Drawdown a drop must reach."""
DROP_FRAC = 0.02
"""Share of the budget a drawdown at the drop depth must last to count as a drop."""
COLLAPSE_DEPTH = 0.5
"""Drawdown that, with no recovery, makes a collapse."""
COLLAPSE_FINAL = 0.9
"""A collapse ends below this share of the running max."""
RECOVERED = 0.05
"""Drawdown below which a drop has recovered."""
ROBUST = 0.9
"""Config score from which a config counts as robust."""
SENSITIVITY_COLUMNS = [
    "hyperparameter",
    "n_values",
    "sensitivity",
    "noise_floor",
    "insensitive",
    "best_value",
    "worst_value",
]
"""Columns of the ``sensitivity`` table (kept when no hyperparameter varies)."""


def smoothed(curve: np.ndarray, window_frac: float = WINDOW_FRAC) -> np.ndarray:
    """Rolling median over ``window_frac`` of the updates, centred (no lag at dips).

    The curve is the per-update score in percent. NaN scores (updates without a
    finished episode) are skipped by the median; NaN stays only where a whole
    window is NaN.
    """
    window = max(1, round(window_frac * len(curve)))
    rolled = pd.Series(np.asarray(curve, dtype=float)).rolling(
        window, center=True, min_periods=1
    )
    out: np.ndarray = rolled.median().to_numpy()
    return out


def _filled(c: np.ndarray) -> np.ndarray:
    """Forward-fill NaN with the last known score (leading NaN stays)."""
    out: np.ndarray = pd.Series(np.asarray(c, dtype=float)).ffill().to_numpy()
    return out


def drawdowns(c: np.ndarray) -> np.ndarray:
    """Relative fall from the running max; NaN while that max is below 1% score.

    ``c`` is the smoothed curve in percent score and may hold NaN: a NaN stretch
    carries the last known score forward (so a drop spanning it stays one drop and
    the running max survives), and leading NaN has no drawdown.
    """
    f = _filled(c)
    m = np.fmax.accumulate(f)
    d = np.full(len(c), np.nan)
    ok = m >= MIN_MAX
    d[ok] = (m[ok] - f[ok]) / m[ok]
    return d


def drops(
    c: np.ndarray, depth: float = DROP_DEPTH, min_frac: float = DROP_FRAC
) -> list[tuple[int, int]]:
    """Maximal intervals ``[start, stop)`` of drawdown >= ``depth``, long enough.

    Indices count updates of the smoothed curve ``c`` (percent score, NaN allowed,
    see ``drawdowns``); an interval lasts long enough at ``min_frac`` of the updates.
    """
    deep = np.concatenate([[False], drawdowns(c) >= depth, [False]])
    edges = np.flatnonzero(deep[1:] != deep[:-1])
    min_len = math.ceil(min_frac * len(c) - 1e-9)
    return [
        (int(a), int(b))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if b - a >= min_len
    ]


def collapsed(c: np.ndarray) -> bool:
    """Drawdown reached 50% and the last known value is below 90% of the running max.

    ``c`` is the smoothed curve in percent score; NaN is carried forward (see
    ``drawdowns``), so a trailing NaN stretch is judged by its last known score.
    """
    f = _filled(c)
    return bool(
        (drawdowns(c) >= COLLAPSE_DEPTH).any() and f[-1] < COLLAPSE_FINAL * np.nanmax(f)
    )


def recovery_times(c: np.ndarray, steps: np.ndarray) -> list[float]:
    """Env steps from each drop's start until the drawdown is below 5% (inf: never).

    Args:
        c: Smoothed curve in percent score, NaN allowed (see ``drawdowns``).
        steps: Env steps of each update, one per entry of ``c``.

    Raises:
        ValueError: If ``steps`` and ``c`` differ in length.
    """
    if len(steps) != len(c):
        msg = f"steps ({len(steps)}) and the curve ({len(c)}) differ in length"
        raise ValueError(msg)
    d = drawdowns(c)
    out = []
    for start, _ in drops(c):
        back = np.flatnonzero(d[start:] < RECOVERED)
        out.append(
            float(steps[start + back[0]] - steps[start]) if back.size else np.inf
        )
    return out


def dispersion(c: np.ndarray) -> float:
    """IQR of the first differences of the curve; differences touching NaN are skipped.

    NaN when fewer than one difference is finite.
    """
    diff = np.diff(np.asarray(c, dtype=float))
    diff = diff[np.isfinite(diff)]
    if not diff.size:
        return float("nan")
    q1, q3 = np.percentile(diff, [25, 75])
    return float(q3 - q1)


def config_scores(tuning_table: pd.DataFrame) -> pd.DataFrame:
    """Per config: seed mean, sd, count and ``r_c`` (mean over the best config's mean).

    Args:
        tuning_table: One row per run with ``config_hash`` and ``final_score``.

    Raises:
        ValueError: If the best config's mean is not positive (``r_c`` is undefined).
    """
    scores = (
        tuning_table
        .groupby("config_hash")["final_score"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    best = scores["mean"].max()
    if not best > 0:
        msg = f"the best config mean must be positive, got {best}"
        raise ValueError(msg)
    scores["r_c"] = scores["mean"] / best
    return scores


def robust_share(scores: pd.DataFrame, threshold: float = ROBUST) -> float:
    """Share of configs with ``r_c`` at or above ``threshold``."""
    return float((scores["r_c"] >= threshold).mean())


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted-key scalar leaves of a config; lists become JSON strings (hashable)."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = (
                json.dumps(value) if isinstance(value, list) else value
            )
    return flat


def _hyperparameters(tuning_table: pd.DataFrame) -> pd.DataFrame:
    """Config values per ``config_hash``: keys constant in a config, varying across.

    A key that differs between the seeds of one config (a seed) is no hyperparameter.
    """
    flat = pd.DataFrame([_flatten(json.loads(j)) for j in tuning_table["config_json"]])
    flat["config_hash"] = tuning_table["config_hash"].to_numpy()
    within = flat.groupby("config_hash").nunique(dropna=False).max()
    firsts = flat.drop_duplicates("config_hash").set_index("config_hash")
    varying = [
        k for k in firsts if within[k] == 1 and firsts[k].nunique(dropna=False) > 1
    ]
    return firsts[varying]


def sensitivity(tuning_table: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity ``S_h`` of each hyperparameter that varies across the configs.

    ``S_h`` is the spread of the mean config score ``r_c`` between the values of
    ``h``. Hyperparameters come from each run's ``config_json`` (the store has no
    params); ``S_h`` below the noise floor reads "insensitive".

    Args:
        tuning_table: The ``runs`` rows of one method's tuning phase, with
            ``config_hash``, ``final_score`` and ``config_json``.

    Returns:
        One row per hyperparameter, most sensitive first: ``hyperparameter``,
        ``n_values``, ``sensitivity``, ``noise_floor`` (mean within-config seed sd
        over the best config mean), ``insensitive``, ``best_value``, ``worst_value``.
    """
    scores = config_scores(tuning_table).set_index("config_hash")
    floor = float(scores["sd"].mean() / scores["mean"].max())
    values = _hyperparameters(tuning_table).loc[scores.index]
    rows = []
    for name in values:
        marginal = scores["r_c"].groupby(values[name], sort=False, dropna=False).mean()
        spread = float(marginal.max() - marginal.min())
        rows.append({
            "hyperparameter": name,
            "n_values": len(marginal),
            "sensitivity": spread,
            "noise_floor": floor,
            "insensitive": spread < floor,
            "best_value": marginal.idxmax(),
            "worst_value": marginal.idxmin(),
        })
    table = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    return table.sort_values("sensitivity", ascending=False, kind="stable").reset_index(
        drop=True
    )
