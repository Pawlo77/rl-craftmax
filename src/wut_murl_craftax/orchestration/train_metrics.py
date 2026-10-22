"""MLflow points of a run's per-update training metrics and its per-run facts.

The full per-update record is ``metrics.npz`` (every update). MLflow gets windows of
``log_every`` updates, so a run holds at most `MAX_POINTS` points per metric: 1M runs
log every update, a 1B run of 15k updates every 8th. A window averages its updates
(returns, lengths and success rates weighted by finished episodes) and is logged at
the env step of its last update. Names: ``train/{env_steps, wall_s, n_episodes,
return, score_pct, length, sps, <loss>, ach/<achievement>}`` (``<loss>``: the loss
columns the trainer reports), method metrics and online families by name
(``ppo/entropy``, ``episode_metrics.metric_keys``), run-level ``compile_s``,
``peak_*_mb``, tags ``{actor,learner}_device``, ``device_kind``, ``host_name``.
"""

import math
import resource
import socket
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import Metric

from ..evaluation.episode_metrics import (
    ONLINE_PREFIX,
    score_pct,
    sum_columns,
    window_metrics,
)
from ..evaluation.milestones import Game
from ..training.records import BASE_COLUMNS, LOSS_COLUMNS
from .runtime import DeviceRoles

MAX_POINTS = 2000
"""Most MLflow points per metric and run; ``log_every`` widens the window to fit."""
_BATCH = 1000
"""MLflow's limit of metrics per ``log_batch`` call."""
_MEAN_PER_EPISODE = ("return", "length")
"""Columns averaged over episodes; the others are means over updates."""

type Points = list[tuple[int, dict[str, float]]]


def log_every(n_updates: int) -> int:
    """Updates per logged window, so that a run has at most `MAX_POINTS` windows."""
    return max(1, math.ceil(n_updates / MAX_POINTS))


def _window(
    w: np.ndarray,
    prev: np.ndarray | None,
    ix: dict[str, int],
    names: Sequence[str],
    loss_columns: Sequence[str],
) -> dict[str, float]:
    """The metrics of the updates in ``w``; ``prev`` is the row before the window."""
    n = w[:, ix["n_episodes"]]
    last = w[-1]
    out = {
        "train/env_steps": last[ix["env_steps"]],
        "train/wall_s": last[ix["wall_s"]],
        "train/n_episodes": n.sum(),
        **{f"train/{c}": w[:, ix[c]].mean() for c in loss_columns},
    }
    start = prev if prev is not None else np.zeros_like(last)
    seconds = last[ix["wall_s"]] - start[ix["wall_s"]]
    if seconds > 0:
        out["train/sps"] = (last[ix["env_steps"]] - start[ix["env_steps"]]) / seconds
    if n.sum():
        share = n / n.sum()
        for c in (*_MEAN_PER_EPISODE, *(f"ach/{a}" for a in names)):
            out[f"train/{c}"] = np.where(n > 0, w[:, ix[c]], 0.0) @ share
    return {k: float(v) for k, v in out.items()}


def _families(
    extras: Mapping[str, Sequence[float]], start: int, end: int, game: Game
) -> dict[str, float]:
    """The online families of updates ``start + 1 .. end``, their episodes pooled.

    The ``online/*`` series are additive sums, so the window's metrics are those of
    one update holding all its episodes; NaN updates (before a resume that added
    them) count as none. Metrics that are NaN (no episodes, deaths) are left out.
    """
    names = [f"{ONLINE_PREFIX}{c}" for c in sum_columns(game)]
    if names[0] not in extras:
        return {}
    total = [np.nansum(np.asarray(extras[n][start:end], float)) for n in names]
    got = window_metrics(np.array(total), game)
    return {k: v for k, v in got.items() if not math.isnan(v)}


def metric_points(
    rows: Sequence[Sequence[float]],
    names: Sequence[str],
    first: int,
    last: int,
    every: int,
    n_updates: int,
    max_reward: float,
    loss_columns: Sequence[str] = LOSS_COLUMNS,
    extras: Mapping[str, Sequence[float]] | None = None,
    game: Game | None = None,
) -> Points:
    """Windows ending in updates ``first + 1 .. last``, as ``(env step, metrics)``.

    ``rows`` is the run log's table (one row per update); a window ends at a multiple
    of ``every`` or at ``n_updates``, so the points do not depend on where chunks or
    resumes cut the run. Windows without finished episodes carry no score. Only
    ``loss_columns`` of the loss layout are reported (a trainer may fill a subset);
    ``extras`` are method metrics by name, per update, averaged over the window under
    their own names (NaN updates skipped). With ``game``, the online families come
    from the window's summed ``online/*`` sums (`_families`).
    """
    cols = [*BASE_COLUMNS, *LOSS_COLUMNS, *(f"ach/{a}" for a in names)]
    ix = {c: i for i, c in enumerate(cols)}
    lo = max(first // every * every - 1, 0)  # the row before the first window
    table = np.array(rows[lo:last], np.float64)
    points: Points = []
    for end in range(first + 1, last + 1):
        if end % every and end != n_updates:
            continue
        start = (end - 1) // every * every
        prev = table[start - 1 - lo] if start else None
        metrics = _window(table[start - lo : end - lo], prev, ix, names, loss_columns)
        for name, series in (extras or {}).items():
            values = np.asarray(series[start:end], float)
            if name.startswith(ONLINE_PREFIX):
                continue
            if (values := values[~np.isnan(values)]).size:
                metrics[name] = float(values.mean())
        if game is not None:
            metrics |= _families(extras or {}, start, end, game)
        if "train/return" in metrics:
            metrics["train/score_pct"] = float(
                score_pct(metrics["train/return"], max_reward)
            )
        points.append((int(metrics["train/env_steps"]), metrics))
    return points


def log_points(points: Points) -> None:
    """Log ``(step, metrics)`` points to the active run in few requests."""
    now = int(time.time() * 1000)
    batch = [Metric(k, v, now, step) for step, m in points for k, v in m.items()]
    run = mlflow.active_run()
    if run is None:
        msg = "no active MLflow run to log training metrics to"
        raise RuntimeError(msg)
    client, run_id = mlflow.MlflowClient(), run.info.run_id
    for i in range(0, len(batch), _BATCH):
        client.log_batch(run_id, metrics=batch[i : i + _BATCH])


def compile_seconds(chunks: Sequence[tuple[int, float]]) -> float | None:
    """Compile time from ``(updates, seconds)`` per chunk: the first chunk's excess.

    The excess over the median steady update time of the later chunks; None when
    only one chunk ran.
    """
    if len(chunks) < 2:
        return None
    steady = statistics.median(t / n for n, t in chunks[1:])
    n0, t0 = chunks[0]
    return max(0.0, t0 - n0 * steady)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (all runs of a worker), MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def _device_peak_mb(roles: DeviceRoles) -> float | None:
    """Peak bytes in use over the role devices that report them, MB."""
    peaks = []
    for device in {roles.actor, roles.learner}:
        stats = device.memory_stats()  # None where the backend keeps no counters
        if stats and "peak_bytes_in_use" in stats:
            peaks.append(stats["peak_bytes_in_use"])
    return max(peaks) / 2**20 if peaks else None


def log_peak_memory(roles: DeviceRoles, step: int) -> None:
    """Log the peak host and device memory so far to the active run."""
    metrics: dict[str, Any] = {"peak_rss_mb": peak_rss_mb()}
    if (device := _device_peak_mb(roles)) is not None:
        metrics["peak_device_mb"] = device
    mlflow.log_metrics(metrics, step=step)


def run_tags(roles: DeviceRoles) -> dict[str, str]:
    """Tags of the machine a run (segment) trains on."""
    return {
        "actor_device": str(roles.actor),
        "learner_device": str(roles.learner),
        "device_kind": roles.learner.device_kind,
        "host_name": socket.gethostname(),
    }
