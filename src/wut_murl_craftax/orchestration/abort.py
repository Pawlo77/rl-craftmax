"""Abort rules of the runner: non-finite losses end a run, flags only tag it.

A NaN or inf in a logged loss (the trainer's ``loss_columns`` and every method metric
whose name holds ``loss``) aborts the run: it ends FAILED with its last ``state/``
kept. The next queue call resumes it from there once; a second abort of the same run
marks it a bug (tag ``abort/bug``) and the queue skips it from then on. Flags
(``flag/<name>`` tags) come from a method's ``abort.flags`` rules, see `flag_hits`;
they never stop a run, and nothing here looks at the score (no early stopping).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..training.records import LOSS_COLUMNS

TAG_ABORT_COUNT = "abort/count"
"""Run tag: how often the run was aborted on a non-finite loss."""
TAG_ABORT_METRIC = "abort/metric"
"""Run tag: the metric that was non-finite at the last abort."""
TAG_ABORT_UPDATE = "abort/update"
"""Run tag: the update (1-based) whose loss was non-finite at the last abort."""
TAG_ABORT_BUG = "abort/bug"
"""Run tag: ``true`` after a second abort; the queue skips such a run."""
FLAG_PREFIX = "flag/"
"""Prefix of the flag tags; the value is the update at which the rule first held."""
MAX_ABORTS = 2
"""Aborts after which a run counts as a bug: the first is retried once."""


def first_nonfinite(
    losses: np.ndarray,
    loss_columns: Sequence[str],
    method_metrics: Mapping[str, np.ndarray],
) -> tuple[int, str] | None:
    """The earliest update (0-based, in the chunk) with a non-finite logged loss.

    ``losses`` is the chunk's ``(updates, len(LOSS_COLUMNS))`` array, of which only
    ``loss_columns`` are checked; of ``method_metrics`` only the names holding
    ``loss``. Returns that update and the metric name, or None.
    """
    series = {c: losses[:, LOSS_COLUMNS.index(c)] for c in loss_columns}
    series |= {k: np.asarray(v) for k, v in method_metrics.items() if "loss" in k}
    hits = [
        (int(np.argmax(bad)), name)
        for name, values in series.items()
        if (bad := ~np.isfinite(values)).any()
    ]
    return min(hits, key=lambda h: h[0]) if hits else None


def _first_run(cond: np.ndarray, length: int) -> int | None:
    """Update (1-based) at which ``cond`` has held for ``length`` updates in a row."""
    if len(cond) < length:
        return None
    full = np.convolve(cond, np.ones(length, int), "valid") == length
    return int(np.argmax(full)) + length if full.any() else None


def flag_hits(
    rules: Sequence[Mapping[str, Any]],
    series: Mapping[str, Sequence[float]],
    n_updates: int,
) -> dict[str, int]:
    """Flag name -> the update (1-based) at which its rule first held.

    A rule is ``{name, metric, above | below, updates, before, after}``: the metric is
    above (below) the threshold for ``updates`` updates in a row (default 1); ``before``
    and ``after`` are budget fractions, and only updates at or before (at or after)
    that share of the ``n_updates`` count. NaN never satisfies a threshold; a metric
    the trainer does not report is skipped.
    """
    hits = {}
    for rule in rules:
        values = np.asarray(series.get(rule["metric"], ()), float)
        if not len(values):
            continue
        with np.errstate(invalid="ignore"):
            cond = values > rule["above"] if "above" in rule else values < rule["below"]
        share = np.arange(1, len(values) + 1) / n_updates
        if "before" in rule:
            cond &= share <= rule["before"]
        if "after" in rule:
            cond &= share >= rule["after"]
        if (update := _first_run(cond, rule.get("updates", 1))) is not None:
            hits[rule["name"]] = update
    return hits
