"""Abort rules of the runner: non-finite losses end a run, flags only tag it.

A NaN or inf in a logged loss (the trainer's ``loss_columns`` and the method metrics the
method YAML lists under ``abort.finite``, none by default) aborts the run: it ends
FAILED with its last ``state/`` kept. The next queue call resumes it from there once;
a second abort of the same run marks it a bug (tag ``abort/bug``) and the queue skips
it from then on. A method that refuses its config at init for the machine's memory
(``MemoryLimitError``) ends the run FAILED with ``abort/memory``, skipped from then on
too. Flags
(``flag/<name>`` tags) come from a method's ``abort.flags`` rules, see `flag_hits`;
they never stop a run, and nothing here looks at the score (no early stopping).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..training.records import LOSS_COLUMNS
from .queue import TAG_ABORT_BUG as TAG_ABORT_BUG  # queue reads it without this module

TAG_ABORT_COUNT = "abort/count"
"""Run tag: how often the run was aborted on a non-finite loss."""
TAG_ABORT_MEMORY = "abort/memory"
"""Run tag: ``true`` when the method's buffers exceed the profile's memory limit; the
queue skips the run for good (change the config or the profile)."""
TAG_ABORT_METRIC = "abort/metric"
"""Run tag: the metric that was non-finite at the last abort."""
TAG_ABORT_UPDATE = "abort/update"
"""Run tag: the update (1-based) whose loss was non-finite at the last abort."""
TAG_ABORT_UNREPORTED = "warn/abort_unreported"
"""Run tag: comma-joined ``abort.finite`` and flag metrics the trainer never reports."""
FLAG_PREFIX = "flag/"
"""Prefix of the flag tags; the value is the update at which the rule first held."""
MAX_ABORTS = 2
"""Aborts after which a run counts as a bug: the first is retried once."""


def first_nonfinite(
    losses: np.ndarray,
    loss_columns: Sequence[str],
    method_metrics: Mapping[str, np.ndarray],
    finite: Sequence[str] = (),
) -> tuple[int, str] | None:
    """The earliest update (0-based, in the chunk) with a non-finite logged loss.

    ``losses`` is the chunk's ``(updates, len(LOSS_COLUMNS))`` array, of which only
    ``loss_columns`` are checked; of ``method_metrics`` only the names in ``finite``
    (a metric may be NaN by design, e.g. a mean over an empty buffer). Returns that
    update and the metric name, or None.
    """
    series = {c: losses[:, LOSS_COLUMNS.index(c)] for c in loss_columns}
    series |= {k: np.asarray(v) for k, v in method_metrics.items() if k in finite}
    hits = [
        (int(np.argmax(bad)), name)
        for name, values in series.items()
        if (bad := ~np.isfinite(values)).any()
    ]
    return min(hits, key=lambda h: h[0]) if hits else None


def unreported(
    rules: Sequence[Mapping[str, Any]], finite: Sequence[str], reported: Iterable[str]
) -> list[str]:
    """The ``finite`` and flag-rule metrics missing from the ``reported`` names."""
    return sorted({*finite, *(r["metric"] for r in rules)} - set(reported))


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
