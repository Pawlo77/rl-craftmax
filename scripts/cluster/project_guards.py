"""Project the guarded table counts of 1B runs to the end of their budget.

The permanent guards `smdp/table_full`, `dyna/model_full` and `count/table_full` end
a run for good, and the counts behind them still grow faster than linearly at 1M. Read
`smdp/keys_visited`, `dyna/pairs_visited` and `count/distinct_keys` of a node store
once a run passed 5-10 % of its budget, fit them against env steps on log-log axes
(from 10 % of the steps so far) and project to the budget. A projection above 0.7 of
the guard means resizing before the other seeds start (compute-budget plan, Task 2):

    uv run python scripts/cluster/project_guards.py --tracking-uri sqlite:///<node>.db
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

ACT = 0.7
"""Share of a guard whose projection calls for a larger table."""
GUARDS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "smdp/keys_visited": lambda c: 0.25 * 2 ** c["log2_rows"],
    "dyna/pairs_visited": lambda c: float(c["model_pairs"]),
    "count/distinct_keys": lambda c: c["max_occupancy"] * 2 ** c["log2_k"],
}
"""Guarded metric -> the count at which its guard ends a run, from the run's config."""


def project(
    steps: Sequence[float], values: Sequence[float], budget: float, since: float = 0.1
) -> float:
    """The log-log fit of the counts at ``budget`` steps; NaN below 5 positive points.

    Fits ``values`` against ``steps`` from ``since`` of the last step on.
    """
    t, v = np.asarray(steps, float), np.asarray(values, float)
    keep = (t >= since * t.max()) & (v > 0) & (t > 0) if len(t) else t > 0
    if keep.sum() < 5:
        return float("nan")
    slope, intercept = np.polyfit(np.log(t[keep]), np.log(v[keep]), 1)
    return float(np.exp(intercept + slope * np.log(budget)))


def main(argv: Sequence[str] | None = None) -> int:
    """Print every run's projections; exit 2 when one is above `ACT` of its guard."""
    from wut_murl_craftax.orchestration.queue import (
        CONFIG_ARTIFACT,
        TAG_BUDGET,
        _client,
        _runs,
        budget_steps,
    )
    from wut_murl_craftax.orchestration.tracking import artifact_path

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tracking-uri", required=True, help="the node's store")
    parser.add_argument("--since", type=float, default=0.1, help="fit from this share")
    args = parser.parse_args(argv)
    client = _client(args.tracking_uri)
    over = False
    for run in _runs(client, ""):
        path = artifact_path(run.info.artifact_uri) / CONFIG_ARTIFACT
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        budget = budget_steps(run.data.tags[TAG_BUDGET])
        for metric, guard_of in GUARDS.items():
            history = client.get_metric_history(run.info.run_id, metric)
            if not history:
                continue
            guard = guard_of(config)
            got = project(
                [m.step for m in history],
                [m.value for m in history],
                budget,
                args.since,
            )
            now = max(history, key=lambda m: m.step)
            flag = got > ACT * guard
            over |= flag
            print(
                f"{run.info.run_name} {metric}: {now.value:,.0f} at {now.step:,}, "
                f"projected {got:,.0f} of guard {guard:,.0f} ({got / guard:.2f})"
                f"{' RESIZE' if flag else ''}"
            )
    return 2 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
