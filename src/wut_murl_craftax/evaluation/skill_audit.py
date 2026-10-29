"""Audit of skill choices in stored eval episodes: ``enchant`` with nothing to enchant.

`enchant_counts` counts, in one episode, the decisions that picked ``enchant`` and
those taken with gear owned but every item already carrying the table's enchantment
(`skills.enchant_redundant`). The start condition excludes that state, so the count is
0 for runs trained with it; without the table-type test such a skill would run to its
500-step timeout. `audit_file` reads one eval file (replaying only the episodes that
chose ``enchant``), `audit_runs` and the CLI (``--help``) every file of a phase.
@see [[Craftax skill library]]
@tested-by [[tests/evaluation/test_skill_audit.py]]
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
import pandas as pd

from ..orchestration.runtime import configure_runtime
from .rollout_metrics import MODES, OUT_ROOT, RunRef, phase_runs, plan_jobs

logger = logging.getLogger(__name__)

AUDIT_COLUMNS = (
    "run_id", "method", "budget", "seed", "update", "mode", "episodes", "decisions",
    "enchant", "redundant", "redundant_steps",
)  # fmt: skip
"""One row per eval file with skill extras: ``decisions`` are all skill decisions of
its episodes, ``enchant`` those that picked the skill, ``redundant`` the part with
nothing to enchant and ``redundant_steps`` the steps those redundant skills ran."""
OUT_FILE = OUT_ROOT.parent / "audit" / "enchant.csv"


class EnchantCounts(NamedTuple):
    """`enchant_counts` of one episode."""

    enchant: int
    """Decisions that picked ``enchant``."""
    redundant: int
    """Of those, with gear owned and every item already enchanted as the table."""
    steps: int
    """Steps the redundant skills ran, up to the next decision or the episode end."""


def enchant_counts(
    obs: np.ndarray, skill: np.ndarray, steps_in_skill: np.ndarray, length: int
) -> EnchantCounts:
    """The ``enchant`` decisions of one episode and the redundant ones among them.

    ``obs`` is the observation each action was chosen on (`replay_episode`), ``skill``
    the running skill id and ``steps_in_skill`` its age at each step (0 at a decision).
    """
    from ..training.skills import SKILL_IDS, decode_obs, enchant_redundant

    decisions = np.flatnonzero(np.asarray(steps_in_skill)[:length] == 0)
    picked = np.asarray(skill)[decisions] == SKILL_IDS["enchant"]
    if not picked.any():
        return EnchantCounts(0, 0, 0)
    ends = np.r_[decisions[1:], length][picked]
    at = decisions[picked]
    views = jax.vmap(decode_obs)(np.asarray(obs)[at])
    redundant = np.asarray(jax.vmap(enchant_redundant)(views))
    return EnchantCounts(
        len(at), int(redundant.sum()), int((ends - at)[redundant].sum())
    )


def audit_file(
    run: Any, arrays: Mapping[str, np.ndarray], head: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The `AUDIT_COLUMNS` row of one eval file, None without skill extras.

    ``run`` is an `EvalRun`; ``head`` holds the identity columns. Only the episodes
    that chose ``enchant`` are replayed.
    """
    from ..training.skills import SKILL_IDS
    from ..visualisation.eval_rollouts import replay_episode

    if "extras/skill" not in arrays:
        return None
    skill, steps = np.asarray(arrays["extras/skill"]), arrays["extras/steps_in_skill"]
    actions, seeds = np.asarray(arrays["actions"]), np.asarray(arrays["seeds"])
    length = np.asarray(arrays["length"])
    total, decisions = np.zeros(3, np.int64), 0
    for i, (seed, n) in enumerate(zip(seeds, length, strict=True)):
        start = steps[i, :n] == 0
        decisions += int(start.sum())
        if (start & (skill[i, :n] == SKILL_IDS["enchant"])).any():
            obs = replay_episode(run, int(seed), actions[i]).obs
            total += enchant_counts(obs, skill[i], steps[i], int(n))
    return {
        **head,
        "episodes": len(seeds),
        "decisions": decisions,
        "enchant": int(total[0]),
        "redundant": int(total[1]),
        "redundant_steps": int(total[2]),
    }


def audit_runs(
    runs: Sequence[RunRef], updates: str | Sequence[int], modes: Sequence[str]
) -> pd.DataFrame:
    """`audit_file` of every eval file of ``runs`` (`plan_jobs` order), as a frame."""
    from ..visualisation.eval_rollouts import EvalRun, load_eval

    rows, run, loaded = [], None, None
    for ref, update, mode in plan_jobs(runs, updates, modes):
        if loaded != ref.dir:
            run, loaded = EvalRun.load(ref.dir), ref.dir
        arrays = load_eval(ref.dir, update, mode == "greedy")
        head = {
            "run_id": ref.run_id, "method": ref.method, "budget": ref.budget,
            "seed": ref.seed, "update": update, "mode": mode,
        }  # fmt: skip
        if row := audit_file(run, arrays, head):
            rows.append(row)
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", action="append", help="tag phases value; repeatable")
    p.add_argument("--run", action="append", help="run id or name; repeatable")
    p.add_argument("--updates", default="all", help="all, final or a comma list")
    p.add_argument("--mode", choices=(*MODES, "both"), default="both")
    p.add_argument("--out", type=Path, default=OUT_FILE, help="the CSV to write")
    p.add_argument("--tracking-uri", required=True)
    a = p.parse_args(argv)
    if not a.phase and not a.run:
        p.error("give --phase or --run")
    return a


def main(argv: Sequence[str] | None = None) -> None:
    """Audit the eval files of a phase's runs and write one CSV row per file."""
    a = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    configure_runtime(host_devices=1)
    runs = phase_runs(a.tracking_uri, a.phase)
    if a.run:
        runs = [r for r in runs if r.run_id in a.run or r.name in a.run]
    modes = MODES if a.mode == "both" else (a.mode,)
    ups = (
        a.updates
        if a.updates in ("all", "final")
        else [int(u) for u in a.updates.split(",")]
    )
    table = audit_runs(runs, ups, modes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    sums = table.groupby("method")[["decisions", "enchant", "redundant"]].sum()
    logger.info("%d files of %d runs -> %s\n%s", len(table), len(runs), a.out, sums)


if __name__ == "__main__":
    main()
