"""Frames for the figure tests: the comparison fixture, achievements, tuning runs.

`figure_frames` adds ``ach/<name>`` columns to the comparison fixture's eval episodes
(F5); `tuning_frames` is a tune-coarse and tune-fine export of a control and three
methods with config grids whose learning rate matters and whose clip does not (F18).
"""

import itertools
import json

import pandas as pd
from _compare_fixture import LEVEL, make_frames

ACHIEVEMENTS = (
    "collect_wood", "place_table", "defeat_zombie", "make_arrow", "enter_dungeon",
    "enter_sewers",
)  # fmt: skip
"""Craftax names of the fixture's ``ach/<name>`` columns: basic, then floors 1 and 3."""
TUNED = {
    "tune-coarse": ("ppo", "sil", "smdp-skills"),
    "tune-fine": ("sil", "smdp-skills"),
}
"""Methods of each tuning phase."""


def figure_frames(
    budgets: tuple[str, ...] = ("1m", "1b"),
) -> dict[str, pd.DataFrame]:
    """The comparison fixture with an ``ach/<name>`` column per `ACHIEVEMENTS` name.

    A method unlocks the first ``round(level)`` achievements of each 4 episodes, so the
    rates differ by method and by achievement.
    """
    frames = make_frames(budgets)
    eps = frames["eval_episodes"].copy()
    slot = (eps["eval_seed"] - 10000) % 4
    level = eps["method"].map(LEVEL)
    for i, name in enumerate(ACHIEVEMENTS):
        eps[f"ach/{name}"] = (slot + i) % 4 < level.round().astype(int)
    frames["eval_episodes"] = eps
    return frames


def tuning_frames() -> dict[str, pd.DataFrame]:
    """``runs`` of tune-coarse (3 methods) and tune-fine (2): 4 configs x 3 seeds."""
    rows = []
    for phase, methods in TUNED.items():
        for method in methods:
            grid = itertools.product((1e-4, 3e-4), (0.1, 0.2))
            for i, (lr, clip) in enumerate(grid):
                for seed in range(3):
                    score = (1.0 if lr == 3e-4 else 0.5) + 0.01 * clip + 0.01 * seed
                    rows.append({
                        "run_id": f"{phase}-{method}-{i}-{seed}",
                        "phase": phase,
                        "phases": phase,
                        "method": method,
                        "budget": "1m",
                        "config_hash": f"{method}-h{i}",
                        "config_ref": f"{method}-h{i}",
                        "seed": seed,
                        "best": i == 3 and seed == 0,
                        "profile": "laptop",
                        "final_score": score,
                        "config_json": json.dumps({
                            "method": method,
                            "lr": lr,
                            "clip": clip,
                            "seed": seed,
                        }),
                    })
    return {"runs": pd.DataFrame(rows)}
