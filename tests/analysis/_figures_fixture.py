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

    Of the 4 eval episodes of a seed, a method unlocks achievement ``i`` in the first
    ``(k + 2 * i) % 5``, ``k`` being the method's position in `LEVEL`: the rate differs
    by achievement and by method, so a misplaced row or column shows.
    """
    frames = make_frames(budgets)
    eps = frames["eval_episodes"].copy()
    slot = (eps["eval_seed"] - 10000) % 4
    offset = eps["method"].map({m: k for k, m in enumerate(LEVEL)})
    for i, name in enumerate(ACHIEVEMENTS):
        eps[f"ach/{name}"] = slot < (offset + 2 * i) % 5
    frames["eval_episodes"] = eps
    return frames


EFFECTS = {
    "ppo": (0.4, 0.002),
    "sil": (0.8, 0.10),
    "smdp-skills": (0.1, 0.50),
}
"""Score gain of the better learning rate and of the better second hyperparameter."""
SECOND = {"ppo": "clip", "sil": "clip", "smdp-skills": "eps_top"}
"""Name of each method's second hyperparameter, so the S_h tick labels differ."""


def tuning_frames() -> dict[str, pd.DataFrame]:
    """``runs`` of tune-coarse (3 methods) and tune-fine (2): 4 configs x 3 seeds.

    Each method has its own effects (`EFFECTS`, halved in tune-fine) and second
    hyperparameter name (`SECOND`), so ``r_c`` and ``S_h`` differ by method and phase.
    """
    rows = []
    for phase, methods in TUNED.items():
        scale = 1.0 if phase == "tune-coarse" else 0.5
        for method in methods:
            lr_gain, second_gain = (g * scale for g in EFFECTS[method])
            grid = itertools.product((1e-4, 3e-4), (0.1, 0.2))
            for i, (lr, second) in enumerate(grid):
                for seed in range(3):
                    score = (
                        1.0
                        + lr_gain * (lr == 3e-4)
                        + second_gain * (second == 0.2)
                        + 0.01 * seed
                    )
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
                            SECOND[method]: second,
                            "seed": seed,
                        }),
                    })
    return {"runs": pd.DataFrame(rows)}
