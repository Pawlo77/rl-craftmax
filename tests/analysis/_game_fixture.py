"""Frames for the in-game figure tests: the figure fixture plus the replay families.

`game_frames` adds to the comparison fixture's final eval episodes the columns F6-F17
read (unlock steps, splits, stalls, fights, coverage, survival, combat, actions,
resources, advanced use) and to its curves the funnel, advanced-achievement and action
entropy series of the methods that log them (never ``ppo`` and ``ppo-rnn``, R78). Every
value is a deterministic function of (method position, seed, eval slot), so a misplaced
row, column or method shows in the tests.
"""

import numpy as np
import pandas as pd
from _compare_fixture import LEVEL, NOT_LOGGED, STEPS
from _figures_fixture import ACHIEVEMENTS, figure_frames

from wut_murl_craftax.evaluation.milestones import ADVANCED_ACHIEVEMENTS, TECH_LADDER

ENTER = ("enter_dungeon", "enter_gnomish_mines", "enter_sewers")
"""``split/<name>`` columns of the floors the fixture reaches, besides the ladder."""
CREATURES = ("zombie", "skeleton")
ACTIONS = ("noop", "left", "right", "do", "sleep")
RESOURCES = ("wood", "stone", "diamond")
FLOOR_COLUMNS = ("coverage/0", "coverage/1", "coverage/2")
STAT_NAMES = ("food", "drink", "energy")


def _episodes(eps: pd.DataFrame) -> pd.DataFrame:
    """The final eval episodes with the replay families (see the module docstring)."""
    out = eps.copy()
    slot = ((out["eval_seed"] - 10000) % 4).to_numpy()
    pos = out["method"].map({m: k for k, m in enumerate(LEVEL)}).to_numpy()
    seed = out["seed"].to_numpy()
    out["env_steps"] = 1_000_000
    out["length"] = 1000 + 100 * slot + 10 * pos
    for i, name in enumerate(
        ACHIEVEMENTS
    ):  # sequential unlocks: a chain a0 -> a1 -> ..
        out[f"unlock_step/{name}"] = np.where(
            out[f"ach/{name}"], 100 * (i + 1) + slot, -1
        ).astype("int32")
    out["n_achievements"] = out[[f"ach/{a}" for a in ACHIEVEMENTS]].sum(axis=1)
    depth = np.minimum(len(TECH_LADDER), (pos + slot + seed % 3) % 11)
    floor = np.minimum(8, (pos + slot) // 2)
    out["tech_depth"], out["max_floor"] = depth, floor
    for j, name in enumerate(TECH_LADDER):
        out[f"split/{name.lower()}"] = np.where(
            depth >= j + 1, 50 * (j + 1) + 7 * slot + 3 * seed, -1
        )
    for f, name in enumerate(ENTER, start=1):
        out[f"split/{name}"] = np.where(floor >= f, 400 * f + 11 * slot + seed, -1)
    out["stalls"] = [
        [
            {
                "start": 0, "end": 300 + 50 * s, "length": 300 + 50 * s,
                "floor": int(f % 3), "ending": "unlock" if s % 2 else "death",
                "ended_by": "collect_wood",
            }
        ]
        * int(s > 0)
        for s, f in zip(slot, floor, strict=True)
    ]  # fmt: skip
    out["n_stalls"] = (slot > 0).astype(int)
    out["fights"] = [
        [{"start": 10, "end": 30, "max_enemies": 2, "damage_taken": 1.0,
          "outcome": "survived"}] * int(s)
        for s in slot
    ]  # fmt: skip
    out["kills/zombie"] = pos + slot
    out["kills/skeleton"] = slot
    out["kills/passive"] = 2 * slot
    out["damage_dealt"] = 3.0 + pos + slot
    out["damage_taken"] = 2.0 + 0.5 * pos + 0.1 * slot
    for c, col in enumerate(FLOOR_COLUMNS):
        out[col] = np.where(floor >= c, 0.1 * (pos + 1) / (c + 1) + 0.01 * slot, np.nan)
    out["new_tiles_per_100"] = 3.0 + pos + 0.1 * slot
    for s, stat in enumerate(STAT_NAMES):
        out[f"low_share/{stat}"] = 0.05 * (s + 1) + 0.01 * pos
        out[f"zero_share/{stat}"] = 0.02 * (s + 1) + 0.005 * slot
        out[f"survival/level/{stat}"] = np.where(pos == 3, np.nan, 2.0 + s + 0.1 * pos)
    counts = {a: (3 + pos + i * slot + 2 * i) for i, a in enumerate(ACTIONS)}
    for a, c in counts.items():
        out[f"action_count/{a}"] = c
    out["action_entropy"] = 1.0 + 0.1 * pos + 0.01 * slot
    for r, res in enumerate(RESOURCES):
        out[f"gathered/{res}"] = 10.0 * (r + 1) + pos
        out[f"used/{res}"] = 5.0 * (r + 1) + 0.5 * pos
    out["steps_per_achievement"] = 200.0 + 10 * pos + slot
    out["wood_to_diamond"] = np.where(slot == 3, -1, 500 + 20 * pos + slot)
    out["adv/cast_fireball"] = (slot > 0).astype(int) * (pos % 2)
    out["adv/drink_potion"] = (slot > 1).astype(int)
    out["wasted_share"] = 0.05 + 0.02 * pos
    return out


def _curves(runs: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Add the funnel, advanced and action-entropy series of the logging methods."""
    parts = [curves]
    for run in runs.itertuples():
        if run.method in NOT_LOGGED:
            continue
        steps = np.linspace(0, STEPS[run.budget], 8).astype("int64")
        ramp = np.linspace(0.0, 1.0, 8)
        pos = list(LEVEL).index(run.method)
        series = (
            {f"train/funnel/depth_{k}": 100 * ramp * 0.9**k for k in range(1, 11)}
            | {f"train/funnel/floor_{f}": 100 * ramp * 0.5**f for f in range(9)}
            | {
                f"train/ach/{a.lower()}": 10 * (pos + 1) * ramp
                for a in ADVANCED_ACHIEVEMENTS
            }
            | {"train/action_entropy": 3.0 - 2 * ramp}
        )
        parts.append(
            pd.DataFrame({
                "run_id": run.run_id,
                "env_steps": np.tile(steps, len(series)),
                "metric": np.repeat(list(series), len(steps)),
                "value": np.concatenate(list(series.values())),
            })
        )
    return pd.concat(parts, ignore_index=True)


def game_frames(
    budgets: tuple[str, ...] = ("1m", "1b"),
) -> dict[str, pd.DataFrame]:
    """`figure_frames` with the in-game families (10 seeds at 1M, 3 at 1B)."""
    frames = figure_frames(budgets)
    return {
        "runs": frames["runs"],
        "curves": _curves(frames["runs"], frames["curves"]),
        "eval_episodes": _episodes(frames["eval_episodes"]),
    }
