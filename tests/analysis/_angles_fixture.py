"""Frames for the analysis-angle figure tests (F21-F28): checkpoints, modes, trees.

`angle_frames` adds to the game fixture's final eval episodes two earlier checkpoints
and a per-episode score. A run's stochastic score at checkpoint ``u`` is its final
score times `RAMP[u]` plus `SLOT_SCORE` per eval slot; `REVERSED` reverses its seeds'
order at the first checkpoint; the greedy score is the stochastic one minus `GAP` times
the method's position in `LEVEL`. `write_distillation` writes policy folders of the
interpretation output for `DISTILLED` (one more folder is unfinished).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from _compare_fixture import LEVEL, STEPS
from _game_fixture import game_frames

UPDATES = (4, 54, 244)
RAMP = {4: 0.2, 54: 0.6, 244: 1.0}
SLOT_SCORE = 0.01
"""Score added per eval slot (0-3): an episode mean is the run's level + 0.015."""
GAP = 0.1
REVERSED = "sil"
"""Its seed ranks at the first checkpoint are the reverse of its final ranks."""
DISTILLED = ("ppo", "sil", "dyna-q")
"""Methods with finished policy folders at 1M; ``pqn-rnn`` has an unfinished one."""
FEATURES = ("water_dist", "food", "prev_action")
"""``prev_action`` is not an input of ``ppo`` (a feed-forward policy)."""


def checkpoint_steps(budget: str, update: int) -> int:
    return round(STEPS[budget] * update / UPDATES[-1])


def _score(eps: pd.DataFrame, runs: pd.DataFrame, update: int) -> np.ndarray:
    """Stochastic score of each episode row at ``update`` (see the module docstring)."""
    final = eps["run_id"].map(runs.set_index("run_id")["final_score"])
    level = final * RAMP[update]
    if update == UPDATES[0]:
        mine = runs[runs["method"] == REVERSED]
        top = mine.groupby("budget")["final_score"].transform("max")
        low = mine.groupby("budget")["final_score"].transform("min")
        flipped = (top + low - mine["final_score"]) * RAMP[update]
        level = level.where(eps["method"] != REVERSED, eps["run_id"].map(
            pd.Series(flipped.to_numpy(), index=mine["run_id"])
        ))  # fmt: skip
    slot = (eps["eval_seed"] - 10000) % 4
    return (level + SLOT_SCORE * slot).to_numpy(float)


def angle_frames(budgets: tuple[str, ...] = ("1m", "1b")) -> dict[str, pd.DataFrame]:
    """`game_frames` with eval episodes at `UPDATES` in both modes and a score."""
    frames = game_frames(budgets)
    runs, final = frames["runs"], frames["eval_episodes"]
    pos = final["method"].map({m: k for k, m in enumerate(LEVEL)})
    parts = []
    for update in UPDATES:
        eps = final.copy()
        eps["update"] = update
        eps["env_steps"] = [checkpoint_steps(b, update) for b in eps["budget"]]
        stochastic = _score(eps, runs, update)
        greedy = eps["mode"] == "greedy"
        eps["score_pct"] = np.where(greedy, stochastic - GAP * pos, stochastic)
        parts.append(eps)
    frames["eval_episodes"] = pd.concat(parts, ignore_index=True)
    return frames


def importance_of(method: str, seed: int, feature: str) -> float:
    """Toy permutation importance; NaN for a feature the method's trees never had."""
    if method == "ppo" and feature == "prev_action":
        return float("nan")
    return 0.01 * (FEATURES.index(feature) + 1) * (len(method) % 5 + 1) + 0.001 * seed


def fidelity_of(method: str, seed: int) -> float:
    return 0.3 + 0.05 * DISTILLED.index(method) + 0.004 * seed


def write_distillation(root: Path, runs: pd.DataFrame) -> Path:
    """Policy folders of the tuned 1M runs of `DISTILLED` under ``root/1m``."""
    tuned = runs[(runs["config_ref"] == "tuned") & (runs["budget"] == "1m")]
    for r in tuned[tuned["method"].isin(DISTILLED)].itertuples():
        folder = root / "1m" / r.method / r.run_id
        folder.mkdir(parents=True, exist_ok=True)
        rows = [
            {"feature": f, "group": "observation", "variant": "base", "std": 0.0,
             "importance": importance_of(r.method, r.seed, f)}
            for f in FEATURES
            if np.isfinite(importance_of(r.method, r.seed, f))
        ]  # fmt: skip
        pd.DataFrame(rows).to_csv(folder / "importance_base.csv", index=False)
        summary = {
            "run_id": r.run_id, "method": r.method, "budget": "1m",
            "config_ref": "tuned", "seed": r.seed, "score": r.final_score,
            "fits": {"base": {"fidelity_256": fidelity_of(r.method, r.seed)}},
        }  # fmt: skip
        (folder / "complexity.json").write_text(json.dumps(summary))
    unfinished = root / "1m" / "pqn-rnn" / "pqn-rnn-tuned-1m-0"
    unfinished.mkdir(parents=True, exist_ok=True)
    (unfinished / "importance_base.csv").write_text("feature,group,importance\n")
    return root
