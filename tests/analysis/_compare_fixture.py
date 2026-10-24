"""In-memory export frames for the comparison tests (contracts.md section 1 columns).

Unlike the shared `_analysis_export` (3 seeds, parquet), these frames hold 10 seeds at
1M and 3 at 1B with noisy scores, so Holm families, 1B rows and a skill-block combo can
be tested. Only the columns the comparison reads are filled.
"""

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

SEEDS = {"1m": 10, "1b": 3}
"""Training seeds per (method, config_ref) and budget."""
STEPS = {"1m": 1_000_000, "1b": 1_000_000_000}
LEVEL = {
    "ppo": 2.1,
    "ppo-rnn": 2.3,
    "pqn-rnn": 1.3,
    "count-bonus-ppo-rnn": 2.4,
    "count-bonus-pqn-rnn": 1.5,
    "sil": 2.2,
    "smdp-skills": 0.9,
    "dyna-q": 1.1,
    "combo": 2.5,
}
"""Mean final score at 1M; 1B multiplies it by `GROWTH`."""
GROWTH = 5.0
NOISE = 0.25
"""Seed sd of the final score."""
ABLATIONS = (
    ("count-bonus-ppo-rnn", "simhash"),
    ("sil", "single-buffer"),
    ("smdp-skills", "random-top"),
    ("smdp-skills", "monte-carlo"),
    ("smdp-skills", "no-flags"),
    ("dyna-q", "kappa0"),
)
ABLATION_COST = 0.3
"""An ablation's score is its tuned one minus this (times `GROWTH` at 1B)."""
MAIN_PAIR = ("count-bonus-ppo-rnn", "sil")
SKILL_PAIR = ("count-bonus-ppo-rnn", "dyna-q")
COLLAPSING = "sil"
"""Seed 0 of this method rises to twice its score, then falls to a fifth of it."""
UNLOCKED = ("collect_wood", "place_table")
"""Achievements of the ``first_unlock/*`` curves (never logged by ppo and ppo-rnn)."""
NOT_LOGGED = ("ppo", "ppo-rnn")
"""Methods whose runs lack the online families (R78); their 1M runs are reused from
tune-coarse."""
TUNE_FINE = ("sil", "smdp-skills")
"""Methods whose tuned 1M seeds below `TUNING_SEEDS` are reused from tune-fine."""
TUNING_SEEDS = 3
RERUN_IN_FINE = "ppo"
"""Its seeds below `TUNING_SEEDS` are in tune-fine too, though it was selected on all 10
in tune-coarse (the store's ppo: an artefact R102 must not read as optimism)."""
EVAL_SEEDS = 4
CAUSE = {"smdp-skills": "zombie"}
"""Cause of death of every death; others die of ``starvation``."""


def _config(method: str, pair: Sequence[str], seed: int) -> str:
    config: dict[str, object] = {"method": method, "seed": seed}
    if method == "combo":
        config["combine"] = {"base": "ppo-rnn", "methods": list(pair)}
    return json.dumps(config)


def runs_frame(
    budgets: Sequence[str] = ("1m", "1b"), pair: Sequence[str] = MAIN_PAIR
) -> pd.DataFrame:
    """One ``runs`` row per run: tuned units of every method plus `ABLATIONS`."""
    rng = np.random.default_rng(7)
    units = [(m, "tuned") for m in LEVEL] + list(ABLATIONS)
    rows = []
    for budget in budgets:
        scale = 1.0 if budget == "1m" else GROWTH
        for method, ref in units:
            phase = f"{'combo' if method == 'combo' else 'final'}-{budget}"
            tuning = ""
            if budget == "1m" and ref == "tuned":
                tuning = "tune-coarse" if method in NOT_LOGGED else ""
                tuning = "tune-fine" if method in TUNE_FINE else tuning
            for seed in range(SEEDS[budget]):
                score = scale * (
                    LEVEL[method] - ABLATION_COST * (ref != "tuned")
                ) + rng.normal(0, NOISE * scale)
                tags = tuning
                if method == RERUN_IN_FINE and tuning and seed < TUNING_SEEDS:
                    tags = f"{tuning},tune-fine"
                rows.append({
                    "run_id": f"{method}-{ref}-{budget}-{seed}",
                    "phase": phase,
                    "phases": f"{tags},{phase}"
                    if tuning and (seed < TUNING_SEEDS or method in NOT_LOGGED)
                    else phase,
                    "method": method,
                    "budget": budget,
                    "config_hash": f"{method}/{ref}",
                    "seed": seed,
                    "config_ref": ref,
                    "best": False,
                    "profile": "laptop",
                    "config_json": _config(method, pair, seed),
                    "online_families": method not in NOT_LOGGED,
                    "final_score": score,
                    "eval_stochastic_score": 0.9 * score,
                    "eval_greedy_score": 0.8 * score,
                    "steady_sps": 1000.0 + 10 * seed,
                    "overall_sps": 800.0 + 10 * seed,
                    "wall_clock_s": STEPS[budget] / (800.0 + 10 * seed),
                    "compile_s": 30.0 + seed,
                    "peak_rss_mb": 3000.0,
                    "peak_host_mem_mb": 9000.0,
                    "peak_device_mem_mb": np.nan,
                    "steps_per_joule": np.nan,
                    "cpu_util_mean": 50.0,
                    "gpu_util_mean": np.nan,
                })
    return pd.DataFrame(rows)


def _score_curve(run: pd.Series) -> np.ndarray:
    """50 per-update scores rising to the final score; `COLLAPSING` seed 0 collapses."""
    ramp = np.linspace(0.0, 1.0, 50) * run["final_score"]
    if run["method"] == COLLAPSING and run["seed"] == 0:
        ramp[20:] = np.r_[np.full(10, 2.0), np.full(20, 0.2)] * run["final_score"]
    return ramp


def curves_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """``train/score_pct`` per run; ``first_unlock/*`` for the methods that log them."""
    parts = []
    for run in runs.itertuples():
        steps = np.linspace(0, STEPS[run.budget], 50).astype("int64")
        parts.append(
            pd.DataFrame({
                "run_id": run.run_id,
                "env_steps": steps,
                "metric": "train/score_pct",
                "value": _score_curve(pd.Series(run._asdict())),
            })
        )
        if run.method in NOT_LOGGED:
            continue
        for i, name in enumerate(UNLOCKED):
            step = float(1000 * (i + 1) * (1 + run.seed) * (1 + len(run.method)))
            parts.append(
                pd.DataFrame({
                    "run_id": run.run_id,
                    "env_steps": [int(step)] * 2,
                    "metric": [f"first_unlock/{name}", f"first_unlock_wall/{name}"],
                    "value": [step, step / 1000],
                })
            )
    return pd.concat(parts, ignore_index=True)


def episodes_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Final-checkpoint eval episodes (update 244, both modes), `EVAL_SEEDS` per run."""
    rows = []
    for run in runs.itertuples():
        for mode in ("stochastic", "greedy"):
            for e in range(EVAL_SEEDS):
                died = e % 2 == 0
                rows.append({
                    "run_id": run.run_id,
                    "method": run.method,
                    "config_ref": run.config_ref,
                    "budget": run.budget,
                    "seed": run.seed,
                    "update": 244,
                    "mode": mode,
                    "eval_seed": 10000 + e,
                    "end_type": "death" if died else "timeout",
                    "cause_of_death": CAUSE.get(run.method, "starvation")
                    if died
                    else "",
                    "max_floor": int(e == 0 and run.seed % 2 == 0),
                    "coverage/0": 0.2 + 0.01 * e,
                    "coverage/1": 0.1 if e == 0 else np.nan,
                    "wasted_share": 0.1 * (e + 1),
                    "adv/cast_fireball": int(e == 0),
                    "adv/drink_potion": 0,
                })
    return pd.DataFrame(rows)


def make_frames(
    budgets: Sequence[str] = ("1m", "1b"), pair: Sequence[str] = MAIN_PAIR
) -> dict[str, pd.DataFrame]:
    """The three frames of `runs_frame`'s runs."""
    runs = runs_frame(budgets, pair)
    return {
        "runs": runs,
        "curves": curves_frame(runs),
        "eval_episodes": episodes_frame(runs),
    }
