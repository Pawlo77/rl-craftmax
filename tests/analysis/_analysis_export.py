"""Builder of the synthetic export tables of contracts.md section 1 (see conftest.py).

One world of runs: the nine registry methods (three controls, hook methods, the skill
methods, ``combo``), two ablation `config_ref` values and three seeds per
(method, config_ref). Each phase export is the world's runs whose ``phases`` tag holds
that phase, so the control runs of ``final-1m`` sit in ``tune-coarse`` too, as in the
store. The tables carry the contract columns, a few members of every column family
and the ``list<struct>`` stalls and fights.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wut_murl_craftax.evaluation.episode_metrics import CAUSES

SEEDS = (0, 1, 2)
"""Training seeds per (method, config_ref)."""
LEVEL = {
    "ppo": 2.1,
    "ppo-rnn": 2.2,
    "pqn-rnn": 1.3,
    "count-bonus-ppo-rnn": 2.4,
    "count-bonus-pqn-rnn": 1.8,
    "sil": 2.25,
    "smdp-skills": 3.0,
    "dyna-q": 2.9,
    "combo": 2.5,
}
"""Final score of seed 0 per method; seed s adds 0.1 * s, an ablation subtracts 0.3."""
FINAL_GROUPS: tuple[tuple[str, str], ...] = (
    *((m, "tuned") for m in LEVEL if m != "combo"),
    ("count-bonus-ppo-rnn", "simhash"),
    ("smdp-skills", "monte-carlo"),
)
"""(method, config_ref) of a ``final-*`` phase; ablations are mixed in."""
COMBO_PAIR = {"base": "ppo-rnn", "methods": ["count-bonus-ppo-rnn", "sil"]}
"""The pair recorded in the config of the fixture's combo runs."""
CONTROL_RUNS_ALSO_IN = "tune-coarse"
"""The phase the ``ppo`` and ``ppo-rnn`` control runs belong to besides ``final-*``."""
UPDATES = (0, 244)
"""Checkpoints of the fixture's eval episodes."""
EVAL_SEEDS = (10000, 10001)
"""Eval seeds per (run, update, mode)."""
ACHIEVEMENTS = ("collect_wood", "place_table", "defeat_zombie")
"""Lower-case enum names of the ``ach/<name>`` and ``unlock_step/<name>`` columns."""
STALL = pa.struct([
    ("start", pa.int64()),
    ("end", pa.int64()),
    ("length", pa.int64()),
    ("floor", pa.int64()),
    ("ending", pa.string()),
    ("ended_by", pa.string()),
])
FIGHT = pa.struct([
    ("start", pa.int64()),
    ("end", pa.int64()),
    ("max_enemies", pa.int64()),
    ("damage_taken", pa.float64()),
    ("outcome", pa.string()),
])
"""Arrow types of the ``stalls`` and ``fights`` columns."""
CURVE_METRICS = ("train/score_pct", "train/entropy", "train/ach/collect_wood")
"""Stored keys the fixture's curves hold (the ppo family logs ``ppo/entropy`` too)."""


def _hex(text: str, n: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def _world(phases: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every run of the requested ``final-*`` and ``combo-*`` phases, one dict each."""
    runs = []
    for phase in (p for p in phases if p.startswith(("final-", "combo-"))):
        budget = phase.split("-")[1]
        groups = [("combo", "tuned")] if phase.startswith("combo") else FINAL_GROUPS
        for method, ref in groups:
            tag = phase
            fine = "tune-fine" in phases
            if method in ("ppo", "ppo-rnn", *(("sil",) if fine else ())) and (
                phase == "final-1m"
            ):
                tag = f"{CONTROL_RUNS_ALSO_IN},{phase}"
                if method != "ppo" and fine:
                    tag = f"{CONTROL_RUNS_ALSO_IN},tune-fine,{phase}"
            for seed in SEEDS:
                name = f"{method}-{budget}-{_hex(f'{method}/{ref}', 8)}-s{seed}"
                config = {"method": method, "budget": budget, "seed": seed}
                if method == "combo":
                    config["combine"] = COMBO_PAIR
                runs.append({
                    "run_id": _hex(name, 32),
                    "run_name": name,
                    "experiment": tag.split(",")[0],
                    "phases": tag,
                    "method": method,
                    "budget": budget,
                    "config_hash": _hex(f"{method}/{ref}", 8),
                    "seed": seed,
                    "config_ref": ref,
                    "config_json": json.dumps(config),
                    "score": LEVEL[method] + 0.1 * seed - (ref != "tuned") * 0.3,
                })
    return runs


def _run_row(
    run: dict[str, Any], phase: str, rng: np.random.Generator
) -> dict[str, Any]:
    """One ``runs`` row; ``config_ref`` is the hash in a tuning phase."""
    score = run["score"]
    tuning = phase.startswith("tune-")
    return {
        **{k: v for k, v in run.items() if k != "score"},
        "phase": phase,
        "config_ref": run["config_hash"] if tuning else run["config_ref"],
        "best": tuning and run["method"] in ("ppo", "ppo-rnn") and run["seed"] == 0,
        "profile": "laptop",
        "device_kind": "Apple GPU",
        "actor_device": "cpu:0",
        "learner_device": "MetalDevice(id=0)",
        "status": "FINISHED",
        "online_families": run["method"] not in ("ppo", "ppo-rnn"),
        "final_score": score,
        "eval_stochastic_score": score,
        "eval_greedy_score": 0.9 * score,
        "eval_stochastic_return": 10 * score,
        "eval_greedy_return": 9 * score,
        "env_steps": 1_000_000,
        "wall_clock_s": 900.0 + rng.uniform(0, 50),
        "steady_sps": 1500.0,
        "overall_sps": 1100.0,
        "compile_s": 40.0,
        "eval_s": np.nan,
        "cpu_util_mean": 55.0,
        "cpu_util_p95": 80.0,
        "gpu_util_mean": np.nan,
        "gpu_util_p95": np.nan,
        "peak_device_mem_mb": np.nan,
        "energy_j": np.nan,
        "steps_per_joule": np.nan,
        "peak_rss_mb": 3000.0,
        "peak_host_mem_mb": 9000.0,
    }


def _curves(run: dict[str, Any]) -> pd.DataFrame:
    steps = np.array([0, 250_000, 500_000, 1_000_000], dtype="int64")
    frames = []
    metrics = CURVE_METRICS + (
        ("ppo/entropy",) if run["method"].startswith("ppo") else ()
    )
    for i, metric in enumerate(metrics):
        value = (run["score"] if "score" in metric else 1.0) * (steps + 1) / 1e6 + i
        frames.append(
            pd.DataFrame({
                "run_id": run["run_id"],
                "env_steps": steps,
                "metric": metric,
                "value": value,
            })
        )
    return pd.concat(frames, ignore_index=True)


def _episodes(
    run: dict[str, Any], phase: str, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    ref = run["config_hash"] if phase.startswith("tune-") else run["config_ref"]
    for update in UPDATES:
        grown = update > 0
        for mode in ("stochastic", "greedy"):
            for eval_seed in EVAL_SEEDS:
                died = bool(rng.integers(2))
                length = int(rng.integers(300, 1500))
                unlocked = {
                    a: int(rng.integers(20, length)) if grown else -1
                    for a in ACHIEVEMENTS
                }
                stalls = (
                    [
                        {
                            "start": 0,
                            "end": 320,
                            "length": 320,
                            "floor": 0,
                            "ending": "unlock",
                            "ended_by": "collect_wood",
                        }
                    ]
                    if grown and length > 400
                    else []
                )
                fights = (
                    [
                        {
                            "start": 50,
                            "end": 80,
                            "max_enemies": 2,
                            "damage_taken": 1.5,
                            "outcome": "survived",
                        }
                    ]
                    if grown
                    else []
                )
                rows.append({
                    "run_id": run["run_id"],
                    "method": run["method"],
                    "config_ref": ref,
                    "budget": run["budget"],
                    "seed": run["seed"],
                    "update": update,
                    "env_steps": 1_000_000 if grown else 0,
                    "mode": mode,
                    "eval_seed": eval_seed,
                    "length": length,
                    "return": run["score"] * 10 + (eval_seed - 10000),
                    "score_pct": run["score"] + 0.01 * (eval_seed - 10000),
                    "end_type": "death" if died else "timeout",
                    "cause_of_death": CAUSES[0] if died else "",
                    "max_floor": int(grown),
                    "tech_depth": 1 + int(grown),
                    "n_achievements": sum(v >= 0 for v in unlocked.values()),
                    **{f"ach/{a}": s >= 0 for a, s in unlocked.items()},
                    **{f"unlock_step/{a}": np.int32(s) for a, s in unlocked.items()},
                    "stalls": stalls,
                    "n_stalls": len(stalls),
                    "max_stall": max((s["length"] for s in stalls), default=0),
                    "fights": fights,
                    "kills/zombie": int(grown),
                    "damage_dealt": 2.0 * grown,
                    "damage_taken": 3.0,
                    "close_calls": 0,
                    "close_calls_recovered": 0,
                    "coverage/0": 0.2 + 0.1 * grown,
                    "coverage/1": 0.1 if grown else np.nan,
                    "new_tiles_per_100": 5.0,
                    "revisit_rate": 0.4,
                    "unique_positions": 120,
                    "low_share/food": 0.1,
                    "zero_share/food": 0.0,
                    "action_count/noop": 30,
                    "action_count/do": 70,
                    "action_entropy": 1.5,
                    "wasted_share": 0.2,
                    "noop_share": 0.1,
                    "forced_share": 0.0,
                    "gathered/wood": 4,
                    "used/wood": 3,
                    "use_ratio": 0.75,
                    "steps_per_achievement": 150.0,
                    "wasted_crafts": 0,
                    "wood_to_diamond": -1,
                    "adv/enchant": 0,
                    "split/floor_1": 400 if grown else -1,
                })
    return pd.DataFrame(rows)


def _write(
    frame: pd.DataFrame, path: Path, schema: dict[str, pa.DataType] | None = None
) -> None:
    """Parquet; list columns get an explicit Arrow type (empty lists lack one)."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for name, kind in (schema or {}).items():
        i = table.schema.get_field_index(name)
        table = table.set_column(
            i, name, pa.array(frame[name].tolist(), type=pa.list_(kind))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_export(
    root: Path, phases: tuple[str, ...] = ("final-1m", "combo-1m", "tune-coarse")
) -> Path:
    """Write ``<root>/<phase>/{runs,curves,eval_episodes}.parquet``; return ``root``.

    ``final-<b>`` and ``combo-<b>`` phases create their runs; ``tune-coarse`` holds the
    runs whose tag lists it (the ppo and ppo-rnn runs of ``final-1m``) with their
    ``config_hash`` as `config_ref` (list ``final-1m`` too, which makes those runs);
    ``tune-fine`` (when listed) holds the ppo-rnn runs and adds the sil runs to both
    tuning phases, so that a tuning phase has a method of its own (T7).
    """
    world = _world(phases)
    for phase in phases:
        rng = np.random.default_rng(int(_hex(phase, 8), 16))
        runs = [r for r in world if phase in r["phases"].split(",")]
        base = root / phase
        _write(
            pd.DataFrame([_run_row(r, phase, rng) for r in runs]), base / "runs.parquet"
        )
        _write(
            pd.concat([_curves(r) for r in runs], ignore_index=True),
            base / "curves.parquet",
        )
        episodes = pd.concat(
            [_episodes(r, phase, rng) for r in runs], ignore_index=True
        )
        _write(
            episodes, base / "eval_episodes.parquet", {"stalls": STALL, "fights": FIGHT}
        )
    return root
