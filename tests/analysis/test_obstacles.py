"""Tests for the obstacle detectors: one firing and one quiet input per detector.

`world` builds the frames of one run of a method plus its base control runs (two
seeds; a control's are the other controls), every series healthy unless overridden,
and the optional inputs (eval actions and the replay frames of evaluation Task 6).
"""

import fnmatch
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wut_murl_craftax.analysis import obstacles as ob
from wut_murl_craftax.analysis.load import BASE_CONTROL, CONTROLS, load_phases
from wut_murl_craftax.analysis.obstacles import (
    ABSENT,
    ACTIONS,
    ALIASES,
    BASIC_ACHIEVEMENTS,
    DETECTED,
    DETECTORS,
    INVENTORY,
    NEEDS_ACTIONS,
    NEEDS_CAUSES,
    NEEDS_REPLAY,
    NOT_APPLICABLE,
    NOT_LOGGED,
    SKILL_ENDS,
    TABLE_COLUMNS,
    Cycle,
    Obstacle,
    count_bonus_detectors,
    detect_all,
    drawdown_floor,
    dyna_detectors,
    eval_actions,
    obstacles_table,
    repeated_cycle,
    shared_detectors,
    sil_detectors,
    skill_detectors,
    table_rows,
)
from wut_murl_craftax.evaluation.episode_metrics import NEEDS
from wut_murl_craftax.evaluation.milestones import achievement_names
from wut_murl_craftax.evaluation.select_pair import SKILL_METHODS
from wut_murl_craftax.training.interface import METHODS

N = 101
"""Logged points per run: every 1% of the budget."""
SCALE = {"1m": 1_000_000, "1b": 1_000_000_000}
UPDATES = (0, 122, 244)
"""Eval checkpoints: start, half and end of the budget."""
EVAL_SEEDS = (10000, 10001, 10002, 10003)
LENGTH = 300
"""Length of every eval episode (all end in death)."""
SKILL_SHARES = {
    "collect_wood": (0.3, 0.8),
    "fight": (0.15, 0.5),
    "unlock_any": (0.1, 0.5),
    "collect_diamond": (0.15, 0.3),
    "descend": (0.1, 0.3),
    "make_iron_pickaxe": (0.2, 0.4),
}
"""Healthy (chosen_share, success_rate) per skill."""
PPO_FAMILY = ("ppo", "ppo-rnn", "count-bonus-ppo-rnn", "sil", "combo")
COMBO_PAIR = {"base": "ppo-rnn", "methods": ["count-bonus-ppo-rnn", "sil"]}
COUNT = ("count-bonus-ppo-rnn", "count-bonus-pqn-rnn")
SKILL = ("smdp-skills", "dyna-q")

type Series = np.ndarray | dict[float, float] | None
"""A metric override: N values, ``{share of budget: value}`` points, or None (drop)."""


def _ramp(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, N)


def _const(v: float) -> np.ndarray:
    return np.full(N, v)


def _healthy(method: str) -> dict[str, Series]:
    """Series of a run that should trigger no detector."""
    ret = _ramp(1.0, 10.0)
    m: dict[str, Series] = {
        "train/score_pct": _ramp(0.0, 2.0),
        "train/return": ret,
        "train/n_episodes": _const(100.0),
        "train/value_loss": _const(0.2),
        "train/ach_reward_share": _const(0.95),
        "train/ach/collect_wood": _ramp(0.0, 80.0),
        "train/ach/defeat_zombie": _const(30.0),
        "first_unlock/collect_wood": {0.01: 10_000.0},
        "first_unlock/place_table": {0.6: 600_000.0},
    }
    if method in PPO_FAMILY:
        m |= {
            "ppo/entropy": _const(2.0),
            "ppo/explained_variance": _const(0.5),
            "ppo/grad_norm": _const(1.0),
        }
    else:
        m |= {
            "train/action_entropy": _const(2.0),
            "pqn/grad_norm": _const(1.0),
            "pqn/q_max_mean": _const(5.0),
        }
    if method in (*COUNT, "combo"):
        m |= {
            "count/r_int_share": _const(0.1),
            "count/new_keys": _const(5.0),
            "count/r_int_mean": _const(0.01),
        }
    if method in ("sil", "combo"):
        m |= {
            "sil/sample_age_max": _const(10.0),
            "sil/active": _const(0.3),
            "sil/value_bias": _const(-0.1),
            "sil/overflow": _const(0.0),
        }
    if method in SKILL:
        eps = np.concatenate([np.linspace(1.0, 0.05, 21), np.full(N - 21, 0.05)])
        m |= {
            "smdp/eps": eps,
            "smdp/decisions_per_episode": _const(50.0),
            "smdp/q_start": 1.5 * ret,
        }
        for s, (chosen, success) in SKILL_SHARES.items():
            m |= {
                f"smdp/skill/{s}/chosen_share": _const(chosen),
                f"smdp/skill/{s}/success_rate": _const(success),
                f"smdp/skill/{s}/end/timeout": _const(0.1),
                f"smdp/skill/{s}/test_agreement": _const(0.99),
            }
    if method == "dyna-q":
        m |= {"dyna/model_accuracy": _const(0.9), "dyna/plan_td_abs": _const(0.01)}
    return m


def _config(method: str, extra: dict[str, Any]) -> str:
    cfg: dict[str, Any] = {"method": method, "envs": 256, "num_steps": 16}
    if method in (*COUNT, "combo"):
        cfg |= {"granularity": "recipe", "key": "progress"}
    if method == "dyna-q":
        cfg["plan_kappa"] = 0.006
    if method == "combo":
        cfg["combine"] = COMBO_PAIR
    return json.dumps(cfg | extra)


def _curves(run_id: str, series: dict[str, Series], scale: int) -> pd.DataFrame:
    steps = np.linspace(0, scale, N).astype("int64")
    parts = []
    for metric, v in series.items():
        if v is None:
            continue
        if isinstance(v, dict):
            at = np.array([int(k * scale) for k in v], dtype="int64")
            vals = np.array(list(v.values())) * (scale / SCALE["1m"])
        else:
            at, vals = steps, np.asarray(v, dtype=float)
        parts.append(
            pd.DataFrame({
                "run_id": run_id,
                "env_steps": at,
                "metric": metric,
                "value": vals,
            })
        )
    return pd.concat(parts, ignore_index=True)


def _episodes(
    run: dict[str, Any], scale: int, final: dict[str, Any], every: dict[str, Any]
) -> pd.DataFrame:
    rows = [
        {
            "run_id": run["run_id"],
            "method": run["method"],
            "config_ref": run["config_ref"],
            "budget": run["budget"],
            "seed": run["seed"],
            "update": u,
            "env_steps": scale * u // UPDATES[-1],
            "mode": "stochastic",
            "eval_seed": s,
            "length": LENGTH,
            "return": 10.0,
            "score_pct": 2.0,
            "end_type": "death",
            "cause_of_death": "zombie",
            "max_floor": 1,
            **every,
            **(final if u == UPDATES[-1] else {}),
        }
        for u in UPDATES
        for s in EVAL_SEEDS
    ]
    return pd.DataFrame(rows)


def _actions(run_id: str, cycle: bool) -> pd.DataFrame:
    rng = np.random.default_rng(len(run_id))
    rows = []
    for s in EVAL_SEEDS:
        a = rng.integers(0, 43, LENGTH)
        if cycle:
            a[100:200] = np.tile([2, 3, 5], 34)[:100]
        rows.append({
            "run_id": run_id,
            "update": UPDATES[-1],
            "mode": "stochastic",
            "eval_seed": s,
            "actions": a,
        })
    return pd.DataFrame(rows)


def _skill_ends(
    run_id: str, scale: int, override: Callable[..., Any] | None
) -> pd.DataFrame:
    """Six successful skills with distinct keys per episode, and one low-stat end."""
    rows = []
    for u in UPDATES[1:]:
        for s in EVAL_SEEDS:
            for i, skill in enumerate(SKILL_SHARES):
                rows.append({
                    "run_id": run_id,
                    "update": u,
                    "env_steps": scale * u // UPDATES[-1],
                    "mode": "stochastic",
                    "eval_seed": s,
                    "start": 40 * i,
                    "end": 40 * i + 30,
                    "skill": skill,
                    "key": 1000 * s + i,
                    "reason": "success",
                    "pair_count": 10,
                })
            rows.append({**rows[-1], "start": 70, "end": 100, "reason": "low_stat"})
    frame = pd.DataFrame(rows)
    return override(frame) if override else frame


def world(
    method: str = "ppo-rnn",
    budget: str = "1m",
    *,
    run: dict[str, Series] | None = None,
    base: dict[str, Series] | None = None,
    final: dict[str, Any] | None = None,
    every: dict[str, Any] | None = None,
    base_episodes: dict[str, Any] | None = None,
    base_by: dict[tuple[str, int], dict[str, Series]] | None = None,
    base_episodes_by: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    cycle: bool = False,
    skill_ends: Callable[..., Any] | None = None,
    inventory: float | None = 0.1,
    inputs: tuple[str, ...] = (ACTIONS, SKILL_ENDS, INVENTORY),
) -> dict[str, pd.DataFrame]:
    """One tuned run of ``method``, two seeds of its base control, optional inputs."""
    scale = SCALE[budget]
    if method in CONTROLS:
        bases = [c for c in CONTROLS if c != method]
    else:
        bases = [COMBO_PAIR["base"] if method == "combo" else BASE_CONTROL[method]]
    specs = [(method, 0, run or {}, final or {}, every or {}, config or {})]
    by, eps_by = base_by or {}, base_episodes_by or {}
    specs += [
        (
            b,
            s,
            (base or {}) | by.get((b, s), {}),
            {},
            (base_episodes or {}) | eps_by.get(b, {}),
            {},
        )
        for b in bases
        for s in (0, 1)
    ]
    runs, curves, episodes, actions, ends, inv = [], [], [], [], [], []
    for m, seed, over, fin, ev, cfg in specs:
        row = {
            "run_id": f"{m}-{seed}",
            "run_name": f"{m}-{budget}-s{seed}",
            "method": m,
            "config_ref": "tuned",
            "budget": budget,
            "seed": seed,
            "phases": f"final-{budget}",
            "config_json": _config(m, cfg),
        }
        runs.append(row)
        curves.append(_curves(row["run_id"], _healthy(m) | over, scale))
        episodes.append(_episodes(row, scale, fin, ev))
        actions.append(_actions(row["run_id"], cycle and m == method))
        if m in SKILL:
            ends.append(
                _skill_ends(row["run_id"], scale, skill_ends if m == method else None)
            )
        steps = pd.DataFrame({"eval_seed": EVAL_SEEDS})
        inv.append(
            steps.assign(
                run_id=row["run_id"],
                update=UPDATES[-1],
                mode="stochastic",
                steps=LENGTH,
                capped_steps=int(LENGTH * (inventory or 0.0)),
            )
        )
    frames = {
        "runs": pd.DataFrame(runs),
        "curves": pd.concat(curves, ignore_index=True),
        "eval_episodes": pd.concat(episodes, ignore_index=True),
    }
    optional = {
        ACTIONS: actions,
        SKILL_ENDS: ends,
        INVENTORY: inv if inventory is not None else [],
    }
    for name in inputs:
        if optional[name]:
            frames[name] = pd.concat(optional[name], ignore_index=True)
    return frames


def result(
    frames: dict[str, pd.DataFrame], method: str, name: str, budget: str = "1m"
) -> Obstacle:
    """The one obstacle ``name`` of the method's run."""
    found = [o for o in detect_all(frames, method, budget) if o.name == name]
    assert len(found) == 1, found
    return found[0]


def _at(*points: tuple[int, int, float], base: np.ndarray) -> np.ndarray:
    """``base`` with ``[a, b)`` set to a value for each ``(a, b, value)``."""
    out = base.copy()
    for a, b, v in points:
        out[a:b] = v
    return out


def _flat_after(i: int, level: float = 1.0) -> np.ndarray:
    """A score rising to ``level`` at point ``i``, flat after it."""
    return np.r_[np.linspace(0.0, level, i), np.full(N - i, level)]


# --- registry ---------------------------------------------------------------------


def test_detector_counts_and_registry() -> None:
    shared = shared_detectors()
    assert len(shared) == 10
    method = [
        *count_bonus_detectors(),
        *sil_detectors(),
        *skill_detectors(),
        *dyna_detectors(),
    ]
    assert len(method) == 23
    assert len({d.name for d in shared + method}) == 33
    assert set(DETECTORS) == set(METHODS)
    names = {m: [d.name for d in ds] for m, ds in DETECTORS.items()}
    shared_names = [d.name for d in shared]
    for control in CONTROLS:
        assert names[control] == shared_names
    by = {
        "count": [d.name for d in count_bonus_detectors()],
        "sil": [d.name for d in sil_detectors()],
        "skill": [d.name for d in skill_detectors()],
        "dyna": [d.name for d in dyna_detectors()],
    }
    q = "Q overestimation"
    assert q in by["count"]
    assert names["count-bonus-ppo-rnn"] == shared_names + [
        n for n in by["count"] if n != q
    ]
    assert names["count-bonus-pqn-rnn"] == shared_names + by["count"]
    assert names["sil"] == shared_names + by["sil"]
    assert names["smdp-skills"] == shared_names + by["skill"]
    assert names["dyna-q"] == shared_names + by["skill"] + by["dyna"]
    assert set(names["combo"]) == set(
        shared_names + by["count"] + by["sil"] + by["skill"] + by["dyna"]
    )
    for d in shared + method:
        assert isinstance(d.REQUIRED_METRICS, tuple)


def test_constants_match_the_game_tables() -> None:
    assert achievement_names("craftax")[:25] == BASIC_ACHIEVEMENTS
    assert NEEDS_CAUSES == NEEDS
    for d in skill_detectors():
        assert d.methods == SKILL_METHODS
    assert ALIASES["policy entropy"] == ("ppo/entropy", "train/entropy")
    assert ALIASES["sil/sample_age"][0] == "sil/sample_age_max"
    assert ALIASES["sil/insert_overflow"] == ("sil/overflow",)
    assert ALIASES["Q values"] == ("pqn/q_max_mean",)
    assert ALIASES["gradient norm"] == (
        "ppo/grad_norm",
        "pqn/grad_norm",
        "train/grad_norm",
    )
    assert ALIASES["explained variance"] == (
        "ppo/explained_variance",
        "train/explained_var",
    )


@pytest.mark.parametrize("method", sorted(METHODS))
def test_a_healthy_run_has_no_obstacle(method: str) -> None:
    found = detect_all(world(method), method, "1m")
    assert {o.status for o in found if o.status == DETECTED} == set(), [
        o for o in found if o.status == DETECTED
    ]
    assert len(found) == len([
        d for d in DETECTORS[method] if method != "combo" or _in_pair(d)
    ])


def _in_pair(d: ob.Detector) -> bool:
    return not d.methods or bool(set(d.methods) & set(COMBO_PAIR["methods"]))


def test_combo_gets_the_detectors_of_its_own_pair() -> None:
    names = {o.name for o in detect_all(world("combo"), "combo", "1m")}
    assert {
        d.name for d in count_bonus_detectors() if d.name != "Q overestimation"
    } <= names
    assert {d.name for d in sil_detectors()} <= names
    assert not names & {d.name for d in skill_detectors() + dyna_detectors()}
    assert "Q overestimation" not in names


def test_detect_all_keeps_tuned_and_ablation_runs_apart() -> None:
    frames = world("sil")
    extra = (
        frames["runs"].iloc[[0]].assign(run_id="sil-abl", config_ref="single-buffer")
    )
    frames["runs"] = pd.concat([frames["runs"], extra], ignore_index=True)
    runs = {o.run for o in detect_all(frames, "sil", "1m")}
    assert runs == {"sil-0"}
    runs = {o.run for o in detect_all(frames, "sil", "1m", config_ref="single-buffer")}
    assert runs == {"sil-abl"}


# --- one firing case per detector -------------------------------------------------

DIP = _ramp(0.0, 2.0) * np.where((np.arange(N) >= 70) & (np.arange(N) <= 80), 0.6, 1.0)
"""A 40% dip once the running max is above 1% (`stability.MIN_MAX`), then recovery."""
FALL = _at((70, N, 0.2), base=_ramp(0.0, 2.0))
LUCKY = {
    "sil/sample_age_max": _const(1000.0),
    "sil/active": _const(0.8),
    "ppo/entropy": _const(1.0),
}
EARLY_UNLOCK = {"first_unlock/place_table": {0.3: 300_000.0}}

FIRES: list[tuple[str, str, dict[str, Any]]] = [
    ("ppo-rnn", "Instability", {"run": {"train/score_pct": DIP}}),
    ("ppo-rnn", "Collapse", {"run": {"train/score_pct": FALL}}),
    ("ppo-rnn", "Plateau", {"run": {"train/score_pct": _flat_after(50, 2.0)}}),
    (
        "pqn-rnn",
        "Entropy collapse",
        {"run": {"train/action_entropy": _at((0, 5, 0.1), base=_const(2.0))}},
    ),
    (
        "ppo-rnn",
        "Value failure",
        {"run": {"ppo/explained_variance": _at((10, 21, -0.2), base=_const(0.5))}},
    ),
    (
        "ppo-rnn",
        "Numerical failure",
        {"run": {"ppo/grad_norm": _at((40, 41, 50.0), base=_const(1.0))}},
    ),
    (
        "pqn-rnn",
        "Numerical failure",
        {"run": {"train/value_loss": _at((30, 31, np.nan), base=_const(0.2))}},
    ),
    (
        "ppo-rnn",
        "Reward hacking",
        {"run": {"train/ach_reward_share": _const(0.6)}, "cycle": True},
    ),
    (
        "count-bonus-ppo-rnn",
        "Reward hacking",
        {
            "run": {
                "train/score_pct": _flat_after(30),
                "count/r_int_mean": _ramp(0.01, 0.5),
            },
            "cycle": True,
        },
    ),
    (
        "ppo-rnn",
        "Exploration failure",
        {"run": EARLY_UNLOCK, "final": {"max_floor": 0}, "every": {"max_floor": 0}},
    ),
    ("ppo-rnn", "Survival failure", {"final": {"cause_of_death": "starvation"}}),
    (
        "ppo-rnn",
        "Skill forgetting",
        {
            "run": {
                "train/ach/collect_wood": _at(
                    (0, 50, 30.0), (50, N, 2.0), base=_const(0)
                )
            }
        },
    ),
    (
        "count-bonus-ppo-rnn",
        "Bonus distraction",
        {
            "run": {
                "count/r_int_share": _const(0.7),
                "train/score_pct": _flat_after(30),
            },
        },
    ),
    ("count-bonus-ppo-rnn", "Hoarding or key cycling", {"cycle": True}),
    (
        "count-bonus-pqn-rnn",
        "Hoarding or key cycling",
        {"config": {"granularity": "log2"}, "inventory": 0.8},
    ),
    (
        "count-bonus-ppo-rnn",
        "Early-race slowdown",
        {"run": {"first_unlock/collect_wood": {0.1: 100_000.0}}},
    ),
    (
        "count-bonus-pqn-rnn",
        "Q overestimation",
        {"run": {"pqn/q_max_mean": _const(30.0)}},
    ),
    ("sil", "Overfitting to early lucky windows", {"run": LUCKY}),
    ("sil", "Forgetting", {"run": {"train/ach/defeat_zombie": _const(15.0)}}),
    ("sil", "Imitated unsafe play", {"every": {"cause_of_death": "skeleton"}}),
    (
        "sil",
        "Value overestimation",
        {
            "run": {
                "sil/value_bias": _const(0.5),
                "ppo/explained_variance": _const(0.2),
            },
        },
    ),
    (
        "sil",
        "Dead SIL",
        {"run": {"sil/active": _at((10, 40, 0.005), base=_const(0.3))}},
    ),
    (
        "sil",
        "Insertion overflow",
        {"run": {"sil/overflow": _at((10, 30, 3.0), base=_const(0.0))}},
    ),
    (
        "smdp-skills",
        "Stuck top level",
        {
            "run": {
                "smdp/skill/collect_diamond/success_rate": _at(
                    (30, 50, 0.01), base=_const(0.3)
                ),
                "smdp/skill/collect_diamond/chosen_share": _const(0.3),
            },
        },
    ),
    (
        "smdp-skills",
        "Fallback collapse",
        {
            "run": {
                "smdp/skill/fight/chosen_share": _const(0.4),
                "smdp/skill/unlock_any/chosen_share": _const(0.3),
            },
        },
    ),
    (
        "smdp-skills",
        "Skill interference",
        {
            "run": {
                "smdp/skill/collect_wood/success_rate": _at(
                    (60, N, 0.01), base=_const(0.8)
                ),
                "smdp/skill/fight/chosen_share": _at((60, N, 0.5), base=_const(0.15)),
            },
        },
    ),
    (
        "dyna-q",
        "Timeout dominance",
        {
            "run": {
                "smdp/skill/descend/end/timeout": _at((40, 60, 0.7), base=_const(0.1))
            }
        },
    ),
    (
        "smdp-skills",
        "Deep-skill starvation",
        {"run": {"smdp/skill/collect_diamond/chosen_share": _const(0.0)}},
    ),
    (
        "smdp-skills",
        "Survival failure despite flags",
        {
            "skill_ends": lambda f: f.assign(
                end=np.where(f["reason"] == "low_stat", 250, f["end"])
            ),
        },
    ),
    (
        "dyna-q",
        "Key aliasing",
        {
            "skill_ends": lambda f: f.assign(
                key=np.where(f["skill"] == "collect_wood", 7, f["key"]),
                reason=np.where(
                    (f["skill"] == "collect_wood") & (f["eval_seed"] % 2 == 1),
                    "timeout",
                    f["reason"],
                ),
            ),
        },
    ),
    (
        "smdp-skills",
        "Blind skill test",
        {"run": {"smdp/skill/fight/test_agreement": _const(0.5)}},
    ),
    ("dyna-q", "Planned-value gap", {"run": {"smdp/q_start": 3.0 * _ramp(1.0, 10.0)}}),
    (
        "dyna-q",
        "Stale model",
        {"run": {"dyna/model_accuracy": _at((50, N, 0.5), base=_const(0.9))}},
    ),
    (
        "dyna-q",
        "Bonus-driven re-trying",
        {
            "run": {"train/score_pct": _flat_after(40)},
            "skill_ends": lambda f: f.assign(
                pair_count=np.where(f["update"] == UPDATES[-1], 5000, 10),
            ),
        },
    ),
    (
        "dyna-q",
        "No planning effect",
        {"run": {"dyna/plan_td_abs": _at((20, 50, 1e-4), base=_const(0.01))}},
    ),
]


def test_every_detector_has_a_firing_case() -> None:
    names = {d.name for ds in DETECTORS.values() for d in ds}
    assert {name for _, name, _ in FIRES} | {"Bonus starvation"} == names


@pytest.mark.parametrize(
    ("method", "name", "kw"), FIRES, ids=[f"{m}:{n}" for m, n, _ in FIRES]
)
def test_detector_fires(method: str, name: str, kw: dict[str, Any]) -> None:
    o = result(world(method, **kw), method, name)
    assert o.status == DETECTED, o
    assert o.run == f"{method}-0"
    assert o.evidence
    assert o.start is None or o.end is None or o.start <= o.end


# --- quiet near misses and thresholds ----------------------------------------------


@pytest.mark.parametrize(
    ("method", "name", "kw"),
    [
        (
            "ppo-rnn",
            "Plateau",
            {
                "run": {
                    "train/score_pct": _flat_after(50, 2.0)
                    + np.r_[np.zeros(50), np.linspace(0, 0.3, N - 50)]
                }
            },
        ),
        (
            "pqn-rnn",
            "Entropy collapse",
            {"run": {"train/action_entropy": _at((20, 25, 0.1), base=_const(2.0))}},
        ),
        (
            "ppo-rnn",
            "Value failure",
            {"run": {"ppo/explained_variance": _at((10, 13, -0.2), base=_const(0.5))}},
        ),
        ("ppo-rnn", "Reward hacking", {"run": {"train/ach_reward_share": _const(0.6)}}),
        ("ppo-rnn", "Exploration failure", {"run": EARLY_UNLOCK}),
        (
            "ppo-rnn",
            "Survival failure",
            {
                "final": {"cause_of_death": "starvation"},
                "every": {"end_type": "timeout"},
            },
        ),
        (
            "count-bonus-ppo-rnn",
            "Bonus distraction",
            {"run": {"count/r_int_share": _const(0.7)}},
        ),
        (
            "count-bonus-ppo-rnn",
            "Early-race slowdown",
            {"run": {"first_unlock/collect_wood": {0.015: 15_000.0}}},
        ),
        (
            "sil",
            "Overfitting to early lucky windows",
            {"run": LUCKY, "base": {"ppo/entropy": _const(0.5)}},
        ),
        ("sil", "Forgetting", {"run": {"train/ach/defeat_zombie": _const(25.0)}}),
        (
            "smdp-skills",
            "Fallback collapse",
            {"run": {"smdp/skill/fight/chosen_share": _const(0.4)}},
        ),
        (
            "dyna-q",
            "Bonus-driven re-trying",
            {
                "skill_ends": lambda f: f.assign(pair_count=5000),
                "run": {"train/score_pct": _flat_after(40)},
            },
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_near_miss_is_absent(method: str, name: str, kw: dict[str, Any]) -> None:
    assert result(world(method, **kw), method, name).status == ABSENT


def test_plateau_is_absolute_at_1b() -> None:
    # gain 1.5 points from 50% to the end: < 10% of the best score, >= 1 point
    score = np.r_[np.linspace(0, 20, 51), np.linspace(20, 21.5, N - 51)[0:]][:N]
    assert (
        result(
            world("ppo-rnn", run={"train/score_pct": score}), "ppo-rnn", "Plateau"
        ).status
        == DETECTED
    )
    frames = world("ppo-rnn", "1b", run={"train/score_pct": score})
    assert result(frames, "ppo-rnn", "Plateau", "1b").status == ABSENT
    flat = np.r_[np.linspace(0, 20, 51), np.linspace(20, 20.5, N - 51)]
    frames = world("ppo-rnn", "1b", run={"train/score_pct": flat})
    assert result(frames, "ppo-rnn", "Plateau", "1b").status == DETECTED


def test_bonus_starvation_is_uninformative_at_1m() -> None:
    starved = {"count/new_keys": _at((30, 50, 0.0), base=_const(5.0))}
    o = result(
        world("count-bonus-ppo-rnn", run=starved),
        "count-bonus-ppo-rnn",
        "Bonus starvation",
    )
    assert o.status == "uninformative at 1M"
    frames = world("count-bonus-ppo-rnn", "1b", run=starved)
    assert (
        result(frames, "count-bonus-ppo-rnn", "Bonus starvation", "1b").status
        == DETECTED
    )
    deep = world("count-bonus-ppo-rnn", "1b", run=starved, final={"max_floor": 2})
    assert (
        result(deep, "count-bonus-ppo-rnn", "Bonus starvation", "1b").status == ABSENT
    )
    healthy = world("count-bonus-ppo-rnn", "1b")
    assert (
        result(healthy, "count-bonus-ppo-rnn", "Bonus starvation", "1b").status
        == ABSENT
    )


def test_reward_hacking_without_eval_actions_is_unconfirmed() -> None:
    frames = world("ppo-rnn", run={"train/ach_reward_share": _const(0.6)}, inputs=())
    assert result(frames, "ppo-rnn", "Reward hacking").status == NEEDS_ACTIONS


def test_hoarding_with_log2_keys_needs_the_inventory_replay() -> None:
    frames = world(
        "count-bonus-ppo-rnn", config={"granularity": "log2"}, inputs=(ACTIONS,)
    )
    o = result(frames, "count-bonus-ppo-rnn", "Hoarding or key cycling")
    assert o.status == NEEDS_REPLAY
    assert o.status == "needs replay (evaluation Task 6)"
    frames = world("count-bonus-ppo-rnn", config={"granularity": "log2"}, inventory=0.1)
    assert (
        result(frames, "count-bonus-ppo-rnn", "Hoarding or key cycling").status
        == ABSENT
    )


@pytest.mark.parametrize(
    ("method", "name"),
    [
        ("smdp-skills", "Survival failure despite flags"),
        ("dyna-q", "Key aliasing"),
        ("dyna-q", "Bonus-driven re-trying"),
    ],
)
def test_replay_detectors_say_so_without_the_replay(method: str, name: str) -> None:
    o = result(world(method, inputs=(ACTIONS,)), method, name)
    assert o.status == NEEDS_REPLAY


def test_retrying_does_not_apply_without_the_bonus() -> None:
    frames = world("dyna-q", config={"plan_kappa": 0.0})
    assert result(frames, "dyna-q", "Bonus-driven re-trying").status == NOT_APPLICABLE


# --- not logged ---------------------------------------------------------------------


def _matches(key: str, group: tuple[str, ...]) -> bool:
    """Whether a stored key is a name, family member or pattern match of ``group``."""
    return any(
        key.startswith(g)
        if g.endswith("/")
        else fnmatch.fnmatchcase(key, g)
        if "*" in g
        else key == g
        for g in group
    )


def _not_logged_cases() -> list[tuple[str, str, tuple[str, ...]]]:
    """Every required group of every (method, detector) the healthy run logs."""
    cases = []
    for method in sorted(METHODS):
        healthy = _healthy(method)
        for d in DETECTORS[method]:
            if method == "combo" and not _in_pair(d):
                continue
            # a group the healthy run lacks is not logged already (PQN EV)
            cases += [
                (method, d.name, g)
                for g in d.REQUIRED_METRICS
                if any(_matches(k, g) for k in healthy)
            ]
    return cases


@pytest.mark.parametrize(("method", "name", "group"), _not_logged_cases())
def test_a_missing_series_is_not_logged(
    method: str, name: str, group: tuple[str, ...]
) -> None:
    drop = {k: None for k in _healthy(method) if _matches(k, group)}
    o = result(world(method, run=drop), method, name)
    assert o.status == NOT_LOGGED
    assert o.status != ABSENT


def test_pqn_value_failure_is_not_logged() -> None:
    assert result(world("pqn-rnn"), "pqn-rnn", "Value failure").status == NOT_LOGGED


def test_reused_runs_use_the_old_metric_names() -> None:
    old = {
        "ppo/entropy": None,
        "ppo/explained_variance": None,
        "ppo/grad_norm": None,
        "train/entropy": _at((0, 5, 0.1), base=_const(2.0)),
        "train/explained_var": _const(0.5),
        "train/grad_norm": _const(1.0),
    }
    frames = world("ppo-rnn", run=old)
    assert result(frames, "ppo-rnn", "Entropy collapse").status == DETECTED
    assert result(frames, "ppo-rnn", "Value failure").status == ABSENT
    assert result(frames, "ppo-rnn", "Numerical failure").status == ABSENT


def test_a_base_control_without_the_series_is_not_logged() -> None:
    # the reused PPO-RNN runs lack first_unlock/* (R78)
    no_race = {"first_unlock/collect_wood": None, "first_unlock/place_table": None}
    frames = world("count-bonus-ppo-rnn", base=no_race)
    o = result(frames, "count-bonus-ppo-rnn", "Early-race slowdown")
    assert o.status == NOT_LOGGED
    assert "base control" in o.evidence
    frames = world(
        "sil", base={"train/ach/collect_wood": None, "train/ach/defeat_zombie": None}
    )
    assert result(frames, "sil", "Forgetting").status == NOT_LOGGED


# --- the cycle check ----------------------------------------------------------------


def test_repeated_cycle_finds_the_shortest_period() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 43, 300)
    a[100:190] = np.tile([4, 7, 9, 4, 7, 9], 15)
    c = repeated_cycle(a)
    assert isinstance(c, Cycle)
    assert c.period == 3
    assert 98 <= c.start <= 100
    assert c.length >= 90


def test_repeated_cycle_needs_50_steps_and_two_actions() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 43, 300)
    short = a.copy()
    short[100:140] = np.tile([1, 2], 20)
    assert repeated_cycle(short) is None
    same = a.copy()
    same[100:200] = 6  # one repeated action (sleep, a wall) is no cycle
    assert repeated_cycle(same) is None
    padded = np.r_[np.tile([1, 2], 30), -np.ones(40, dtype=int)]
    assert repeated_cycle(padded) == Cycle(2, 0, 60)


def test_eval_actions_reads_an_eval_file(tmp_path: Path) -> None:
    d = tmp_path / "eval" / "stochastic"
    d.mkdir(parents=True)
    acts = np.array([[1, 2, 3, -1], [4, 5, -1, -1]])
    np.savez(
        d / "00244.npz",
        seeds=np.array([10000, 10001]),
        actions=acts,
        length=np.array([3, 2]),
    )
    frame = eval_actions(tmp_path, "r1", 244)
    assert list(frame.columns) == ["run_id", "update", "mode", "eval_seed", "actions"]
    assert frame["actions"].map(list).tolist() == [[1, 2, 3], [4, 5]]
    assert frame["eval_seed"].tolist() == [10000, 10001]


# --- the table ----------------------------------------------------------------------


def test_obstacles_table_renders_the_template_columns() -> None:
    obs = [
        Obstacle("Collapse", "r1", 420_000, 510_000, "drawdown 0.62 | no recovery"),
        Obstacle("Collapse", "r2", None, None, "", ABSENT),
        Obstacle("Plateau", "r1", None, None, "", ABSENT),
        Obstacle(
            "Value failure",
            "r1",
            None,
            None,
            "not logged: ppo/explained_variance",
            NOT_LOGGED,
        ),
    ]
    rows = table_rows(obs, "1m")
    assert [r["Obstacle"] for r in rows] == ["Collapse", "Value failure"]
    assert rows[0]["Runs affected"] == "1 / 2"
    assert rows[0]["Benchmark"] == "1M"
    assert rows[0]["When (env steps)"] == "420k to 510k"
    assert rows[1]["Runs affected"] == "not logged: 1"
    text = obstacles_table(rows)
    lines = text.splitlines()
    assert lines[0] == "| " + " | ".join(TABLE_COLUMNS) + " |"
    assert TABLE_COLUMNS == (
        "Obstacle",
        "Benchmark",
        "Runs affected",
        "When (env steps)",
        "Evidence",
        "Mitigation tried",
        "Outcome",
        "Replay or plot",
    )
    assert lines[1] == "|" + "---|" * len(TABLE_COLUMNS)
    assert len(lines) == 4
    assert "drawdown 0.62 \\| no recovery" in lines[2]


def test_table_of_a_detected_world() -> None:
    frames = world("ppo-rnn", run={"train/score_pct": FALL})
    text = obstacles_table(table_rows(detect_all(frames, "ppo-rnn", "1m"), "1m"))
    assert "| Collapse | 1M | 1 / 1 |" in text


def test_detect_all_runs_on_the_synthetic_export(export_root: Path) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    known = {DETECTED, ABSENT, NOT_LOGGED, NEEDS_REPLAY, NEEDS_ACTIONS, NOT_APPLICABLE}
    for method in sorted(METHODS):
        found = detect_all(frames, method, "1m")
        runs = frames["runs"]
        tuned = runs[(runs["method"] == method) & (runs["config_ref"] == "tuned")]
        assert {o.run for o in found} == set(tuned["run_id"])
        assert {o.status for o in found} <= known | {"uninformative at 1M"}


# --- review fixes: drawdown floor, base pools ------------------------------------

LOW_DIP = DIP * 0.4
"""The instability dip on a run peaking near 0.8%, below the 1-point floor."""


def test_instability_at_1m_uses_a_relative_floor() -> None:
    o = result(
        world("ppo-rnn", run={"train/score_pct": LOW_DIP}), "ppo-rnn", "Instability"
    )
    assert o.status == DETECTED
    assert "floor 0.2" in o.evidence  # 0.25 x the best smoothed score of 0.8
    low_fall = FALL * 0.4
    o = result(
        world("ppo-rnn", run={"train/score_pct": low_fall}), "ppo-rnn", "Collapse"
    )
    assert o.status == DETECTED
    assert "floor" in o.evidence


def test_instability_at_1b_keeps_the_one_point_floor() -> None:
    frames = world("ppo-rnn", "1b", run={"train/score_pct": LOW_DIP})
    assert result(frames, "ppo-rnn", "Instability", "1b").status == ABSENT
    frames = world("ppo-rnn", "1b", run={"train/score_pct": DIP})
    o = result(frames, "ppo-rnn", "Instability", "1b")
    assert o.status == DETECTED
    assert "floor 1" in o.evidence


@pytest.mark.parametrize(
    ("budget", "curve", "floor"),
    [
        ("1m", [0.2, 0.8, np.nan], 0.2),  # 0.25 x the best score
        ("1m", [0.1, 0.2], 0.1),  # never below 0.1
        ("1m", [np.nan, np.nan], 0.1),  # no score: the minimum
        ("1b", [0.2, 0.8], 1.0),  # absolute budgets keep 1 point
    ],
)
def test_drawdown_floor(budget: str, curve: list[float], floor: float) -> None:
    assert drawdown_floor(budget, np.array(curve)) == pytest.approx(floor)


def test_a_base_run_without_first_unlocks_is_left_out_of_the_median() -> None:
    # seed 1 of the PPO-RNN base is a reused run without first_unlock/* (R78)
    unlogged = {"first_unlock/collect_wood": None, "first_unlock/place_table": None}
    frames = world(
        "count-bonus-ppo-rnn",
        run={"first_unlock/collect_wood": {0.1: 100_000.0}},
        base_by={("ppo-rnn", 1): unlogged},
    )
    o = result(frames, "count-bonus-ppo-rnn", "Early-race slowdown")
    assert o.status == DETECTED, o
    assert "collect_wood x10" in o.evidence
    assert "over 1 of 2 runs; 1 not logged" in o.evidence


def test_a_control_is_compared_with_each_other_control() -> None:
    early = {"run": EARLY_UNLOCK, "final": {"max_floor": 0}, "every": {"max_floor": 0}}
    mixed = world("ppo-rnn", **early, base_episodes_by={"pqn-rnn": {"max_floor": 0}})
    assert result(mixed, "ppo-rnn", "Exploration failure").status == ABSENT
    o = result(world("ppo-rnn", **early), "ppo-rnn", "Exploration failure")
    assert o.status == DETECTED
    assert "vs ppo 1, pqn-rnn 1" in o.evidence


def test_reward_hacking_starts_where_the_aux_reward_rises() -> None:
    kw = {
        "run": {
            "train/score_pct": _flat_after(30),
            "count/r_int_mean": _ramp(0.01, 0.5),
        },
        "cycle": True,
    }
    o = result(
        world("count-bonus-ppo-rnn", **kw), "count-bonus-ppo-rnn", "Reward hacking"
    )
    assert o.start is not None
    assert 300_000 <= o.start <= 400_000  # the score is flat from 30% of the budget
    assert o.end == SCALE["1m"]


def test_q_overestimation_compares_magnitudes() -> None:
    frames = world("count-bonus-pqn-rnn", run={"pqn/q_max_mean": _const(-30.0)})
    o = result(frames, "count-bonus-pqn-rnn", "Q overestimation")
    assert o.status == DETECTED


def test_a_missing_deep_skill_is_not_logged() -> None:
    drop = {"smdp/skill/descend/chosen_share": None}
    o = result(world("smdp-skills", run=drop), "smdp-skills", "Deep-skill starvation")
    assert o.status == NOT_LOGGED
    assert "descend" in o.evidence


def test_importing_the_module_loads_no_mlflow() -> None:
    code = (
        "import sys, wut_murl_craftax.analysis.obstacles; "
        "print('mlflow' in sys.modules, 'jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]
