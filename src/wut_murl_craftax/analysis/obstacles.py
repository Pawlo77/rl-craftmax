"""Obstacle detectors of the training analysis, their registry and the obstacles table.

A detector reads one run's curves (stored names through `ALIASES`), its eval episodes,
its base control (seed medians) where the rule compares, and optional per-step inputs:
eval actions (`ACTIONS`) and the replay frames of evaluation Task 6 (`SKILL_ENDS`,
`INVENTORY`). A missing series reads "not logged" and a missing replay "needs replay
(evaluation Task 6)", never absent (R78). Thresholds are module constants; those that
assume 1B have a 1M variant or read "uninformative at 1M" (R79). `detect_all` keeps
tuned and ablation runs apart (R77).

@see [[2026-10-20-craftax-training-analysis]]
"""

import fnmatch
import functools
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..evaluation.stability import (
    COLLAPSE_DEPTH,
    MIN_MAX,
    collapsed,
    drawdowns,
    drops,
    smoothed,
)
from . import load
from .load import Frames, final_checkpoint, pair_of, select

DETECTED = "detected"
ABSENT = "absent"
NOT_LOGGED = "not logged"
NEEDS_REPLAY = "needs replay (evaluation Task 6)"
NEEDS_ACTIONS = "needs eval actions"
NOT_APPLICABLE = "not applicable"
"""Obstacle statuses; "uninformative at <budget>" is built from the budget."""
ACTIONS = "eval_actions"
"""Optional frame: one row per eval episode, ``run_id update mode eval_seed actions``
(an int array; `eval_actions` reads it from an eval file)."""
SKILL_ENDS = "skill_ends"
"""Optional replay frame (evaluation Task 6): one row per finished skill in an eval
episode; columns and types in `SKILL_ENDS_COLUMNS` and the module note."""
SKILL_ENDS_COLUMNS = (
    "run_id", "update", "env_steps", "mode", "eval_seed", "start", "end", "skill",
    "key", "reason",
)  # fmt: skip
"""Columns of `SKILL_ENDS`; Dyna-Q rows add ``pair_count`` from the checkpoint's
``eval/dyna_model_graph.parquet``."""
INVENTORY = "eval_inventory"
"""Optional replay frame (evaluation Task 6): one row per eval episode; columns in
`INVENTORY_COLUMNS`."""
INVENTORY_COLUMNS = ("run_id", "update", "mode", "eval_seed", "steps", "capped_steps")
"""Columns of `INVENTORY`: episode steps, and steps with a material at its
`progress_key.CAPS` cap."""
REPLAY = (SKILL_ENDS, INVENTORY)
INPUTS = (ACTIONS, *REPLAY)
"""Optional frames `detect_all` passes on to the detectors."""
BUDGET_STEPS = {"1m": 1_000_000, "1b": 1_000_000_000}
"""Env steps of a ``budget`` tag."""
EVAL_MODE = "stochastic"
"""Eval mode the episode rules read (final stochastic eval, as the thresholds)."""

ALIASES: dict[str, tuple[str, ...]] = {
    "policy entropy": ("ppo/entropy", "train/entropy"),
    "action entropy": ("train/action_entropy",),
    "explained variance": ("ppo/explained_variance", "train/explained_var"),
    "gradient norm": ("ppo/grad_norm", "pqn/grad_norm", "train/grad_norm"),
    "approx KL": ("ppo/approx_kl", "train/approx_kl"),
    "clip fraction": ("ppo/clip_frac", "train/clip_frac"),
    "Q values": ("pqn/q_max_mean",),
    "sil/sample_age": ("sil/sample_age_max", "sil/sample_age_mean"),
    "sil/insert_overflow": ("sil/overflow",),
    "sil/active": ("sil/active",),
    "death shares": ("train/death/",),
    "funnel": ("train/funnel/",),
    "first unlock": ("first_unlock/",),
}
"""Plan metric name -> stored names in order (contracts.md section 5); a reader takes
the first one the run logged; a name ending in ``/`` is a family (any member)."""
SCORE = ("train/score_pct",)
ENTROPY = ALIASES["policy entropy"] + ALIASES["action entropy"]
EV = ALIASES["explained variance"]
GRAD = ALIASES["gradient norm"]
RETURN = ("train/return",)
AUX_REWARD = ("count/r_int_mean", "dyna/bonus_mean")
"""Auxiliary reward per step of the methods that add one."""
BASIC_ACHIEVEMENTS: tuple[str, ...] = (
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword", "make_arrow", "make_torch", "place_torch",
)  # fmt: skip
"""Craftax achievements 0-24 (reward 1, [[Craftax achievements]]); spelled out because
`milestones.achievement_names` imports Craftax."""
NEEDS_CAUSES = ("starvation", "dehydration", "exhaustion")
"""`episode_metrics.NEEDS` (its module imports JAX): the needs-drain causes."""

ABSOLUTE_BUDGETS = ("1b",)
"""Budgets whose scores are large enough for the plan's absolute point thresholds."""
DRAWDOWN_FLOOR_MIN, DRAWDOWN_FLOOR_SHARE = 0.1, 0.25
"""Drawdown floor outside `ABSOLUTE_BUDGETS` (R92): ``max(0.1, 0.25 x`` the run's best
smoothed score); at those budgets it is `stability.MIN_MAX` (1 point)."""
PLATEAU_AT = 0.5
PLATEAU_POINTS = 1.0
PLATEAU_SHARE = 0.1
"""Plateau: score gain from ``PLATEAU_AT`` of the budget to the end below
``PLATEAU_POINTS`` (absolute budgets) or ``PLATEAU_SHARE`` of the best smoothed
score."""
ENTROPY_MIN, ENTROPY_BEFORE = 0.2, 0.1
"""Entropy collapse: below ``ENTROPY_MIN`` nats before ``ENTROPY_BEFORE`` of budget."""
EV_FRAC = 0.05
"""Value failure: explained variance < 0 for this share of the budget."""
GRAD_SPIKE, GRAD_WARMUP = 10.0, 5
"""Numerical failure: gradient norm above ``GRAD_SPIKE`` x the running median of the
earlier points, from the ``GRAD_WARMUP``-th point on."""
AUX_SHARE, AUX_RISE_FRAC = 0.2, 0.1
"""Reward hacking: non-achievement share of return above ``AUX_SHARE`` (second half),
or auxiliary reward rising while the score is flat for ``AUX_RISE_FRAC``."""
CYCLE_STEPS, CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD = 50, 2, 25
"""Repeated cycle: the same action period over ``CYCLE_STEPS`` consecutive steps; one
repeated action (sleeping, a wall) is no cycle."""
CYCLE_SHARE = 0.1
"""Share of final eval episodes with a cycle that confirms a cycle rule."""
EXPLORE_LATE = 0.5
"""Exploration failure: no new first unlock after this share of the budget."""
NEEDS_SHARE = 0.5
"""Survival failure: share of final-checkpoint deaths by needs drain."""
FORGET_HIGH, FORGET_LOW = 20.0, 5.0
"""Skill forgetting: ``train/ach/<name>`` (% of episodes) from >= high to < low."""
FLAT_WINDOW, FLAT_GAIN = 0.1, 0.05
"""Flat score: gain over the trailing ``FLAT_WINDOW`` of the budget below ``FLAT_GAIN``
of the run's best smoothed score."""
R_INT_SHARE, R_INT_AFTER, DISTRACTION_FRAC = 0.5, 0.2, 0.05
"""Bonus distraction: ``count/r_int_share`` above 0.5 after 20% of the budget while
the score is flat, for ``DISTRACTION_FRAC`` of the budget."""
CAPPED_SHARE = 0.5
"""Hoarding: share of eval steps with a material at its recipe cap (log2 keys)."""
STARVATION_FRAC, STARVATION_FLOOR, STARVATION_BUDGETS = 0.1, 2, ("1b",)
"""Bonus starvation: no new key for ``STARVATION_FRAC`` while no eval episode reaches
``STARVATION_FLOOR``; informative only at these budgets (1M never reaches floor 2)."""
RACE_RATIO = 2.0
"""Early-race slowdown: basic first unlocks later than the base control's by this."""
Q_RATIO = 2.0
"""Q overestimation: ``pqn/q_max_mean`` above this times the largest return."""
AGE_SHARE, SIL_ACTIVE = 0.5, 0.5
"""SIL overfitting: oldest sample older than ``AGE_SHARE`` of the updates so far while
``sil/active`` (scalar, R78) is above ``SIL_ACTIVE``."""
BASE_GAP, FINAL_FRAC = 10.0, 0.1
"""Points a rate trails the base control's by (means of the last ``FINAL_FRAC``)."""
DEAD_SIL, DEAD_FRAC = 0.01, 0.2
OVERFLOW_SHARE = 0.1
STUCK_SUCCESS, STUCK_SHARE, STUCK_FRAC = 0.05, 0.2, 0.1
FALLBACK_SKILLS, FALLBACK_SHARE = ("fight", "unlock_any"), 0.6
SKILL_HIGH, SKILL_LOW = 0.2, 0.05
"""Skill interference: ``success_rate`` (a fraction) from >= high to < low."""
TIMEOUT_SHARE, TIMEOUT_FRAC = 0.5, 0.1
DEEP_SKILLS, DEEP_MIN = ("collect_diamond", "descend", "make_iron_pickaxe"), 100
"""Deep-skill starvation: chosen fewer than ``DEEP_MIN`` times in the run, an absolute
count at every budget (R92); the curves cannot tell ``descend`` from floor 1 apart,
so every descend counts."""
FLAG_DEATH_STEPS, FLAG_DEATH_SHARE = 100, 0.3
ALIAS_SHARE = 0.4
AGREEMENT_MIN = 0.9
GAP_RATIO, GAP_FRAC = 2.0, 0.1
STALE_DROP, STALE_WINDOW = 0.2, 0.1
RETRY_COUNT, RETRY_SUCCESS, RETRY_RISE = 1000, 0.9, 0.1
"""Bonus-driven re-trying: decisions on pairs with model count > 1000 and success
> 0.9, their share rising by ``RETRY_RISE`` from mid budget to the end, the score
flat. The model keeps no success flag, so success is the eval share of the pair's
ends at that checkpoint."""
PLAN_TD_MIN, PLAN_TD_FRAC = 1e-3, 0.2
TABLE_COLUMNS = (
    "Obstacle",
    "Benchmark",
    "Runs affected",
    "When (env steps)",
    "Evidence",
    "Mitigation tried",
    "Outcome",
    "Replay or plot",
)
"""Columns of the obstacles table template."""

type Group = tuple[str, ...]
"""Aliases of one input, first logged wins; ``prefix/`` is a family and a name with
``*`` a pattern (any match counts as logged)."""


@dataclass(frozen=True, slots=True)
class Obstacle:
    """One detector's verdict on one run; ``start``/``end`` are env steps."""

    name: str
    run: str
    start: int | None
    end: int | None
    evidence: str
    status: str = DETECTED


class Hit(NamedTuple):
    """What a rule found: a detection, or a status the rule itself decided."""

    start: int | None
    end: int | None
    evidence: str
    status: str = DETECTED


class Cycle(NamedTuple):
    """A repeated action period: ``length`` steps from step ``start``."""

    period: int
    start: int
    length: int


@dataclass(frozen=True)
class RunData:
    """One run (or a base control's pooled runs) as the rules read it."""

    run: pd.Series
    curves: dict[str, pd.Series]
    """Stored metric -> values indexed by env steps (seed medians for a pool)."""
    episodes: pd.DataFrame
    budget_steps: int
    bases: tuple["RunData", ...] = ()
    """Base control pools, one per method (a control has the other controls)."""
    members: tuple["RunData", ...] = ()
    """The runs of a pool, for per-run statistics; empty for one run."""
    inputs: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def logged(self, group: Group) -> bool:
        """Whether the run logged any name of ``group``."""
        return any(
            any(k.startswith(g) for k in self.curves)
            if g.endswith("/")
            else bool(fnmatch.filter(self.curves, g))
            if "*" in g
            else g in self.curves
            for g in group
        )

    def pick(self, group: Group) -> tuple[str, pd.Series]:
        """The first logged name of ``group`` and its series."""
        name = next(g for g in group if g in self.curves)
        return name, self.curves[name]

    def series(self, group: Group) -> pd.Series:
        """The series of the first logged name of ``group``."""
        return self.pick(group)[1]

    def family(self, prefix: str, suffix: str = "") -> dict[str, pd.Series]:
        """``<prefix><name><suffix>`` series keyed by ``name``."""
        return {
            k[len(prefix) : len(k) - len(suffix)]: v
            for k, v in self.curves.items()
            if k.startswith(prefix) and k.endswith(suffix)
        }

    def final(self) -> pd.DataFrame:
        """Eval episodes of the last checkpoint in `EVAL_MODE`."""
        e = self.episodes[self.episodes["mode"] == EVAL_MODE]
        return final_checkpoint(e) if len(e) else e

    def config(self) -> dict[str, Any]:
        """The run's ``config_json``."""
        out: dict[str, Any] = json.loads(self.run["config_json"])
        return out


type Rule = Callable[[RunData], Hit | None]


@dataclass(frozen=True)
class Detector:
    """A named rule with the inputs it needs; calling it judges one run."""

    name: str
    rule: Rule
    REQUIRED_METRICS: tuple[Group, ...] = ()
    """Curves the run must log, else "not logged"."""
    BASE_METRICS: tuple[Group, ...] | None = None
    """Curves the base control must log; None when the rule needs no base control."""
    inputs: tuple[str, ...] = ()
    """Optional frames (`INPUTS`) the rule needs."""
    budgets: tuple[str, ...] | None = None
    """Budgets where the rule is informative; None for all."""
    methods: tuple[str, ...] = ()
    """Methods owning a method detector (combo runs keep those of their pair)."""

    def __call__(self, d: RunData) -> Obstacle:
        """The obstacle verdict for ``d``, checking its inputs first."""
        run = str(d.run["run_id"])
        if gaps := [g[0] for g in self.REQUIRED_METRICS if not d.logged(g)]:
            return Obstacle(
                self.name, run, None, None, f"not logged: {', '.join(gaps)}", NOT_LOGGED
            )
        if self.BASE_METRICS is not None:
            if not d.bases:
                return Obstacle(
                    self.name, run, None, None, "no base control runs", NOT_LOGGED
                )
            if gaps := [
                f"{b.run['method']} lacks {g[0]}"
                for b in d.bases
                for g in self.BASE_METRICS
                if not b.logged(g)
            ]:
                text = f"base control {'; '.join(gaps)}"
                return Obstacle(self.name, run, None, None, text, NOT_LOGGED)
        if gaps := [i for i in self.inputs if i not in d.inputs]:
            status = NEEDS_REPLAY if set(gaps) & set(REPLAY) else NEEDS_ACTIONS
            return Obstacle(self.name, run, None, None, f"no {', '.join(gaps)}", status)
        budget = str(d.run["budget"])
        if self.budgets is not None and budget not in self.budgets:
            text = f"rule assumes {'/'.join(b.upper() for b in self.budgets)} scale"
            return Obstacle(
                self.name, run, None, None, text, f"uninformative at {budget.upper()}"
            )
        hit = self.rule(d)
        if hit is None:
            return Obstacle(self.name, run, None, None, "", ABSENT)
        return Obstacle(self.name, run, *hit)


# --- helpers --------------------------------------------------------------------


def _fmt(steps: float) -> str:
    """Env steps as ``420k``, ``1.5M`` or ``2B``."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(steps) >= size:
            return f"{steps / size:.3g}{unit}"
    return f"{steps:.0f}"


def _at(s: pd.Series, steps: np.ndarray) -> np.ndarray:
    """``s`` linearly interpolated at ``steps`` over its finite points (NaN if none)."""
    ok = np.isfinite(s.to_numpy(dtype=float))
    if not ok.any():
        return np.full(len(steps), np.nan)
    out: np.ndarray = np.interp(steps, s.index.to_numpy()[ok], s.to_numpy(float)[ok])
    return out


def _stretches(
    s: pd.Series, mask: np.ndarray, min_steps: float
) -> list[tuple[int, int]]:
    """Maximal runs of True points covering >= ``min_steps`` env steps, (first, last).

    A point covers the steps until the next point; the last one the median spacing.
    """
    steps = s.index.to_numpy()
    gaps = np.diff(steps)
    width = np.append(gaps, np.median(gaps) if gaps.size else 1)
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return [
        (int(steps[a]), int(steps[b - 1]))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if width[a:b].sum() >= min_steps - 1e-9
    ]


def _first(found: list[tuple[int, int]], what: str) -> Hit | None:
    """A hit spanning every stretch in ``found``; None if there is none."""
    if not found:
        return None
    n = f" ({len(found)} stretches)" if len(found) > 1 else ""
    return Hit(found[0][0], found[-1][1], f"{what} from {_fmt(found[0][0])}{n}")


def _score(d: RunData) -> pd.Series:
    """The smoothed per-update score (`stability.smoothed`), indexed by env steps."""
    s = d.series(SCORE)
    return pd.Series(smoothed(s.to_numpy(dtype=float)), index=s.index)


def _flat(d: RunData, steps: np.ndarray) -> np.ndarray:
    """Where the score gained less than `FLAT_GAIN` of its best over `FLAT_WINDOW`."""
    c = _score(d)
    window = FLAT_WINDOW * d.budget_steps
    best = np.nanmax(c.to_numpy()) if np.isfinite(c.to_numpy()).any() else 0.0
    gain = _at(c, steps) - _at(c, steps - window)
    flat: np.ndarray = (steps >= window) & (gain < FLAT_GAIN * max(best, 0.0))
    return flat


def _forgotten(
    rates: dict[str, pd.Series], high: float, low: float
) -> list[tuple[str, int, int]]:
    """``(name, step >= high, later step < low)`` of each rate that fell, smoothed."""
    out = []
    for name, s in rates.items():
        v = smoothed(s.to_numpy(dtype=float))
        top = np.flatnonzero(v >= high)
        if top.size and (fell := np.flatnonzero(v[top[0] :] < low)).size:
            out.append((name, int(s.index[top[0]]), int(s.index[top[0] + fell[0]])))
    return out


def _late_mean(s: pd.Series, budget: int, frac: float) -> float:
    """Mean of the points in the last ``frac`` of the budget."""
    return float(s[s.index >= (1 - frac) * budget].mean())


def _first_unlocks(d: RunData) -> dict[str, float]:
    """Env step of each basic achievement's first unlock in the run."""
    fu = d.family("first_unlock/")
    return {n: float(s.index.min()) for n, s in fu.items() if n in BASIC_ACHIEVEMENTS}


def _base_unlocks(base: RunData) -> tuple[dict[str, float], int, int]:
    """Median over the base runs of each basic first unlock, the runs used, all runs.

    A run that never logged ``first_unlock/*`` is not logged and left out (R78); a
    logged run that never unlocked a name counts as inf for it.
    """
    members = base.members or (base,)
    runs = [_first_unlocks(m) for m in members if m.logged(("first_unlock/",))]
    names = set().union(*runs)
    medians = {n: float(np.median([r.get(n, np.inf) for r in runs])) for n in names}
    return medians, len(runs), len(members)


def _max_floor(d: RunData) -> float:
    """Deepest floor of any eval episode; for a pool, the median over its runs."""
    return float(np.median([m.episodes["max_floor"].max() for m in d.members or (d,)]))


def _base(d: RunData) -> RunData:
    """The one base control pool of a method run, checked by `Detector` first."""
    if len(d.bases) != 1:
        msg = f"run {d.run['run_id']} has {len(d.bases)} base pools, not one"
        raise ValueError(msg)
    return d.bases[0]


def _final_step(d: RunData) -> int | None:
    """Env steps of the run's final eval checkpoint."""
    fin = d.final()
    return int(fin["env_steps"].max()) if len(fin) else None


def _cycles(d: RunData) -> tuple[list[Cycle], int]:
    """Cycles found in the final-checkpoint eval actions, and the episode count."""
    a = d.inputs[ACTIONS]
    a = a[a["mode"] == EVAL_MODE]
    last = a[a["update"] == a["update"].max()]
    found = [c for c in map(repeated_cycle, last["actions"]) if c is not None]
    return found, len(last)


def _cycle_text(found: list[Cycle], n: int) -> str:
    return (
        f"action cycle (period {min(c.period for c in found)}, up to "
        f"{max(c.length for c in found)} steps) in {len(found)} / {n} final episodes"
    )


def repeated_cycle(
    actions: np.ndarray | Sequence[int], min_steps: int = CYCLE_STEPS
) -> Cycle | None:
    """The shortest action period repeated over >= ``min_steps`` consecutive steps.

    The lag-``p`` autocorrelation of the action sequence as exact matches
    ``a[t] == a[t + p]``: a run of ``r`` matches spans ``r + p`` steps. Periods from
    `CYCLE_MIN_PERIOD` to `CYCLE_MAX_PERIOD`; a stretch of one repeated action does not
    count. Padding (negative actions) is dropped.
    """
    a = np.asarray(actions)
    a = a[a >= 0]
    for p in range(CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD + 1):
        if len(a) < max(min_steps, 2 * p):
            break
        edges = np.flatnonzero(np.diff(np.r_[False, a[:-p] == a[p:], False]))
        for s, e in zip(edges[::2], edges[1::2], strict=True):
            span = int(e - s + p)
            if span >= min_steps and np.unique(a[s : s + span]).size > 1:
                return Cycle(p, int(s), span)
    return None


def eval_actions(
    run_dir: Path, run_id: str, update: int, mode: str = EVAL_MODE
) -> pd.DataFrame:
    """The `ACTIONS` rows of one eval file, unpadded.

    ``run_dir`` is the run's artifacts directory
    (``mlruns/<experiment>/<run_id>/artifacts``); the file is
    ``<run_dir>/eval/<mode>/<update:05d>.npz``.
    """
    with np.load(Path(run_dir) / "eval" / mode / f"{update:05d}.npz") as z:
        seeds, actions, length = z["seeds"], z["actions"], z["length"]
    return pd.DataFrame({
        "run_id": run_id,
        "update": update,
        "mode": mode,
        "eval_seed": seeds,
        "actions": [row[:n] for row, n in zip(actions, length, strict=True)],
    })


# --- shared detectors -------------------------------------------------------------


def drawdown_floor(budget: str, c: np.ndarray) -> float:
    """Drawdown floor of a smoothed score curve (R92), shared with comparison T3.

    1 point at `ABSOLUTE_BUDGETS`, else ``max(0.1, 0.25 x`` the curve's best score).
    """
    if budget in ABSOLUTE_BUDGETS:
        return MIN_MAX
    best = float(np.nanmax(c)) if np.isfinite(c).any() else 0.0
    return max(DRAWDOWN_FLOOR_MIN, DRAWDOWN_FLOOR_SHARE * best)


def _floor(d: RunData, c: np.ndarray) -> float:
    return drawdown_floor(str(d.run["budget"]), c)


def _instability(d: RunData) -> Hit | None:
    s = d.series(SCORE)
    c = smoothed(s.to_numpy(dtype=float))
    floor = _floor(d, c)
    if not (found := drops(c, floor=floor)):
        return None
    depth = np.nanmax(drawdowns(c, floor))
    steps = s.index.to_numpy()
    a, b = found[0]
    text = f"{len(found)} drops, deepest drawdown {depth:.2f} (floor {floor:.2g})"
    return Hit(int(steps[a]), int(steps[b - 1]), text)


def _collapse(d: RunData) -> Hit | None:
    s = d.series(SCORE)
    c = smoothed(s.to_numpy(dtype=float))
    floor = _floor(d, c)
    if not collapsed(c, floor):
        return None
    dd = drawdowns(c, floor)
    start = int(s.index[np.flatnonzero(dd >= COLLAPSE_DEPTH)[0]])
    text = (
        f"drawdown {np.nanmax(dd):.2f}, final {c[-1]:.2f} of best "
        f"{np.nanmax(c):.2f} (floor {floor:.2g})"
    )
    return Hit(start, int(s.index[-1]), text)


def _plateau(d: RunData) -> Hit | None:
    c = _score(d)
    ok = c[np.isfinite(c.to_numpy())]
    if ok.empty:
        return None
    mid_step = PLATEAU_AT * d.budget_steps
    gain = float(ok.iloc[-1] - _at(ok, np.array([mid_step]))[0])
    if str(d.run["budget"]) in ABSOLUTE_BUDGETS:
        limit = PLATEAU_POINTS
    else:
        limit = PLATEAU_SHARE * float(ok.max())
    if gain >= limit:
        return None
    text = f"score gain {gain:.2f} from {_fmt(mid_step)} to the end (< {limit:.2f})"
    return Hit(int(mid_step), int(ok.index[-1]), text)


def _entropy_collapse(d: RunData) -> Hit | None:
    name, s = d.pick(ENTROPY)
    early = s[s.index <= ENTROPY_BEFORE * d.budget_steps]
    low = early[early < ENTROPY_MIN]
    if low.empty:
        return None
    text = f"{name} {low.min():.3f} nats before {_fmt(ENTROPY_BEFORE * d.budget_steps)}"
    return Hit(int(low.index[0]), int(low.index[-1]), text)


def _value_failure(d: RunData) -> Hit | None:
    name, s = d.pick(EV)
    found = _stretches(s, s.to_numpy() < 0, EV_FRAC * d.budget_steps)
    return _first(found, f"{name} < 0 for >= {EV_FRAC:.0%} of the budget")


def _numerical_failure(d: RunData) -> Hit | None:
    losses = {
        k: v for k, v in d.curves.items() if k.rsplit("/", 1)[-1].endswith("loss")
    }
    name, gn = d.pick(GRAD)
    bad = [
        (int(v.index[~np.isfinite(v.to_numpy())][0]), f"{k} not finite")
        for k, v in losses.items()
        if not np.isfinite(v.to_numpy()).all()
    ]
    g = gn.to_numpy(dtype=float)
    median = gn.expanding().median().shift(1).to_numpy()
    spike = (~np.isfinite(g) | (g > GRAD_SPIKE * median)) & (
        np.arange(len(g)) >= GRAD_WARMUP
    )
    if spike.any():
        i = int(np.flatnonzero(spike)[0])
        bad.append((
            int(gn.index[i]),
            f"{name} {g[i]:.3g} > {GRAD_SPIKE:g} x running median {median[i]:.3g}",
        ))
    if not bad:
        return None
    bad.sort()
    return Hit(bad[0][0], bad[-1][0], "; ".join(t for _, t in bad))


def _aux_suspicion(d: RunData) -> tuple[str, int] | None:
    """Why reward hacking is suspected and from which env step, before the cycles."""
    share = d.series(("train/ach_reward_share",))
    other = 1 - _late_mean(share, d.budget_steps, 0.5)
    if other > AUX_SHARE:
        why = f"non-achievement share of return {other:.0%} in the second half"
        return why, int(0.5 * d.budget_steps)
    for name in AUX_REWARD:
        if name in d.curves:
            aux = d.curves[name]
            steps = aux.index.to_numpy()
            window = FLAT_WINDOW * d.budget_steps
            rising = _at(aux, steps) > _at(aux, steps - window)
            found = _stretches(
                aux, rising & _flat(d, steps), AUX_RISE_FRAC * d.budget_steps
            )
            if found:
                return f"{name} rising while the score is flat", found[0][0]
    return None


def _reward_hacking(d: RunData) -> Hit | None:
    if (suspect := _aux_suspicion(d)) is None:
        return None
    why, start = suspect
    text = f"{why} from {_fmt(start)}"
    if ACTIONS not in d.inputs:
        return Hit(
            start, None, f"{text}; unconfirmed without eval actions", NEEDS_ACTIONS
        )
    found, n = _cycles(d)
    if len(found) < CYCLE_SHARE * n or not found:
        return None
    return Hit(start, _final_step(d), f"{text}; {_cycle_text(found, n)}")


def _exploration_failure(d: RunData) -> Hit | None:
    last = max(float(s.index.max()) for s in d.family("first_unlock/").values())
    floors = _max_floor(d)
    # a control is compared with each other control on its own, never a mixed pool
    bases = {str(b.run["method"]): _max_floor(b) for b in d.bases}
    if last >= EXPLORE_LATE * d.budget_steps or floors >= min(bases.values()):
        return None
    vs = ", ".join(f"{m} {f:g}" for m, f in bases.items())
    text = f"last first unlock {_fmt(last)}; deepest floor {floors:g} vs {vs}"
    return Hit(int(last), d.budget_steps, text)


def _survival_failure(d: RunData) -> Hit | None:
    fin = d.final()
    deaths = fin[fin["end_type"] == "death"]
    if deaths.empty:
        return None
    share = float(deaths["cause_of_death"].isin(NEEDS_CAUSES).mean())
    if share <= NEEDS_SHARE:
        return None
    step = int(fin["env_steps"].max())
    return Hit(
        step,
        step,
        f"{share:.0%} of {len(deaths)} final-checkpoint deaths by needs drain",
    )


def _skill_forgetting(d: RunData) -> Hit | None:
    fell = _forgotten(d.family("train/ach/"), FORGET_HIGH, FORGET_LOW)
    if not fell:
        return None
    names = ", ".join(f"{n} ({_fmt(a)} to {_fmt(b)})" for n, a, b in fell)
    return Hit(
        min(a for _, a, _ in fell), max(b for *_, b in fell), f">= 20% to < 5%: {names}"
    )


def shared_detectors() -> list[Detector]:
    """The ten detectors every method and control gets."""
    return [
        Detector("Instability", _instability, (SCORE,)),
        Detector("Collapse", _collapse, (SCORE,)),
        Detector("Plateau", _plateau, (SCORE,)),
        Detector("Entropy collapse", _entropy_collapse, (ENTROPY,)),
        Detector("Value failure", _value_failure, (EV,)),
        Detector("Numerical failure", _numerical_failure, (GRAD,)),
        Detector(
            "Reward hacking", _reward_hacking, (("train/ach_reward_share",), SCORE)
        ),
        Detector(
            "Exploration failure", _exploration_failure, (("first_unlock/",),), ()
        ),
        Detector("Survival failure", _survival_failure),
        Detector("Skill forgetting", _skill_forgetting, (("train/ach/",),)),
    ]


# --- count-bonus ------------------------------------------------------------------

COUNT_METHODS = ("count-bonus-ppo-rnn", "count-bonus-pqn-rnn")


def _bonus_distraction(d: RunData) -> Hit | None:
    s = d.series(("count/r_int_share",))
    steps = s.index.to_numpy()
    mask = (
        (s.to_numpy() > R_INT_SHARE)
        & (steps >= R_INT_AFTER * d.budget_steps)
        & _flat(d, steps)
    )
    found = _stretches(s, mask, DISTRACTION_FRAC * d.budget_steps)
    return _first(found, f"count/r_int_share > {R_INT_SHARE} with a flat score")


def _hoarding(d: RunData) -> Hit | None:
    found, n = _cycles(d)
    parts = (
        [f"key cycling: {_cycle_text(found, n)}"]
        if found and len(found) >= CYCLE_SHARE * n
        else []
    )
    # hoarding only pays under log2 keys: recipe keys stop counting at the caps
    if d.config().get("granularity") == "log2":
        if INVENTORY not in d.inputs:
            if not parts:
                return Hit(None, None, f"hoarding: no {INVENTORY}", NEEDS_REPLAY)
            parts.append(f"hoarding unchecked: no {INVENTORY}")
        elif (share := _capped_share(d)) > CAPPED_SHARE:
            parts.append(
                f"hoarding: a material at its recipe cap in {share:.0%} of steps"
            )
    if not parts:
        return None
    end = _final_step(d)
    return Hit(end, end, "; ".join(parts))


def _capped_share(d: RunData) -> float:
    """Share of final eval steps with a material at its recipe cap (`INVENTORY`)."""
    inv = d.inputs[INVENTORY]
    inv = inv[(inv["mode"] == EVAL_MODE) & (inv["update"] == inv["update"].max())]
    return float(inv["capped_steps"].sum() / max(inv["steps"].sum(), 1))


def _bonus_starvation(d: RunData) -> Hit | None:
    s = d.series(("count/new_keys",))
    found = _stretches(s, s.to_numpy() == 0, STARVATION_FRAC * d.budget_steps)
    floors = d.episodes["max_floor"].max()
    if not found or floors >= STARVATION_FLOOR:
        return None
    return _first(
        found, f"count/new_keys = 0 for >= 10% of the budget, deepest floor {floors}"
    )


def _race_slowdown(d: RunData) -> Hit | None:
    base_run = _base(d)
    run, (base, used, total) = _first_unlocks(d), _base_unlocks(base_run)
    slow = {
        n: run.get(n, math.inf) / b
        for n, b in base.items()
        if b > 0
        and RACE_RATIO * b < d.budget_steps
        and run.get(n, math.inf) > RACE_RATIO * b
    }
    if not slow:
        return None
    text = ", ".join(
        f"{n} x{r:.1f}" for n, r in sorted(slow.items(), key=lambda x: -x[1])
    )
    starts = [run[n] for n in slow if n in run] or [d.budget_steps]
    return Hit(
        int(min(starts)),
        int(max(starts)),
        f"first unlocks later than the base control's: {text} "
        f"(base median over {used} of {total} runs; {total - used} not logged)",
    )


def _q_overestimation(d: RunData) -> Hit | None:
    q = d.series(ALIASES["Q values"]).abs()
    largest = float(d.episodes["return"].abs().max())
    if RETURN[0] in d.curves:
        largest = max(largest, float(d.curves[RETURN[0]].abs().max()))
    over = q[q > Q_RATIO * largest]
    if over.empty:
        return None
    text = (
        f"|pqn/q_max_mean| up to {q.max():.3g} > {Q_RATIO:g} x |return| {largest:.3g}"
    )
    return Hit(int(over.index[0]), int(over.index[-1]), text)


def count_bonus_detectors() -> list[Detector]:
    """Count-bonus detectors (both variants), and Q overestimation (the PQN-RNN one)."""
    return [
        Detector(
            "Bonus distraction",
            _bonus_distraction,
            (("count/r_int_share",), SCORE),
            methods=COUNT_METHODS,
        ),
        Detector(
            "Hoarding or key cycling",
            _hoarding,
            inputs=(ACTIONS,),
            methods=COUNT_METHODS,
        ),
        Detector(
            "Bonus starvation",
            _bonus_starvation,
            (("count/new_keys",),),
            budgets=STARVATION_BUDGETS,
            methods=COUNT_METHODS,
        ),
        Detector(
            "Early-race slowdown",
            _race_slowdown,
            (("first_unlock/",),),
            (("first_unlock/",),),
            methods=COUNT_METHODS,
        ),
        Detector(
            "Q overestimation",
            _q_overestimation,
            (ALIASES["Q values"],),
            methods=("count-bonus-pqn-rnn",),
        ),
    ]


# --- sil --------------------------------------------------------------------------


def _steps_per_update(d: RunData) -> float:
    """Env steps per update, from the eval checkpoints (``env_steps / update``)."""
    e = d.episodes[d.episodes["update"] > 0]
    return float((e["env_steps"] / e["update"]).median()) if len(e) else math.nan


def _sil_overfitting(d: RunData) -> Hit | None:
    base_run = _base(d)
    name, age = d.pick(ALIASES["sil/sample_age"])
    steps = age.index.to_numpy()
    updates = steps / _steps_per_update(d)
    active = _at(d.series(("sil/active",)), steps)
    below = _at(d.series(ENTROPY), steps) < _at(base_run.series(ENTROPY), steps)
    mask = (age.to_numpy() > AGE_SHARE * updates) & (active > SIL_ACTIVE) & below
    found = _stretches(age, mask, 0)
    return _first(
        found,
        f"{name} > {AGE_SHARE:.0%} of the updates, sil/active > {SIL_ACTIVE}, "
        "entropy below the base control's",
    )


def _sil_forgetting(d: RunData) -> Hit | None:
    base_run = _base(d)
    run, base = d.family("train/ach/"), base_run.family("train/ach/")
    gaps = {
        n: _late_mean(base[n], d.budget_steps, FINAL_FRAC)
        - _late_mean(run[n], d.budget_steps, FINAL_FRAC)
        for n in BASIC_ACHIEVEMENTS
        if n in run and n in base
    }
    low = {n: g for n, g in gaps.items() if g > BASE_GAP}
    if not low:
        return None
    text = ", ".join(
        f"{n} -{g:.0f} points" for n, g in sorted(low.items(), key=lambda x: -x[1])
    )
    start = int((1 - FINAL_FRAC) * d.budget_steps)
    return Hit(
        start, d.budget_steps, f"basic success rates below the base control's: {text}"
    )


def _death_shares(e: pd.DataFrame, since: float) -> pd.Series:
    """% of eval deaths per cause from env step ``since`` on."""
    dead = e[
        (e["mode"] == EVAL_MODE)
        & (e["end_type"] == "death")
        & (e["env_steps"] >= since)
    ]
    return dead["cause_of_death"].value_counts(normalize=True) * 100


def _imitated_unsafe_play(d: RunData) -> Hit | None:
    base_run = _base(d)
    active = d.series(("sil/active",))
    on = active[active > 0]
    if on.empty:
        return None
    start = float(on.index[0])
    run, base = (
        _death_shares(d.episodes, start),
        _death_shares(base_run.episodes, start),
    )
    rise = run.sub(base, fill_value=0.0)
    rise = rise[rise > BASE_GAP]
    if rise.empty:
        return None
    text = ", ".join(
        f"{c} +{r:.0f} points" for c, r in rise.sort_values(ascending=False).items()
    )
    return Hit(
        int(start),
        d.budget_steps,
        f"cause-of-death shares above the base control's after SIL starts: {text}",
    )


def _value_overestimation(d: RunData) -> Hit | None:
    base_run = _base(d)
    bias = _late_mean(d.series(("sil/value_bias",)), d.budget_steps, 0.5)
    ev = _late_mean(d.series(EV), d.budget_steps, 0.5)
    base_ev = _late_mean(base_run.series(EV), d.budget_steps, 0.5)
    if not (bias > 0 and ev < base_ev):
        return None
    text = (
        f"second half: sil/value_bias {bias:.3g} > 0, explained variance {ev:.2f} "
        f"< base control's {base_ev:.2f}"
    )
    return Hit(int(0.5 * d.budget_steps), d.budget_steps, text)


def _dead_sil(d: RunData) -> Hit | None:
    s = d.series(("sil/active",))
    found = _stretches(s, s.to_numpy() < DEAD_SIL, DEAD_FRAC * d.budget_steps)
    return _first(
        found, f"sil/active < {DEAD_SIL} for >= {DEAD_FRAC:.0%} of the budget"
    )


def _insertion_overflow(d: RunData) -> Hit | None:
    s = d.series(ALIASES["sil/insert_overflow"])
    over = s[s > 0]
    share = len(over) / max(len(s), 1)
    if share <= OVERFLOW_SHARE:
        return None
    return Hit(
        int(over.index[0]),
        int(over.index[-1]),
        f"sil/overflow > 0 in {share:.0%} of logged updates",
    )


def sil_detectors() -> list[Detector]:
    """The six SIL detectors."""
    m = ("sil",)
    return [
        Detector(
            "Overfitting to early lucky windows",
            _sil_overfitting,
            (ALIASES["sil/sample_age"], ("sil/active",), ENTROPY),
            (ENTROPY,),
            methods=m,
        ),
        Detector(
            "Forgetting",
            _sil_forgetting,
            (("train/ach/",),),
            (("train/ach/",),),
            methods=m,
        ),
        Detector(
            "Imitated unsafe play",
            _imitated_unsafe_play,
            (("sil/active",),),
            (),
            methods=m,
        ),
        Detector(
            "Value overestimation",
            _value_overestimation,
            (("sil/value_bias",), EV),
            (EV,),
            methods=m,
        ),
        Detector("Dead SIL", _dead_sil, (("sil/active",),), methods=m),
        Detector(
            "Insertion overflow",
            _insertion_overflow,
            (ALIASES["sil/insert_overflow"],),
            methods=m,
        ),
    ]


# --- smdp-skills and dyna-q ------------------------------------------------------

SKILL_PREFIX = "smdp/skill/"


def _stat_group(stat: str) -> Group:
    """The required-metric pattern of one per-skill stat."""
    return (f"{SKILL_PREFIX}*/{stat}",)


def _stat(d: RunData, stat: str) -> dict[str, pd.Series]:
    """``smdp/skill/<name>/<stat>`` per skill name."""
    return d.family(SKILL_PREFIX, f"/{stat}")


def _decayed(d: RunData) -> float:
    """First env step at which ``smdp/eps`` reached its final value."""
    eps = d.series(("smdp/eps",))
    return float(eps.index[np.flatnonzero(eps.to_numpy() <= eps.iloc[-1] + 1e-9)[0]])


def _stuck_top_level(d: RunData) -> Hit | None:
    chosen = _stat(d, "chosen_share")
    found = []
    for name, sr in _stat(d, "success_rate").items():
        if name in chosen:
            mask = (sr.to_numpy() < STUCK_SUCCESS) & (
                _at(chosen[name], sr.index.to_numpy()) > STUCK_SHARE
            )
            found += [
                (a, b, name)
                for a, b in _stretches(sr, mask, STUCK_FRAC * d.budget_steps)
            ]
    if not found:
        return None
    names = ", ".join(sorted({n for *_, n in found}))
    return Hit(
        min(a for a, *_ in found),
        max(b for _, b, _ in found),
        f"success < 5% with > 20% of decisions: {names}",
    )


def _fallback_collapse(d: RunData) -> Hit | None:
    chosen = _stat(d, "chosen_share")
    after = _decayed(d)
    parts = [s[s.index >= after] for n, s in chosen.items() if n in FALLBACK_SKILLS]
    share = float(pd.concat(parts, axis=1).sum(axis=1).mean()) if parts else 0.0
    if not share > FALLBACK_SHARE:
        return None
    return Hit(
        int(after),
        d.budget_steps,
        f"{' + '.join(FALLBACK_SKILLS)} take {share:.0%} of decisions after eps decay",
    )


def _skill_interference(d: RunData) -> Hit | None:
    fell = _forgotten(_stat(d, "success_rate"), SKILL_HIGH, SKILL_LOW)
    if not fell:
        return None
    chosen = _stat(d, "chosen_share")
    parts = []
    for name, a, b in fell:
        rise = {
            n: float(_at(s, np.array([b]))[0] - _at(s, np.array([a]))[0])
            for n, s in chosen.items()
            if n != name
        }
        top = max(rise, key=rise.__getitem__) if rise else "none"
        parts.append(
            f"{name} ({_fmt(a)} to {_fmt(b)}; {top} chosen +{rise.get(top, 0.0):.0%})"
        )
    return Hit(
        min(a for _, a, _ in fell),
        max(b for *_, b in fell),
        "success from >= 20% to < 5%: " + ", ".join(parts),
    )


def _timeout_dominance(d: RunData) -> Hit | None:
    found = [
        (a, b, n)
        for n, s in _stat(d, "end/timeout").items()
        for a, b in _stretches(
            s, s.to_numpy() > TIMEOUT_SHARE, TIMEOUT_FRAC * d.budget_steps
        )
    ]
    if not found:
        return None
    names = ", ".join(sorted({n for *_, n in found}))
    return Hit(
        min(a for a, *_ in found),
        max(b for _, b, _ in found),
        f"timeouts > 50% of ends for >= 10% of the budget: {names}",
    )


def _deep_skill_starvation(d: RunData) -> Hit | None:
    chosen = _stat(d, "chosen_share")
    dpe = d.series(("smdp/decisions_per_episode",))
    eps = d.series(("train/n_episodes",))
    decisions = _at(dpe, eps.index.to_numpy()) * eps.to_numpy()
    counts = {
        n: float(np.nansum(_at(chosen[n], eps.index.to_numpy()) * decisions))
        for n in DEEP_SKILLS
    }
    low = {n: c for n, c in counts.items() if c < DEEP_MIN}
    if not low:
        return None
    text = ", ".join(f"{n} about {c:.0f}" for n, c in low.items())
    return Hit(
        None, d.budget_steps, f"deep skills chosen fewer than {DEEP_MIN} times: {text}"
    )


def _final_ends(d: RunData) -> pd.DataFrame:
    """`SKILL_ENDS` rows of the last checkpoint in `EVAL_MODE`."""
    e = d.inputs[SKILL_ENDS]
    e = e[e["mode"] == EVAL_MODE]
    return e[e["update"] == e["update"].max()]


def _survival_despite_flags(d: RunData) -> Hit | None:
    ends = _final_ends(d)
    low = ends[ends["reason"] == "low_stat"]
    if low.empty:
        return None
    keys = ["update", "mode", "eval_seed"]
    ep = d.episodes[[*keys, "length", "end_type", "env_steps"]]
    low = low.drop(columns=["env_steps"], errors="ignore").merge(ep, on=keys)
    died = (low["end_type"] == "death") & (
        low["length"] - low["end"] <= FLAG_DEATH_STEPS
    )
    share = float(died.mean())
    if share <= FLAG_DEATH_SHARE:
        return None
    step = int(low["env_steps"].max())
    return Hit(
        step,
        step,
        f"{share:.0%} of {len(low)} low-stat interruptions followed by death "
        f"within {FLAG_DEATH_STEPS} steps",
    )


def _key_aliasing(d: RunData) -> Hit | None:
    ends = _final_ends(d)
    outcome = ends.assign(
        ok=ends["reason"] == "success", timeout=ends["reason"] == "timeout"
    )
    pairs = outcome.groupby(["skill", "key"]).agg(
        n=("ok", "size"), ok=("ok", "any"), timeout=("timeout", "any")
    )
    pairs["aliased"] = pairs["ok"] & pairs["timeout"]
    per = pairs.groupby("skill").apply(
        lambda g: (g["n"] * g["aliased"]).sum() / g["n"].sum(), include_groups=False
    )
    bad = per[per > ALIAS_SHARE]
    if bad.empty:
        return None
    step = int(ends["env_steps"].max())
    text = ", ".join(f"{n} {s:.0%}" for n, s in bad.items())
    return Hit(step, step, f"tries on keys with both success and timeout: {text}")


def _blind_skill_test(d: RunData) -> Hit | None:
    means = {n: float(s.mean()) for n, s in _stat(d, "test_agreement").items()}
    bad = {n: m for n, m in means.items() if m < AGREEMENT_MIN}
    if not bad:
        return None
    text = ", ".join(f"{n} {m:.2f}" for n, m in bad.items())
    return Hit(
        None, d.budget_steps, f"mean test_agreement below {AGREEMENT_MIN}: {text}"
    )


def skill_detectors() -> list[Detector]:
    """The eight detectors of the skill methods (smdp-skills; Dyna-Q has them too)."""
    # imported here: select_pair pulls in MLflow and the orchestration code
    from ..evaluation.select_pair import SKILL_METHODS

    m = SKILL_METHODS
    success, chosen = _stat_group("success_rate"), _stat_group("chosen_share")
    fallback = tuple((f"{SKILL_PREFIX}{s}/chosen_share",) for s in FALLBACK_SKILLS)
    deep = tuple((f"{SKILL_PREFIX}{s}/chosen_share",) for s in DEEP_SKILLS)
    counts = (("smdp/decisions_per_episode",), ("train/n_episodes",))
    return [
        Detector("Stuck top level", _stuck_top_level, (success, chosen), methods=m),
        Detector(
            "Fallback collapse",
            _fallback_collapse,
            (*fallback, ("smdp/eps",)),
            methods=m,
        ),
        Detector(
            "Skill interference", _skill_interference, (success, chosen), methods=m
        ),
        Detector(
            "Timeout dominance",
            _timeout_dominance,
            (_stat_group("end/timeout"),),
            methods=m,
        ),
        Detector(
            "Deep-skill starvation",
            _deep_skill_starvation,
            (*deep, *counts),
            methods=m,
        ),
        Detector(
            "Survival failure despite flags",
            _survival_despite_flags,
            inputs=(SKILL_ENDS,),
            methods=m,
        ),
        Detector("Key aliasing", _key_aliasing, inputs=(SKILL_ENDS,), methods=m),
        Detector(
            "Blind skill test",
            _blind_skill_test,
            (_stat_group("test_agreement"),),
            methods=m,
        ),
    ]


# --- dyna-q -----------------------------------------------------------------------


def _planned_value_gap(d: RunData) -> Hit | None:
    q = d.series(("smdp/q_start",))
    steps = q.index.to_numpy()
    mask = (q.to_numpy() > GAP_RATIO * _at(d.series(RETURN), steps)) & (
        steps >= _decayed(d)
    )
    found = _stretches(q, mask, GAP_FRAC * d.budget_steps)
    return _first(
        found, f"smdp/q_start > {GAP_RATIO:g} x mean return after the epsilon decay"
    )


def _stale_model(d: RunData) -> Hit | None:
    acc = d.series(("dyna/model_accuracy",))
    steps, v = acc.index.to_numpy(), acc.to_numpy(dtype=float)
    lo = np.searchsorted(steps, steps - STALE_WINDOW * d.budget_steps)
    fall = np.array([np.nanmax(v[a : i + 1]) - v[i] for i, a in enumerate(lo)])
    hit = np.flatnonzero(fall > STALE_DROP)
    if not hit.size:
        return None
    i = int(hit[0])
    return Hit(
        int(steps[lo[i]]),
        int(steps[i]),
        f"dyna/model_accuracy fell {fall[hit].max():.2f} within 10% of the budget",
    )


def _retrying(d: RunData) -> Hit | None:
    if not d.config().get("plan_kappa", 0.0) > 0:
        return Hit(None, None, "plan_kappa is 0: no exploration bonus", NOT_APPLICABLE)
    if SKILL_ENDS not in d.inputs:
        return Hit(None, None, f"no {SKILL_ENDS}", NEEDS_REPLAY)
    e = d.inputs[SKILL_ENDS]
    e = e[e["mode"] == EVAL_MODE]
    if "pair_count" not in e:
        return Hit(None, None, f"no pair_count in {SKILL_ENDS}", NEEDS_REPLAY)
    ok = (e["reason"] == "success").groupby([e["env_steps"], e["skill"], e["key"]])
    sat = (e["pair_count"] > RETRY_COUNT) & (ok.transform("mean") > RETRY_SUCCESS)
    share = sat.groupby(e["env_steps"]).mean()
    later = share[share.index >= PLATEAU_AT * d.budget_steps]
    if len(later) < 2:
        return None
    rise = float(later.iloc[-1] - later.iloc[0])
    flat = _flat(d, later.index.to_numpy(dtype=float))[-1]
    if rise <= RETRY_RISE or not flat:
        return None
    text = f"share of decisions on saturated pairs +{rise:.0%} while the score is flat"
    return Hit(int(later.index[0]), int(later.index[-1]), text)


def _no_planning_effect(d: RunData) -> Hit | None:
    s = d.series(("dyna/plan_td_abs",))
    found = _stretches(s, s.to_numpy() < PLAN_TD_MIN, PLAN_TD_FRAC * d.budget_steps)
    return _first(
        found,
        f"dyna/plan_td_abs < {PLAN_TD_MIN:g} for >= {PLAN_TD_FRAC:.0%} of the budget",
    )


def dyna_detectors() -> list[Detector]:
    """The four Dyna-Q detectors on top of the skill ones."""
    m = ("dyna-q",)
    return [
        Detector(
            "Planned-value gap",
            _planned_value_gap,
            (("smdp/q_start",), RETURN, ("smdp/eps",)),
            methods=m,
        ),
        Detector("Stale model", _stale_model, (("dyna/model_accuracy",),), methods=m),
        Detector("Bonus-driven re-trying", _retrying, (SCORE,), methods=m),
        Detector(
            "No planning effect",
            _no_planning_effect,
            (("dyna/plan_td_abs",),),
            methods=m,
        ),
    ]


@functools.cache
def _registry() -> dict[str, list[Detector]]:
    shared, count = shared_detectors(), count_bonus_detectors()
    sil, skill, dyna = sil_detectors(), skill_detectors(), dyna_detectors()
    return {
        "ppo": shared,
        "ppo-rnn": shared,
        "pqn-rnn": shared,
        "count-bonus-ppo-rnn": shared
        + [x for x in count if "count-bonus-ppo-rnn" in x.methods],
        "count-bonus-pqn-rnn": shared + count,
        "sil": shared + sil,
        "smdp-skills": shared + skill,
        "dyna-q": shared + skill + dyna,
        "combo": shared + count + sil + skill + dyna,
    }


DETECTORS: dict[str, list[Detector]]
"""Registry method -> its detectors: the shared ones, then the method's; ``combo``
holds every method detector and a combo run keeps those of its own pair. Built on
first access (the skill owners come from `select_pair.SKILL_METHODS`)."""


def __getattr__(name: str) -> object:
    """Build `DETECTORS` on first access."""
    if name == "DETECTORS":
        return _registry()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# --- running them -----------------------------------------------------------------


def _series(curves: pd.DataFrame) -> dict[str, pd.Series]:
    """Long curves of one run (or a pool) as metric -> values by env steps."""
    return {
        str(m): g.set_index("env_steps")["value"].sort_index()
        for m, g in curves.groupby("metric", sort=False)
    }


class _Builder:
    """`RunData` of the runs of one budget, base-control pools cached."""

    def __init__(self, frames: Frames, budget: str) -> None:
        self.frames = frames
        self.budget = budget
        self.steps = BUDGET_STEPS[budget]
        self.curves = dict(tuple(frames["curves"].groupby("run_id", sort=False)))
        self.episodes = dict(
            tuple(frames["eval_episodes"].groupby("run_id", sort=False))
        )
        self.inputs = {
            k: dict(tuple(frames[k].groupby("run_id", sort=False)))
            for k in INPUTS
            if k in frames
        }
        self.pools: dict[str, RunData | None] = {}

    def run(self, row: pd.Series, bases: tuple[RunData, ...] = ()) -> RunData:
        rid = row["run_id"]
        empty = self.frames["curves"].iloc[:0]
        return RunData(
            row,
            _series(self.curves.get(rid, empty)),
            self.episodes.get(rid, self.frames["eval_episodes"].iloc[:0]),
            self.steps,
            bases,
            inputs={k: v[rid] for k, v in self.inputs.items() if rid in v},
        )

    def pool(self, method: str) -> RunData | None:
        """The tuned runs of ``method`` pooled: seed-median curves, all episodes."""
        if method not in self.pools:
            sel = select(
                self.frames, method=method, config_ref="tuned", budget=self.budget
            )
            rows = [r for _, r in sel["runs"].iterrows()]
            members = tuple(self.run(r) for r in rows)
            if not members:
                self.pools[method] = None
            else:
                med = (
                    sel["curves"]
                    .groupby(["metric", "env_steps"], sort=False)["value"]
                    .median()
                )
                self.pools[method] = RunData(
                    rows[0],
                    _series(med.reset_index()),
                    sel["eval_episodes"],
                    self.steps,
                    members=members,
                )
        return self.pools[method]


def _base_methods(row: pd.Series) -> tuple[str, ...]:
    """The methods of a run's base control pools; a control's are the other controls."""
    if row["method"] in load.CONTROLS:
        return tuple(c for c in load.CONTROLS if c != row["method"])
    return (load.base_control(row),)


def detect_all(
    frames: Frames, method: str, budget: str, *, config_ref: str = "tuned"
) -> list[Obstacle]:
    """Every detector of ``method`` on each of its runs of ``budget``, ``config_ref``.

    ``frames`` are the export tables (`load.load_phases`) plus any of the optional
    `INPUTS`; base controls are the tuned runs of the same frames. One unit per call
    (R77): pass an ablation's ``config_ref`` for its own table.

    Raises:
        KeyError: If ``method`` has no detectors or ``budget`` is unknown.
    """
    detectors = _registry()[method]
    runs = frames["runs"]
    mine = select({"runs": runs}, method=method, config_ref=config_ref, budget=budget)
    rows = [r for _, r in mine["runs"].iterrows()]
    bases = [_base_methods(r) for r in rows]
    pooled = select(
        {"runs": runs},
        method=sorted(set().union(*bases)),
        budget=budget,
        config_ref="tuned",
    )
    # one pass over the large frames: keep only the runs and base runs needed
    ids = set(mine["runs"]["run_id"]) | set(pooled["runs"]["run_id"])
    build = _Builder({k: v[v["run_id"].isin(ids)] for k, v in frames.items()}, budget)
    out = []
    for row, base in zip(rows, bases, strict=True):
        pools = (build.pool(m) for m in base)
        d = build.run(row, tuple(p for p in pools if p is not None))
        owners = set((pair_of(row) or {}).get("methods", [method]))
        out += [
            det(d) for det in detectors if not det.methods or owners & set(det.methods)
        ]
    return out


# --- the table --------------------------------------------------------------------


def table_rows(obstacles: Sequence[Obstacle], budget: str) -> list[dict[str, str]]:
    """One `TABLE_COLUMNS` row per obstacle found in a run or not judged in some.

    "Runs affected" counts detected runs over judged ones (detected or absent) and
    names the runs the detector could not judge; an obstacle absent from every run has
    no row. Mitigation, outcome and plot are left for the analysis note.
    """
    groups: dict[str, list[Obstacle]] = {}
    for o in obstacles:
        groups.setdefault(o.name, []).append(o)
    rows = []
    for name, obs in groups.items():
        hit = [o for o in obs if o.status == DETECTED]
        judged = sum(o.status in (DETECTED, ABSENT) for o in obs)
        other = Counter(
            o.status for o in obs if o.status not in (DETECTED, ABSENT, NOT_APPLICABLE)
        )
        if not hit and not other:
            continue
        counts = ([f"{len(hit)} / {judged}"] if judged else []) + [
            f"{s}: {n}" for s, n in other.items()
        ]
        starts = [o.start for o in hit if o.start is not None]
        ends = [o.end for o in hit if o.end is not None]
        when = f"{_fmt(min(starts))} to {_fmt(max(ends))}" if starts and ends else ""
        first = hit[0] if hit else next(o for o in obs if o.status in other)
        more = f" (run {first.run}; {len(hit) - 1} more)" if len(hit) > 1 else ""
        rows.append(
            dict(
                zip(
                    TABLE_COLUMNS,
                    (
                        name,
                        budget.upper(),
                        "; ".join(counts),
                        when,
                        first.evidence + more,
                        "",
                        "",
                        "",
                    ),
                    strict=True,
                )
            )
        )
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def obstacles_table(rows: Sequence[Mapping[str, str]]) -> str:
    """The rows as a Markdown table with exactly the `TABLE_COLUMNS`."""
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "|" + "---|" * len(TABLE_COLUMNS)]
    lines += [
        "| " + " | ".join(_cell(r.get(c, "")) for c in TABLE_COLUMNS) + " |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"
