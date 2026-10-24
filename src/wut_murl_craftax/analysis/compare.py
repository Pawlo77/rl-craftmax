"""Result tables of the method comparison (T1-T4, T6, T7, PoI matrix, ablations).

Every table runs per benchmark and block (`block_of`); the three controls are reference
rows of both blocks and no method meets one of the other block. Statistics are per
(method, ``config_ref``): the tables use the tuned runs, `ablation_table` each ablation
against its method's tuned runs (R77). IQM, PoI and Mann-Whitney with Holm (one family
per block) exist at 1M only; 1B rows give mean and median with CIs (indicative).
`reproduction_gate` reports and never raises; laptop timings are context only (R97).

@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_compare.py]]
"""

import itertools
import re
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.hardware import TUNING_PHASES, reused_from_tuning
from ..evaluation.reproduction import PUBLISHED_SCORES, band
from ..evaluation.select_pair import COMBO, SKILL_METHODS
from ..evaluation.stability import (
    MIN_MAX,
    collapsed,
    config_scores,
    drawdowns,
    drops,
    robust_share,
    sensitivity,
    smoothed,
)
from ..evaluation.stats import (
    BOOTSTRAP_N,
    Interval,
    iqm,
    mann_whitney_holm,
    mean_se,
    probability_of_improvement,
    seed_spread,
    stratified_bootstrap_ci,
    t_ci,
)
from . import load as _load
from .literature import COMPARABLE_BLOCKS, comparable_rows
from .load import Frames, base_control, final_checkpoint, in_phase, pair_of
from .obstacles import (
    ABSOLUTE_BUDGETS,
    DRAWDOWN_FLOOR_MIN,
    DRAWDOWN_FLOOR_SHARE,
    SCORE,
)

MAIN, SKILLS = "main", "skills"
BLOCKS = (MAIN, SKILLS)
"""The two blocks of every table; `SKILL_METHODS` set the second apart."""
TUNED = "tuned"
SIGNIFICANCE_BUDGETS = ("1m",)
"""Budgets with 10 seeds: IQM, PoI and p-values; the others (3 seeds) are indicative."""
SCORE_KINDS = {"training": "final_score", "stochastic": "eval_stochastic_score"}
"""Score kind of `literature.COMPARABLE_BLOCKS` -> the ``runs`` column it reads."""
PROTOCOLS = {
    "training": "training-episode return",
    "stochastic": "separate evaluation of the trained policy",
}
"""Our protocol per score kind, worded as the literature table's ``protocol``."""
REFERENCE_CODE = {
    ("pqn-rnn", "1m"): (
        "no published 1M value; reference-code comparison passed "
        "(original 1.509, ours 1.27, band 1.005)"
    ),
}
"""Gate rows settled by a reference-code run instead of a published value."""
LAPTOP = "laptop"
"""`hardware.profile_of` of the Apple-GPU runs."""
CONTEXT_NOTE = "two concurrent workers, context only"
"""Every laptop run trained beside a second queue worker (R97)."""
SUMMARY_COLUMNS = [
    "mean", "se", "mean_lo", "mean_hi", "iqm", "iqm_lo", "iqm_hi",
    "median", "median_lo", "median_hi", "t_lo", "t_hi",
]  # fmt: skip
"""Seed statistics of a unit: IQM at significance budgets, median and t CI elsewhere."""
_HEAD = ["benchmark", "block", "method", "role"]
POI_COLUMNS = ["benchmark", "block", "method", "versus", "poi", "poi_lo", "poi_hi"]
ABLATION_COLUMNS = [
    "benchmark", "block", "method", "config_ref", "indicative", "seeds",
    *SUMMARY_COLUMNS, "tuned_seeds", "tuned_mean", "tuned_iqm", "diff", "diff_lo",
    "diff_hi", "poi_tuned", "poi_tuned_lo", "poi_tuned_hi", "p", "p_holm",
]  # fmt: skip
GATE_COLUMNS = [
    "control", "benchmark", "seeds", "mean", "published", "s_pub", "gap", "h",
    "status", "check", "note",
]  # fmt: skip
LITERATURE_TABLE_COLUMNS = [
    "score_kind", "benchmark", "block", "source", "role", "name", "note", "protocol",
    "seeds", "score_pct", "ci_lo", "ci_hi", "spread", "spread_kind", "gate", "gap",
    "h", "ours_above", "ours_within", "ours_below",
]  # fmt: skip
"""T2: ``block`` is ours (main, skills) or the literature block (A, A2, D)."""
STABILITY_COLUMNS = [
    *_HEAD, "seeds", "sd", "cv", "curve_runs", "max_drawdown_median",
    "runs_with_drops", "runs_with_collapse",
]  # fmt: skip
HARDWARE_METRICS = (
    "steady_sps", "overall_sps", "wall_clock_s", "compile_s", "peak_rss_mb",
    "peak_host_mem_mb", "peak_device_mem_mb", "steps_per_joule", "cpu_util_mean",
    "gpu_util_mean",
)  # fmt: skip
"""Medians over runs; NaN is "n.m." (`hardware.show_metric`)."""
HARDWARE_COLUMNS = [
    "benchmark", "profile", "block", "method", "role", "runs", "reused_runs",
    *HARDWARE_METRICS, "note",
]  # fmt: skip
RACE_COLUMNS = [
    *_HEAD, "achievement", "profile", "n", "k", "eligible", "median_steps",
    "median_wall", "winner_steps", "winner_wall",
]  # fmt: skip
GAME_COLUMNS = [
    *_HEAD, "race_wins_steps", "race_wins_wall", "deepest_floor",
    "deepest_floor_share", "top_death_cause", "top_death_share", "coverage",
    "wasted_share", "adoption",
]  # fmt: skip
SENSITIVITY_TABLE_COLUMNS = [
    "phase", *_HEAD, "configs", "runs", "best_config", "best_mean",
    "selected_config", "robust_share", "noise_floor", "top1", "top1_s",
    "top1_insensitive", "top2", "top2_s", "top2_insensitive",
]  # fmt: skip
"""T7 per tuning phase; ``best_config`` has the top mean, ``selected_config`` the tag
``best``."""
_UNLOCK, _UNLOCK_WALL = "first_unlock/", "first_unlock_wall/"
_FIRSTS_COLUMNS = ["run_id", "achievement", "steps", "wall", "method", "profile"]
MIXED = "mixed"
"""Race profile of a method whose runs span profiles: it runs no wall-clock race."""
_NAN = float("nan")
_VERSUS = ("poi", "poi_lo", "poi_hi", "p", "p_holm", "diff", "diff_lo", "diff_hi")
"""Per-control quantities of T1: ``<quantity>_<control>``."""
type Member = tuple[str, str, str]
"""(block, method, role) of one table row; role ``control`` or ``method``."""


def block_of(method: str, combo_pair: Mapping[str, Any] | None) -> str:
    """``skills`` for a skill method or a combo whose pair holds one, else ``main``.

    ``combo_pair`` is the run's own ``{base, methods}`` (`load.pair_of`, R80a).

    Raises:
        ValueError: For combo without its pair.
    """
    if method == COMBO:
        if combo_pair is None:
            msg = "a combo run's block needs its own pair (load.pair_of)"
            raise ValueError(msg)
        return SKILLS if set(combo_pair["methods"]) & set(SKILL_METHODS) else MAIN
    return SKILLS if method in SKILL_METHODS else MAIN


def _budgets(runs: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Each budget of ``runs`` (1M before 1B) with its runs."""
    for budget in sorted(set(runs["budget"]), key=lambda b: ("mb".find(b[-1]), b)):
        yield budget, runs[runs["budget"] == budget]


def _tuned(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[runs["config_ref"] == TUNED]


def _members(runs: pd.DataFrame) -> list[Member]:
    """Rows of a table: per block, the controls, then its methods (combo last).

    A block without a method of its own is left out.
    """
    present = set(runs["method"])
    controls = [c for c in _load.CONTROLS if c in present]
    blocks = {
        str(m): block_of(str(m), pair_of(g.iloc[0]))
        for m, g in runs.groupby("method")
        if m not in controls
    }
    out: list[Member] = []
    for block in BLOCKS:
        mine = sorted(
            (m for m, b in blocks.items() if b == block), key=lambda m: (m == COMBO, m)
        )
        if mine:
            out += [(block, c, "control") for c in controls]
            out += [(block, m, "method") for m in mine]
    return out


def _scores(runs: pd.DataFrame, column: str = "final_score") -> dict[str, np.ndarray]:
    return {str(m): g[column].to_numpy(dtype=float) for m, g in runs.groupby("method")}


def _boot(
    data: Mapping[str, np.ndarray],
    statistic: Callable[[dict[str, np.ndarray]], float],
    n_boot: int,
    seed: int,
) -> Interval:
    return stratified_bootstrap_ci(
        {k: v.tolist() for k, v in data.items()}, statistic, n_boot, seed
    )


def _poi(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Interval:
    return _boot(
        {"x": x, "y": y},
        lambda d: probability_of_improvement(list(d["x"]), list(d["y"])),
        n_boot,
        seed,
    )


def _diff(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Interval:
    return _boot(
        {"x": x, "y": y}, lambda d: float(d["x"].mean() - d["y"].mean()), n_boot, seed
    )


def _summary(x: np.ndarray, budget: str, n_boot: int, seed: int) -> dict[str, float]:
    """`SUMMARY_COLUMNS` of one unit's scores."""
    out = dict.fromkeys(SUMMARY_COLUMNS, _NAN)
    out["mean"], out["se"] = mean_se(x.tolist())
    data = {"x": x}
    _, out["mean_lo"], out["mean_hi"] = _boot(
        data, lambda d: float(d["x"].mean()), n_boot, seed
    )
    if budget in SIGNIFICANCE_BUDGETS:
        out["iqm"], out["iqm_lo"], out["iqm_hi"] = _boot(
            data, lambda d: iqm(d["x"]), n_boot, seed
        )
        return out
    out["median"], out["median_lo"], out["median_hi"] = _boot(
        data, lambda d: float(np.median(d["x"])), n_boot, seed
    )
    if len(x) > 1:
        _, out["t_lo"], out["t_hi"] = t_ci(x.tolist())
    return out


def main_columns() -> list[str]:
    """T1 columns; the per-control ones follow `load.CONTROLS`."""
    versus = [f"{q}_{c}" for c in _load.CONTROLS for q in _VERSUS]
    return [
        *_HEAD, "base_control", "indicative", "seeds", *SUMMARY_COLUMNS,
        "eval_stochastic", "eval_greedy", *versus,
    ]  # fmt: skip


type Pairs = dict[Hashable, tuple[Sequence[float], Sequence[float]]]
"""Mann-Whitney comparisons keyed by (method, other); first sample is the method's."""
type Tests = dict[Hashable, tuple[float, float]]
"""Key of `Pairs` -> (p, Holm p)."""


def _holm(families: Sequence[Pairs]) -> Tests:
    """Key -> (p, Holm p), each mapping of ``families`` one Holm family."""
    out: Tests = {}
    for pairs in families:
        if pairs:
            tests = mann_whitney_holm(pairs)
            p = zip(tests["p"], tests["p_holm"], strict=True)
            out |= dict(zip(pairs, p, strict=True))
    return out


def _versus_controls(
    members: Sequence[Member], scores: Mapping[str, np.ndarray], controls: Sequence[str]
) -> list[Pairs]:
    """One family per block: its methods against each control."""
    return [
        {
            (m, c): (scores[m].tolist(), scores[c].tolist())
            for b, m, role in members
            if b == block and role == "method"
            for c in controls
        }
        for block in BLOCKS
    ]


def _versus(
    method: str,
    scores: Mapping[str, np.ndarray],
    budget: str,
    tests: Tests,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """PoI and Holm p (1M) or the difference of means (1B) against each control."""
    out: dict[str, float] = {}
    for c in _load.CONTROLS:
        values = [_NAN] * len(_VERSUS)
        if c in scores and c != method:
            x, y = scores[method], scores[c]
            if budget in SIGNIFICANCE_BUDGETS:
                p = tests.get((method, c), (_NAN, _NAN))
                values[:5] = [*_poi(x, y, n_boot, seed), *p]
            else:
                values[5:] = _diff(x, y, n_boot, seed)
        out |= {f"{q}_{c}": v for q, v in zip(_VERSUS, values, strict=True)}
    return out


def main_table(
    frames: Frames, *, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> pd.DataFrame:
    """T1: one row per (benchmark, block, method) of the tuned runs.

    Final score (training-episode, 2% rule) per seed: mean +- s.e., and with 95% CIs
    the mean and IQM (1M) or the mean, median and t interval (1B); the final-checkpoint
    eval means; against each control the PoI with CI and Holm p (1M, one Holm family per
    block; control rows get PoI only) or the difference of means with CI (1B).
    """
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        scores = _scores(runs)
        members = _members(runs)
        controls = [c for c in _load.CONTROLS if c in scores]
        tests = (
            _holm(_versus_controls(members, scores, controls))
            if budget in SIGNIFICANCE_BUDGETS
            else {}
        )
        for block, method, role in members:
            g = runs[runs["method"] == method]
            rows.append({
                "benchmark": budget,
                "block": block,
                "method": method,
                "role": role,
                "base_control": base_control(g.iloc[0]),
                "indicative": budget not in SIGNIFICANCE_BUDGETS,
                "seeds": len(g),
                **_summary(scores[method], budget, n_boot, seed),
                "eval_stochastic": g["eval_stochastic_score"].mean(),
                "eval_greedy": g["eval_greedy_score"].mean(),
                **_versus(method, scores, budget, tests, n_boot, seed),
            })
    return pd.DataFrame(rows, columns=main_columns())


def poi_matrix(
    frames: Frames, *, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> pd.DataFrame:
    """PoI with CI between every two members of one block (F3), at 1M only.

    No cross-block cell and no significance; P(b > a) is 1 - P(a > b) in every
    resample, so it is derived, not bootstrapped again.
    """
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        if budget not in SIGNIFICANCE_BUDGETS:
            continue
        scores, members = _scores(runs), _members(runs)
        for block in BLOCKS:
            names = [m for b, m, _ in members if b == block]
            for a, b in itertools.combinations(names, 2):
                p, lo, hi = _poi(scores[a], scores[b], n_boot, seed)
                rows += [
                    (budget, block, a, b, p, lo, hi),
                    (budget, block, b, a, 1 - p, 1 - hi, 1 - lo),
                ]
    return pd.DataFrame(rows, columns=POI_COLUMNS)


def _ablation_row(
    budget: str,
    x: np.ndarray,
    tuned: np.ndarray | None,
    test: tuple[float, float],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """The statistics columns of one ablation row (NaN where the budget has none)."""
    sig = budget in SIGNIFICANCE_BUDGETS
    out: dict[str, Any] = dict.fromkeys(ABLATION_COLUMNS[6:], _NAN)
    out |= {"indicative": not sig, "seeds": len(x), **_summary(x, budget, n_boot, seed)}
    if tuned is None:
        return out
    out["tuned_seeds"], out["tuned_mean"] = len(tuned), float(tuned.mean())
    out["diff"], out["diff_lo"], out["diff_hi"] = _diff(x, tuned, n_boot, seed)
    if sig:
        out["tuned_iqm"] = iqm(tuned)
        out["poi_tuned"], out["poi_tuned_lo"], out["poi_tuned_hi"] = _poi(
            x, tuned, n_boot, seed
        )
        out["p"], out["p_holm"] = test
    return out


def ablation_table(
    frames: Frames, *, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> pd.DataFrame:
    """R77: each ablation (method, ``config_ref``) against its method's tuned runs.

    The difference of means (ablation minus tuned) with CI at every budget; at 1M also
    IQM, PoI against tuned and Mann-Whitney with Holm over the ablations of one block.
    """
    rows = []
    for budget, runs in _budgets(frames["runs"]):
        tuned = _scores(_tuned(runs))
        units = {
            (str(m), str(r)): g
            for (m, r), g in runs.groupby(["method", "config_ref"])
            if r != TUNED
        }
        blocks = {u: block_of(u[0], pair_of(g.iloc[0])) for u, g in units.items()}
        scores = {u: g["final_score"].to_numpy(dtype=float) for u, g in units.items()}
        tests: Tests = {}
        if budget in SIGNIFICANCE_BUDGETS:
            tests = _holm([
                {
                    u: (x.tolist(), tuned[u[0]].tolist())
                    for u, x in scores.items()
                    if u[0] in tuned and blocks[u] == block
                }
                for block in BLOCKS
            ])
        for u in sorted(units, key=lambda u: (BLOCKS.index(blocks[u]), u)):
            stats = _ablation_row(
                budget,
                scores[u],
                tuned.get(u[0]),
                tests.get(u, (_NAN, _NAN)),
                n_boot,
                seed,
            )
            rows.append({
                "benchmark": budget,
                "block": blocks[u],
                "method": u[0],
                "config_ref": u[1],
                **stats,
            })
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)


def _gate(
    x: np.ndarray, entry: Mapping[str, Any] | None, reference: str | None
) -> dict[str, Any]:
    """Gate columns of one control and benchmark (see `reproduction_gate`)."""
    out: dict[str, Any] = {
        "seeds": len(x),
        "mean": float(x.mean()),
        **dict.fromkeys(("published", "s_pub", "gap", "h"), _NAN),
        "status": "no published value",
        "check": "",
        "note": "",
    }
    if entry is None or entry.get("s_pub") is None:
        if reference:
            out |= {"status": "pass", "check": "reference code", "note": reference}
        return out
    out |= {"published": float(entry["score"]), "s_pub": float(entry["s_pub"])}
    out |= {"gap": out["mean"] - out["published"], "check": "band"}
    if len(x) < 2:  # band() needs a standard error
        return out | {"status": "too few seeds"}
    h, passed = band(x.tolist(), out["published"], out["s_pub"])
    return out | {"h": h, "status": "pass" if passed else "fail"}


def reproduction_gate(
    frames: Frames, published: Path = PUBLISHED_SCORES
) -> pd.DataFrame:
    """Each control's tuned final score against its published value, per benchmark.

    ``pass`` or ``fail`` with gap (our mean minus published) and the half-width ``h`` of
    `reproduction.band`; ``too few seeds`` under 2, a `REFERENCE_CODE` note where no
    value is published. Budgets without runs have no row (R86). It never raises on a
    fail: the comparison continues with our control as the base.
    """
    ref = yaml.safe_load(published.read_text(encoding="utf-8")) or {}
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        scores = _scores(runs)
        rows += [
            {
                "control": c,
                "benchmark": budget,
                **_gate(
                    scores[c],
                    (ref.get(c) or {}).get(budget),
                    REFERENCE_CODE.get((c, budget)),
                ),
            }
            for c in _load.CONTROLS
            if c in scores
        ]
    return pd.DataFrame(rows, columns=GATE_COLUMNS)


_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
"""``[[target]]`` or ``[[target|alias]]``: the shown text is group 1."""


def _our_ci(x: np.ndarray, budget: str, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI of the mean: bootstrap at 1M, t at 1B (3 seeds resample too coarsely)."""
    if budget in SIGNIFICANCE_BUDGETS:
        _, lo, hi = _boot({"x": x}, lambda d: float(d["x"].mean()), n_boot, seed)
        return lo, hi
    if len(x) < 2:
        return _NAN, _NAN
    _, lo, hi = t_ci(x.tolist())
    return lo, hi


def _literature_row(lit: Any, ours: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One published row and where our CIs lie against its value."""
    v = lit.score_pct
    named = {r["name"]: r for r in ours}.values()  # a control's two rows are equal

    def which(test: Callable[[dict[str, Any]], bool]) -> str:
        return ", ".join(r["name"] for r in named if test(r))

    return {
        "block": lit.block,
        "source": "literature",
        "role": "",
        "name": _WIKILINK.sub(r"\1", lit.implementation),
        "note": lit.note,
        "protocol": lit.protocol,
        "seeds": lit.seeds,
        "score_pct": v,
        "ci_lo": _NAN,
        "ci_hi": _NAN,
        "spread": lit.spread,
        "spread_kind": lit.spread_kind,
        "gate": "",
        "gap": _NAN,
        "h": _NAN,
        "ours_above": which(lambda r: r["ci_lo"] > v),
        "ours_within": which(lambda r: r["ci_lo"] <= v <= r["ci_hi"]),
        "ours_below": which(lambda r: r["ci_hi"] < v),
    }


def _our_rows(
    kind: str,
    runs: pd.DataFrame,
    budget: str,
    gate: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Our tuned rows of one score kind; training control rows carry the gate."""
    scores = _scores(runs, SCORE_KINDS[kind])
    ci = {m: _our_ci(x, budget, n_boot, seed) for m, x in scores.items()}
    rows = []
    for block, method, role in _members(runs):
        x = scores[method]
        g = gate.loc[(method, budget)] if (method, budget) in gate.index else None
        g = g if kind == "training" else None
        rows.append({
            "block": block,
            "source": "ours",
            "role": role,
            "name": method,
            "note": "",
            "protocol": PROTOCOLS[kind],
            "seeds": len(x),
            "score_pct": float(x.mean()),
            "ci_lo": ci[method][0],
            "ci_hi": ci[method][1],
            "spread": mean_se(x.tolist())[1],
            "spread_kind": "s.e.",
            "gate": "" if g is None else g["status"],
            "gap": _NAN if g is None else g["gap"],
            "h": _NAN if g is None else g["h"],
            **dict.fromkeys(("ours_above", "ours_within", "ours_below"), ""),
        })
    return rows


def literature_table(
    frames: Frames,
    *,
    published: Path = PUBLISHED_SCORES,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> pd.DataFrame:
    """T2: per comparable (score kind, benchmark), our tuned rows, then its one block.

    Our rows: mean with s.e. and 95% CI; control rows of the training kind carry the
    `reproduction_gate` (status, gap, h). Literature rows keep note, protocol, seeds and
    spread kind as copied (wikilinks unwrapped); ``ours_above``, ``ours_within`` and
    ``ours_below`` name our methods whose CI lies above, holds or lies below the value.
    """
    gate = reproduction_gate(frames, published).set_index(["control", "benchmark"])
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        for kind, at in COMPARABLE_BLOCKS:
            if at != budget:
                continue
            ours = _our_rows(kind, runs, budget, gate, n_boot, seed)
            lit = [
                _literature_row(r, ours)
                for r in comparable_rows(kind, budget).itertuples()
            ]
            rows += [{"score_kind": kind, "benchmark": budget, **r} for r in ours + lit]
    return pd.DataFrame(rows, columns=LITERATURE_TABLE_COLUMNS)


def _drawdown_floor(c: np.ndarray, budget: str) -> float:
    """R92 as in the training analysis: 1 point at 1B, else ``max(0.1, 0.25 best)``."""
    if budget in ABSOLUTE_BUDGETS:
        return MIN_MAX
    best = float(np.nanmax(c)) if np.isfinite(c).any() else 0.0
    return max(DRAWDOWN_FLOOR_MIN, DRAWDOWN_FLOOR_SHARE * best)


def _curve_stability(curve: np.ndarray, budget: str) -> tuple[float, bool, bool]:
    """Deepest drawdown (NaN below the floor), any drop, collapse of one score curve."""
    c = smoothed(curve)
    floor = _drawdown_floor(c, budget)
    d = drawdowns(c, floor)
    deepest = float(np.nanmax(d)) if np.isfinite(d).any() else _NAN
    return deepest, bool(drops(c, floor=floor)), collapsed(c, floor)


def stability_table(frames: Frames) -> pd.DataFrame:
    """T3: seed sd and CV of the final score; per-run drawdowns, drops and collapses.

    Curves are ``train/score_pct`` per update, smoothed as `stability.smoothed`, with
    the drawdown floor of the training analysis (R92); a run without the curve is left
    out of the curve columns (``curve_runs`` counts the rest).
    """
    curves = frames["curves"]
    score = curves[curves["metric"] == SCORE[0]].sort_values(["run_id", "env_steps"])
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        mine = score[score["run_id"].isin(runs["run_id"])]
        per_run = {
            r: _curve_stability(g["value"].to_numpy(dtype=float), budget)
            for r, g in mine.groupby("run_id")
        }
        for block, method, role in _members(runs):
            g = runs[runs["method"] == method]
            spread = seed_spread(g["final_score"].to_numpy(dtype=float))
            found = [per_run[r] for r in g["run_id"] if r in per_run]
            rows.append({
                "benchmark": budget,
                "block": block,
                "method": method,
                "role": role,
                "seeds": len(g),
                "sd": spread["sd"],
                "cv": spread["cv"],
                "curve_runs": len(found),
                "max_drawdown_median": pd.Series([f[0] for f in found]).median(),
                "runs_with_drops": sum(f[1] for f in found),
                "runs_with_collapse": sum(f[2] for f in found),
            })
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def hardware_table(frames: Frames) -> pd.DataFrame:
    """T4 per benchmark and ``profile`` (never compared across): run medians.

    ``reused_runs`` counts runs reused from tuning; laptop rows carry `CONTEXT_NOTE`
    (R97). NaN means not measured.
    """
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        for profile, prof in runs.groupby("profile"):
            for block, method, role in _members(prof):
                g = prof[prof["method"] == method]
                rows.append({
                    "benchmark": budget,
                    "profile": profile,
                    "block": block,
                    "method": method,
                    "role": role,
                    "runs": len(g),
                    "reused_runs": sum(map(reused_from_tuning, g["phases"])),
                    **g[list(HARDWARE_METRICS)].median().to_dict(),
                    "note": CONTEXT_NOTE if profile == LAPTOP else "",
                })
    return pd.DataFrame(rows, columns=HARDWARE_COLUMNS)


def _first_unlocks(curves: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per run and achievement: ``steps`` and ``wall`` of the first unlock.

    With the run's ``method`` and ``profile``; runs without the family are absent.
    """
    fam = curves[
        curves["run_id"].isin(runs["run_id"])
        & curves["metric"].str.startswith((_UNLOCK, _UNLOCK_WALL))
    ]
    first = (
        fam
        .sort_values("env_steps")
        .groupby(["run_id", "metric"], sort=False)["value"]
        .first()
        .reset_index()
    )
    if first.empty:
        return pd.DataFrame(columns=_FIRSTS_COLUMNS)
    parts = first["metric"].str.split("/", n=1, expand=True)
    kind = np.where(parts[0] == _UNLOCK.rstrip("/"), "steps", "wall")
    wide = (
        first
        .assign(kind=kind, achievement=parts[1])
        .pivot_table(index=["run_id", "achievement"], columns="kind", values="value")
        .reindex(columns=["steps", "wall"])
        .reset_index()
    )
    return wide.merge(runs[["run_id", "method", "profile"]], on="run_id")[
        _FIRSTS_COLUMNS
    ]


def _race_block(firsts: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """`race_table` rows of one block's methods (no rows for a never-logged method)."""
    mine = firsts[firsts["method"].isin(names)]
    n = mine.groupby("method")["run_id"].nunique()
    profiles = mine.groupby("method")["profile"].agg(
        lambda p: p.iloc[0] if p.nunique() == 1 else MIXED
    )
    stats = (
        mine
        .dropna(subset=["steps"])
        .groupby(["achievement", "method"])
        .agg(
            k=("run_id", "nunique"),
            median_steps=("steps", "median"),
            median_wall=("wall", "median"),
        )
        .reset_index()
    )
    stats["n"] = stats["method"].map(n)
    stats["profile"] = stats["method"].map(profiles)
    stats["eligible"] = 2 * stats["k"] >= stats["n"]
    ok = stats[stats["eligible"]]
    best = stats["achievement"].map(ok.groupby("achievement")["median_steps"].min())
    stats["winner_steps"] = stats["eligible"] & (stats["median_steps"] == best)
    timed = ok[ok["profile"] != MIXED]
    fastest = timed.groupby(["achievement", "profile"])["median_wall"].min()
    key = pd.MultiIndex.from_frame(stats[["achievement", "profile"]])
    stats["winner_wall"] = stats["eligible"] & (
        stats["median_wall"].to_numpy() == fastest.reindex(key).to_numpy()
    )
    order = {m: i for i, m in enumerate(names)}
    return stats.sort_values(
        ["achievement", "method"],
        key=lambda c: c.map(order) if c.name == "method" else c,
    )


def race_table(frames: Frames) -> pd.DataFrame:
    """First-unlock race per benchmark, block and achievement (metrics-plan rule).

    ``n`` counts a method's tuned runs that logged ``first_unlock/*`` (a run without
    the family is left out, R78) and ``k`` those that unlocked the achievement; medians
    run over those k. Eligible at k >= n / 2; the eligible method(s) with the lowest
    median win in env steps and, among methods of one profile, in wall-clock. Methods
    race only inside their block.
    """
    parts = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        firsts = _first_unlocks(frames["curves"], runs)
        members = _members(runs)
        for block in BLOCKS:
            names = [m for b, m, _ in members if b == block]
            stats = _race_block(firsts, names)
            role = stats["method"].map({m: r for b, m, r in members if b == block})
            parts.append(stats.assign(benchmark=budget, block=block, role=role))
    if not parts:
        return pd.DataFrame(columns=RACE_COLUMNS)
    return pd.concat(parts, ignore_index=True).reindex(columns=RACE_COLUMNS)


def _game_headlines(eps: pd.DataFrame) -> dict[str, Any]:
    """T6 columns from one unit's final stochastic eval episodes (see the table)."""
    out: dict[str, Any] = dict.fromkeys(GAME_COLUMNS[6:], _NAN)
    out["top_death_cause"] = ""
    if eps.empty:
        return out
    run = eps["run_id"]
    deepest = int(eps["max_floor"].max())
    out["deepest_floor"] = deepest
    out["deepest_floor_share"] = (
        (eps["max_floor"] == deepest).groupby(run).mean().mean()
    )
    deaths = eps.loc[eps["end_type"] == "death", "cause_of_death"]
    if len(deaths):
        counts = deaths.value_counts()
        top = min(counts.index, key=lambda c: (-counts[c], c))
        out["top_death_cause"], out["top_death_share"] = top, counts[top] / len(deaths)
    coverage = [c for c in eps if c.startswith("coverage/")]
    if coverage:
        out["coverage"] = eps[coverage].mean(axis=1).groupby(run).median().median()
    out["wasted_share"] = eps["wasted_share"].groupby(run).median().median()
    if adv := [c for c in eps if c.startswith("adv/")]:
        out["adoption"] = (eps[adv] > 0).groupby(run).mean().mean(axis=1).mean()
    return out


def game_summary_table(frames: Frames) -> pd.DataFrame:
    """T6: one row per (benchmark, block, method) of the tuned runs.

    Race wins count the achievements a method wins in its block (`race_table`); NaN
    when no run of it logged ``first_unlock/*`` (R78). The rest reads the final
    checkpoint's stochastic eval episodes, each seed weighted equally: deepest floor and
    the share of episodes reaching it, the most common cause of death and its share of
    deaths, coverage (mean over visited floors) and wasted share (medians of seed
    medians), adoption (share of episodes using a mechanic, mean over mechanics).
    """
    race = race_table(frames)
    wins = race.groupby(["benchmark", "block", "method"])[
        ["winner_steps", "winner_wall"]
    ].sum()
    episodes = frames["eval_episodes"]
    final = final_checkpoint(episodes[episodes["mode"] == "stochastic"])
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        for block, method, role in _members(runs):
            key = (budget, block, method)
            won = wins.loc[key] if key in wins.index else None
            ids = runs.loc[runs["method"] == method, "run_id"]
            rows.append({
                "benchmark": budget,
                "block": block,
                "method": method,
                "role": role,
                "race_wins_steps": _NAN if won is None else won["winner_steps"],
                "race_wins_wall": _NAN if won is None else won["winner_wall"],
                **_game_headlines(final[final["run_id"].isin(ids)]),
            })
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def _sensitivity_row(runs: pd.DataFrame) -> dict[str, Any]:
    """T7 columns of one method's runs of one tuning phase."""
    scores = config_scores(runs)
    top = scores.loc[scores["r_c"].idxmax()]
    sens = sensitivity(runs)
    selected = sorted(set(runs.loc[runs["best"].astype(bool), "config_hash"]))
    out: dict[str, Any] = {
        "configs": len(scores),
        "runs": len(runs),
        "best_config": top["config_hash"],
        "best_mean": float(top["mean"]),
        "selected_config": ", ".join(selected),
        "robust_share": robust_share(scores),
        "noise_floor": sens["noise_floor"].iloc[0] if len(sens) else _NAN,
    }
    for i in (1, 2):
        h = sens.iloc[i - 1] if len(sens) >= i else None
        out |= {
            f"top{i}": "" if h is None else h["hyperparameter"],
            f"top{i}_s": _NAN if h is None else h["sensitivity"],
            f"top{i}_insensitive": _NAN if h is None else bool(h["insensitive"]),
        }
    return out


def sensitivity_table(frames: Frames) -> pd.DataFrame:
    """T7 per tuning phase (``phases`` tag), benchmark and method.

    Configs and runs tried, the config with the top mean and the one tagged ``best``,
    robust share, the noise floor and the two most sensitive hyperparameters with
    ``S_h`` (`stability.sensitivity`; hyperparameters from ``config_json``). Load the
    tuning phases on their own: a run read from a final phase has ``best`` False.
    """
    runs = frames["runs"]
    rows = []
    for phase in sorted(TUNING_PHASES):
        for budget, tuning in _budgets(runs[in_phase(runs, phase)]):
            for block, method, role in _members(tuning):
                rows.append({
                    "phase": phase,
                    "benchmark": budget,
                    "block": block,
                    "method": method,
                    "role": role,
                    **_sensitivity_row(tuning[tuning["method"] == method]),
                })
    return pd.DataFrame(rows, columns=SENSITIVITY_TABLE_COLUMNS)
