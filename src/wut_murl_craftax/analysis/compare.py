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

import functools
import itertools
import re
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from ..evaluation.hardware import TUNING_PHASES, reused_from_tuning
from ..evaluation.reproduction import (
    PUBLISHED_SCORES,
    band,
    published_config_hashes,
)
from ..evaluation.select_pair import COMBO, SKILL_METHODS
from ..evaluation.stability import (
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
    CI_LEVEL,
    Interval,
    iqm,
    mann_whitney_holm,
    mean_se,
    probability_of_improvement,
    seed_spread,
    stratified_bootstrap_ci,
    t_ci,
)
from ..orchestration.queue import METHODS_DIR
from . import load as _load
from .literature import COMPARABLE_BLOCKS, comparable_rows
from .load import (
    FINAL_PHASES,
    Frames,
    base_control,
    final_checkpoint,
    in_phase,
    pair_of,
)
from .obstacles import SCORE, drawdown_floor

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
NOT_REPORTED = "protocol not reported"
"""T2 ``protocol_match`` of a literature row whose protocol is ``n.r.``."""
REFERENCE_CODE = {
    ("pqn-rnn", "1m"): (
        "no published 1M value; reference-code comparison passed "
        "(original 1.509, ours 1.27 in tune-coarse, 3 seeds; band 1.005)"
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
    "score_kind", "benchmark", "indicative", "block", "source", "role", "name",
    "note", "protocol",
    "seeds", "score_pct", "ci_lo", "ci_hi", "spread", "spread_kind", "gate", "gap",
    "h", "protocol_match", "ours_above", "ours_within", "ours_below",
]  # fmt: skip
"""T2: ``block`` is ours (main, skills) or the literature block (A, A2, D); the
``ours_*`` verdicts exist only for rows of our protocol (``protocol_match``)."""
STABILITY_COLUMNS = [
    *_HEAD, "indicative", "seeds", "sd", "cv", "curve_runs", "max_drawdown_median",
    "runs_with_drops", "runs_with_collapse",
]  # fmt: skip
RUN_STABILITY_COLUMNS = [
    "benchmark", "run_id", "method", "seed", "max_drawdown", "drops", "collapsed",
]  # fmt: skip
"""One row per tuned run with a score curve (`run_stability`)."""
HARDWARE_METRICS = (
    "steady_sps", "overall_sps", "wall_clock_s", "compile_s", "peak_rss_mb",
    "peak_host_mem_mb", "peak_device_mem_mb", "steps_per_joule", "cpu_util_mean",
    "gpu_util_mean",
)  # fmt: skip
"""Medians over runs; NaN is "n.m." (`hardware.show_metric`)."""
HARDWARE_COLUMNS = [
    "benchmark", "profile", "block", "method", "role", "indicative", "runs",
    "reused_runs",
    *HARDWARE_METRICS, "note",
]  # fmt: skip
RACE_COLUMNS = [
    *_HEAD, "achievement", "profile", "n", "k", "eligible", "median_steps",
    "median_wall", "winner_steps", "winner_wall", "wall_note",
]  # fmt: skip
"""Medians are lower medians over the ``n`` logged runs, +inf for a never-unlocked one;
``wall_note`` is `CONTEXT_NOTE` for laptop wall-clock (R97)."""
GAME_COLUMNS = [
    *_HEAD, "indicative", "race_wins_steps", "race_wins_wall", "wall_note",
    "deepest_floor", "deepest_floor_share", "top_death_cause", "top_death_share",
    "coverage", "wasted_share", "adoption",
]  # fmt: skip
_HEADLINES = GAME_COLUMNS[8:]
"""T6 columns read from the eval episodes."""
SENSITIVITY_TABLE_COLUMNS = [
    "phase", *_HEAD, "configs", "runs", "best_config", "best_mean",
    "selected_config", "robust_share", "noise_floor", "top1", "top1_s",
    "top1_insensitive", "top2", "top2_s", "top2_insensitive",
]  # fmt: skip
"""T7 per tuning phase; ``best_config`` has the top mean, ``selected_config`` the tag
``best``."""
TUNING_OPTIMISM_COLUMNS = [
    *_HEAD, "config_hash", "tuning_phase", "tuning_seeds", "tuning_mean",
    "final_seeds", "fresh_seeds", "fresh_mean", "optimism", "optimism_note",
]  # fmt: skip
"""R101, R102: the tuned config's tuning mean (3 seeds) beside the mean of its fresh
final seeds (not reused from tuning); ``optimism`` is tuning minus fresh, or NaN with
an ``n/a`` note."""
MIN_FRESH = 5
"""Fresh final seeds below which the optimism reads n/a (R102)."""
TUNING_ORDER = ("tune-fine", "tune-coarse")
"""Tuning phase of the tuning mean: the fine phase when it ran the tuned config."""
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


def budgets(runs: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Each budget of ``runs`` (1M before 1B) with its runs, for the figures."""
    yield from _budgets(runs)


def tuned(runs: pd.DataFrame) -> pd.DataFrame:
    """The tuned runs of ``runs`` (no ablation, no tuning config)."""
    return _tuned(runs)


def members(runs: pd.DataFrame) -> list[Member]:
    """Rows of every table: per block, the controls, then its methods (combo last)."""
    return _members(runs)


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


def _welch(x: np.ndarray, y: np.ndarray) -> Interval:
    """Difference of means with its Welch t interval (3 seeds resample too coarsely)."""
    diff = float(x.mean() - y.mean())
    if min(len(x), len(y)) < 2:
        return Interval(diff, _NAN, _NAN)
    ci = sps.ttest_ind(x, y, equal_var=False).confidence_interval(CI_LEVEL)
    return Interval(diff, float(ci.low), float(ci.high))


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
                values[5:] = _welch(x, y)
        out |= {f"{q}_{c}": v for q, v in zip(_VERSUS, values, strict=True)}
    return out


def main_table(
    frames: Frames, *, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> pd.DataFrame:
    """T1: one row per (benchmark, block, method) of the tuned runs.

    Final score (training-episode, 2% rule) per seed: mean +- s.e., and with 95% CIs
    the mean and IQM (1M) or the mean, median and t interval (1B); the final-checkpoint
    eval means; against each control the PoI with CI and Holm p (1M, one Holm family per
    block; control rows get PoI only) or the difference of means with its Welch t
    interval (1B).
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
    out["diff"], out["diff_lo"], out["diff_hi"] = (
        _diff(x, tuned, n_boot, seed) if sig else _welch(x, tuned)
    )
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

    Only runs of a final phase count (a tuning config is no ablation). The difference
    of means (ablation minus tuned) with a bootstrap CI at 1M (Welch t at 1B); at 1M
    also IQM, PoI against tuned and Mann-Whitney with Holm over one block's ablations.
    """
    rows = []
    every = frames["runs"]
    final = np.logical_or.reduce([in_phase(every, p) for p in FINAL_PHASES])
    for budget, runs in _budgets(every[final]):
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
        "mean": float(x.mean()) if len(x) else _NAN,
        **dict.fromkeys(("published", "s_pub", "gap", "h"), _NAN),
        "status": "no published value",
        "check": "",
        "note": "",
    }
    if entry is None:
        if reference:
            out |= {"status": "pass", "check": "reference code", "note": reference}
        return out
    out["published"] = float(entry["score"])
    if entry.get("s_pub") is None:
        return out | {"status": "no published spread"}
    out |= {"s_pub": float(entry["s_pub"]), "check": "band"}
    if not len(x):
        return out | {"status": "published config not run"}
    out["gap"] = out["mean"] - out["published"]
    if len(x) < 2:  # band() needs a standard error
        return out | {"status": "too few seeds"}
    h, passed = band(x.tolist(), out["published"], out["s_pub"])
    return out | {"h": h, "status": "pass" if passed else "fail"}


@functools.cache
def _published_hash(control: str, budget: str) -> str:
    """Config hash of a control's published config at ``budget``."""
    return published_config_hashes(control, budget, METHODS_DIR)[control]


def reproduction_gate(
    frames: Frames,
    published: Path = PUBLISHED_SCORES,
    *,
    published_hash: Callable[[str, str], str] = _published_hash,
) -> pd.DataFrame:
    """Each control's final score against its published value, per benchmark.

    The runs are those of the published config (``published_hash``, by default
    `reproduction.published_config_hashes`; ``published config not run`` without
    them). ``pass`` or ``fail`` with gap (our mean minus published) and the half-width
    ``h`` of `reproduction.band`; ``too few seeds`` under 2, ``no published spread``
    without ``s_pub``, a `REFERENCE_CODE` note where no value is published (its row
    reads the tuned runs). Budgets without runs have no row (R86). It never raises on
    a fail: the comparison continues with our control as the base.
    """
    ref = yaml.safe_load(published.read_text(encoding="utf-8")) or {}
    rows = []
    for budget, runs in _budgets(frames["runs"]):
        for c in _load.CONTROLS:
            mine = runs[runs["method"] == c]
            if mine.empty:
                continue
            entry = (ref.get(c) or {}).get(budget)
            ours = (
                mine[mine["config_hash"] == published_hash(c, budget)]
                if entry is not None
                else _tuned(mine)
            )
            x = ours["final_score"].to_numpy(dtype=float)
            gate = _gate(x, entry, REFERENCE_CODE.get((c, budget)))
            rows.append({"control": c, "benchmark": budget, **gate})
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


def _literature_row(
    lit: Any, ours: Sequence[dict[str, Any]], kind: str
) -> dict[str, Any]:
    """One published row and, if it is of our protocol, where our CIs lie against it."""
    v = lit.score_pct
    named = {r["name"]: r for r in ours}.values()  # a control's two rows are equal
    same = lit.protocol == PROTOCOLS[kind]
    match = "same" if same else "different protocol"
    match = NOT_REPORTED if lit.protocol == "n.r." else match

    def which(test: Callable[[dict[str, Any]], bool]) -> str:
        return ", ".join(r["name"] for r in named if same and test(r))

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
        "protocol_match": match,
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
            **dict.fromkeys(
                ("protocol_match", "ours_above", "ours_within", "ours_below"), ""
            ),
        })
    return rows


def literature_table(
    frames: Frames,
    *,
    published: Path = PUBLISHED_SCORES,
    published_hash: Callable[[str, str], str] = _published_hash,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> pd.DataFrame:
    """T2: per comparable (score kind, benchmark), our tuned rows, then its one block.

    Our rows: mean with s.e. and 95% CI; control rows of the training kind carry the
    `reproduction_gate` (status, gap, h). Literature rows keep note, protocol, seeds and
    spread kind as copied (wikilinks unwrapped). For a row of our protocol,
    ``ours_above``, ``ours_within`` and ``ours_below`` name our methods whose CI lies
    above, holds or lies below the value; other rows read ``different protocol`` (or
    `NOT_REPORTED` for ``n.r.``) in ``protocol_match`` and give no verdict. 1B groups
    are ``indicative``.
    """
    gate = reproduction_gate(frames, published, published_hash=published_hash)
    gate = gate.set_index(["control", "benchmark"])
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        for kind, at in COMPARABLE_BLOCKS:
            if at != budget:
                continue
            ours = _our_rows(kind, runs, budget, gate, n_boot, seed)
            lit = [
                _literature_row(r, ours, kind)
                for r in comparable_rows(kind, budget).itertuples()
            ]
            head = {
                "score_kind": kind,
                "benchmark": budget,
                "indicative": budget not in SIGNIFICANCE_BUDGETS,
            }
            rows += [head | r for r in ours + lit]
    return pd.DataFrame(rows, columns=LITERATURE_TABLE_COLUMNS)


def _curve_stability(curve: np.ndarray, budget: str) -> tuple[float, bool, bool]:
    """Deepest drawdown (NaN below the floor), any drop, collapse of one score curve."""
    c = smoothed(curve)
    floor = drawdown_floor(budget, c)
    d = drawdowns(c, floor)
    deepest = float(np.nanmax(d)) if np.isfinite(d).any() else _NAN
    return deepest, bool(drops(c, floor=floor)), collapsed(c, floor)


def run_stability(frames: Frames) -> pd.DataFrame:
    """The per-run part of T3: deepest drawdown, any drop and collapse of each run.

    One row per tuned run with a ``train/score_pct`` curve (`RUN_STABILITY_COLUMNS`);
    ``max_drawdown`` is NaN below `obstacles.drawdown_floor` (R92). A run without the
    curve has no row.
    """
    curves = frames["curves"]
    score = curves[curves["metric"] == SCORE[0]].sort_values(["run_id", "env_steps"])
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        mine = score[score["run_id"].isin(runs["run_id"])]
        info = runs.set_index("run_id")
        for run_id, g in mine.groupby("run_id"):
            deepest, dropped, collapse = _curve_stability(
                g["value"].to_numpy(dtype=float), budget
            )
            rows.append({
                "benchmark": budget,
                "run_id": run_id,
                "method": info.at[run_id, "method"],
                "seed": info.at[run_id, "seed"],
                "max_drawdown": deepest,
                "drops": dropped,
                "collapsed": collapse,
            })
    return pd.DataFrame(rows, columns=RUN_STABILITY_COLUMNS)


def stability_table(frames: Frames) -> pd.DataFrame:
    """T3: seed sd and CV of the final score; per-run drawdowns, drops and collapses.

    Curves are ``train/score_pct`` per update, smoothed as `stability.smoothed`, with
    `obstacles.drawdown_floor` (R92; `run_stability`); a run without the curve is left
    out of the curve columns (``curve_runs`` counts the rest).
    """
    per_run = run_stability(frames).set_index("run_id")
    rows = []
    for budget, runs in _budgets(_tuned(frames["runs"])):
        for block, method, role in _members(runs):
            g = runs[runs["method"] == method]
            spread = seed_spread(g["final_score"].to_numpy(dtype=float))
            found = per_run.loc[per_run.index.intersection(g["run_id"])]
            rows.append({
                "benchmark": budget,
                "block": block,
                "method": method,
                "role": role,
                "indicative": budget not in SIGNIFICANCE_BUDGETS,
                "seeds": len(g),
                "sd": spread["sd"],
                "cv": spread["cv"],
                "curve_runs": len(found),
                "max_drawdown_median": found["max_drawdown"].median(),
                "runs_with_drops": int(found["drops"].sum()),
                "runs_with_collapse": int(found["collapsed"].sum()),
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
                    "indicative": budget not in SIGNIFICANCE_BUDGETS,
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


def first_unlocks(curves: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per run and achievement: ``steps`` and ``wall`` of the first unlock."""
    return _first_unlocks(curves, runs)


def _lower_median(v: pd.Series) -> float:
    return float(np.sort(v.to_numpy())[(len(v) - 1) // 2])


def _race_block(firsts: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """`race_table` rows of one block's methods (no rows for a never-logged method)."""
    mine = firsts[firsts["method"].isin(names)]
    if mine.empty:
        return pd.DataFrame(columns=[c for c in RACE_COLUMNS if c not in _HEAD[:2]])
    runs = mine.drop_duplicates("run_id").set_index("run_id")
    method = runs["method"]
    achievements = sorted(set(mine["achievement"]))

    def wide(column: str) -> pd.DataFrame:
        """Run x achievement, +inf where the run never unlocked it."""
        table = mine.pivot_table(
            index="run_id", columns="achievement", values=column, aggfunc="first"
        )
        return table.reindex(index=runs.index, columns=achievements).fillna(np.inf)

    steps, wall = wide("steps"), wide("wall")
    stats = pd.DataFrame({
        "k": np.isfinite(steps).groupby(method).sum().stack(),
        "median_steps": steps.groupby(method).agg(_lower_median).stack(),
        "median_wall": wall.groupby(method).agg(_lower_median).stack(),
    }).reset_index()
    stats["n"] = stats["method"].map(method.value_counts())
    profiles = runs.groupby("method")["profile"].agg(
        lambda p: p.iloc[0] if p.nunique() == 1 else MIXED
    )
    stats["profile"] = stats["method"].map(profiles)
    stats["eligible"] = 2 * stats["k"] >= stats["n"]
    ok = stats[stats["eligible"]]
    best = stats["achievement"].map(ok.groupby("achievement")["median_steps"].min())
    stats["winner_steps"] = stats["eligible"] & (stats["median_steps"] == best)
    timed = ok[(ok["profile"] != MIXED) & np.isfinite(ok["median_wall"])]
    fastest = timed.groupby(["achievement", "profile"])["median_wall"].min()
    key = pd.MultiIndex.from_frame(stats[["achievement", "profile"]])
    stats["winner_wall"] = (
        stats["eligible"]
        & (stats["median_wall"].to_numpy() == fastest.reindex(key).to_numpy())
        & (stats["profile"] != LAPTOP)  # R97: laptop timings are context, never ranked
    )
    stats["wall_note"] = np.where(stats["profile"] == LAPTOP, CONTEXT_NOTE, "")
    order = {m: i for i, m in enumerate(names)}
    return stats.sort_values(
        ["achievement", "method"],
        key=lambda c: c.map(order) if c.name == "method" else c,
    )


def race_table(frames: Frames) -> pd.DataFrame:
    """First-unlock race per benchmark, block and achievement (metrics-plan rule).

    ``n`` counts a method's tuned runs that logged ``first_unlock/*`` (a run without
    the family is left out, R78) and ``k`` those that unlocked the achievement. Medians
    are lower medians over all n runs, +inf for a run that never unlocked it (as
    `obstacles`' base medians), so a method is ranked on its slow seeds too. Eligible at
    k >= n / 2 (then the median is finite); the eligible method(s) with the lowest
    median win in env steps and, among methods of one profile, in wall-clock; laptop
    rows keep their times with `CONTEXT_NOTE` but win nothing (R97, two workers shared
    the machine). Methods race only inside their block.
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
    out: dict[str, Any] = dict.fromkeys(_HEADLINES, _NAN)
    out["top_death_cause"] = ""
    if eps.empty:
        return out
    run = eps["run_id"]
    deepest = int(eps["max_floor"].max())
    out["deepest_floor"] = deepest
    out["deepest_floor_share"] = (
        (eps["max_floor"] == deepest).groupby(run).mean().mean()
    )
    deaths = eps[eps["end_type"] == "death"]
    if len(deaths):
        share = pd.crosstab(
            deaths["run_id"], deaths["cause_of_death"], normalize="index"
        ).mean()
        top = min(share.index, key=lambda c: (-share[c], c))
        out["top_death_cause"], out["top_death_share"] = top, float(share[top])
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
    when no run of it logged ``first_unlock/*`` (R78); ``wall_note`` is the race's R97
    note. The rest reads the final checkpoint's stochastic eval episodes, each seed
    weighted equally: deepest floor and the share of episodes reaching it, the cause of
    death with the largest mean per-seed share of deaths (seeds with a death), coverage
    (mean over visited floors) and wasted share (medians of seed medians), adoption
    (share of episodes using a mechanic, mean over mechanics).
    """
    race = race_table(frames)
    wins = race.groupby(["benchmark", "block", "method"]).agg(
        winner_steps=("winner_steps", "sum"),
        winner_wall=("winner_wall", "sum"),
        wall_note=("wall_note", "first"),
    )
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
                "indicative": budget not in SIGNIFICANCE_BUDGETS,
                "race_wins_steps": _NAN if won is None else won["winner_steps"],
                "race_wins_wall": (
                    _NAN if won is None or won["wall_note"] else won["winner_wall"]
                ),
                "wall_note": "" if won is None else won["wall_note"],
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
    ``S_h`` (`stability.sensitivity`; hyperparameters from ``config_json``). A method
    tuned in another phase has an empty ``selected_config``.

    Raises:
        ValueError: If no run of a tuning phase is tagged ``best``: the frames were not
            loaded from the tuning phases on their own (a final phase gives ``best``
            False).
    """
    runs = frames["runs"]
    rows = []
    for phase in sorted(TUNING_PHASES):
        phase_runs = runs[in_phase(runs, phase)]
        if len(phase_runs) and not phase_runs["best"].astype(bool).any():
            msg = (
                f"no {phase} run is tagged best: load the tuning phases on their own "
                "(a run read from a final phase has best False)"
            )
            raise ValueError(msg)
        for budget, tuning in _budgets(phase_runs):
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


def _tuning_runs(
    runs: pd.DataFrame, method: str, config_hash: str
) -> tuple[str, pd.DataFrame]:
    """The first phase of `TUNING_ORDER` that ran ``config_hash`` and its runs of it."""
    mine = runs[(runs["method"] == method) & (runs["config_hash"] == config_hash)]
    for phase in TUNING_ORDER:
        if (found := mine[in_phase(mine, phase)]).size:
            return phase, found
    return "", mine.iloc[:0]


def _optimism(tuning: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    """Columns from ``tuning_mean`` on; fresh seeds were not reused from tuning."""
    fresh = final[~final["phases"].map(reused_from_tuning)]
    tuned = float(tuning["final_score"].mean()) if len(tuning) else _NAN
    out: dict[str, Any] = {
        "tuning_mean": tuned,
        "final_seeds": len(final),
        "fresh_seeds": len(fresh),
        "fresh_mean": float(fresh["final_score"].mean()) if len(fresh) else _NAN,
        "optimism": _NAN,
        "optimism_note": "",
    }
    if tuning.empty:
        return out | {"optimism_note": "n/a: never tuned"}
    if fresh.empty:
        return out | {"optimism_note": "n/a: selected on all its final seeds"}
    if len(fresh) < MIN_FRESH:
        return out | {"optimism_note": f"n/a: fewer than {MIN_FRESH} fresh seeds"}
    return out | {"optimism": tuned - out["fresh_mean"]}


def tuning_optimism(frames: Frames) -> pd.DataFrame:
    """R101, R102: per tuned unit, its tuning mean beside its fresh final mean.

    The tuning mean is over the tuned config's runs (by ``config_hash``) of tune-fine,
    else tune-coarse (3 seeds). Final seeds reused from tuning (`reused_from_tuning`)
    are left out of the comparison: ``fresh_mean`` is over the others and ``optimism``
    = tuning - fresh, the selection's winner's curse. With no fresh seed (a config
    selected on all its final seeds, as the controls) or fewer than `MIN_FRESH`, or a
    unit never tuned (combo), ``optimism`` is NaN and ``optimism_note`` says why. A
    budget without any tuning run (1B) has no rows.
    """
    rows: list[dict[str, Any]] = []
    every = frames["runs"]
    for budget, runs in _budgets(_tuned(every)):
        mine = []
        for block, method, role in _members(runs):
            final = runs[runs["method"] == method]
            config = str(final["config_hash"].iloc[0])
            phase, tuning = _tuning_runs(
                every[every["budget"] == budget], method, config
            )
            mine.append({
                "benchmark": budget,
                "block": block,
                "method": method,
                "role": role,
                "config_hash": config,
                "tuning_phase": phase,
                "tuning_seeds": len(tuning),
                **_optimism(tuning, final),
            })
        rows += mine if any(r["tuning_phase"] for r in mine) else []
    return pd.DataFrame(rows, columns=TUNING_OPTIMISM_COLUMNS)
