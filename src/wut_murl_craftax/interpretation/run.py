"""Distil every final policy into decision trees: fidelity, complexity and rules.

``python -m wut_murl_craftax.interpretation.run --phases final-1m combo-1m --method all
--out outputs/interpretation --tracking-uri sqlite:///<db>?timeout=300 [--dagger]``
distils each run of a ``config_ref`` (default ``tuned``; the export's ``runs``, R77) at
its final checkpoint into ``<out>/<budget>/<method>/<run_id>/`` (skipped once done,
unless ``--force``), sums each method up, writes ``<out>/<budget>/complexity.csv`` and
the figure ``<budget>/fidelity_vs_leaves`` (copied to ``--figures/<budget>/``, each
folder with a ``captions.csv``) and logs each
method folder, datasets left out, as ``distill-<budget>-<method>`` in ``analysis``.
@see [[2026-10-20-craftax-interpretation-and-ruleset]]
"""

import argparse
import json
import logging
import math
import pickle  # nosec B403 - only trees this module wrote (scikit-learn persists by pickle)
import shutil
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..analysis.cases import run_dirs
from ..analysis.figures import (
    _figure,
    _labels,
    _save,
    _style,
    _style_of,
)
from ..analysis.figures import _theme as figure_theme
from ..analysis.latex import MIN_UNCENSORED
from ..analysis.load import METRICS_ROOT, atomic_path, load_phases, write_text_atomic
from ..evaluation.select_pair import SKILL_METHODS
from ..orchestration.tracking import log_analysis_run
from ..training.smdp import GREEDY_TABLE_FILE
from ..visualisation.charts import Shape, capture_captions, write_captions
from ..visualisation.eval_rollouts import EvalRun
from .dataset import (
    _FIELDS,
    MAX_STATES,
    SPLITS,
    Dataset,
    collect,
    relabel,
)
from .distill import LEAF_GRID, FittedTree, complexity, fidelity_curve, fit_grid
from .rules import (
    ACTION_GROUPS,
    TopLevel,
    failure_breakdown,
    importance,
    rules_text,
    top_level_rules,
)
from .tree_policy import CLOSED_LOOP_LEAVES, closed_loop, tree_rollouts

logger = logging.getLogger(__name__)

type Json = dict[str, Any]

DEFAULT_PHASES = ("final-1m", "combo-1m")
"""The phases that exist; the 1B ones are accepted but not assumed (R75, R86)."""
OUT = Path("outputs/interpretation")
FIGURES = Path("outputs/results/figures")
"""Where the comparison track reads the budget figure (report F5)."""
CAPS: dict[str, int] = {"train": MAX_STATES, "val": MAX_STATES, "test": 10_000}
"""States kept per episode and split. The plan caps training states only. Validation
states only score (``val_top1``), so the training cap serves. Test states give every
reported fidelity: 10,000 keeps whole 1M episodes (about 300 steps, at most 1,500)
and bounds the memory of a long 1B one (the limit is 100,000 steps)."""
KEPT = (16, CLOSED_LOOP_LEAVES)
"""Leaf-grid trees kept as files; the other 19 fits are reduced to the curve."""
FIDELITY_AT = (16, 64, CLOSED_LOOP_LEAVES)
BASE, NO_MEMORY, DAGGER = "base", "no-memory", "dagger"
"""Fits: default columns (memory proxies for recurrent policies), recurrent policies
without them (a feed-forward policy's is its base fit), one DAgger round on base."""
VARIANTS = (BASE, NO_MEMORY, DAGGER)
CENSORED = "> 4096"
"""How a level never reached on the grid is printed."""
SUMMARY, LOGGED = "complexity.json", ".logged"
"""A policy's summary (written last: its folder is finished) and the logging marker."""
TABLES = ("curve", "closed_loop", "importance")
"""Per-fit CSV files that the method folder concatenates."""
COLUMNS = [
    "budget", "method", "config_ref", "variant", "n_seeds",
    "l80_median", "l90_median", "l95_median", "l90_lo", "l90_hi", "l90_spread",
    "depth_at_l90_median", "fidelity_16", "fidelity_64", "fidelity_256",
    "expected_fidelity_256", "closed_loop_256", "spearman_score_l90", "spearman_p",
    "top_keys", "top_skills", "top_l90", "top_optimistic_share", "l90_censored",
    "majority", "gain_16", "gain_64", "gain_256", "peak_val_leaves",
    *(f"group_{g}" for g in ACTION_GROUPS),
    "spearman_score_fid256", "spearman_fid256_p",
]  # fmt: skip
"""`complexity.csv`: contracts section 3, then the optimistic share of the default
top-level view and the seeds whose L90 is above the grid, then the R106 numbers:
the majority-class fidelity and each fidelity's gain over it, the leaf count at the
validation peak, the 256-leaf recall per network action group and the Spearman of
score vs fidelity at 256 leaves."""


@dataclass(slots=True)
class DistilSummary:
    """What `distil_all` did."""

    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    """One ``<run_id>: <error>`` per policy that could not be distilled."""


# ------------------------------------------------------------------ censored levels


def _censor(levels: Sequence[float | None]) -> list[float]:
    return [math.inf if v is None else float(v) for v in levels]


def censored_median(levels: Sequence[float | None]) -> float:
    """Lower median of L-values, None (never reached, "> 4096") above every size.

    The lower median is a grid size, and it is inf exactly when more than half the
    seeds never reach the level. NaN without values.
    """
    return float(statistics.median_low(_censor(levels))) if levels else math.nan


def level_text(level: float) -> str:
    """A level as printed: the size, ``> 4096`` when censored, ``n/a`` for NaN."""
    if math.isnan(level):
        return "n/a"
    return CENSORED if math.isinf(level) else str(int(level))


def uncensored_median(values: Sequence[float]) -> float:
    """Median of the seeds that reached a level; NaN under `MIN_UNCENSORED` of them."""
    return float(np.median(values)) if len(values) >= MIN_UNCENSORED else math.nan


def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman's rho and p; NaN (no scipy warning) if either list is constant."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def spearman(
    scores: Sequence[float], levels: Sequence[float | None]
) -> tuple[float, float]:
    """Spearman's rho and p of score vs L90; None ranks above every size.

    NaN for fewer than 3 seeds, fewer than `MIN_UNCENSORED` reached levels (with the
    rest tied above the grid, the rank order rests on one or two seeds) or constant
    scores or levels.
    """
    if len(scores) < 3 or sum(v is not None for v in levels) < MIN_UNCENSORED:
        return math.nan, math.nan
    return _rank_correlation(scores, _censor(levels))


# ----------------------------------------------------------------------- files


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"cannot write a {type(value).__name__} as JSON"
    raise TypeError(msg)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace ``path`` atomically with ``frame`` (a reader never sees half a CSV)."""
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)


def _write_json(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=1, default=_plain) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _tree_path(folder: Path, leaves: int, variant: str) -> Path:
    return folder / f"tree_{leaves}_{variant}.pkl"


# --------------------------------------------------------------------- one policy


def _datasets(
    run: EvalRun, update: int, folder: Path, seeds: Mapping[str, Sequence[int]]
) -> dict[str, Dataset]:
    """The split datasets of the policy, collected once (`CAPS`) and then read."""
    data = {}
    for split, chosen in seeds.items():
        path = folder / f"{split}.npz"
        if path.is_file():
            data[split] = Dataset.load(path)
        else:
            data[split] = collect(run, update, chosen, CAPS[split])
            data[split].save(path)
    return data


def _union(a: Dataset, b: Dataset) -> Dataset:
    """The rows of both datasets (same columns)."""
    if a.columns != b.columns:
        msg = "a union needs datasets with the same columns"
        raise ValueError(msg)
    joined = {k: np.concatenate([getattr(a, k), getattr(b, k)]) for k in _FIELDS}
    return Dataset(columns=a.columns, family=a.family, recurrent=a.recurrent, **joined)


def _dagger_states(
    run: EvalRun, update: int, folder: Path, seeds: Sequence[int]
) -> Dataset:
    """States the base 256-leaf tree visits on ``seeds``, labelled by the network."""
    path = folder / "dagger.npz"
    if path.is_file():
        return Dataset.load(path)
    with _tree_path(folder, CLOSED_LOOP_LEAVES, BASE).open("rb") as f:
        tree: FittedTree = pickle.load(f)  # nosec B301 - written by _save_fit
    ds = relabel(run, update, tree_rollouts(tree, run, update, seeds), CAPS["train"])
    ds.save(path)
    return ds


def _save_fit(
    folder: Path,
    variant: str,
    leaf: Mapping[int, FittedTree],
    test: Dataset,
    game: Any,
) -> None:
    """The kept trees, the 16-leaf rules and the 256-leaf importance and failures."""
    for n in KEPT:
        with _tree_path(folder, n, variant).open("wb") as f:
            pickle.dump(leaf[n], f)
    write_text_atomic(
        folder / f"rules_16_{variant}.txt", rules_text(leaf[16], game=game) + "\n"
    )
    tree = leaf[CLOSED_LOOP_LEAVES]
    rows = pd.DataFrame(importance(tree, test)).assign(variant=variant)
    rows.to_csv(folder / f"importance_{variant}.csv", index=False)
    _write_json(folder / f"failure_{variant}.json", failure_breakdown(tree, test))


def _fit(
    run: EvalRun,
    update: int,
    data: Mapping[str, Dataset],
    variant: str,
    folder: Path,
    *,
    workers: int,
    seeds: Mapping[str, Sequence[int]],
    network: Sequence[float] | None,
) -> Json:
    """One fit of the grid, scored and played; ``fit_<variant>.json`` once done."""
    path = folder / f"fit_{variant}.json"
    if path.is_file():
        return dict(_read_json(path))
    train = data["train"]
    if variant == DAGGER:
        train = _union(train, _dagger_states(run, update, folder, seeds["train"]))
    memory = False if variant == NO_MEMORY else None
    trees = fit_grid(train, data["val"], memory=memory, workers=workers)
    curve = fidelity_curve(trees, data["test"])
    leaf = {int(t.size or 0): t for t in trees if t.kind == "leaves"}
    net = None if network is None else np.asarray(network)
    loop = closed_loop(leaf[CLOSED_LOOP_LEAVES], run, update, seeds["test"], net)
    _save_fit(folder, variant, leaf, data["test"], run.game)
    pd.DataFrame(curve).assign(variant=variant).to_csv(
        folder / f"curve_{variant}.csv", index=False
    )
    pd.DataFrame({
        "variant": variant, "eval_seed": loop.seeds, "tree": loop.tree,
        "network": loop.network,
    }).to_csv(folder / f"closed_loop_{variant}.csv", index=False)  # fmt: skip
    rows = {int(r["size"]): r for r in curve if r["kind"] == "leaves"}
    fit = {
        "variant": variant,
        "memory": rows[16]["memory"],
        "skill": rows[16]["skill"],
        "train_states": len(train.label),
        **complexity(curve),
        **{f"fidelity_{n}": rows[n]["top1"] for n in FIDELITY_AT},
        f"expected_{CLOSED_LOOP_LEAVES}": rows[CLOSED_LOOP_LEAVES]["expected"],
        f"closed_loop_{CLOSED_LOOP_LEAVES}": loop.fidelity,
        "tree_score": float(loop.tree.mean()),
        "network_score": float(loop.network.mean()),
        "network": loop.network,
    }
    _write_json(path, fit)
    return fit


def _top_json(top: TopLevel) -> Json:
    keys = ("update", "n_rules", "skills", "counts", "levels", "curve", "tree_text")
    return {k: getattr(top, k) for k in (*keys, "top", "n_optimistic")} | {
        "optimistic_share": top.optimistic_share
    }


def _top_text(view: str, top: TopLevel) -> list[str]:
    l90 = level_text(censored_median([top.levels["L90"]]))
    return [
        (
            f"## {view} view: {top.n_rules} keys, {len(top.skills)} greedy skills, "
            f"key-tree L90 {l90}"
        ),
        "",
        top.tree_text,
        "",
        "Highest-value keys:",
        *(f"- {r['text']}" for r in top.top),
        "",
    ]


def _top_level(update: int, run: EvalRun, folder: Path) -> Json:
    """Both views of the top level at the final checkpoint; the tried one is main.

    Raises:
        RuntimeError: If the greedy table holds no key of the final checkpoint.
    """
    path = folder / "top_level.json"
    if path.is_file():
        return dict(_read_json(path))
    table = pd.read_parquet(run.dir / "eval" / GREEDY_TABLE_FILE)
    tried = top_level_rules(table, update=update, tried=True)
    default = top_level_rules(table, update=update)
    if not (tried.update == default.update == update and default.n_rules):
        msg = f"{run.dir}: the greedy table holds no key of update {update}"
        raise RuntimeError(msg)
    share = default.optimistic_share
    lines = [
        f"# Top level of {run.method} at update {update}",
        "",
        (
            "The tried view is the main one: the default argmax lands on an untried "
            f"optimistic entry for {share:.1%} of the keys (optimistic share)."
        ),
        "",
        *_top_text("tried", tried),
        *_top_text("default", default),
    ]
    write_text_atomic(folder / "top_level_rules.txt", "\n".join(lines))
    data = {"main": "tried", "tried": _top_json(tried), "default": _top_json(default)}
    _write_json(path, data)
    return data


def distil_policy(
    run: EvalRun,
    row: Mapping[str, Any],
    folder: Path,
    *,
    dagger: bool = False,
    workers: int = 1,
    seeds: Mapping[str, Sequence[int]] = SPLITS,
) -> Json:
    """Distil one run at its final checkpoint into ``folder``; its summary.

    ``row`` is the run's ``runs`` row. Datasets, fits and the top level are each
    written once, so an interrupted policy resumes; ``complexity.json`` comes last.
    """
    update = run.updates[-1]  # the final checkpoint, passed on explicitly
    folder.mkdir(parents=True, exist_ok=True)
    data = _datasets(run, update, folder, seeds)
    train = data["train"]
    variants = [BASE, *[NO_MEMORY] * train.recurrent, *[DAGGER] * dagger]
    fits: dict[str, Json] = {}
    for variant in variants:
        network = fits[BASE]["network"] if fits else None
        fits[variant] = _fit(
            run, update, data, variant, folder, workers=workers, seeds=seeds,
            network=network,
        )  # fmt: skip
    top = _top_level(update, run, folder) if run.method in SKILL_METHODS else None
    summary = {
        "run_id": str(row["run_id"]),
        "method": str(row["method"]),
        "budget": str(row["budget"]),
        "config_ref": str(row["config_ref"]),
        "seed": int(row["seed"]),
        "score": float(row["final_score"]),
        "update": update,
        "env_steps": run.env_steps(update),
        "family": train.family,
        "recurrent": train.recurrent,
        "split_seeds": _split_counts(seeds),
        "states": {k: len(v.label) for k, v in data.items()},
        "fits": fits,
        "top_level": top,
    }
    _write_json(folder / SUMMARY, summary)
    return summary


# -------------------------------------------------------------------- summing up


def _median_run(summaries: Sequence[Json]) -> Json:
    """The seed with the (lower) median final score."""
    ranked = sorted(summaries, key=lambda s: (s["score"], s["run_id"]))
    return ranked[(len(ranked) - 1) // 2]


def _concat(folder: Path, summaries: Sequence[Json], table: str) -> pd.DataFrame:
    """A per-fit table of every policy, with the run's identity in front."""
    parts = []
    for s in summaries:
        for variant in s["fits"]:
            part = pd.read_csv(folder / s["run_id"] / f"{table}_{variant}.csv")
            ident = {k: s[k] for k in ("run_id", "seed", "method", "config_ref")}
            parts.append(part.assign(**ident))
    frame = pd.concat(parts, ignore_index=True)
    front = ["run_id", "seed", "method", "config_ref", "variant"]
    return frame[front + [c for c in frame.columns if c not in front]]


def _rules_text(folder: Path, median: Json) -> str:
    lines = [
        (
            f"# 16-leaf rules of {median['method']}: run {median['run_id']} "
            f"(seed {median['seed']}, median final score {median['score']:.3f}, "
            f"update {median['update']})"
        ),
    ]
    for variant in median["fits"]:
        path = folder / median["run_id"] / f"rules_16_{variant}.txt"
        lines += ["", f"## {variant}", "", path.read_text().rstrip()]
    return "\n".join(lines) + "\n"


FIDELITY_STEM = "fidelity_vs_leaves"
"""File stem of the distillation's fidelity figure (budget folder and method folder)."""
FIDELITY_ID = "FID"
"""Figure id of the fidelity figure in ``captions.csv`` (as `analysis.results`)."""
FIDELITY_TITLE = "Distillation fidelity"
FIDELITY_DESCRIPTION = (
    "Test top-1 fidelity of the distilled decision trees (base fit) against their "
    "number of leaves on a log2 axis: how much of the policy a tree of that size "
    "reproduces. One line per unit, the median over its seeds{seeds}; the muted "
    "horizontal line is the 0.9 level (L90). Solid lines are tuned methods, dashed "
    "lines controls, dotted lines with hollow markers ablations; colours: method "
    "legend."
)


def _plot_fidelity(
    curves: pd.DataFrame, out_dir: Path, name: str, *, per_seed: bool
) -> None:
    """Test top-1 fidelity of the base fits over the leaf grid, with the 0.9 level.

    One line per unit (median over seeds), or with ``per_seed`` one thin line per
    seed under the method's median. No legend, title or label on the image: the method
    legend figure gives the colours, the caption the rest (`FIDELITY_DESCRIPTION`).
    """
    th = figure_theme("report")
    fig = _figure(th, shape=Shape.CURVE)
    ax = _style(fig.add_subplot(), th, grid="both")
    leaf = curves[(curves["kind"] == "leaves") & (curves["variant"] == BASE)]
    for (method, ref), unit in leaf.groupby(["method", "config_ref"], sort=True):
        style = _style_of(th, str(method), str(ref))
        if per_seed:
            for _, seed in unit.groupby("run_id"):
                ax.plot(seed["size"], seed["top1"], color=style["color"], lw=0.5,
                        alpha=0.4)  # fmt: skip
        med = unit.groupby("size")["top1"].median()
        ax.plot(med.index, med.to_numpy(), lw=th.lw, ms=th.ms, **style)
    # solid: dashed lines are controls (contracts section 6)
    ax.axhline(0.9, color=th.muted, lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEAF_GRID[::2], [str(n) for n in LEAF_GRID[::2]])
    ax.set_ylim(0, 1.02)
    _labels(ax, th, "leaves (log2)", "test top-1 fidelity")
    seeds = ", with a thin line per seed" if per_seed else ""
    _save(
        fig, th, out_dir, name, title=FIDELITY_TITLE,
        description=FIDELITY_DESCRIPTION.format(seeds=seeds),
    )  # fmt: skip
    (out_dir / f"{name}.pdf").unlink(missing_ok=True)  # figures are PNG only now


def sum_up_method(folder: Path) -> None:
    """The method folder's tables, rules, top level and figure from its policies."""
    summaries = [dict(_read_json(p)) for p in sorted(folder.glob(f"*/{SUMMARY}"))]
    if not summaries:
        return
    median = _median_run(summaries)
    curves = _concat(folder, summaries, "curve")
    for table in TABLES:
        frame = curves if table == "curve" else _concat(folder, summaries, table)
        _write_csv(frame, folder / f"{table}.csv")
    _write_json(folder / SUMMARY, {"median_run": median["run_id"], "seeds": summaries})
    write_text_atomic(folder / "rules_16.txt", _rules_text(folder, median))
    if median["top_level"] is not None:
        shutil.copyfile(
            folder / median["run_id"] / "top_level_rules.txt",
            folder / "top_level_rules.txt",
        )
        seeds = [
            {k: s[k] for k in ("run_id", "seed", "score")} | s["top_level"]
            for s in summaries
        ]
        _write_json(
            folder / "top_level.json", {"median_run": median["run_id"], "seeds": seeds}
        )
    with capture_captions() as captions:
        _plot_fidelity(curves, folder, FIDELITY_STEM, per_seed=True)
    write_captions(folder, FIDELITY_ID, captions)


def _fit_of(summary: Json, variant: str) -> Json | None:
    """A policy's fit of ``variant``; a feed-forward policy's no-memory fit is base."""
    fits = summary["fits"]
    if variant == NO_MEMORY and not summary["recurrent"]:
        return dict(fits[BASE])
    fit = fits.get(variant)
    return None if fit is None else dict(fit)


def _quantiles(levels: Sequence[float | None], budget: str) -> tuple[float, float]:
    """IQR bounds at 1M, min and max at 1B (contracts section 3); no interpolation."""
    values = np.asarray(_censor(levels))
    if budget == "1b":
        return float(values.min()), float(values.max())
    lo = np.quantile(values, 0.25, method="lower")
    return float(lo), float(np.quantile(values, 0.75, method="higher"))


def _top_columns(group: Sequence[Json]) -> dict[str, float]:
    tops = [s["top_level"] for s in group if s["top_level"] is not None]
    if not tops:
        return dict.fromkeys(("top_keys", "top_skills", "top_l90"), math.nan) | {
            "top_optimistic_share": math.nan
        }
    tried = [t["tried"] for t in tops]
    return {
        "top_keys": float(np.median([t["n_rules"] for t in tried])),
        "top_skills": float(np.median([len(t["skills"]) for t in tried])),
        "top_l90": censored_median([t["levels"]["L90"] for t in tried]),
        "top_optimistic_share": float(
            np.median([t["default"]["optimistic_share"] for t in tops])
        ),
    }


GROUP_COLUMNS = [f"group_{g}" for g in ACTION_GROUPS]
EXTRA_KEYS = ("majority", "peak_val_leaves", *GROUP_COLUMNS)
"""The per-policy numbers `policy_extras` reads from a policy's files."""


def _median(values: Sequence[float]) -> float:
    """Median ignoring NaN; NaN without values."""
    arr = np.asarray(values, float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if len(arr) else math.nan


def policy_extras(folder: Path, variant: str) -> dict[str, float]:
    """R106 numbers of one policy and fit, from the files of its folder (NaN if absent).

    ``majority``: the share of the commonest network action among the test states
    (what a constant tree scores); ``peak_val_leaves``: the leaf count with the best
    validation fidelity (the smaller one on a tie); ``group_<name>``: the 256-leaf
    tree's recall on the states where the network chose an action of that group
    (the diagonal of ``failure_<variant>.json``'s confusion over its row sums).
    """
    out = dict.fromkeys(EXTRA_KEYS, math.nan)
    test = folder / "test.npz"
    if test.is_file():
        with np.load(test) as z:
            label = z["label"]
        if len(label):
            out["majority"] = float(np.bincount(label).max() / len(label))
    curve = folder / f"curve_{variant}.csv"
    if curve.is_file():
        frame = pd.read_csv(curve)
        leaf = frame[frame["kind"] == "leaves"].dropna(subset=["val_top1"])
        if len(leaf):  # idxmax is the first maximum: the smallest tree
            out["peak_val_leaves"] = float(
                leaf.loc[leaf["val_top1"].idxmax(), "leaves"]
            )
    failure = folder / f"failure_{variant}.json"
    if failure.is_file():
        data = _read_json(failure)
        conf = np.asarray(data["confusion"], float)
        rows = conf.sum(axis=1)
        for i, name in enumerate(data["groups"]):
            if rows[i]:
                out[f"group_{name}"] = float(conf[i, i] / rows[i])
    return out


def _spearman_values(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman's rho and p of two lists without their NaN pairs; NaN for under 3."""
    pairs = [
        (a, b)
        for a, b in zip(x, y, strict=True)
        if not (math.isnan(a) or math.isnan(b))
    ]
    if len(pairs) < 3:
        return math.nan, math.nan
    x, y = zip(*pairs, strict=True)
    return _rank_correlation(x, y)


def _row(
    budget: str,
    variant: str,
    group: Sequence[tuple[Json, Json]],
    extras: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> Json:
    """The `complexity.csv` row of one unit and variant (no Spearman yet)."""
    fits = [f for _, f in group]
    extra = [(extras or {}).get((s["run_id"], variant), {}) for s, _ in group]
    nan = math.nan
    majority = [e.get("majority", nan) for e in extra]
    lo, hi = _quantiles([f["L90"] for f in fits], budget)
    depths = [f["depth_L90"] for f in fits if f["depth_L90"] is not None]

    def median(key: str) -> float:
        values = np.array([f[key] for f in fits], float)
        values = values[~np.isnan(values)]  # NaN: the network scored 0 in closed loop
        return float(np.median(values)) if len(values) else math.nan

    return {
        "budget": budget,
        "method": group[0][0]["method"],
        "config_ref": group[0][0]["config_ref"],
        "variant": variant,
        "n_seeds": len(group),
        **{
            f"{k.lower()}_median": censored_median([f[k] for f in fits])
            for k in ("L80", "L90", "L95")
        },
        "l90_lo": lo,
        "l90_hi": hi,
        "l90_spread": "range" if budget == "1b" else "iqr",
        "depth_at_l90_median": uncensored_median(depths),
        **{f"fidelity_{n}": median(f"fidelity_{n}") for n in FIDELITY_AT},
        "expected_fidelity_256": median(f"expected_{CLOSED_LOOP_LEAVES}"),
        "closed_loop_256": median(f"closed_loop_{CLOSED_LOOP_LEAVES}"),
        **_top_columns([s for s, _ in group]),
        "l90_censored": sum(f["L90"] is None for f in fits),
        "majority": _median(majority),
        **{
            f"gain_{n}": _median([
                f[f"fidelity_{n}"] - m for f, m in zip(fits, majority, strict=True)
            ])
            for n in FIDELITY_AT
        },
        "peak_val_leaves": _median([e.get("peak_val_leaves", nan) for e in extra]),
        **{c: _median([e.get(c, nan) for e in extra]) for c in GROUP_COLUMNS},
    }


def complexity_table(
    summaries: Sequence[Json],
    budget: str,
    extras: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """One row per (method, config_ref, variant); Spearman over each variant's seeds.

    Key-tree levels stay in ``top_l90``, out of the L90 columns and the Spearman.
    ``extras`` maps (``run_id``, variant) to `policy_extras`; without them the R106
    columns are NaN. The Spearman of score vs fidelity at 256 leaves is R106's
    replacement for the all-censored score vs L90 one.
    """
    rows = []
    for variant in VARIANTS:
        fitted = [(s, f) for s in summaries if (f := _fit_of(s, variant))]
        if not fitted:
            continue
        rho, p = spearman(
            [s["score"] for s, _ in fitted], [f["L90"] for _, f in fitted]
        )
        rho_f, p_f = _spearman_values(
            [s["score"] for s, _ in fitted],
            [f[f"fidelity_{CLOSED_LOOP_LEAVES}"] for _, f in fitted],
        )
        units: dict[tuple[str, str], list[tuple[Json, Json]]] = {}
        for s, f in fitted:
            units.setdefault((s["method"], s["config_ref"]), []).append((s, f))
        rows += [
            _row(budget, variant, units[k], extras)
            | {
                "spearman_score_l90": rho,
                "spearman_p": p,
                "spearman_score_fid256": rho_f,
                "spearman_fid256_p": p_f,
            }
            for k in sorted(units)
        ]
    return pd.DataFrame(rows, columns=COLUMNS)


def _read_curve(path: Path) -> pd.DataFrame | None:
    """A method's ``curve.csv``; None (and a warning) if it is empty or half written."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.warning("%s is empty or torn (being rewritten?): left out", path)
        return None


def _copy_atomic(src: Path, dst: Path) -> None:
    with atomic_path(dst) as tmp:
        shutil.copyfile(src, tmp)


def policy_table(summaries: Sequence[tuple[Json, Path]]) -> pd.DataFrame:
    """One row per policy and variant: score, fidelities and the `policy_extras`."""
    rows = []
    for s, folder in summaries:
        for variant in VARIANTS:
            fit = _fit_of(s, variant)
            if fit is None:
                continue
            ex = policy_extras(folder, BASE if _fit_is_base(s, variant) else variant)
            rows.append({
                **{k: s[k] for k in ("run_id", "seed", "method", "config_ref")},
                "variant": variant,
                "score": s["score"],
                **{f"fidelity_{n}": fit[f"fidelity_{n}"] for n in FIDELITY_AT},
                **ex,
            })
    return pd.DataFrame(rows)


def _fit_is_base(summary: Json, variant: str) -> bool:
    """A feed-forward policy's no-memory fit is its base fit (files included)."""
    return variant == NO_MEMORY and not summary["recurrent"]


def sum_up_budget(folder: Path, figures: Path) -> None:
    """``complexity.csv``, ``policies.csv`` and the budget figure from ``folder``."""
    budget = folder.name
    paths = sorted(folder.glob(f"*/*/{SUMMARY}"))
    if not paths:
        return
    summaries = [(dict(_read_json(p)), p.parent) for p in paths]
    extras = {
        (s["run_id"], v): policy_extras(d, BASE if _fit_is_base(s, v) else v)
        for s, d in summaries
        for v in VARIANTS
    }
    _write_csv(
        complexity_table([s for s, _ in summaries], budget, extras),
        folder / "complexity.csv",
    )
    _write_csv(policy_table(summaries), folder / "policies.csv")
    parts = [_read_curve(p) for p in sorted(folder.glob("*/curve.csv"))]
    curves = pd.concat([c for c in parts if c is not None], ignore_index=True)
    with capture_captions() as captions:
        _plot_fidelity(curves, folder, FIDELITY_STEM, per_seed=False)
    write_captions(folder, FIDELITY_ID, captions)
    dest = figures / budget  # the comparison's per-benchmark figure folder
    dest.mkdir(parents=True, exist_ok=True)
    _copy_atomic(folder / f"{FIDELITY_STEM}.png", dest / f"{FIDELITY_STEM}.png")
    write_captions(dest, FIDELITY_ID, captions)


# ------------------------------------------------------------------------- driver


def _split_counts(seeds: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {k: len(v) for k, v in seeds.items()}


def _recorded(folder: Path) -> dict[str, int] | None:
    """The seed counts a policy's summary was written for; None if unknown."""
    path = folder / SUMMARY
    return _read_json(path).get("split_seeds") if path.is_file() else None


def _stale(folder: Path, seeds: Mapping[str, Sequence[int]]) -> bool:
    """Whether the summary was written for fewer seeds than requested (a smoke run).

    A summary from before the counts were recorded is not stale, and one with more
    seeds than requested is never: a ``--max-seeds`` call must not replace a full one.
    """
    have, want = _recorded(folder), _split_counts(seeds)
    return have is not None and any(have.get(k, 0) < n for k, n in want.items())


def _larger(folder: Path, seeds: Mapping[str, Sequence[int]]) -> bool:
    """Whether the summary holds more seeds than requested (kept, not redone)."""
    have, want = _recorded(folder), _split_counts(seeds)
    return have is not None and any(have.get(k, 0) > n for k, n in want.items())


def _finished(folder: Path, dagger: bool, seeds: Mapping[str, Sequence[int]]) -> bool:
    path = folder / SUMMARY
    return (
        path.is_file()
        and not _stale(folder, seeds)
        and (not dagger or DAGGER in _read_json(path)["fits"])
    )


def _distil_method(
    runs: pd.DataFrame,
    folder: Path,
    summary: DistilSummary,
    *,
    tracking_uri: str,
    dagger: bool,
    workers: int,
    seeds: Mapping[str, Sequence[int]],
    force: bool,
) -> bool:
    """Distil the method's unfinished policies; whether any was (re)computed."""
    rows = runs.sort_values(["config_ref", "seed"]).to_dict("records")
    todo = [
        r for r in rows if force or not _finished(folder / r["run_id"], dagger, seeds)
    ]
    summary.skipped += [r["run_id"] for r in rows if r not in todo]
    for r in rows:
        if r not in todo and _larger(folder / r["run_id"], seeds):
            logger.warning(
                "%s: finished with more seeds than requested, kept", r["run_id"]
            )
    if not todo:
        return False
    dirs = run_dirs([r["run_id"] for r in todo], tracking_uri)  # one short read
    for r in todo:
        rid, t0 = str(r["run_id"]), time.time()
        if force or _stale(folder / rid, seeds):  # no datasets of other seed counts
            shutil.rmtree(folder / rid, ignore_errors=True)
        try:
            run = EvalRun.load(dirs[rid])
            distil_policy(
                run, r, folder / rid, dagger=dagger, workers=workers, seeds=seeds
            )
        except Exception as e:  # one bad policy must not stop the others
            logger.exception("%s: distillation failed", rid)
            summary.failed.append(f"{rid}: {type(e).__name__}: {e}")
            continue
        logger.info("%s %s: distilled in %.0fs", folder, rid, time.time() - t0)
        summary.done.append(rid)
    return True


def _log_method(folder: Path, phases: Sequence[str], tracking_uri: str) -> None:
    """Log the method folder without its datasets as ``distill-<budget>-<method>``."""
    budget, method = folder.parent.name, folder.name
    tags = {
        "analysis/kind": "distill",
        "analysis/phases": ",".join(phases),
        "analysis/budget": budget,
        "analysis/method": method,
    }
    skip = shutil.ignore_patterns("*.npz", "*.tmp", LOGGED)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / method
        shutil.copytree(folder, stage, ignore=skip)
        name = f"distill-{budget}-{method}"
        paths = sorted(stage.iterdir())
        run_id = log_analysis_run(stage, paths, name, tags, tracking_uri)
    write_text_atomic(folder / LOGGED, run_id + "\n")
    logger.info("logged %s to run %s", folder, run_id)


def distil_all(
    runs: pd.DataFrame,
    *,
    out: Path,
    tracking_uri: str,
    figures: Path = FIGURES,
    phases: Sequence[str] = (),
    dagger: bool = False,
    workers: int = 1,
    seeds: Mapping[str, Sequence[int]] = SPLITS,
    force: bool = False,
    log: bool = True,
) -> DistilSummary:
    """Distil ``runs`` per budget and method, sum up and log each method folder.

    The budget files cover every policy found under ``<out>/<budget>``; ``phases``
    names the loaded phases in the run tag.
    """
    summary = DistilSummary()
    for budget in sorted(runs["budget"].astype(str).unique()):
        sel = runs[runs["budget"] == budget]
        for method in sorted(sel["method"].astype(str).unique()):
            folder = out / budget / method
            changed = _distil_method(
                sel[sel["method"] == method], folder, summary,
                tracking_uri=tracking_uri, dagger=dagger, workers=workers,
                seeds=seeds, force=force,
            )  # fmt: skip
            sum_up_method(folder)
            if log and (changed or not (folder / LOGGED).is_file()):
                _log_method(folder, phases, tracking_uri)
        sum_up_budget(out / budget, figures)
    return summary


# ---------------------------------------------------------------------------- CLI


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--phases", nargs="+", default=list(DEFAULT_PHASES))
    p.add_argument("--method", nargs="+", default=["all"], help="names, or all")
    p.add_argument("--refs", nargs="+", default=["tuned"], help="config_ref, or all")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--metrics-root", type=Path, default=METRICS_ROOT)
    p.add_argument("--figures", type=Path, default=FIGURES)
    p.add_argument(
        "--tracking-uri",
        required=True,
        help="sqlite:/// store (run folders, logging); ?timeout=<s>: busy timeout",
    )
    p.add_argument("--dagger", action="store_true", help="add one DAgger round")
    p.add_argument("--workers", type=int, default=4, help="threads per tree grid")
    p.add_argument(
        "--max-seeds", type=int, default=None, help="first N seeds per split (smoke)"
    )
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--force", action="store_true", help="redo finished policies")
    p.add_argument("--no-log", action="store_true", help="write the files only")
    a = p.parse_args(argv)
    if not a.tracking_uri.partition("?")[0].startswith("sqlite:///"):
        p.error(f"--tracking-uri must be a sqlite:/// store, got {a.tracking_uri}")
    return a


def _selected(runs: pd.DataFrame, methods: list[str], refs: list[str]) -> pd.DataFrame:
    """The runs of ``methods`` and ``refs`` (``all`` keeps every one).

    Raises:
        SystemExit: If a method has no run in the loaded phases.
    """
    if unknown := sorted(set(methods) - {"all"} - set(runs["method"])):
        msg = f"no runs of {', '.join(unknown)} in the loaded phases"
        raise SystemExit(msg)
    keep = runs["method"].isin(runs["method"] if methods == ["all"] else methods)
    keep &= runs["config_ref"].isin(runs["config_ref"] if refs == ["all"] else refs)
    return runs[keep]


def main(argv: Sequence[str] | None = None) -> None:
    """Distil the runs of the given phases, methods and ``config_ref`` values.

    Raises:
        SystemExit: If a method is unknown or some policy failed (the others finish;
            rerun to retry, finished policies are skipped).
    """
    a = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    runs = load_phases(a.phases, a.metrics_root, skip_missing=a.skip_missing)["runs"]
    loaded = [p for p in a.phases if p in set(runs["phase"])]
    seeds = {k: v[: a.max_seeds] for k, v in SPLITS.items()}
    summary = distil_all(
        _selected(runs, a.method, a.refs), out=a.out, tracking_uri=a.tracking_uri,
        figures=a.figures, phases=loaded, dagger=a.dagger, workers=a.workers,
        seeds=seeds, force=a.force, log=not a.no_log,
    )  # fmt: skip
    logger.info(
        "%d policies distilled, %d skipped, %d failed",
        len(summary.done), len(summary.skipped), len(summary.failed),
    )  # fmt: skip
    if summary.failed:
        msg = f"{len(summary.failed)} policies failed: {summary.failed}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
