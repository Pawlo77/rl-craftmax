"""Decision-tree distillation: the tree grid, fidelity curves and complexity levels.

`fit_grid` fits `DecisionTreeClassifier`s of growing size on the labelled states of
one policy (`dataset.Dataset`), `fidelity` scores a tree on held-out states and
`complexity` reads L80, L90 and L95 off the curve. Plan
[[2026-10-20-craftax-interpretation-and-ruleset]]. Memory proxies enter the trees of
recurrent policies only (`feature_columns`), so a report can show a recurrent
policy with and without them.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, NamedTuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .dataset import MEMORY_FEATURES, SKILL_FEATURES, Dataset

type Row = dict[str, Any]
"""One tree of the curve: kind, size, leaves, depth, top1, expected, val_top1."""

LEAF_GRID: tuple[int, ...] = tuple(2**k for k in range(1, 13))
"""``max_leaf_nodes`` of the size grid: 2, 4, ..., 4096."""
DEPTH_GRID: tuple[int | None, ...] = (2, 4, 6, 8, 10, 12, 16, 20, None)
"""``max_depth`` of the depth grid (None: unlimited)."""
LEVELS: dict[str, float] = {"L80": 0.8, "L90": 0.9, "L95": 0.95}
"""Complexity levels: the top-1 fidelity a tree must reach."""
MIN_SAMPLES_LEAF = 5
"""Smallest leaf, so a leaf is a rule backed by several states."""


class FittedTree(NamedTuple):
    """A tree of the grid with the columns it reads and its validation fidelity."""

    kind: Literal["leaves", "depth"]
    """Which limit the grid varies: ``max_leaf_nodes`` or ``max_depth``."""
    size: int | None
    """The limit's value (None: unlimited depth)."""
    model: DecisionTreeClassifier
    columns: tuple[str, ...]
    """Feature names in the order the model reads them; datasets are matched by name."""
    val_top1: float
    """Top-1 fidelity on the validation states (tree-size choice)."""

    @property
    def leaves(self) -> int:
        """Leaves of the fitted tree (at most ``size`` on the leaf grid)."""
        return int(self.model.get_n_leaves())

    @property
    def depth(self) -> int:
        """Depth of the fitted tree."""
        return int(self.model.get_depth())


def feature_columns(
    ds: Dataset, memory: bool | None = None, skill: bool | None = None
) -> tuple[str, ...]:
    """The columns a tree may read.

    ``memory`` defaults to the dataset's ``recurrent`` flag: memory proxies are
    for recurrent policies only, and False drops them to report apart. ``skill``
    (default True) drops the current-skill columns of two-level policies.

    Raises:
        ValueError: If ``memory`` is True on a policy that is not recurrent.
    """
    if memory and not ds.recurrent:
        msg = f"memory proxies need a recurrent policy, not {ds.family!r}"
        raise ValueError(msg)
    drop: set[str] = set()
    if not (ds.recurrent if memory is None else memory):
        drop |= set(MEMORY_FEATURES)
    if skill is False:
        drop |= set(SKILL_FEATURES)
    return tuple(c for c in ds.columns if c not in drop)


def _matrix(ds: Dataset, columns: tuple[str, ...]) -> np.ndarray:
    """The feature rows of ``ds`` restricted to ``columns``, in that order."""
    if columns == ds.columns:
        return ds.features
    return ds.features[:, [ds.columns.index(c) for c in columns]]


def _top1(model: DecisionTreeClassifier, x: np.ndarray, label: np.ndarray) -> float:
    return float((model.predict(x) == label).mean())


def _fit(
    kind: Literal["leaves", "depth"],
    size: int | None,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    columns: tuple[str, ...],
) -> FittedTree:
    limit = {"max_leaf_nodes": size} if kind == "leaves" else {"max_depth": size}
    model = DecisionTreeClassifier(
        criterion="gini", min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0, **limit
    ).fit(*train)
    return FittedTree(kind, size, model, columns, _top1(model, *val))


def fit_grid(
    train: Dataset,
    val: Dataset,
    memory: bool | None = None,
    skill: bool | None = None,
    workers: int = 1,
) -> list[FittedTree]:
    """Fit the leaf grid (`LEAF_GRID`) then the depth grid (`DEPTH_GRID`) on ``train``.

    ``memory`` and ``skill`` pick the columns as in `feature_columns`. The fits
    are independent (a fixed ``random_state``): ``workers`` threads speed them up
    without changing a tree, as scikit-learn's tree builder releases the GIL.

    Raises:
        ValueError: If ``train`` and ``val`` have different columns.
    """
    if train.columns != val.columns:
        msg = "train and val need the same columns"
        raise ValueError(msg)
    columns = feature_columns(train, memory, skill)
    fit = (_matrix(train, columns), train.label)
    check = (_matrix(val, columns), val.label)
    jobs: list[tuple[Literal["leaves", "depth"], int | None]] = [
        *(("leaves", s) for s in LEAF_GRID),
        *(("depth", s) for s in DEPTH_GRID),
    ]
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_fit, k, s, fit, check, columns) for k, s in jobs]
        return [f.result() for f in futures]


def _predict(tree: FittedTree, ds: Dataset) -> np.ndarray:
    """The tree's action at each state of ``ds`` (columns matched by name)."""
    return np.asarray(tree.model.predict(_matrix(ds, tree.columns)))


def fidelity(tree: FittedTree, test: Dataset) -> dict[str, float]:
    """Agreement of a tree with the policy on ``test`` states.

    ``top1`` is the share of states where the tree plays the network's greedy action;
    ``expected`` the mean probability the stochastic policy gives the tree's action.
    """
    pred = _predict(tree, test)
    return {
        "top1": float((pred == test.label).mean()),
        "expected": float(test.probs[np.arange(len(pred)), pred].mean()),
    }


def fidelity_curve(trees: Sequence[FittedTree], test: Dataset) -> list[Row]:
    """One row per tree: kind, size, leaves, depth, top1, expected and val_top1."""
    return [
        {
            "kind": t.kind,
            "size": t.size,
            "leaves": t.leaves,
            "depth": t.depth,
            **fidelity(t, test),
            "val_top1": t.val_top1,
        }
        for t in trees
    ]


def complexity(curve: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    """L80, L90, L95 and the depth at L90 of a `fidelity_curve`.

    Lx is the smallest ``max_leaf_nodes`` of the leaf grid with test top-1 fidelity
    of at least x (None: never reached, reported as "> 4096"); ``depth_L90`` is the
    depth of that tree.
    """
    leaf = sorted((r for r in curve if r["kind"] == "leaves"), key=lambda r: r["size"])
    hits = {
        name: next((r for r in leaf if r["top1"] >= level), None)
        for name, level in LEVELS.items()
    }
    out = {n: None if r is None else int(r["size"]) for n, r in hits.items()}
    out["depth_L90"] = None if hits["L90"] is None else int(hits["L90"]["depth"])
    return out
