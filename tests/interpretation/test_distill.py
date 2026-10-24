"""Tests for the tree grid, fidelity curves and complexity of distilled policies."""

import numpy as np
import pytest

from wut_murl_craftax.interpretation import (
    DEPTH_GRID,
    LEAF_GRID,
    MEMORY_FEATURES,
    SKILL_FEATURES,
    Dataset,
    FittedTree,
    complexity,
    feature_columns,
    fidelity,
    fidelity_curve,
    fit_grid,
)

N_ACTIONS = 17
N_FEATURES = 5
LEAVES = 8
STATES = 6000
HALF = 5
"""Features are digits 0-9, so every threshold of the rule is exactly recoverable."""


def _rule(x: np.ndarray) -> np.ndarray:
    """A known 8-leaf tree over all 5 features; every leaf plays its own action."""
    left, a, b, c, d = (x[:, i] < HALF for i in range(N_FEATURES))
    return np.where(
        left,
        np.where(a, np.where(d, 0, 1), np.where(b, 2, 3)),
        np.where(d, np.where(a, 4, 5), np.where(c, 6, 7)),
    ).astype(np.int8)


def _dataset(
    seed: int,
    *,
    one_hot: bool = True,
    extra: tuple[str, ...] = (),
    recurrent: bool = False,
) -> Dataset:
    """States of a synthetic policy: uniform features, labels from `_rule`."""
    rng = np.random.default_rng(seed)
    feats = rng.integers(0, 10, (STATES, N_FEATURES + len(extra))).astype(np.float32)
    label = _rule(feats)
    eye = np.eye(N_ACTIONS, dtype=np.float32)
    probs = eye[label]
    if not one_hot:  # soft policy: 70% on the greedy action, the rest uniform
        probs = 0.7 * probs + 0.3 / N_ACTIONS
    return Dataset(
        features=feats,
        columns=tuple(f"f{i}" for i in range(N_FEATURES)) + extra,
        label=label,
        probs=probs.astype(np.float32),
        episode=np.repeat(np.arange(STATES // 100, dtype=np.int32), 100),
        step=np.tile(np.arange(100, dtype=np.int32), STATES // 100),
        floor=np.zeros(STATES, np.int8),
        family="ppo-rnn" if recurrent else "ppo",
        recurrent=recurrent,
    )


@pytest.fixture(scope="module")
def splits() -> tuple[Dataset, Dataset, Dataset]:
    return _dataset(0), _dataset(1), _dataset(2)


@pytest.fixture(scope="module")
def trees(splits: tuple[Dataset, Dataset, Dataset]) -> list[FittedTree]:
    return fit_grid(splits[0], splits[1])


def test_grid_matches_the_plan(trees: list[FittedTree]) -> None:
    assert tuple(2**k for k in range(1, 13)) == LEAF_GRID
    assert DEPTH_GRID == (2, 4, 6, 8, 10, 12, 16, 20, None)
    assert [(t.kind, t.size) for t in trees] == [
        *(("leaves", s) for s in LEAF_GRID),
        *(("depth", s) for s in DEPTH_GRID),
    ]
    for t in trees:
        params = t.model.get_params()
        assert params["min_samples_leaf"] == 5
        assert params["criterion"] == "gini"
        assert params["random_state"] == 0
        if t.kind == "leaves":
            assert params["max_leaf_nodes"] == t.size
            assert t.leaves <= t.size
        else:
            assert params["max_depth"] == t.size
            assert t.size is None or t.depth <= t.size


def test_known_rule_is_recovered_from_eight_leaves(
    splits: tuple[Dataset, Dataset, Dataset],
    trees: list[FittedTree],
) -> None:
    test = splits[2]
    top1 = {t.size: fidelity(t, test)["top1"] for t in trees if t.kind == "leaves"}
    assert top1[2] < 0.9  # two leaves cannot express 8 actions
    assert all(v == 1.0 for s, v in top1.items() if s >= LEAVES)
    got = complexity(fidelity_curve(trees, test))
    assert got["L90"] is not None
    assert got["L90"] <= LEAVES
    assert got["L95"] == LEAVES
    assert got["depth_L90"] is not None
    assert got["depth_L90"] <= 3  # a balanced 8-leaf rule is three questions deep


def test_expected_fidelity_equals_top1_for_a_one_hot_policy(
    splits: tuple[Dataset, Dataset, Dataset],
    trees: list[FittedTree],
) -> None:
    for t in trees[:5]:  # includes sizes where fidelity is below 1
        got = fidelity(t, splits[2])
        assert got["expected"] == pytest.approx(got["top1"])


def test_expected_fidelity_averages_the_policy_probability_of_the_tree_action() -> None:
    train, val, test = _dataset(0), _dataset(1), _dataset(2, one_hot=False)
    tree = fit_grid(train, val)[LEAF_GRID.index(LEAVES)]
    got = fidelity(tree, test)
    # a perfect tree plays the greedy action, which carries 0.7 + 0.3 / n of the mass
    assert got["top1"] == 1.0
    assert got["expected"] == pytest.approx(0.7 + 0.3 / N_ACTIONS, abs=1e-6)
    # a 2-leaf tree plays the wrong action on some states: lower, but above the floor
    stump = fit_grid(train, val)[0]
    low = fidelity(stump, test)
    assert 0.3 / N_ACTIONS < low["expected"] < got["expected"]
    pred = stump.model.predict(test.features)
    by_hand = test.probs[np.arange(len(pred)), pred].mean()
    assert low["expected"] == pytest.approx(by_hand)


def test_validation_fidelity_is_stored_and_matches(
    splits: tuple[Dataset, Dataset, Dataset],
    trees: list[FittedTree],
) -> None:
    for t in trees:
        assert t.val_top1 == fidelity(t, splits[1])["top1"]


def test_curve_has_one_row_per_tree(
    splits: tuple[Dataset, Dataset, Dataset],
    trees: list[FittedTree],
) -> None:
    rows = fidelity_curve(trees, splits[2])
    assert len(rows) == len(LEAF_GRID) + len(DEPTH_GRID)
    first = rows[0]
    assert first["kind"] == "leaves"
    assert first["size"] == 2
    assert first["leaves"] == trees[0].leaves
    assert first["depth"] == trees[0].depth
    assert set(first) >= {"top1", "expected", "val_top1"}


def test_fitting_is_identical_across_worker_counts(
    splits: tuple[Dataset, Dataset, Dataset],
    trees: list[FittedTree],
) -> None:
    again = fit_grid(splits[0], splits[1], workers=3)
    for a, b in zip(trees, again, strict=True):
        assert (a.kind, a.size, a.columns) == (b.kind, b.size, b.columns)
        np.testing.assert_array_equal(
            a.model.predict(splits[2].features), b.model.predict(splits[2].features)
        )


def test_memory_proxies_only_for_recurrent_policies() -> None:
    memory = MEMORY_FEATURES
    base = _dataset(0, extra=memory)
    assert feature_columns(base) == base.columns[:N_FEATURES]
    rnn = _dataset(0, extra=memory, recurrent=True)
    assert feature_columns(rnn) == rnn.columns
    # reported apart: a recurrent policy can also be fitted without them
    assert feature_columns(rnn, memory=False) == rnn.columns[:N_FEATURES]
    with pytest.raises(ValueError, match="recurrent"):
        feature_columns(base, memory=True)


def test_skill_columns_follow_the_dataset_and_can_be_dropped() -> None:
    extra = (*MEMORY_FEATURES, *SKILL_FEATURES)
    skills = _dataset(0, extra=extra, recurrent=True)
    assert feature_columns(skills) == skills.columns
    assert (
        feature_columns(skills, skill=False)
        == skills.columns[: N_FEATURES + len(MEMORY_FEATURES)]
    )
    plain = _dataset(0)
    assert feature_columns(plain) == plain.columns


def test_memory_columns_are_used_by_recurrent_trees_only() -> None:
    """A label that is a function of a memory proxy is learnable only with it."""

    def make(seed: int, recurrent: bool) -> Dataset:
        ds = _dataset(seed, extra=("prev_action",), recurrent=recurrent)
        feats = ds.features.copy()
        feats[:, -1] = np.random.default_rng(seed).integers(0, 8, STATES)
        return ds._replace(features=feats, label=feats[:, -1].astype(np.int8))

    for recurrent, reaches in ((False, False), (True, True)):
        train, val, test = (make(s, recurrent) for s in (0, 1, 2))
        trees = fit_grid(train, val)
        top1 = fidelity(trees[LEAF_GRID.index(8)], test)["top1"]
        assert (top1 == 1.0) is reaches
        assert ("prev_action" in trees[0].columns) is recurrent


def test_trees_use_the_columns_of_their_dataset_by_name() -> None:
    train, val, test = _dataset(0), _dataset(1), _dataset(2)
    tree = fit_grid(train, val)[LEAF_GRID.index(LEAVES)]
    order = [3, 1, 4, 0, 2]  # same data, columns shuffled
    shuffled = test._replace(
        features=test.features[:, order],
        columns=tuple(test.columns[i] for i in order),
    )
    assert fidelity(tree, shuffled) == fidelity(tree, test)


def test_mismatched_columns_are_refused() -> None:
    train, val = _dataset(0), _dataset(1)
    with pytest.raises(ValueError, match="columns"):
        fit_grid(train, val._replace(columns=("a", "b", "c", "d", "e")))


def test_complexity_takes_the_smallest_grid_size_reaching_each_level() -> None:
    top1 = [0.5, 0.82, 0.9, 0.93, 0.951, 0.97, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99]
    rows = [
        {"kind": "leaves", "size": s, "depth": d, "top1": v}
        for s, d, v in zip(LEAF_GRID, range(1, 13), top1, strict=True)
    ]
    # depth rows must not move the leaf thresholds
    rows += [{"kind": "depth", "size": 2, "depth": 2, "top1": 1.0}]
    got = complexity(rows)
    assert got == {"L80": 4, "L90": 8, "L95": 32, "depth_L90": 3}


def test_complexity_is_none_when_a_level_is_never_reached() -> None:
    rows = [{"kind": "leaves", "size": s, "depth": 1, "top1": 0.5} for s in LEAF_GRID]
    assert complexity(rows) == {
        "L80": None,
        "L90": None,
        "L95": None,
        "depth_L90": None,
    }


def test_a_level_hit_exactly_counts() -> None:
    rows = [{"kind": "leaves", "size": 2, "depth": 1, "top1": 0.9}]
    assert complexity(rows)["L90"] == 2
