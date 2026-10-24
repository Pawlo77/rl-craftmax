"""Tests for tree rules, importance, failure breakdown and top-level rules."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from craftax.craftax.constants import Action

from wut_murl_craftax.interpretation import (
    ACTION_GROUPS,
    LEAF_GRID,
    MEMORY_FEATURES,
    Condition,
    Dataset,
    FittedTree,
    Rule,
    condition_text,
    extract_rules,
    failure_breakdown,
    fit_grid,
    importance,
    paths_to,
    rules_text,
    top_level_rules,
)

N_ACTIONS = 17
STATES = 6000
RULE_COLUMNS = ("direction", "wood", "tree_dist", "facing_creature", "facing_block")
NOISE = ("stone", "coal", "iron")
COLUMNS = (*RULE_COLUMNS, *NOISE)
DO, SLEEP, NOOP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, LEFT, PICKAXE = (
    Action.DO.value,
    Action.SLEEP.value,
    Action.NOOP.value,
    Action.PLACE_STONE.value,
    Action.PLACE_TABLE.value,
    Action.PLACE_FURNACE.value,
    Action.LEFT.value,
    Action.MAKE_WOOD_PICKAXE.value,
)


def _rule(x: np.ndarray, merge: bool = False) -> np.ndarray:
    """A known 8-leaf policy over the 5 rule features, decoded units, one action a leaf.

    With ``merge`` the leaf that places a furnace plays ``DO`` as well: two chains
    end in the same action.
    """
    direction, wood, tree, creature, block = (x[:, i] for i in range(5))
    sideways = direction <= 2  # left or right
    label = np.where(
        sideways,
        np.where(
            wood >= 5,
            np.where(creature >= 1, DO, PLACE_TABLE),
            np.where(tree <= 2, SLEEP, NOOP),
        ),
        np.where(
            block <= 4,
            np.where(tree <= 3, PICKAXE, PLACE_STONE),
            np.where(wood >= 3, DO if merge else PLACE_FURNACE, LEFT),
        ),
    )
    return label.astype(np.int8)


def _dataset(
    seed: int,
    *,
    merge: bool = False,
    extra: tuple[str, ...] = (),
    states: int = STATES,
) -> Dataset:
    """States of the planted policy: independent uniform features, one-hot labels."""
    rng = np.random.default_rng(seed)
    high = np.array([5, 10, 9, 5, 17, 10, 10, 10, *[10] * len(extra)])
    low = np.array([1, 0, 1, 0, 0, 0, 0, 0, *[0] * len(extra)])
    feats = rng.integers(low, high, (states, len(high))).astype(np.float32)
    label = _rule(feats, merge)
    probs = np.eye(N_ACTIONS, dtype=np.float32)[label]
    return Dataset(
        features=feats,
        columns=(*COLUMNS, *extra),
        label=label,
        probs=probs,
        episode=np.repeat(np.arange(states // 100, dtype=np.int32), 100),
        step=np.tile(np.arange(100, dtype=np.int32), states // 100),
        floor=np.zeros(states, np.int8),
        family="ppo-rnn" if extra else "ppo",
        recurrent=bool(extra),
    )


def _tree(leaves: int, **kwargs: object) -> tuple[FittedTree, Dataset]:
    """The grid tree with ``leaves`` leaves (fitted on seed 0) and test states."""
    trees = fit_grid(_dataset(0, **kwargs), _dataset(1, **kwargs))  # type: ignore[arg-type]  # test kwargs
    return trees[LEAF_GRID.index(leaves)], _dataset(2, **kwargs)  # type: ignore[arg-type]  # test kwargs


def _holds(c: Condition, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    col = x[:, columns.index(c.feature)]
    ok = np.ones(len(x), bool)
    if c.lo is not None:
        ok &= col >= c.lo
    if c.hi is not None:
        ok &= col <= c.hi
    return ok


def _matches(rule: Rule, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    ok = np.ones(len(x), bool)
    for c in rule.conditions:
        ok &= _holds(c, x, columns)
    return ok


@pytest.fixture(scope="module")
def planted() -> tuple[FittedTree, Dataset]:
    return _tree(8)


def test_the_planted_policy_is_recovered_by_eight_leaves(
    planted: tuple[FittedTree, Dataset],
) -> None:
    tree, test = planted
    assert tree.leaves == 8
    assert (tree.model.predict(test.features) == test.label).all()


def test_rules_text_lists_every_leaf_in_decoded_units(
    planted: tuple[FittedTree, Dataset],
) -> None:
    tree, _ = planted
    lines = rules_text(tree).splitlines()
    assert len(lines) == 8
    assert [line.split(".")[0] for line in lines] == [str(i) for i in range(1, 9)]
    text = "\n".join(lines)
    used = {c.feature for r in extract_rules(tree) for c in r.conditions}
    assert used == set(RULE_COLUMNS)  # none of the noise columns
    assert "direction is left or right" in text
    assert "direction is up or down" in text
    assert "wood >= 5" in text
    assert "wood <= 4" in text
    for action in ("DO", "PLACE_TABLE", "SLEEP", "NOOP", "PLACE_STONE", "LEFT"):
        assert f"THEN {action} " in text
    # no raw float thresholds: every threshold is an integer of the decoded unit
    assert not re.search(r"\d\.\d+ ", re.sub(r"\(.*?\)", "", text))
    for ordinal in ("direction", "facing_creature", "facing_block"):
        assert not re.search(rf"{ordinal} (<|>|=)", text)


def test_every_leaf_line_carries_its_share_and_purity(
    planted: tuple[FittedTree, Dataset],
) -> None:
    tree, _ = planted
    rules = extract_rules(tree)
    assert sum(r.samples for r in rules) == tree.model.tree_.n_node_samples[0]
    assert sum(r.share for r in rules) == pytest.approx(1.0)
    assert all(r.purity == 1.0 for r in rules)
    first = rules_text(tree).splitlines()[0]
    assert f"({rules[0].share:.1%} of states, purity 1.00)" in first


def test_the_conditions_of_a_leaf_select_exactly_the_states_it_plays(
    planted: tuple[FittedTree, Dataset],
) -> None:
    tree, test = planted
    x = test.features
    seen = np.zeros(len(x), int)
    for rule in extract_rules(tree):
        mask = _matches(rule, x, test.columns)
        assert mask.any()
        assert (tree.model.predict(x[mask]) == rule.action).all()
        seen += mask
    assert (seen == 1).all()  # the leaves partition the states


def test_repeated_features_on_a_path_merge_into_one_interval() -> None:
    train, val = _dataset(0), _dataset(1)
    deep = fit_grid(train, val)[LEAF_GRID.index(64)]
    for rule in extract_rules(deep):
        names = [c.feature for c in rule.conditions]
        assert len(names) == len(set(names))
        for c in rule.conditions:
            assert c.lo is not None or c.hi is not None
            assert c.lo is None or c.hi is None or c.lo <= c.hi


@pytest.mark.parametrize(
    ("condition", "game", "text"),
    [
        (Condition("direction", None, 2), "craftax", "direction is left or right"),
        (Condition("direction", 3, None), "craftax", "direction is up or down"),
        (Condition("direction", 2, 3), "craftax", "direction is right or up"),
        (Condition("direction", 1, 1), "craftax", "direction is left"),
        (Condition("direction", None, 3), "craftax", "direction is not down"),
        (Condition("facing_creature", None, 0), "craftax", "facing_creature is none"),
        (
            Condition("facing_creature", 1, None),
            "craftax",
            "facing_creature is not none",
        ),
        (
            Condition("facing_creature", 1, 2),
            "craftax",
            "facing_creature is melee or passive",
        ),
        (
            Condition("facing_block", None, 4),
            "craftax",
            "facing_block is invalid, out_of_bounds, grass, water or stone",
        ),
        (
            Condition("facing_block", 5, None),
            "classic",
            "facing_block is not invalid, out_of_bounds, grass, water or stone",
        ),
        (Condition("facing_block", 16, None), "classic", "facing_block is ripe_plant"),
        (Condition("tree_row_sign", None, -1), "craftax", "tree_row_sign is above"),
        (
            Condition("hostile_col_sign", 1, None),
            "craftax",
            "hostile_col_sign is right",
        ),
        (
            Condition("prev_action", None, 4),
            "craftax",
            "prev_action is start, NOOP, LEFT, RIGHT, UP or DOWN [memory proxy]",
        ),
        (
            Condition("steps_since_unlock", 20, None),
            "craftax",
            "steps_since_unlock >= 20 [memory proxy]",
        ),
        (Condition("sleeping", 1, None), "craftax", "sleeping"),
        (Condition("walkable_left", None, 0), "craftax", "not walkable_left"),
        (Condition("skill_collect_wood", 1, None), "craftax", "skill is collect_wood"),
        (
            Condition("skill_collect_wood", None, 0),
            "craftax",
            "skill is not collect_wood",
        ),
        (Condition("wood", 5, None), "craftax", "wood >= 5"),
        (Condition("wood", None, 2), "craftax", "wood <= 2"),
        (Condition("wood", 2, 6), "craftax", "2 <= wood <= 6"),
        (Condition("wood", 3, 3), "craftax", "wood = 3"),
        (Condition("tree_dist", None, 3), "craftax", "tree_dist <= 3"),
    ],
)
def test_condition_text_names_the_codes_of_ordinal_features(
    condition: Condition, game: str, text: str
) -> None:
    assert condition_text(condition, game) == text  # type: ignore[arg-type]  # test literal


def test_a_condition_that_selects_no_named_code_is_refused() -> None:
    with pytest.raises(ValueError, match="direction"):
        condition_text(Condition("direction", 5, None))


def test_the_game_is_read_from_the_columns() -> None:
    """A tree with Classic-only columns names Classic's 17 block types."""

    def make(seed: int) -> Dataset:
        ds = _dataset(seed)
        block = ds.features[:, COLUMNS.index("facing_block")]
        return ds._replace(label=(block >= 16).astype(np.int8) + DO)

    stump = fit_grid(make(0), make(1))[0]
    classic = rules_text(stump)
    assert "facing_block is ripe_plant" in classic
    mana = tuple("mana" if c == "iron" else c for c in stump.columns)
    craftax = rules_text(stump._replace(columns=mana))
    assert "facing_block is ripe_plant" not in craftax
    assert "facing_block is not invalid, out_of_bounds, grass" in craftax


def test_paths_to_returns_every_chain_ending_in_the_action() -> None:
    tree, test = _tree(32, merge=True)  # two planted leaves play DO: several chains
    rules = extract_rules(tree)
    chains = paths_to(tree, DO)
    assert len(chains) >= 2
    assert [r for r in rules if r.action == DO] == chains
    mask = np.zeros(len(test.features), bool)
    for rule in chains:
        mask |= _matches(rule, test.features, test.columns)
    np.testing.assert_array_equal(mask, tree.model.predict(test.features) == DO)
    assert paths_to(tree, "DO") == paths_to(tree, DO)
    assert paths_to(tree, "do") == paths_to(tree, DO)
    assert sum(len(paths_to(tree, a)) for a in {r.action for r in rules}) == len(rules)


def test_paths_to_finds_the_single_chain_of_a_planted_leaf() -> None:
    tree, test = _tree(8)
    (chain,) = paths_to(tree, "SLEEP")
    mask = _matches(chain, test.features, test.columns)
    np.testing.assert_array_equal(mask, test.label == SLEEP)
    assert paths_to(tree, "PLACE_PLANT") == []  # a valid action no leaf plays


def test_paths_to_refuses_an_unknown_action_name() -> None:
    tree, _ = _tree(8)
    with pytest.raises(ValueError, match="DANCE"):
        paths_to(tree, "DANCE")


def test_importance_ranks_the_rule_features_above_the_rest() -> None:
    tree, test = _tree(8, extra=MEMORY_FEATURES)
    rows = importance(tree, test)
    assert {r["feature"] for r in rows[:5]} == set(RULE_COLUMNS)
    assert all(r["importance"] > 0.02 for r in rows[:5])
    assert all(r["importance"] == 0.0 for r in rows[5:])
    assert [r["importance"] for r in rows] == sorted(
        (r["importance"] for r in rows), reverse=True
    )
    assert {r["feature"] for r in rows} == set(tree.columns)
    groups = {r["feature"]: r["group"] for r in rows}
    assert {groups[m] for m in MEMORY_FEATURES} == {"memory"}
    assert {groups[f] for f in COLUMNS} == {"observation"}


def test_importance_of_the_only_informative_feature_is_the_accuracy_it_carries() -> (
    None
):
    """Four equally likely classes read off one feature: shuffling leaves 1/4."""

    def make(seed: int) -> Dataset:
        ds = _dataset(seed)
        feats = ds.features.copy()
        feats[:, 0] = np.random.default_rng(seed).integers(1, 5, len(feats))
        return ds._replace(features=feats, label=(feats[:, 0] - 1).astype(np.int8))

    trees = fit_grid(make(0), make(1))
    rows = importance(trees[LEAF_GRID.index(4)], make(2), repeats=10)
    assert rows[0]["feature"] == "direction"
    assert rows[0]["importance"] == pytest.approx(0.75, abs=0.03)
    assert rows[0]["std"] < 0.03


def test_importance_is_reproducible_and_seeded() -> None:
    tree, test = _tree(8)
    assert importance(tree, test) == importance(tree, test)
    assert importance(tree, test, seed=1) != importance(tree, test, seed=2)


def test_importance_matches_test_columns_by_name() -> None:
    tree, test = _tree(8)
    order = [4, 2, 0, 7, 1, 3, 6, 5]
    shuffled = test._replace(
        features=test.features[:, order],
        columns=tuple(test.columns[i] for i in order),
    )
    a = {r["feature"]: r["importance"] for r in importance(tree, test)}
    b = {r["feature"]: r["importance"] for r in importance(tree, shuffled)}
    assert a.keys() == b.keys()
    for f in a:
        assert a[f] == pytest.approx(b[f], abs=0.02)


def test_importance_needs_test_states() -> None:
    tree, test = _tree(8)
    empty = test._replace(
        features=test.features[:0],
        label=test.label[:0],
        probs=test.probs[:0],
        episode=test.episode[:0],
        step=test.step[:0],
        floor=test.floor[:0],
    )
    with pytest.raises(ValueError, match="test states"):
        importance(tree, empty)


def test_action_groups_cover_every_action_once() -> None:
    names = [n for group in ACTION_GROUPS.values() for n in group]
    assert sorted(names) == sorted(a.name for a in Action)
    assert list(ACTION_GROUPS)[:3] == ["noop", "move", "do"]


def test_failure_breakdown_counts_confusion_by_group_and_fidelity_by_floor() -> None:
    train, val, test = _dataset(0), _dataset(1), _dataset(2)
    stump = fit_grid(train, val)[0]  # two leaves: most actions are missed
    test = test._replace(floor=(np.arange(STATES) % 3).astype(np.int8))
    got = failure_breakdown(stump, test)
    pred = stump.model.predict(test.features)
    group_of = {
        a.value: next(g for g, names in ACTION_GROUPS.items() if a.name in names)
        for a in Action
    }
    groups = tuple(ACTION_GROUPS)
    assert got["groups"] == groups
    expected = np.zeros((len(groups), len(groups)), int)
    for network, tree in zip(test.label, pred, strict=True):
        expected[
            groups.index(group_of[int(network)]), groups.index(group_of[int(tree)])
        ] += 1
    np.testing.assert_array_equal(got["confusion"], expected)
    assert got["confusion"].sum() == STATES
    assert got["top1"] == pytest.approx((pred == test.label).mean())
    assert [r["floor"] for r in got["by_floor"]] == [0, 1, 2]
    for row in got["by_floor"]:
        sel = test.floor == row["floor"]
        assert row["states"] == sel.sum()
        assert row["top1"] == pytest.approx((pred[sel] == test.label[sel]).mean())
    assert got["top1"] < 0.9  # the stump does fail


def test_failure_breakdown_of_a_perfect_tree_is_diagonal() -> None:
    tree, test = _tree(8)
    got = failure_breakdown(tree, test)
    conf = got["confusion"]
    assert got["top1"] == 1.0
    assert (conf == np.diag(np.diag(conf))).all()
    assert [r["top1"] for r in got["by_floor"]] == [1.0]


# --- top level -----------------------------------------------------------------

SKILLS = ("collect_wood", "place_table", "collect_stone", "place_furnace")
FIELDS = ("floor", "ladder_open", "pickaxe", "wood", "stone")


def _frame(
    update: int, pickaxe: np.ndarray, wood: np.ndarray, skill: np.ndarray
) -> pd.DataFrame:
    n = len(skill)
    rng = np.random.default_rng(update)
    value = rng.permutation(n) / 4.0
    tried = skill.astype(object).copy()
    tried[wood == 9] = ""  # a key whose entries were all untried
    return pd.DataFrame({
        "update": np.full(n, update, np.int32),
        "key": rng.permutation(10_000)[:n].astype(np.int32),
        "floor": np.zeros(n, int),
        "ladder_open": np.zeros(n, int),
        "pickaxe": pickaxe,
        "wood": wood,
        "stone": np.zeros(n, int),
        "greedy_skill": skill,
        "value": value,
        "tried": np.ones(n, np.int32),
        "greedy_tried_skill": tried,
        "greedy_tried_value": np.where(wood == 9, np.nan, value - 1.0),
    })


def _table() -> pd.DataFrame:
    """Two checkpoints: 40 keys at the last whose greedy skill is a 4-leaf rule."""
    pick, wood = (a.ravel() for a in np.meshgrid(range(4), range(10), indexing="ij"))
    skill = np.where(
        pick == 0,
        np.where(wood < 5, SKILLS[0], SKILLS[1]),
        np.where(wood < 5, SKILLS[2], SKILLS[3]),
    )
    old = _frame(0, np.arange(25) % 4, np.arange(25) % 10, np.full(25, "descend"))
    return pd.concat([old, _frame(100, pick, wood, skill)], ignore_index=True)


def test_top_level_reports_rules_skills_complexity_and_highest_values() -> None:
    table = _table()
    got = top_level_rules(table)
    assert got.update == 100  # the final checkpoint only
    assert got.n_rules == 40
    assert set(got.skills) == set(SKILLS)
    assert got.counts == {
        "collect_stone": 15,
        "place_furnace": 15,
        "collect_wood": 5,
        "place_table": 5,
    }
    assert got.skills == tuple(got.counts)  # most frequent first, ties by name
    assert got.levels["L90"] == 4
    assert got.levels["L95"] == 4
    assert got.levels["depth_L90"] == 2
    two = next(r for r in got.curve if r["size"] == 2)
    assert two["top1"] < 0.9
    assert next(r for r in got.curve if r["size"] == 4)["top1"] == 1.0
    assert len(got.rules) == 4
    assert len(got.tree_text.splitlines()) == 4
    for line in got.tree_text.splitlines():
        assert "of keys, purity 1.00)" in line


def test_the_key_tree_rules_partition_the_keys_and_name_the_skill() -> None:
    final = _table().query("update == 100")
    got = top_level_rules(final)
    x = final[list(FIELDS)].to_numpy(np.float32)
    seen = np.zeros(len(x), int)
    for rule in got.rules:
        mask = _matches(rule, x, FIELDS)
        assert (final["greedy_skill"].to_numpy()[mask] == rule.action).all()
        assert {c.feature for c in rule.conditions} == {"pickaxe", "wood"}
        seen += mask
    assert (seen == 1).all()


def test_the_ten_highest_value_rules_are_in_words() -> None:
    final = _table().query("update == 100")
    got = top_level_rules(final)
    best = final.sort_values("value", ascending=False).head(10)
    assert [r["key"] for r in got.top] == best["key"].tolist()
    assert [r["value"] for r in got.top] == best["value"].tolist()
    assert [r["skill"] for r in got.top] == best["greedy_skill"].tolist()
    row = best.iloc[0]
    assert got.top[0]["text"] == (
        f"{row['greedy_skill']} (value {row['value']:.2f}) when floor 0, "
        f"pickaxe={row['pickaxe']}, wood={row['wood']}"
    )


def test_top_count_is_a_parameter() -> None:
    assert len(top_level_rules(_table(), top=3).top) == 3
    assert len(top_level_rules(_table(), top=100).top) == 40


def test_key_words_name_flags_and_skip_zero_fields() -> None:
    table = _table()
    table.loc[table["update"] == 100, "ladder_open"] = 1
    table.loc[table["update"] == 100, "floor"] = 2
    text = top_level_rules(table).top[0]["text"]
    assert "when floor 2, ladder_open, pickaxe=" in text
    assert "stone=" not in text  # zero fields are left out


def test_an_older_checkpoint_can_be_asked_for() -> None:
    got = top_level_rules(_table(), update=0)
    assert got.update == 0
    assert got.n_rules == 25
    assert got.skills == ("descend",)


def test_a_table_file_is_read_like_the_frame(tmp_path: Path) -> None:
    table = _table()
    path = tmp_path / "smdp_greedy_table.parquet"
    table.to_parquet(path)
    assert top_level_rules(path) == top_level_rules(table)
    assert top_level_rules(str(path)) == top_level_rules(table)


def test_the_tried_view_drops_keys_without_a_tried_skill() -> None:
    got = top_level_rules(_table(), tried=True)
    assert got.n_rules == 36  # the 4 keys with wood == 9 have none
    assert got.top[0]["value"] == pytest.approx(
        _table().query("update == 100 and wood < 9")["greedy_tried_value"].max()
    )


def test_a_checkpoint_without_keys_has_no_rules() -> None:
    got = top_level_rules(_table(), update=7)
    assert got.n_rules == 0
    assert got.skills == ()
    assert got.counts == {}
    assert got.levels == {"L80": None, "L90": None, "L95": None, "depth_L90": None}
    assert got.rules == []
    assert got.top == []
    assert got.tree_text == ""


def test_one_skill_everywhere_is_one_rule() -> None:
    table = _table().query("update == 100")
    table = table.assign(greedy_skill="collect_wood")
    got = top_level_rules(table)
    assert got.skills == ("collect_wood",)
    assert got.levels["L90"] == 2  # the grid's smallest size; the tree has one leaf
    assert got.curve[0]["leaves"] == 1
    assert got.tree_text == "1. ALWAYS collect_wood (100.0% of keys, purity 1.00)"


def test_a_table_without_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="no rows"):
        top_level_rules(_table().iloc[:0])
