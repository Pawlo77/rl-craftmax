"""Rules and importance of distilled trees, and the top-level rules of skill methods.

`extract_rules` reads a `distill.FittedTree` leaf by leaf into `Condition` intervals
that `condition_text` prints in decoded units. Ordinal codes (`direction`,
`facing_creature`, `facing_block`, offset signs, `prev_action`) never print a
threshold: ``direction <= 2.5`` becomes the named set it selects ("left or right").
`importance` is permutation importance on test states, `failure_breakdown` the
confusion by action group and fidelity per floor, `top_level_rules` the rules of a
greedy table's final checkpoint. Plan [[2026-10-20-craftax-interpretation-and-ruleset]];
code tables in [[interpretation]].
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from craftax.craftax.constants import Action, BlockType
from sklearn.tree import DecisionTreeClassifier

from ..visualisation.rollout import GameKind
from .dataset import MEMORY_FEATURES, SKILL_FEATURES, Dataset
from .distill import LEAF_GRID, FittedTree, Row, complexity
from .features import CREATURES, FEATURES

ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "noop": ("NOOP",),
    "move": ("LEFT", "RIGHT", "UP", "DOWN"),
    "do": ("DO",),
    "rest": ("SLEEP", "REST"),
    "place": tuple(a.name for a in Action if a.name.startswith("PLACE_")),
    "craft": tuple(a.name for a in Action if a.name.startswith("MAKE_")),
    "ladder": ("DESCEND", "ASCEND"),
    "combat": ("SHOOT_ARROW", "CAST_FIREBALL", "CAST_ICEBALL"),
    "potion": tuple(a.name for a in Action if a.name.startswith("DRINK_POTION_")),
    "upgrade": (
        "READ_BOOK",
        "ENCHANT_SWORD",
        "ENCHANT_ARMOUR",
        "ENCHANT_BOW",
        "LEVEL_UP_DEXTERITY",
        "LEVEL_UP_STRENGTH",
        "LEVEL_UP_INTELLIGENCE",
    ),
}
"""Action group -> action names; the groups of `failure_breakdown`, in its order."""
_NAMES = {a.value: a.name for a in Action}
_GROUP_OF = np.zeros(max(_NAMES) + 1, np.int64)
_GROUP_OF[[Action[n].value for names in ACTION_GROUPS.values() for n in names]] = [
    i for i, names in enumerate(ACTION_GROUPS.values()) for _ in names
]
_FLAGS = frozenset({
    *("sleeping", "resting", "learned_fireball", "learned_iceball", "ladder_open"),
    *("boss_vulnerable", "armour_enchanted", "bow", "hostile_adjacent"),
    *(f"walkable_{d}" for d in ("left", "right", "up", "down")),
    *("spell_fireball", "spell_iceball", "ench_sword", "ench_bow", "ench_armour"),
    *("low_food", "low_drink", "low_energy", "low_health"),
})
"""0/1 columns (observation flags and key fields): printed as the bare name."""
_SIGNS = {
    "_row_sign": {-1: "above", 0: "level or none", 1: "below"},
    "_col_sign": {-1: "left", 0: "level or none", 1: "right"},
}
"""Offset-sign columns by suffix; 0 is also the value with nothing in view."""
_KEY_META = frozenset({
    *("update", "key", "greedy_skill", "value", "tried"),
    *("greedy_tried_skill", "greedy_tried_value"),
})
"""Greedy-table columns that are not decoded key fields."""
_FRACTIONAL = frozenset({"light_level"})
"""The one non-integer column: its rules keep the float threshold."""
KEY_LEAVES = 16
"""Leaves of the key tree printed when it never reaches L90 (the rules-tree size)."""


class Condition(NamedTuple):
    """One feature restricted to an inclusive interval; None: unbounded.

    Bounds are integers, except on `_FRACTIONAL` columns where they are the float32
    values that reproduce the tree's cut exactly.
    """

    feature: str
    lo: float | None
    hi: float | None


class Rule(NamedTuple):
    """A leaf: the conditions that lead to it and what it plays."""

    conditions: tuple[Condition, ...]
    action: int | str
    """Primitive action id of a policy tree, skill name of a key tree."""
    samples: int
    """Training rows in the leaf."""
    share: float
    """``samples`` over the tree's training rows."""
    purity: float
    """Share of the leaf's rows that carry its action."""


class TopLevel(NamedTuple):
    """The top level of a skill method at one checkpoint (see `top_level_rules`)."""

    update: int
    n_rules: int
    """Visited keys, each one a rule."""
    skills: tuple[str, ...]
    """Distinct greedy skills, most frequent first."""
    counts: dict[str, int]
    """Keys per greedy skill."""
    levels: dict[str, int | None]
    """`distill.complexity` of the key tree: L80, L90, L95 over keys, depth_L90."""
    curve: list[Row]
    """Key trees of the leaf grid: size, leaves, depth, top1 (on the keys)."""
    rules: list[Rule]
    """Leaves of the tree printed (the L90 tree, else the 16-leaf one)."""
    tree_text: str
    top: list[Row]
    """The highest-value keys: key, skill, value and the rule in words."""
    n_optimistic: int
    """Keys whose greedy skill is not their greedy tried skill, so the argmax landed
    on an untried optimistic entry (counted on all keys of the checkpoint)."""
    optimistic_share: float
    """``n_optimistic`` over all keys of the checkpoint (0 without keys)."""


def action_id(action: int | str) -> int:
    """The primitive action id of an id or an action name (any case).

    Raises:
        ValueError: If a name is not an action of the game.
    """
    if isinstance(action, str):
        try:
            return int(Action[action.upper()].value)
        except KeyError:
            msg = f"unknown action name {action!r}"
            raise ValueError(msg) from None
    return int(action)


def _game(columns: Sequence[str]) -> GameKind:
    """Classic when every observation column is one of its 60, else Craftax."""
    own = set(columns) - set(MEMORY_FEATURES) - set(SKILL_FEATURES)
    return "classic" if own <= set(FEATURES["classic"]) else "craftax"


def _codes(feature: str, game: GameKind) -> dict[int, str] | None:
    """Value -> name of an ordinal column; None for a numeric one."""
    n_actions = 17 if game == "classic" else len(Action)
    if feature == "direction":
        return {1: "left", 2: "right", 3: "up", 4: "down"}
    if feature == "facing_creature":
        return dict(enumerate(CREATURES))
    if feature == "facing_block":
        blocks = [b for b in BlockType if game == "craftax" or b.value < 17]
        names = {b.value: b.name.lower() for b in blocks}
        return names | {0: "unlit/invalid"}  # an unlit tile decodes as block 0
    if feature == "prev_action":
        return {-1: "start", **{i: _NAMES[i] for i in range(n_actions)}}
    return next((c for s, c in _SIGNS.items() if feature.endswith(s)), None)


def _join(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


def condition_text(c: Condition, game: GameKind = "craftax") -> str:
    """A condition in words and decoded units; memory proxies get a suffix.

    Ordinal columns name the codes they select (the smaller of the set and its
    complement), flags print as their name, other columns as an integer comparison.

    Raises:
        ValueError: If an ordinal column's interval selects none of its codes.
    """
    text = _text(c, game)
    return f"{text} [memory proxy]" if c.feature in MEMORY_FEATURES else text


def _text(c: Condition, game: GameKind) -> str:
    f, lo, hi = c
    codes = _codes(f, game)
    if codes is not None:
        chosen = [
            n
            for v, n in codes.items()
            if (lo is None or v >= lo) and (hi is None or v <= hi)
        ]
        rest = [n for n in codes.values() if n not in chosen]
        if not chosen:
            msg = f"{f} in [{lo}, {hi}] selects none of its codes"
            raise ValueError(msg)
        return (
            f"{f} is not {_join(rest)}"
            if len(rest) < len(chosen)
            else f"{f} is {_join(chosen)}"
        )
    if f in SKILL_FEATURES[:-1]:  # the running-skill one-hot
        return f"skill is {'' if lo else 'not '}{f.removeprefix('skill_')}"
    if f in _FLAGS:
        return f if lo else f"not {f}"
    low, high = (None if b is None else _number(f, b) for b in (lo, hi))
    if low is None:
        return f"{f} <= {high}"
    if high is None:
        return f"{f} >= {low}"
    return f"{f} = {low}" if lo == hi else f"{low} <= {f} <= {high}"


def _number(feature: str, v: float) -> str:
    """An integer, or for a fractional column the shortest float32 text of ``v``."""
    if feature in _FRACTIONAL:
        return str(np.format_float_positional(np.float32(v), trim="-"))
    return str(int(v))


def _merge(path: Sequence[Condition]) -> tuple[Condition, ...]:
    """One interval per feature, in order of first appearance on the path."""
    bounds: dict[str, list[float | None]] = {}
    for f, lo, hi in path:
        b = bounds.setdefault(f, [None, None])
        if lo is not None:
            b[0] = lo if b[0] is None else max(b[0], lo)
        if hi is not None:
            b[1] = hi if b[1] is None else min(b[1], hi)
    return tuple(Condition(f, *b) for f, b in bounds.items())


def _extract(model: DecisionTreeClassifier, columns: Sequence[str]) -> list[Rule]:
    """Leaves of ``model``, left to right.

    An integer column is cut at the floor of the threshold; a fractional one keeps the
    float32 values on either side of it, so a rule selects what the tree sends there.
    """
    t = model.tree_
    total = int(t.n_node_samples[0])
    rules: list[Rule] = []
    stack: list[tuple[int, tuple[Condition, ...]]] = [(0, ())]
    while stack:
        node, path = stack.pop()
        left, right = int(t.children_left[node]), int(t.children_right[node])
        if left == -1:
            counts = t.value[node, 0]
            n = int(t.n_node_samples[node])
            action = model.classes_[counts.argmax()].item()
            purity = float(counts.max() / counts.sum())
            rules.append(Rule(_merge(path), action, n, n / total, purity))
            continue
        feature, cut = columns[t.feature[node]], float(t.threshold[node])
        below, above = _cut(feature, cut)
        stack.append((right, (*path, Condition(feature, above, None))))
        stack.append((left, (*path, Condition(feature, None, below))))
    return rules


def _cut(feature: str, t: float) -> tuple[float, float]:
    """Largest value the left child takes and smallest the right child takes."""
    if feature not in _FRACTIONAL:
        return math.floor(t), math.floor(t) + 1
    f, inf = np.float32(t), np.float32(np.inf)  # the tree compares float32 features
    below = f if f <= t else np.nextafter(f, -inf)
    above = f if f > t else np.nextafter(f, inf)
    return float(below), float(above)


def extract_rules(tree: FittedTree) -> list[Rule]:
    """Every leaf of a fitted tree as a rule, left to right."""
    return _extract(tree.model, tree.columns)


def _action_name(action: int | str) -> str:
    return action if isinstance(action, str) else _NAMES.get(action, f"ACTION_{action}")


def _line(i: int, rule: Rule, game: GameKind, unit: str) -> str:
    then = (
        f"{_action_name(rule.action)} "
        f"({rule.share:.1%} of {unit}, purity {rule.purity:.2f})"
    )
    if not rule.conditions:
        return f"{i}. ALWAYS {then}"
    ifs = " AND ".join(condition_text(c, game) for c in rule.conditions)
    return f"{i}. IF {ifs} THEN {then}"


def rules_text(tree: FittedTree, game: GameKind | None = None) -> str:
    """One numbered line per leaf: conditions, action, share of states and purity.

    ``game`` defaults to the one the tree's columns belong to; it sets the block
    types a ``facing_block`` condition can name.
    """
    game = game or _game(tree.columns)
    return "\n".join(
        _line(i, r, game, "states") for i, r in enumerate(extract_rules(tree), 1)
    )


def paths_to(tree: FittedTree, action: int | str) -> list[Rule]:
    """Every leaf that plays ``action`` (an id or a name such as ``"DO"``).

    Raises:
        ValueError: If ``action`` is an unknown name.
    """
    want = action_id(action)
    return [r for r in extract_rules(tree) if r.action == want]


def _x(ds: Dataset, tree: FittedTree) -> np.ndarray:
    """The rows of ``ds`` in the tree's columns, matched by name (a fresh copy)."""
    return ds.features[:, [ds.columns.index(c) for c in tree.columns]]


def _group_of(column: str) -> str:
    if column in MEMORY_FEATURES:
        return "memory"
    return "skill" if column in SKILL_FEATURES else "observation"


def importance(
    tree: FittedTree, test: Dataset, repeats: int = 5, seed: int = 0
) -> list[Row]:
    """Permutation importance of every column on the test states, largest first.

    Each column the tree splits on is shuffled ``repeats`` times; ``importance`` is
    the drop in top-1 fidelity (mean over shuffles), ``std`` its spread. A column
    the tree never reads cannot matter and is 0 without a shuffle. Rows carry the
    ``group`` (observation, memory or skill) so memory proxies and the current skill
    are reported apart.

    Raises:
        ValueError: If ``test`` has no states.
    """
    if not len(test.label):
        msg = "importance needs test states"
        raise ValueError(msg)
    model, x = tree.model, _x(test, tree)
    base = float((model.predict(x) == test.label).mean())
    drops: dict[int, tuple[float, float]] = {}
    rng = np.random.default_rng(seed)
    for j in sorted(int(f) for f in set(model.tree_.feature) if f >= 0):
        original = x[:, j].copy()
        scores = []
        for _ in range(repeats):
            x[:, j] = rng.permutation(original)
            scores.append(float((model.predict(x) == test.label).mean()))
        x[:, j] = original
        drops[j] = (base - float(np.mean(scores)), float(np.std(scores)))
    rows = [
        {
            "feature": c,
            "group": _group_of(c),
            "importance": drops.get(j, (0.0, 0.0))[0],
            "std": drops.get(j, (0.0, 0.0))[1],
        }
        for j, c in enumerate(tree.columns)
    ]
    return sorted(rows, key=lambda r: -r["importance"])


def failure_breakdown(tree: FittedTree, test: Dataset) -> dict[str, Any]:
    """Where a tree departs from the network on the test states.

    ``confusion[i, j]`` counts states where the network's action is in group ``i``
    and the tree's in group ``j`` (`ACTION_GROUPS` order, ``groups``); ``by_floor``
    holds the states and top-1 fidelity per floor, floors ascending; ``top1`` is the
    overall fidelity.
    """
    pred = np.asarray(tree.model.predict(_x(test, tree)))
    hit = pred == test.label
    n = len(ACTION_GROUPS)
    confusion = np.zeros((n, n), np.int64)
    np.add.at(confusion, (_GROUP_OF[test.label], _GROUP_OF[pred]), 1)
    floors = [
        {
            "floor": int(f),
            "states": int((m := test.floor == f).sum()),
            "top1": float(hit[m].mean()),
        }
        for f in np.unique(test.floor)
    ]
    return {
        "top1": float(hit.mean()),
        "groups": tuple(ACTION_GROUPS),
        "confusion": confusion,
        "by_floor": floors,
    }


def _key_words(fields: Mapping[str, Any]) -> str:
    """A key in words: floor, flags that are set and counts that are not zero."""
    parts = []
    for name, v in fields.items():
        v = int(v)
        if name == "floor":
            parts.append(f"floor {v}")
        elif name in _FLAGS:
            parts += [name] * bool(v)
        elif v:
            parts.append(f"{name}={v}")
    return ", ".join(parts) or "an empty key"


def _fit_keys(
    x: np.ndarray, y: np.ndarray
) -> tuple[list[Row], dict[int, DecisionTreeClassifier]]:
    """Key trees of the leaf grid, scored on the keys themselves, until one is exact.

    Every visited key is a rule of the top level, so there is no held-out set: the
    tree is a compression of the table and ``min_samples_leaf`` is 1.
    """
    curve: list[Row] = []
    models: dict[int, DecisionTreeClassifier] = {}
    for size in LEAF_GRID:
        model = DecisionTreeClassifier(
            criterion="gini", max_leaf_nodes=size, random_state=0
        ).fit(x, y)
        top1 = float((model.predict(x) == y).mean())
        models[size] = model
        curve.append({
            "kind": "leaves",
            "size": size,
            "leaves": int(model.get_n_leaves()),
            "depth": int(model.get_depth()),
            "top1": top1,
        })
        if top1 == 1.0:  # a larger tree is the same tree
            break
    return curve, models


def top_level_rules(
    table: pd.DataFrame | str | Path,
    update: int | None = None,
    tried: bool = False,
    top: int = 10,
) -> TopLevel:
    """The top-level rules of a skill method from its greedy table.

    ``table`` is the frame or parquet file of ``eval/smdp_greedy_table.parquet``,
    which holds every checkpoint: rows are filtered to ``update`` (default the
    largest, the final checkpoint; pass it when the final one stored no keys). Each
    visited key is one rule; reported are their number, the distinct greedy skills,
    a tree over the decoded key fields predicting the skill (its L80, L90 and L95
    over the keys) and the ``top`` highest-value keys in words. The greedy skill is
    the unmasked argmax of the key's row, so an untried optimistic entry can win:
    ``tried=True`` uses the best tried skill and its value instead, dropping keys
    without one. At ``q_init`` 10 most final keys were never updated off it, so the
    argmax is a tie broken by skill order: the tried view is the informative one and
    ``n_optimistic`` / ``optimistic_share`` say how many keys the default misreads.

    Raises:
        ValueError: If the table has no rows.
    """
    df = table if isinstance(table, pd.DataFrame) else pd.read_parquet(table)
    if df.empty:
        msg = "the greedy table has no rows"
        raise ValueError(msg)
    update = int(df["update"].max()) if update is None else update
    skill, value = (
        ("greedy_tried_skill", "greedy_tried_value")
        if tried
        else ("greedy_skill", "value")
    )
    df = df[df["update"] == update]
    n_optimistic = int((df["greedy_skill"] != df["greedy_tried_skill"]).sum())
    share = n_optimistic / len(df) if len(df) else 0.0
    df = df[df[skill] != ""]  # no tried entry
    fields = [c for c in df.columns if c not in _KEY_META]
    counts = Counter(df[skill])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    curve: list[Row] = []
    shown: list[Rule] = []
    models: dict[int, DecisionTreeClassifier] = {}
    if len(df):
        curve, models = _fit_keys(
            df[fields].to_numpy(np.float32), df[skill].to_numpy(str)
        )
    levels = complexity(curve)
    if curve:
        shown = _extract(models[levels["L90"] or KEY_LEAVES], fields)
    order = np.lexsort((df["key"].to_numpy(), -df[value].to_numpy()))[:top]
    best = [
        {
            "key": int(r["key"]),
            "skill": r[skill],
            "value": float(r[value]),
            "text": f"{r[skill]} (value {r[value]:.2f}) when "
            f"{_key_words({f: r[f] for f in fields})}",
        }
        for _, r in df.iloc[order].iterrows()
    ]
    text = "\n".join(_line(i, r, "craftax", "keys") for i, r in enumerate(shown, 1))
    return TopLevel(
        update=update,
        n_rules=len(df),
        skills=tuple(s for s, _ in ranked),
        counts=dict(ranked),
        levels=levels,
        curve=curve,
        rules=shown,
        tree_text=text,
        top=best,
        n_optimistic=n_optimistic,
        optimistic_share=share,
    )
