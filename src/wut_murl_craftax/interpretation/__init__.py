"""Policy interpretation: features, datasets, decision-tree distillation and rules."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .dataset import (
        MAX_STATES,
        MEMORY_FEATURES,
        RECURRENT,
        SKILL_FEATURES,
        SPLITS,
        TEST_SEEDS,
        TRAIN_SEEDS,
        VAL_SEEDS,
        Dataset,
        collect,
        dataset_path,
        relabel,
    )
    from .distill import (
        DEPTH_GRID,
        LEAF_GRID,
        LEVELS,
        FittedTree,
        complexity,
        feature_columns,
        fidelity,
        fidelity_curve,
        fit_grid,
    )
    from .features import CREATURES, FEATURES, featurize
    from .rules import (
        ACTION_GROUPS,
        KEY_LEAVES,
        Condition,
        Rule,
        TopLevel,
        action_id,
        condition_text,
        extract_rules,
        failure_breakdown,
        importance,
        paths_to,
        rules_text,
        top_level_rules,
    )
    from .tree_policy import (
        CLOSED_LOOP_LEAVES,
        ClosedLoop,
        TreeCarry,
        TreePolicy,
        closed_loop,
        tree_rollouts,
    )

_EXPORTS: dict[str, str] = {
    "ACTION_GROUPS": "rules",
    "CLOSED_LOOP_LEAVES": "tree_policy",
    "CREATURES": "features",
    "ClosedLoop": "tree_policy",
    "Condition": "rules",
    "DEPTH_GRID": "distill",
    "Dataset": "dataset",
    "FEATURES": "features",
    "FittedTree": "distill",
    "KEY_LEAVES": "rules",
    "LEAF_GRID": "distill",
    "LEVELS": "distill",
    "MAX_STATES": "dataset",
    "MEMORY_FEATURES": "dataset",
    "RECURRENT": "dataset",
    "Rule": "rules",
    "SKILL_FEATURES": "dataset",
    "SPLITS": "dataset",
    "TEST_SEEDS": "dataset",
    "TRAIN_SEEDS": "dataset",
    "TopLevel": "rules",
    "TreeCarry": "tree_policy",
    "TreePolicy": "tree_policy",
    "VAL_SEEDS": "dataset",
    "action_id": "rules",
    "closed_loop": "tree_policy",
    "collect": "dataset",
    "complexity": "distill",
    "condition_text": "rules",
    "dataset_path": "dataset",
    "extract_rules": "rules",
    "failure_breakdown": "rules",
    "feature_columns": "distill",
    "featurize": "features",
    "fidelity": "distill",
    "fidelity_curve": "distill",
    "fit_grid": "distill",
    "importance": "rules",
    "paths_to": "rules",
    "relabel": "dataset",
    "rules_text": "rules",
    "top_level_rules": "rules",
    "tree_rollouts": "tree_policy",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "ACTION_GROUPS",
    "CLOSED_LOOP_LEAVES",
    "CREATURES",
    "DEPTH_GRID",
    "FEATURES",
    "KEY_LEAVES",
    "LEAF_GRID",
    "LEVELS",
    "MAX_STATES",
    "MEMORY_FEATURES",
    "RECURRENT",
    "SKILL_FEATURES",
    "SPLITS",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "VAL_SEEDS",
    "ClosedLoop",
    "Condition",
    "Dataset",
    "FittedTree",
    "Rule",
    "TopLevel",
    "TreeCarry",
    "TreePolicy",
    "action_id",
    "closed_loop",
    "collect",
    "complexity",
    "condition_text",
    "dataset_path",
    "extract_rules",
    "failure_breakdown",
    "feature_columns",
    "featurize",
    "fidelity",
    "fidelity_curve",
    "fit_grid",
    "importance",
    "paths_to",
    "relabel",
    "rules_text",
    "top_level_rules",
    "tree_rollouts",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
