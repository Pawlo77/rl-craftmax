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

_EXPORTS: dict[str, str] = {
    "CREATURES": "features",
    "DEPTH_GRID": "distill",
    "Dataset": "dataset",
    "FEATURES": "features",
    "FittedTree": "distill",
    "LEAF_GRID": "distill",
    "LEVELS": "distill",
    "MAX_STATES": "dataset",
    "MEMORY_FEATURES": "dataset",
    "RECURRENT": "dataset",
    "SKILL_FEATURES": "dataset",
    "SPLITS": "dataset",
    "TEST_SEEDS": "dataset",
    "TRAIN_SEEDS": "dataset",
    "VAL_SEEDS": "dataset",
    "collect": "dataset",
    "complexity": "distill",
    "dataset_path": "dataset",
    "feature_columns": "distill",
    "featurize": "features",
    "fidelity": "distill",
    "fidelity_curve": "distill",
    "fit_grid": "distill",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "CREATURES",
    "DEPTH_GRID",
    "FEATURES",
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
    "Dataset",
    "FittedTree",
    "collect",
    "complexity",
    "dataset_path",
    "feature_columns",
    "featurize",
    "fidelity",
    "fidelity_curve",
    "fit_grid",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
