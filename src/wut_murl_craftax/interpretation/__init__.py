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
    from .features import CREATURES, FEATURES, featurize

_EXPORTS: dict[str, str] = {
    "CREATURES": "features",
    "Dataset": "dataset",
    "FEATURES": "features",
    "MAX_STATES": "dataset",
    "MEMORY_FEATURES": "dataset",
    "RECURRENT": "dataset",
    "SKILL_FEATURES": "dataset",
    "SPLITS": "dataset",
    "TEST_SEEDS": "dataset",
    "TRAIN_SEEDS": "dataset",
    "VAL_SEEDS": "dataset",
    "collect": "dataset",
    "dataset_path": "dataset",
    "featurize": "features",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "CREATURES",
    "FEATURES",
    "MAX_STATES",
    "MEMORY_FEATURES",
    "RECURRENT",
    "SKILL_FEATURES",
    "SPLITS",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "VAL_SEEDS",
    "Dataset",
    "collect",
    "dataset_path",
    "featurize",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
