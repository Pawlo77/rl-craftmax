"""Policy interpretation: features, datasets, decision-tree distillation and rules."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .features import CREATURES, FEATURES, featurize

_EXPORTS: dict[str, str] = {
    "CREATURES": "features",
    "FEATURES": "features",
    "featurize": "features",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = ["CREATURES", "FEATURES", "featurize"]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
