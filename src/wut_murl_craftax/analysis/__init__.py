"""Phase 4 analysis: run loading, training analysis, cases and method comparison."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real names for type checkers; runtime imports stay lazy
    from .load import (
        BASE_CONTROL,
        CONTROLS,
        FINAL_PHASES,
        METRICS_ROOT,
        UNIT_COLUMNS,
        Frames,
        MissingPhaseError,
        SchemaError,
        base_control,
        final_checkpoint,
        in_phase,
        load_phase,
        load_phases,
        pair_of,
        select,
    )

_EXPORTS: dict[str, str] = {
    "BASE_CONTROL": "load",
    "CONTROLS": "load",
    "FINAL_PHASES": "load",
    "Frames": "load",
    "METRICS_ROOT": "load",
    "MissingPhaseError": "load",
    "SchemaError": "load",
    "UNIT_COLUMNS": "load",
    "base_control": "load",
    "final_checkpoint": "load",
    "in_phase": "load",
    "load_phase": "load",
    "load_phases": "load",
    "pair_of": "load",
    "select": "load",
}
"""Public name -> submodule; imported on first access so ``python -m`` runs of a
submodule do not import it twice."""

__all__ = [
    "BASE_CONTROL",
    "CONTROLS",
    "FINAL_PHASES",
    "METRICS_ROOT",
    "UNIT_COLUMNS",
    "Frames",
    "MissingPhaseError",
    "SchemaError",
    "base_control",
    "final_checkpoint",
    "in_phase",
    "load_phase",
    "load_phases",
    "pair_of",
    "select",
]


def __getattr__(name: str) -> object:
    """Import a public name from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(importlib.import_module(f".{module}", __name__), name)
