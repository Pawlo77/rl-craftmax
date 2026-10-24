"""Fixtures of the synthetic export (`_analysis_export`) for the analysis tests."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from _analysis_export import write_export

# the case tests replay the tiny eval runs of the visualisation tests (`_learning_run`)
sys.path.append(str(Path(__file__).parents[1] / "visualisation"))


@pytest.fixture(scope="session")
def export_writer() -> Callable[..., Path]:
    """`write_export`, for tests that need another phase set or a damaged export."""
    return write_export


@pytest.fixture(scope="session")
def export_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export of ``final-1m``, ``combo-1m`` and ``tune-coarse``; read-only."""
    return write_export(tmp_path_factory.mktemp("metrics"))
