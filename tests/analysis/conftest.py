"""Fixtures of the synthetic export (`_export`), shared by every analysis test."""

from collections.abc import Callable
from pathlib import Path

import pytest
from _export import write_export


@pytest.fixture(scope="session")
def export_writer() -> Callable[..., Path]:
    """`write_export`, for tests that need another phase set or a damaged export."""
    return write_export


@pytest.fixture(scope="session")
def export_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export of ``final-1m``, ``combo-1m`` and ``tune-coarse``; read-only."""
    return write_export(tmp_path_factory.mktemp("metrics"))
