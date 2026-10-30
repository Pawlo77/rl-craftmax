"""Tests for MLflow tracking URI handling."""

from pathlib import Path

import pytest

from wut_murl_craftax.orchestration.tracking import _resolve_tracking_uri


def test_sqlite_busy_timeout_survives_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``?timeout=`` query stays outside the resolved database path."""
    monkeypatch.chdir(tmp_path)
    uri = _resolve_tracking_uri("sqlite:///mlruns.db?timeout=300")
    db = (tmp_path / "mlruns.db").resolve().as_posix()
    assert uri == f"sqlite:///{db}?timeout=300"
