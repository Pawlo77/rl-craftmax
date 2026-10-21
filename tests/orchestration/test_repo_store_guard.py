"""The tests never use the repo's MLflow store; the CLIs refuse it under pytest."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _fake_trainer import FakeTrainer
from conftest import ROOT, store_listing
from test_runner import PROFILE, Store

from wut_murl_craftax.orchestration import queue, runner
from wut_murl_craftax.orchestration import store as store_cli
from wut_murl_craftax.orchestration.tracking import (
    UNDER_PYTEST_ENV,
    _resolve_tracking_uri,
)


def test_the_session_isolates_mlflow_from_the_repo() -> None:
    uri = os.environ["MLFLOW_TRACKING_URI"]
    assert uri.startswith("sqlite:///")
    assert not Path(uri.removeprefix("sqlite:///")).is_relative_to(ROOT)
    assert os.environ[UNDER_PYTEST_ENV] == "1"
    assert "MLFLOW_RUN_ID" not in os.environ
    assert "MLFLOW_ACTIVE_RUN_ID" not in os.environ


@pytest.mark.parametrize("uri", ["sqlite:///mlruns.db", "mlruns", "sqlite:///x/y.db"])
def test_a_store_inside_the_repo_is_refused(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    with pytest.raises(RuntimeError, match="inside the repo under pytest"):
        _resolve_tracking_uri(uri)
    assert not (ROOT / "x").exists()


def test_a_store_outside_the_repo_is_fine(tmp_path: Path) -> None:
    got = _resolve_tracking_uri(f"sqlite:///{tmp_path / 'm.db'}")
    assert got.endswith("m.db")


def test_every_cli_refuses_the_shipped_repo_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    with pytest.raises(RuntimeError, match="inside the repo"):
        queue.coarse_results("ppo")
    with pytest.raises(RuntimeError, match="inside the repo"):
        store_cli._main_uri(None)
    harness = Store(tmp_path, monkeypatch)
    specs = harness.phase("tune-coarse", ["published"])
    trainer = FakeTrainer()
    with pytest.raises(RuntimeError, match="inside the repo"):
        runner.main(
            specs,
            PROFILE,
            experiment_config=queue.EXPERIMENT_CONFIG,
            methods={"toy": trainer},
            evaluate=harness.evaluate,
        )
    assert trainer.updates == []


def test_a_subprocess_cli_inherits_the_refusal() -> None:
    code = "from wut_murl_craftax.orchestration.queue import _client; _client(None)"
    done = subprocess.run(  # fixed interpreter and code
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "inside the repo under pytest" in done.stderr


def test_the_guard_listing_sees_changed_added_and_removed_store_files(
    tmp_path: Path,
) -> None:
    assert store_listing(tmp_path) == {}  # no store: nothing to guard
    (tmp_path / "mlruns.db").write_bytes(b"x")
    (tmp_path / "mlruns" / "e" / "r").mkdir(parents=True)
    ckpt = tmp_path / "mlruns" / "e" / "r" / "00001.msgpack"
    ckpt.write_bytes(b"a")
    before = store_listing(tmp_path)
    assert set(before) == {"mlruns.db", "mlruns/e/r/00001.msgpack"}
    ckpt.write_bytes(b"ab")
    assert store_listing(tmp_path) != before
    ckpt.unlink()
    assert set(store_listing(tmp_path)) == {"mlruns.db"}
