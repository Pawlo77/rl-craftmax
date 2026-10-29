"""Pytest configuration: tests never touch the repo's MLflow store.

A live ``make queue`` writes ``mlruns.db`` and ``mlruns/`` in the repo root. The tests
get a throwaway store instead (`isolated_mlflow`), the runner, queue and store CLIs
refuse the repo-root store under pytest (``orchestration.tracking``), and
`repo_store_untouched` fails the session if the real store changed meanwhile (a live
queue does that too: set ``WUT_MURL_STORE_GUARD=off`` to run tests beside one).
"""

import gc
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from wut_murl_craftax.orchestration.tracking import UNDER_PYTEST_ENV

ROOT = Path(__file__).resolve().parents[1]
GUARD_ENV = "WUT_MURL_STORE_GUARD"
_STORE_FILES = ("mlruns.db", "mlruns.db-wal", "mlruns.db-shm", "mlruns.db-journal")


def store_listing(root: Path = ROOT) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of the repo's MLflow database and mlruns/."""
    paths = [root / name for name in _STORE_FILES]
    if (root / "mlruns").is_dir():
        paths += [p for p in (root / "mlruns").rglob("*") if p.is_file()]
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
        if p.is_file()
    }


def pytest_configure(config: pytest.Config) -> None:
    """Build Craftax's texture caches once, before any xdist worker starts.

    Craftax writes them inside the installed package on first use, so in a fresh
    virtual environment (CI) parallel workers would read a cache that another worker
    is still writing.
    A subprocess keeps the JAX import out of this process (XLA_FLAGS stay open).
    """
    if hasattr(config, "workerinput"):  # an xdist worker: the controller already warmed
        return
    env = os.environ | {"JAX_PLATFORMS": "cpu"}
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.runtime"]  # make warm
    subprocess.run(cmd, check=True, capture_output=True, env=env)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Iterator[None]:
    """Drop JAX's compiled executables once a test file is done.

    Every config a test compiles stays cached for the life of the process: an xdist
    worker grew past 12 GB over its share of the suite, more than a CI runner holds.
    Clearing per file keeps the executables a file's tests share (clearing after
    every test made the serial suite 36% slower).
    """
    yield
    jax = sys.modules.get("jax")
    if jax is not None and (nextitem is None or nextitem.path != item.path):
        jax.clear_caches()
        gc.collect()


@pytest.fixture(autouse=True, scope="session")
def isolated_mlflow(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """A temporary MLflow store for any code that reads the environment."""
    tmp = tmp_path_factory.mktemp("mlflow")
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp / 'mlruns.db'}")
    # the default artifact root of a store created without an explicit location
    mp.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", (tmp / "artifacts").as_uri())
    mp.delenv("MLFLOW_RUN_ID", raising=False)
    mp.delenv("MLFLOW_ACTIVE_RUN_ID", raising=False)
    mp.setenv(UNDER_PYTEST_ENV, "1")  # inherited by the CLIs tests start
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _restore_mlflow_env() -> Iterator[None]:
    """Undo what MLflow's fluent setters write into ``os.environ`` during a test.

    ``set_tracking_uri`` and ``set_experiment`` export MLFLOW_TRACKING_URI and
    MLFLOW_EXPERIMENT_ID for subprocesses; without this they outlive the test and
    reach later tests and the subprocesses those start.
    """
    keys = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID")
    before = {k: os.environ.get(k) for k in keys}
    yield
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True, scope="session")
def repo_store_untouched() -> Iterator[None]:
    """Fail the session if a test changed the repo's ``mlruns.db`` or ``mlruns/``."""
    before = store_listing()
    yield
    if os.environ.get(GUARD_ENV) == "off":
        return
    after = store_listing()
    changed = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    if changed:
        shown = ", ".join(changed[:10])
        msg = (
            f"the tests changed the repo's MLflow store ({len(changed)} paths): {shown}"
        )
        raise AssertionError(msg)
