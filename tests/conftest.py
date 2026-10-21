"""Pytest configuration: tests never touch the repo's MLflow store.

A live ``make queue`` writes ``mlruns.db`` and ``mlruns/`` in the repo root. The tests
get a throwaway store instead (`isolated_mlflow`), the runner, queue and store CLIs
refuse the repo-root store under pytest (``orchestration.tracking``), and
`repo_store_untouched` fails the session if the real store changed meanwhile (a live
queue does that too: set ``WUT_MURL_STORE_GUARD=off`` to run tests beside one).
"""

import os
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
