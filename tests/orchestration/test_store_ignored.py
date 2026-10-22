"""The runner's MLflow store and the merge-mlflow node stores stay untracked."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path",
    ["mlruns.db", "mlruns-node-3.db", "mlruns-node-3.db.array.json", "mlruns/smoke/x"],
)
def test_store_files_are_git_ignored(path: str) -> None:
    done = subprocess.run(  # fixed git command on a fixed path
        ["git", "check-ignore", "--quiet", path],
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, f"{path} is not ignored: a commit could add the store"
