"""Tests for the code version the runner tags: real git checkouts, no mocks."""

import subprocess
from pathlib import Path

import pytest

from wut_murl_craftax.orchestration.reproducibility import code_version

_GIT_CONFIG = ("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false")
_GIT_CONFIG += ("-c", "user.name=t", "-c", "user.email=t@example.com")


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *_GIT_CONFIG, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """A git worktree on a branch whose ref is packed (as after ``git gc``)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "worktree", "add", "-q", "-b", "fix", str(tmp_path / "wt"))
    _git(repo, "pack-refs", "--all")
    return tmp_path / "wt"


def test_a_worktree_with_packed_refs_has_its_commit_and_dirty_flag(
    worktree: Path,
) -> None:
    commit = _git(worktree, "rev-parse", "HEAD")
    assert len(commit) == 40
    assert code_version(worktree) == (commit, False)
    (worktree / "a.txt").write_text("b\n")
    assert code_version(worktree) == (commit, True)
