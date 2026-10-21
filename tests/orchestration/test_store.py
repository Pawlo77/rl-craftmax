"""Tests for the run status of a phase and the merge of node MLflow stores."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pytest

from wut_murl_craftax.orchestration import queue
from wut_murl_craftax.orchestration.store import merge_stores, phase_status
from wut_murl_craftax.orchestration.tracking import _artifact_location, artifact_path

CONFIGS = Path(__file__).parent / "fixtures" / "configs"
METHODS = CONFIGS / "methods"
ENTRIES = CONFIGS / "phases" / "entries.yaml"
REPO = Path(__file__).resolve().parents[2]


class Node:
    """A throwaway SQLite store whose artifacts live beside its database."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True)
        self.uri = f"sqlite:///{root / 'mlruns.db'}"
        self.client = mlflow.MlflowClient(tracking_uri=self.uri)

    def add(
        self,
        name: str,
        phases: str,
        *,
        experiment: str = "tune-coarse",
        status: str = "FINISHED",
        files: dict[str, str] | None = None,
        metrics: dict[str, list[float]] | None = None,
    ) -> str:
        exp = self.client.get_experiment_by_name(experiment)
        exp_id = (
            exp.experiment_id
            if exp
            else self.client.create_experiment(
                experiment, artifact_location=_artifact_location(self.uri, experiment)
            )
        )
        run = self.client.create_run(
            exp_id, tags={"phases": phases, "method": "toy"}, run_name=name
        )
        run_id = run.info.run_id
        self.client.log_param(run_id, "lr", "0.0003")
        for key, values in (metrics or {}).items():
            for step, value in enumerate(values):
                self.client.log_metric(run_id, key, value, step=step)
        out = artifact_path(run.info.artifact_uri)
        for rel, text in (files or {"config.json": "{}"}).items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(text)
        self.client.set_terminated(run_id, status)
        return run_id

    def runs(self) -> dict[str, Any]:
        """Every active run by name; a duplicate fails the test."""
        ids = [e.experiment_id for e in self.client.search_experiments()]
        runs = self.client.search_runs(ids)
        assert len({r.info.run_name for r in runs}) == len(runs), "duplicate run"
        return {r.info.run_name: r for r in runs}


@pytest.fixture
def main_store(tmp_path: Path) -> Node:
    return Node(tmp_path / "main")


@pytest.fixture
def node(tmp_path: Path) -> Node:
    return Node(tmp_path / "node")


def _phases(run: Any) -> set[str]:
    return set(run.data.tags["phases"].split(","))


# --- merge ----------------------------------------------------------------------


def test_merge_copies_a_run_with_its_history_and_artifacts(
    main_store: Node, node: Node
) -> None:
    node.add(
        "ppo-1m-aaaaaaaa-s0",
        "tune-coarse",
        files={"config.json": "{}", "ckpt/00000010.npz": "p", "eval/a.json": "e"},
        metrics={"train/return": [1.0, 2.0, 3.0]},
    )
    counts = merge_stores(node.uri, main_store.uri)
    assert counts == {"copied": 1, "joined": 0}
    run = main_store.runs()["ppo-1m-aaaaaaaa-s0"]
    assert run.info.status == "FINISHED"
    assert run.data.params == {"lr": "0.0003"}
    assert run.data.tags["phases"] == "tune-coarse"
    history = main_store.client.get_metric_history(run.info.run_id, "train/return")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 2.0), (2, 3.0)]
    out = artifact_path(run.info.artifact_uri)
    assert out.is_relative_to(main_store.root)
    assert (out / "ckpt" / "00000010.npz").read_text() == "p"
    assert (out / "eval" / "a.json").read_text() == "e"
    experiment = main_store.client.get_experiment(run.info.experiment_id)
    assert experiment.name == "tune-coarse"


def test_merging_two_nodes_and_again_leaves_no_duplicates(
    main_store: Node, tmp_path: Path
) -> None:
    first, second = Node(tmp_path / "n1"), Node(tmp_path / "n2")
    first.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    second.add("a-1m-aaaaaaaa-s1", "tune-coarse")
    second.add("b-1m-bbbbbbbb-s0", "finals", experiment="finals")
    for _ in range(2):
        merge_stores(first.uri, main_store.uri)
        merge_stores(second.uri, main_store.uri)
    assert set(main_store.runs()) == {
        "a-1m-aaaaaaaa-s0",
        "a-1m-aaaaaaaa-s1",
        "b-1m-bbbbbbbb-s0",
    }


def test_a_run_in_both_stores_merges_its_phases(main_store: Node, node: Node) -> None:
    main_store.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    node.add("a-1m-aaaaaaaa-s0", "finals,tune-coarse", experiment="finals")
    counts = merge_stores(node.uri, main_store.uri)
    assert counts == {"copied": 0, "joined": 1}
    assert _phases(main_store.runs()["a-1m-aaaaaaaa-s0"]) == {"tune-coarse", "finals"}


def test_a_finished_node_run_replaces_a_killed_main_run(
    main_store: Node, node: Node
) -> None:
    main_store.add("a-1m-aaaaaaaa-s0", "tune-coarse", status="KILLED")
    node.add(
        "a-1m-aaaaaaaa-s0",
        "finals",
        files={"config.json": "{}", "ckpt/last.npz": "done"},
    )
    merge_stores(node.uri, main_store.uri)
    run = main_store.runs()["a-1m-aaaaaaaa-s0"]
    assert run.info.status == "FINISHED"
    assert _phases(run) == {"tune-coarse", "finals"}
    assert (artifact_path(run.info.artifact_uri) / "ckpt" / "last.npz").exists()


def test_a_killed_node_run_never_replaces_a_finished_main_run(
    main_store: Node, node: Node
) -> None:
    main_store.add("a-1m-aaaaaaaa-s0", "tune-coarse", files={"config.json": "main"})
    node.add("a-1m-aaaaaaaa-s0", "finals", status="KILLED")
    counts = merge_stores(node.uri, main_store.uri)
    run = main_store.runs()["a-1m-aaaaaaaa-s0"]
    assert counts["copied"] == 0
    assert run.info.status == "FINISHED"
    assert (artifact_path(run.info.artifact_uri) / "config.json").read_text() == "main"
    assert _phases(run) == {"tune-coarse", "finals"}


def test_merge_reads_artifacts_beside_a_moved_store(
    main_store: Node, tmp_path: Path
) -> None:
    """A store copied off the cluster records artifact paths that do not exist here."""
    node = Node(tmp_path / "cluster")
    node.add("a-1m-aaaaaaaa-s0", "tune-coarse", files={"ckpt/x.npz": "x"})
    moved = tmp_path / "copied"
    shutil.copytree(node.root, moved)
    shutil.rmtree(node.root)
    merge_stores(f"sqlite:///{moved / 'mlruns.db'}", main_store.uri)
    run = main_store.runs()["a-1m-aaaaaaaa-s0"]
    assert (artifact_path(run.info.artifact_uri) / "ckpt" / "x.npz").read_text() == "x"


def test_merge_accepts_a_plain_database_path(main_store: Node, node: Node) -> None:
    node.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    merge_stores(node.root / "mlruns.db", main_store.uri)
    assert set(main_store.runs()) == {"a-1m-aaaaaaaa-s0"}


def test_merge_fails_before_writing_when_artifacts_are_missing(
    main_store: Node, node: Node
) -> None:
    node.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    shutil.rmtree(node.root / "mlruns")
    with pytest.raises(FileNotFoundError, match="a-1m-aaaaaaaa-s0"):
        merge_stores(node.uri, main_store.uri)
    assert main_store.runs() == {}


def test_merge_ignores_deleted_runs(main_store: Node, node: Node) -> None:
    run_id = node.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    node.client.delete_run(run_id)
    assert merge_stores(node.uri, main_store.uri)["copied"] == 0


def test_merge_cli_reports_the_counts(main_store: Node, node: Node) -> None:
    node.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.store", "merge"]
    cmd += [str(node.root / "mlruns.db"), "--dst", main_store.uri]
    out = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "copied 1, joined 0"
    assert set(main_store.runs()) == {"a-1m-aaaaaaaa-s0"}


# --- status ---------------------------------------------------------------------


def _names() -> list[str]:
    return [s.name for s in queue.expand_phase(ENTRIES, methods_dir=METHODS)]


def test_status_counts_only_runs_tagged_with_the_phase(main_store: Node) -> None:
    ppo, pqn, toy, *_ = _names()
    main_store.add(ppo, "entries,other", status="RUNNING", experiment="entries")
    main_store.add(pqn, "entries")
    # finished under another phase only: it still has to join this one
    main_store.add(toy, "other", experiment="other")
    status = phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)
    want = ["running", "finished", "pending", "pending", "pending"]
    assert status == dict(zip(_names(), want, strict=True))


@pytest.mark.parametrize(
    ("mlflow_status", "shown"),
    [("FINISHED", "finished"), ("KILLED", "killed"), ("FAILED", "failed")],
)
def test_status_names(main_store: Node, mlflow_status: str, shown: str) -> None:
    name = _names()[1]
    main_store.add(name, "entries", status=mlflow_status)
    assert phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)[name] == shown


def test_status_prefers_a_finished_run_among_same_named_ones(
    main_store: Node, node: Node
) -> None:
    name = _names()[1]
    main_store.add(name, "entries", status="KILLED")
    main_store.add(name, "entries", status="FINISHED", experiment="entries")
    assert (
        phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)[name] == "finished"
    )


def test_status_cli_prints_each_run_and_the_counts(main_store: Node) -> None:
    ppo, _, toy, *_ = _names()
    main_store.add(ppo, "entries", status="KILLED")
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.store", "status"]
    cmd += ["--phase", str(ENTRIES), "--methods-dir", str(METHODS)]
    cmd += ["--tracking-uri", main_store.uri]
    out = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    assert f"killed   {ppo}" in out
    assert f"pending  {toy}" in out
    assert out.strip().splitlines()[-1] == "pending 4, running 0, finished 0, killed 1"
