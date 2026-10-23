"""Tests for the run status of a phase and the merge of node MLflow stores."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest

from wut_murl_craftax.orchestration import queue, store
from wut_murl_craftax.orchestration.store import (
    EVAL_MODES,
    merge_stores,
    phase_status,
)
from wut_murl_craftax.orchestration.tracking import _artifact_location, artifact_path
from wut_murl_craftax.training.checkpoints import save_state, saved_update

CONFIGS = Path(__file__).parent / "fixtures" / "configs"
METHODS = CONFIGS / "methods"
ENTRIES = CONFIGS / "phases" / "entries.yaml"
REPO = Path(__file__).resolve().parents[2]


def evaluated(*updates: int) -> dict[str, str]:
    """Run files of checkpoints ``updates`` with both modes' eval files."""
    files = {
        "config.json": "{}",
        "ckpt/index.json": json.dumps([{"update": u} for u in updates]),
    }
    return files | {
        f"eval/{mode}/{u:05d}.npz": "e" for u in updates for mode in EVAL_MODES
    }


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
        for rel, text in (evaluated(3) if files is None else files).items():
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


def _saved_at(node: Node, run_id: str, update: int) -> None:
    """A ``state/`` saved after ``update``, as the runner lays it out."""
    out = artifact_path(node.client.get_run(run_id).info.artifact_uri)
    save_state(out, {"trainer": {"w": np.zeros(2)}, "update": update, "wall_s": 1.0})


@pytest.mark.parametrize(
    ("main_at", "node_at", "copied"), [(100, 200, 1), (200, 100, 0)]
)
def test_of_two_unfinished_copies_the_one_further_on_is_kept(
    main_store: Node, node: Node, main_at: int, node_at: int, copied: int
) -> None:
    """infra-6 review M2: a node's progress is not dropped for a staler main copy."""
    name = "a-1b-aaaaaaaa-s0"
    _saved_at(main_store, main_store.add(name, "final-1b", status="KILLED"), main_at)
    _saved_at(node, node.add(name, "final-1b", status="KILLED"), node_at)
    counts = merge_stores(node.uri, main_store.uri)
    assert counts == {"copied": copied, "joined": 1 - copied}
    run = main_store.runs()[name]
    assert run.info.status == "KILLED"
    assert saved_update(artifact_path(run.info.artifact_uri)) == max(main_at, node_at)


_KILLED_COPY = """
import os, signal, shutil, sys
from pathlib import Path
from wut_murl_craftax.orchestration import store

def cut(src, dst, **kw):
    Path(dst, "state").mkdir(parents=True)
    Path(dst, "state", "state.npz").write_bytes(b"PK half a state")
    os.kill(os.getpid(), signal.SIGKILL)

shutil.copytree = cut
store.merge_stores(sys.argv[1], sys.argv[2])
"""


def test_a_seed_killed_mid_copy_is_redone_by_the_next_seed(
    main_store: Node, node: Node
) -> None:
    """Review p2b I1: SIGKILL (node loss) during seeding leaves a tagged half copy,
    which lookups skip and the next seed replaces with a whole one."""
    name = "a-1b-aaaaaaaa-s0"
    run_id = main_store.add(name, "final-1b", status="KILLED", files={"m.npz": "m"})
    _saved_at(main_store, run_id, 5)
    cmd = [sys.executable, "-c", _KILLED_COPY, str(main_store.root / "mlruns.db")]
    cut = subprocess.run([*cmd, node.uri], capture_output=True, text=True, check=False)
    assert cut.returncode == -9, cut.stderr[-2000:]
    (half,) = node.client.search_runs([
        e.experiment_id for e in node.client.search_experiments()
    ])
    assert half.data.tags[store.TAG_MERGE_PARTIAL] == "true"
    assert store._active_runs(node.client) == []
    assert merge_stores(main_store.root / "mlruns.db", node.uri)["copied"] == 1
    run = node.runs()[name]  # the half copy is gone: no duplicate
    assert store.TAG_MERGE_PARTIAL not in run.data.tags
    out = artifact_path(run.info.artifact_uri)
    assert (out / "m.npz").read_text() == "m"
    assert saved_update(out) == 5
    assert list(node.root.rglob("*.staging")) == []


def test_an_unreadable_state_loses_to_a_readable_copy_and_never_raises(
    main_store: Node, node: Node
) -> None:
    """A cut copy's truncated ``state.npz`` counts as no state (-1)."""
    name = "a-1b-aaaaaaaa-s0"
    _saved_at(main_store, main_store.add(name, "final-1b", status="KILLED"), 5)
    broken = node.add(name, "final-1b", status="KILLED")
    state = artifact_path(node.client.get_run(broken).info.artifact_uri) / "state"
    state.mkdir()
    (state / "state.npz").write_bytes(b"PK truncated")
    assert saved_update(state.parent) == -1
    assert merge_stores(node.uri, main_store.uri)["joined"] == 1  # main's stays
    assert saved_update(_dir_of(main_store, name)) == 5
    assert merge_stores(main_store.root / "mlruns.db", node.uri)["copied"] == 1
    assert saved_update(_dir_of(node, name)) == 5


def _dir_of(copy: Node, name: str) -> Path:
    return artifact_path(copy.runs()[name].info.artifact_uri)


def test_a_failed_copy_leaves_the_main_store_as_it_was(
    main_store: Node, node: Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    """infra-6 review M1: no half-copied RUNNING run, the replaced one stays."""
    name = "a-1m-aaaaaaaa-s0"
    main_store.add(name, "tune-coarse", status="KILLED", files={"config.json": "old"})
    node.add(name, "tune-coarse", files={"config.json": "new", "ckpt/x.npz": "x"})
    real = shutil.copytree

    def disk_full(src: Path, dst: Path, **kw: Any) -> Any:
        Path(dst, "ckpt").mkdir(parents=True)
        Path(dst, "ckpt", "x.npz").write_text("half")
        msg = "No space left on device"
        raise OSError(msg)

    monkeypatch.setattr(store.shutil, "copytree", disk_full)
    with pytest.raises(OSError, match="No space"):
        merge_stores(node.uri, main_store.uri)
    (run,) = main_store.runs().values()
    assert run.info.status == "KILLED"
    out = artifact_path(run.info.artifact_uri)
    assert (out / "config.json").read_text() == "old"
    leftovers = {p.parent.name for p in main_store.root.rglob("x.npz")}
    assert leftovers == set()
    monkeypatch.setattr(store.shutil, "copytree", real)
    assert merge_stores(node.uri, main_store.uri)["copied"] == 1
    assert main_store.runs()[name].info.status == "FINISHED"


def test_merge_ignores_deleted_runs(main_store: Node, node: Node) -> None:
    run_id = node.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    node.client.delete_run(run_id)
    assert merge_stores(node.uri, main_store.uri)["copied"] == 0


def _history(store: Node, name: str, key: str) -> list[tuple[int, float]]:
    run_id = store.runs()[name].info.run_id
    return sorted(
        (m.step, m.value) for m in store.client.get_metric_history(run_id, key)
    )


def test_a_join_copies_the_eval_files_and_metrics_the_main_copy_lacks(
    main_store: Node, node: Node
) -> None:
    """Phase 0 review I5: a node's log-spaced evaluation of a reused run survives."""
    name = "a-1m-aaaaaaaa-s0"
    main_store.add(
        name,
        "tune-coarse",
        files={"config.json": "{}", "eval/final/3.npz": "main"},
        metrics={"eval/final/return": [1.0]},
    )
    node.add(
        name,
        "tune-coarse,final-1m",
        experiment="final-1m",
        files={"eval/final/3.npz": "node", "eval/final/1.npz": "n1"},
        metrics={"eval/final/return": [9.0, 2.0], "train/return": [5.0]},
    )
    assert merge_stores(node.uri, main_store.uri) == {"copied": 0, "joined": 1}
    out = artifact_path(main_store.runs()[name].info.artifact_uri)
    assert (out / "eval/final/3.npz").read_text() == "main"  # main's file stays
    assert (out / "eval/final/1.npz").read_text() == "n1"
    # step 0 is main's; only evaluation series are copied on a join
    assert _history(main_store, name, "eval/final/return") == [(0, 1.0), (1, 2.0)]
    assert _history(main_store, name, "train/return") == []
    assert merge_stores(node.uri, main_store.uri) == {"copied": 0, "joined": 1}
    assert _history(main_store, name, "eval/final/return") == [(0, 1.0), (1, 2.0)]


@pytest.mark.parametrize(
    ("main_pending", "node_pending", "node_phases", "want"),
    [
        (False, True, "tune-coarse,final-1m", True),  # the node's evaluation was cut
        (True, False, "tune-coarse,final-1m", False),  # the node completed it
        (True, False, "final-1m", True),  # the node never saw main's phase
        (False, False, "final-1m", False),
    ],
)
def test_a_join_carries_the_pending_evaluation_tag(
    main_store: Node,
    node: Node,
    main_pending: bool,
    node_pending: bool,
    node_phases: str,
    want: bool,
) -> None:
    name = "a-1m-aaaaaaaa-s0"
    main_id = main_store.add(name, "tune-coarse")
    node_id = node.add(name, node_phases, experiment="final-1m")
    for copy, run_id, pending in (
        (main_store, main_id, main_pending),
        (node, node_id, node_pending),
    ):
        if pending:
            copy.client.set_tag(run_id, "eval/pending", "true")
            copy.client.set_tag(run_id, "fail/error", "OSError: disk full")
    merge_stores(node.uri, main_store.uri)
    tags = main_store.runs()[name].data.tags
    assert ("eval/pending" in tags) == want
    assert ("fail/error" in tags) == (want and main_pending)


def test_merge_of_named_runs_seeds_a_node_store(main_store: Node, node: Node) -> None:
    """The reverse merge copies only the runs a node's worker will reuse or resume."""
    main_store.add("a-1m-aaaaaaaa-s0", "tune-coarse")
    main_store.add("b-1m-bbbbbbbb-s0", "tune-coarse")
    counts = merge_stores(
        main_store.root / "mlruns.db", node.uri, names={"a-1m-aaaaaaaa-s0", "c"}
    )
    assert counts == {"copied": 1, "joined": 0}
    assert set(node.runs()) == {"a-1m-aaaaaaaa-s0"}


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


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        ("abort/guard", "dyna/divergence"),
        ("abort/bug", "true"),
        ("abort/memory", "true"),
    ],
)
def test_status_shows_a_run_aborted_for_good_apart(
    main_store: Node, tag: str, value: str
) -> None:
    ppo, pqn, *_ = _names()
    main_store.client.set_tag(
        main_store.add(ppo, "entries", status="FAILED"), tag, value
    )
    main_store.add(pqn, "entries", status="FAILED")
    status = phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)
    assert (status[ppo], status[pqn]) == ("aborted", "failed")
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.store", "status"]
    cmd += ["--phase", str(ENTRIES), "--methods-dir", str(METHODS)]
    # a plain .db path, as the chain passes it
    cmd += ["--tracking-uri", main_store.uri.removeprefix("sqlite:///")]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
    last = out.stdout.strip().splitlines()[-1]
    assert last == "pending 3, running 0, finished 0, killed 0, failed 1, aborted 1"


def test_status_shows_a_finished_run_with_a_pending_evaluation_apart(
    main_store: Node,
) -> None:
    ppo, pqn, *_ = _names()
    rid = main_store.add(ppo, "entries")
    main_store.client.set_tag(rid, "eval/pending", "true")
    main_store.add(pqn, "entries")
    status = phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)
    assert (status[ppo], status[pqn]) == ("eval-pending", "finished")
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.store", "status"]
    cmd += ["--phase", str(ENTRIES), "--methods-dir", str(METHODS)]
    cmd += ["--tracking-uri", main_store.uri]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
    last = out.stdout.strip().splitlines()[-1]
    assert last == "pending 3, running 0, eval-pending 1, finished 1, killed 0"


def test_status_shows_a_finished_run_without_every_log_spaced_eval_apart(
    main_store: Node, tmp_path: Path
) -> None:
    """I5 ruling: a final phase's run needs both modes of every checkpoint."""
    ppo, pqn, toy, *_ = _names()
    main_store.add(ppo, "entries", files=evaluated(1, 3))
    partial = evaluated(1, 3)
    del partial["eval/greedy/00001.npz"]
    main_store.add(pqn, "entries", files=partial)
    main_store.add(toy, "entries", files={"config.json": "{}"})  # no checkpoints
    status = phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)
    assert [status[n] for n in (ppo, pqn, toy)] == [
        "finished",
        "eval-incomplete",
        "eval-incomplete",
    ]
    final = tmp_path / "entries.yaml"  # same stem; a final-only phase
    final.write_text(ENTRIES.read_text().replace("log_spaced", "final"))
    status = phase_status(final, main_store.uri, methods_dir=METHODS)
    assert {status[n] for n in (ppo, pqn, toy)} == {"finished"}
    cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.store", "status"]
    cmd += ["--phase", str(ENTRIES), "--methods-dir", str(METHODS)]
    cmd += ["--tracking-uri", main_store.uri]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
    last = out.stdout.strip().splitlines()[-1]
    want = "pending 2, running 0, eval-incomplete 2, finished 1, killed 0"
    assert last == want


def test_status_eval_modes_are_the_evaluations() -> None:
    from wut_murl_craftax.visualisation.eval_rollouts import MODES

    assert set(EVAL_MODES) == set(MODES)


def test_status_prefers_a_finished_run_among_same_named_ones(
    main_store: Node, node: Node
) -> None:
    name = _names()[1]
    main_store.add(name, "entries", status="KILLED")
    main_store.add(name, "entries", status="FINISHED", experiment="entries")
    assert (
        phase_status(ENTRIES, main_store.uri, methods_dir=METHODS)[name] == "finished"
    )


def test_status_lists_a_fine_phase_from_the_store_without_expanding(
    main_store: Node,
) -> None:
    """``make status PHASE=tune-fine`` without FINE: the runs tagged with the phase."""
    fine = CONFIGS / "phases" / "fine.yaml"
    main_store.add("toy-1m-aaaaaaaa-s0", "tune-coarse,fine", status="RUNNING")
    main_store.add("toy-1m-bbbbbbbb-s1", "fine", experiment="fine")
    main_store.add("toy-1m-cccccccc-s0", "tune-coarse")  # not in the phase
    want = {"toy-1m-aaaaaaaa-s0": "running", "toy-1m-bbbbbbbb-s1": "finished"}
    assert phase_status(fine, main_store.uri, methods_dir=METHODS) == want
    assert phase_status(fine, main_store.uri, method="toy", methods_dir=METHODS) == want
    with pytest.raises(queue.QueueConfigError, match="no phase entry"):
        phase_status(fine, main_store.uri, method="donor", methods_dir=METHODS)


def test_status_of_a_mixed_phase_lists_its_specs_then_the_fine_runs(
    main_store: Node, tmp_path: Path
) -> None:
    path = tmp_path / "mixed.yaml"
    path.write_text(ENTRIES.read_text() + "  waiting: {configs: fine}\n")
    main_store.add("waiting-1m-dddddddd-s0", "mixed")  # tagged method toy: not fine
    run_id = main_store.add("waiting-1m-eeeeeeee-s0", "mixed", status="KILLED")
    main_store.client.set_tag(run_id, "method", "waiting")
    status = phase_status(path, main_store.uri, methods_dir=METHODS)
    assert list(status) == [*_names(), "waiting-1m-eeeeeeee-s0"]
    assert status["waiting-1m-eeeeeeee-s0"] == "killed"
    only = phase_status(path, main_store.uri, method="waiting", methods_dir=METHODS)
    assert only == {"waiting-1m-eeeeeeee-s0": "killed"}


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
