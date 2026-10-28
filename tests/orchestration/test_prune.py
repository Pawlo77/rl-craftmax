"""Tests for the storage-retention prune: what it keeps, what it deletes, the marker."""

import json
import re
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from wut_murl_craftax.orchestration import prune
from wut_murl_craftax.orchestration.store import EVAL_MODES
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training.checkpoints import missing_checkpoints, pruned_updates

FINAL_SCORES = {"F0": 10.0, "F1": 50.0, "F2": 30.0, "F3": 20.0, "F4": 40.0}
"""Ascending: F0 F3 F2 F4 F1, so F1 is best and F2 the (lower) median."""
EVAL_SCORES = {"F0": 99.0, "F1": 5.0, "F2": 4.0, "F3": 3.0, "F4": 2.0}
"""F0 is best by its stochastic evaluation."""


class Store:
    """A throwaway store with the run kinds the prune tells apart."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.uri = f"sqlite:///{root / 'mlruns.db'}"
        self.client = mlflow.MlflowClient(tracking_uri=self.uri)
        self.metrics = root / "metrics"
        self.analysis = root / "analysis"
        self.dirs: dict[str, Path] = {}

    def add(
        self,
        name: str,
        phases: str,
        updates: list[int],
        *,
        status: str = "FINISHED",
        log_spaced: bool = True,
    ) -> str:
        experiment = phases.split(",")[-1]
        found = self.client.get_experiment_by_name(experiment)
        exp_id = (
            found.experiment_id
            if found
            else self.client.create_experiment(
                experiment,
                artifact_location=(self.root / "mlruns" / experiment).as_uri(),
            )
        )
        run = self.client.create_run(exp_id, run_name=name, tags={"phases": phases})
        out = artifact_path(run.info.artifact_uri)
        (out / "ckpt").mkdir(parents=True)
        index = []
        for u in updates:
            (out / "ckpt" / f"{u:05d}.msgpack").write_bytes(b"p" * 1000)
            index.append({
                "update": u,
                "env_steps": u,
                "wall_s": 0.0,
                "file": f"{u:05d}.msgpack",
            })
        (out / "ckpt" / "index.json").write_text(json.dumps(index))
        for u in updates if log_spaced else updates[-1:]:
            for mode in EVAL_MODES:
                (out / "eval" / mode).mkdir(parents=True, exist_ok=True)
                (out / "eval" / mode / f"{u:05d}.npz").write_bytes(b"e")
        (out / "state").mkdir()
        (out / "state" / "state.npz").write_bytes(b"s" * 5000)
        (out / "metrics.npz").write_bytes(b"m")
        self.client.set_terminated(run.info.run_id, status)
        self.dirs[name] = out
        return run.info.run_id

    def files(self) -> dict[str, bytes]:
        return {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in sorted((self.root / "mlruns").rglob("*"))
            if p.is_file()
        }


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path)
    s.add("T", "tune-coarse", [0, 1, 3], log_spaced=False)
    ids = {n: s.add(n, "final-1m", [0, 1, 2, 3]) for n in ("F0", "F1", "F2", "F3")}
    ids["F4"] = s.add("F4", "tune-coarse,final-1m", [0, 1, 2, 3])
    s.add("U", "final-1m", [0, 1, 2, 3])  # not exported yet
    s.add("K", "final-1m", [0, 1], status="KILLED")
    s.add("P", "final-1m", [0, 1, 2, 3])
    (s.dirs["P"] / "eval" / "greedy" / "00001.npz").unlink()  # evaluation incomplete
    rows = [
        {
            "run_id": ids[n],
            "budget": "1m",
            "method": "m",
            "config_ref": "tuned",
            "seed": i,
            "final_score": FINAL_SCORES[n],
            "eval_stochastic_score": EVAL_SCORES[n],
        }
        for i, n in enumerate(FINAL_SCORES)
    ]
    (s.metrics / "final-1m").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(s.metrics / "final-1m" / "runs.parquet")
    (s.analysis / "1m").mkdir(parents=True)
    cases = pd.DataFrame({"run_id": [ids["F3"]], "update": [1], "id": ["c"]})
    cases.to_csv(s.analysis / "1m" / "cases.csv", index=False)
    return s


def _args(store: Store, *more: str) -> list[str]:
    return [
        "--tracking-uri",
        f"{store.uri}?timeout=30",
        "--metrics-root",
        str(store.metrics),
        "--analysis-root",
        str(store.analysis),
        *more,
    ]


def _plan(store: Store, *more: str) -> set[tuple[str, str, str]]:
    plan = prune.plan_from_args(prune.parse_args(_args(store, *more)))
    return {(i.run, i.kind, i.path.name) for i in plan.items}


def _ckpt(*updates: int) -> list[str]:
    return [f"{u:05d}.msgpack" for u in updates]


def test_the_plan_keeps_what_later_steps_read(store: Store) -> None:
    got = _plan(store)
    want = {(n, "state", "state") for n in ("T", "F0", "F1", "F2", "F3", "F4", "U")}
    want |= {("T", "ckpt_tuning", f) for f in _ckpt(0, 1, 3)}
    want |= {("F3", "ckpt_final", f) for f in _ckpt(0, 2)}  # last and case kept
    want |= {("F4", "ckpt_final", f) for f in _ckpt(0, 1, 2)}
    assert got == want


def test_a_dry_run_deletes_nothing_and_prints_the_sizes_per_class(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    before = store.files()
    prune.main(_args(store))
    assert store.files() == before
    out = capsys.readouterr().out
    assert "dry run" in out
    assert re.search(r"state\s+7\s+7\s+35,000\s", out), out
    assert re.search(r"ckpt_tuning\s+3\s+1\s+3,000\s", out), out
    assert re.search(r"ckpt_final\s+5\s+2\s+5,000\s", out), out
    assert re.search(r"total\s+15\s+7\s+43,000\s", out), out
    assert "kept whole: 3 runs" in out
    assert "not in the export tables, every checkpoint kept: U" in out
    assert "evaluation incomplete, skipped: P" in out


def test_apply_marks_then_deletes_and_keeps_the_index(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    index = {n: (d / "ckpt" / "index.json").read_bytes() for n, d in store.dirs.items()}
    prune.main(_args(store, "--apply"))
    assert "deleted" in capsys.readouterr().out
    d = store.dirs
    assert sorted(p.name for p in (d["T"] / "ckpt").iterdir()) == [
        "index.json",
        "pruned.json",
    ]
    assert pruned_updates(d["T"]) == {0, 1, 3}
    assert pruned_updates(d["F3"]) == {0, 2}
    assert pruned_updates(d["F4"]) == {0, 1, 2}
    assert sorted(p.name for p in (d["F3"] / "ckpt").glob("*.msgpack")) == _ckpt(1, 3)
    for n in ("F0", "F1", "F2", "U"):
        assert sorted(p.name for p in (d[n] / "ckpt").glob("*.msgpack")) == _ckpt(
            0, 1, 2, 3
        )
        assert pruned_updates(d[n]) == frozenset()
    for n in ("T", "F0", "F1", "F2", "F3", "F4", "U"):
        assert not (d[n] / "state").exists()
        assert missing_checkpoints(d[n]) == []
    for n in ("K", "P"):
        assert (d[n] / "state" / "state.npz").is_file()
        assert len(list((d[n] / "ckpt").glob("*.msgpack"))) == len(json.loads(index[n]))
    assert {n: (x / "ckpt" / "index.json").read_bytes() for n, x in d.items()} == index
    marker = json.loads((d["T"] / "ckpt" / "pruned.json").read_text())
    assert marker["reason"] == prune.REASON
    after = store.files()
    prune.main(_args(store, "--apply"))  # idempotent
    assert store.files() == after
    assert _plan(store) == set()


def test_phases_limit_the_scope_to_runs_tagged_with_one(store: Store) -> None:
    got = _plan(store, "--phases", "tune-coarse")
    assert {run for run, _, _ in got} == {"T", "F4"}
    assert ("F4", "ckpt_final", "00001.msgpack") in got  # a final run all the same


def test_seed_picks_rank_by_final_score_and_by_the_stochastic_eval() -> None:
    runs = pd.DataFrame({
        "run_id": ["a", "b", "c", "d", "x"],
        "budget": ["1m", "1m", "1m", "1m", "1b"],
        "method": ["m"] * 5,
        "config_ref": ["tuned"] * 5,
        "final_score": [1.0, 4.0, 3.0, 2.0, 0.0],
        "eval_stochastic_score": [9.0, 1.0, float("nan"), 2.0, 0.0],
    })
    picks = prune.seed_picks(runs)
    assert picks == {
        "b": {"best"},
        "d": {"median"},  # ascending a d c b: the lower median
        "a": {"best-eval"},
        "x": {"best", "median", "best-eval"},  # its own budget
    }
