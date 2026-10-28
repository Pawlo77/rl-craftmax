"""Tests for the training-analysis CLI: outputs, resume, MLflow logging, replay inputs.

The orchestration tests run the CLI on a synthetic export of the four final phases
(`_analysis_export`) with the replay stubbed (`fake_replay`: cases write small files
instead of rendering), a temporary MLflow store and temporary output roots. One test
runs the real replay on the tiny Classic PPO run of the visualisation tests.
"""

import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wut_murl_craftax.analysis import training
from wut_murl_craftax.analysis.cases import CASE_COLUMNS
from wut_murl_craftax.analysis.load import MissingPhaseError
from wut_murl_craftax.analysis.obstacles import TABLE_COLUMNS

PHASES = ("final-1m", "final-1b", "combo-1m", "combo-1b")
METHODS = (
    "count-bonus-pqn-rnn", "count-bonus-ppo-rnn", "combo", "dyna-q", "ppo",
    "ppo-rnn", "pqn-rnn", "sil", "smdp-skills",
)  # fmt: skip
FINAL = 244
"""The last checkpoint of the synthetic eval episodes."""


def _helper(name: str) -> ModuleType:
    """A ``tests/visualisation`` helper module, loaded once by path."""
    if name not in sys.modules:
        path = Path(__file__).parents[1] / "visualisation" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


_learning_run = _helper("_learning_run")
needs_ffmpeg, tiny_run = _learning_run.needs_ffmpeg, _learning_run.tiny_run


def make_cases_possible(root: Path) -> None:
    """Add what a case needs and the synthetic export lacks: game, floor unlocks."""
    for path in root.glob("*/eval_episodes.parquet"):
        t = pq.read_table(path)
        floor = np.asarray(t["max_floor"])
        step = np.where(floor >= 1, 100, -1).astype("int32")
        t = t.append_column("ach/enter_dungeon", pa.array(floor >= 1))
        t = t.append_column("unlock_step/enter_dungeon", pa.array(step))
        pq.write_table(t, path)
    for path in root.glob("*/runs.parquet"):
        runs = pd.read_parquet(path)
        runs["config_json"] = [
            json.dumps(json.loads(c) | {"env": "Craftax-Symbolic-v1"})
            for c in runs["config_json"]
        ]
        runs.to_parquet(path)


@pytest.fixture(scope="module")
def metrics(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export of the four final phases whose episodes can all become cases."""
    from _analysis_export import write_export  # tests/analysis, via conftest

    root = write_export(tmp_path_factory.mktemp("metrics"), PHASES)
    make_cases_possible(root)
    return root


class Replay(SimpleNamespace):
    """What the stubbed replay saw: exported and described case ids, opened runs."""


@pytest.fixture
def fake_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Replay:
    """Cases write placeholder files; run directories are empty tmp folders.

    Eval files go into ``<tmp>/runs/<run_id>/eval/stochastic`` (see `write_eval`).
    """
    seen = Replay(
        exported=[], described=[], opened=[], fail=set(), root=tmp_path / "runs"
    )

    def export(case: Any, out_dir: Path, run: Any = None, **kw: Any) -> dict[str, Any]:
        cid = str(case["id"])
        root = kw["root"]
        if cid in seen.fail:  # fails late, after the flat rollout was copied
            flat = root / "visualisation" / "runs" / f"case-{cid}.npz"
            flat.parent.mkdir(parents=True, exist_ok=True)
            flat.write_bytes(b"x")
            msg = f"replay of {cid} failed"
            raise ValueError(msg)
        files = {
            "rollout": root / "visualisation" / "runs" / f"case-{cid}.npz",
            "master": out_dir / "masters" / f"case-{cid}.mkv",
            "gif": out_dir / "gifs" / f"{cid}.gif",
        }
        if case["kind"] == "stall":
            files["overlay"] = out_dir / "overlays" / f"{cid}.mp4"
        for p in [*files.values(), out_dir / "rollouts" / f"case-{cid}.npz"]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        seen.exported.append(cid)
        return {
            **dict(case),
            "overlay": "",
            **{k: str(p.relative_to(root)) for k, p in files.items()},
        }

    def describe(case: Any, run: Any = None, **kw: Any) -> dict[str, Any]:
        seen.described.append(str(case["id"]))
        return {"id": case["id"], "kind": case["kind"], "peak": int(case["peak"])}

    def dirs(run_ids: Any, tracking_uri: str) -> dict[str, Path]:
        out = {r: seen.root / r for r in dict.fromkeys(run_ids)}
        for p in out.values():
            p.mkdir(parents=True, exist_ok=True)
        return out

    def opened(run_dir: Path) -> Path:
        seen.opened.append(run_dir.name)
        return run_dir

    monkeypatch.setattr(training, "export_case", export)
    monkeypatch.setattr(training, "describe_case", describe)
    monkeypatch.setattr(training, "run_dirs", dirs)
    monkeypatch.setattr(training, "_open_run", opened)
    return seen


@pytest.fixture
def store(tmp_path: Path) -> str:
    """A tracking URI of a temporary store, with the busy-timeout query of R103."""
    return f"sqlite:///{tmp_path / 'store' / 'mlruns.db'}?timeout=300"


def run_cli(
    metrics: Path,
    out: Path,
    store: str,
    *extra: str,
    phases: tuple[str, ...] = ("final-1m", "combo-1m"),
    method: str = "all",
) -> None:
    training.main([
        "--phases", *phases,
        "--method", method,
        "--out", str(out / "analysis"),
        "--metrics-root", str(metrics),
        "--tracking-uri", store,
        *extra,
    ])  # fmt: skip


def folder(out: Path, budget: str, method: str) -> Path:
    return out / "analysis" / budget / method


def write_eval(replay: Replay, run_id: str, actions: np.ndarray) -> None:
    """The final stochastic eval file of a run: two episodes with the same actions."""
    path = replay.root / run_id / "eval" / "stochastic"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / f"{FINAL:05d}.npz",
        seeds=np.array([10000, 10001]),
        actions=np.tile(actions, (2, 1)),
        length=np.array([len(actions)] * 2),
    )


def obstacle_rows(out: Path, budget: str, method: str) -> pd.DataFrame:
    return pd.read_csv(
        folder(out, budget, method) / "obstacles.csv", keep_default_na=False
    )


# ------------------------------------------------------------------- outputs


def test_every_method_gets_its_folder_in_each_budget(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(metrics, tmp_path / "out", store, phases=PHASES)
    for budget in ("1m", "1b"):
        for method in METHODS:
            d = folder(tmp_path / "out", budget, method)
            assert (d / "obstacles.md").is_file(), (budget, method)
            assert (d / "hard" / "floor_reach.csv").is_file()
            assert (d / "plots" / "floor_reach.png").is_file()
            assert (d / "cases").is_dir()
            assert (d / "cases.csv").is_file()
    # the combination is analysed like the others, in the budget of its own runs
    assert (folder(tmp_path / "out", "1m", "combo") / "hard").is_dir()


def test_the_hard_tables_and_plots_cover_the_plan_questions(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(metrics, tmp_path / "out", store, "--no-log", method="sil")
    d = folder(tmp_path / "out", "1m", "sil")
    names = {p.name for p in (d / "hard").glob("*.csv")}
    assert names == {
        "floor_reach.csv", "achievement_status.csv", "death_causes.csv",
        "stalls.csv", "stall_stats.csv", "stall_starts.csv", "stall_ccdf.csv",
    }  # fmt: skip
    plots = {p.name for p in (d / "plots").glob("*.png")}
    assert plots == {
        "floor_reach.png", "achievement_status_tuned.png",
        "death_causes_over_training_tuned.png", "stall_lengths.png",
    }  # fmt: skip
    captions = pd.read_csv(d / "plots" / "captions.csv")
    assert set(captions["stem"]) == {p[:-4] for p in plots}
    assert set(captions["id"]) == {"HARD"}
    floors = pd.read_csv(d / "hard" / "floor_reach.csv")
    assert set(floors["method"]) == {"sil"}
    assert set(floors["budget"]) == {"1m"}


def test_units_stay_apart_in_every_table(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(metrics, tmp_path / "out", store, "--no-log", method="count-bonus-ppo-rnn")
    d = folder(tmp_path / "out", "1m", "count-bonus-ppo-rnn")
    stats = pd.read_csv(d / "hard" / "stall_stats.csv")
    assert sorted(stats["config_ref"]) == ["simhash", "tuned"]
    text = (d / "obstacles.md").read_text()
    # tuned first, then the ablation, one table each with the template columns
    assert text.index("## tuned") < text.index("## simhash")
    assert text.count("| " + " | ".join(TABLE_COLUMNS) + " |") == 2
    runs = obstacle_rows(tmp_path / "out", "1m", "count-bonus-ppo-rnn")
    assert set(runs["config_ref"]) == {"tuned", "simhash"}
    assert runs.groupby("config_ref")["run_id"].nunique().to_dict() == {
        "simhash": 3,
        "tuned": 3,
    }


def test_cases_of_all_units_of_a_method_are_written_in_one_table(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", method="smdp-skills")
    run_cli(metrics, out, store, "--no-log", method="ppo")
    budget = pd.read_csv(out / "analysis" / "1m" / "cases.csv", keep_default_na=False)
    assert tuple(budget.columns) == CASE_COLUMNS
    assert set(budget["method"]) == {
        "smdp-skills",
        "ppo",
    }  # the second call kept the first
    mine = pd.read_csv(
        folder(out, "1m", "smdp-skills") / "cases.csv", keep_default_na=False
    )
    assert set(mine["config_ref"]) == {"tuned", "monte-carlo"}
    assert mine["id"].is_unique
    assert (mine["rollout"] != "").all()  # export filled the file columns
    assert set(mine["id"]) == set(budget.loc[budget["method"] == "smdp-skills", "id"])
    # every case was replayed and described once, its facts kept under cases/
    facts = {
        p.stem for p in (folder(out, "1m", "smdp-skills") / "cases").glob("*.json")
    }
    assert facts == set(mine["id"])
    case_ids = set(mine["id"]) | set(budget.loc[budget["method"] == "ppo", "id"])
    assert sorted(fake_replay.exported) == sorted(case_ids)
    assert fake_replay.described == fake_replay.exported


def test_one_replay_is_opened_per_run(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(
        metrics, tmp_path / "out", store, "--no-log", method="sil", phases=("final-1m",)
    )
    assert len(fake_replay.opened) == len(set(fake_replay.opened)) <= 3


def test_a_method_name_selects_and_an_unknown_one_fails(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(metrics, tmp_path / "out", store, "--no-log", method="sil")
    assert [p.name for p in (tmp_path / "out/analysis/1m").glob("*/")] == ["sil"]
    with pytest.raises(SystemExit, match="nope"):
        run_cli(metrics, tmp_path / "out", store, "--no-log", method="nope")


# -------------------------------------------------------------------- phases


def test_a_missing_phase_fails_with_the_loaders_error(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    with pytest.raises(MissingPhaseError, match="final-9b"):
        run_cli(
            metrics,
            tmp_path / "out",
            store,
            "--no-log",
            phases=("final-1m", "final-9b"),
        )
    assert not (tmp_path / "out").exists()


def test_skip_missing_runs_the_phases_that_exist(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(
        metrics, tmp_path / "out", store, "--no-log", "--skip-missing",
        phases=("final-1m", "final-9b"), method="ppo",
    )  # fmt: skip
    assert [p.name for p in (tmp_path / "out/analysis").iterdir()] == ["1m"]


def test_the_1m_phases_alone_are_enough(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(
        metrics, tmp_path / "out", store, "--no-log",
        phases=("final-1m", "combo-1m"), method="all",
    )  # fmt: skip
    assert {
        p.name for p in (tmp_path / "out/analysis/1m").iterdir() if p.is_dir()
    } == set(METHODS)
    assert not (tmp_path / "out/analysis/1b").exists()


# -------------------------------------------------------------------- MLflow


def runs_of(store: str) -> dict[str, Any]:
    """Run name -> MLflow run of experiment ``analysis`` (all lifecycle stages)."""
    client = mlflow.MlflowClient(store)
    exp = client.get_experiment_by_name("analysis")
    assert exp is not None
    found = client.search_runs([exp.experiment_id], max_results=1000)
    names = [r.info.run_name for r in found]
    assert len(names) == len(set(names)), "a rerun made a second run of one name"
    return {r.info.run_name: r for r in found}


def test_every_method_is_logged_under_its_own_run_name(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(metrics, tmp_path / "out", store, phases=("final-1m", "combo-1m"))
    found = runs_of(store)
    assert set(found) == {f"training-analysis-1m-{m}" for m in METHODS}
    run = found["training-analysis-1m-count-bonus-ppo-rnn"]
    tags = run.data.tags
    assert tags["analysis/kind"] == "training"
    assert tags["analysis/phases"] == "final-1m,combo-1m"
    assert (tags["analysis/budget"], tags["analysis/method"]) == (
        "1m",
        "count-bonus-ppo-rnn",
    )
    # loaders select training runs by these tags: an analysis run never carries them
    assert not {"phases", "method", "budget", "seed", "config_hash"} & set(tags)
    assert run.info.status == "FINISHED"
    client = mlflow.MlflowClient(store)
    top = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"obstacles.md", "hard", "plots", "cases", "cases.csv"} <= top
    # artifacts live beside the store, in the experiment's own folder
    db = Path(store.removeprefix("sqlite:///").partition("?")[0])
    assert (db.parent / "mlruns" / "analysis").is_dir()


def test_masters_stay_out_of_the_logged_artifacts_unless_asked(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, phases=("final-1m",), method="sil")
    client = mlflow.MlflowClient(store)

    def top() -> set[str]:
        run = runs_of(store)["training-analysis-1m-sil"]
        return {a.path for a in client.list_artifacts(run.info.run_id)}

    assert "masters" not in top()
    assert {"gifs", "rollouts", "hard", "obstacles.md"} <= top()
    assert any((folder(out, "1m", "sil") / "masters").iterdir())  # kept on disk
    run_cli(
        metrics,
        out,
        store,
        "--force",
        "--log-masters",
        phases=("final-1m",),
        method="sil",
    )
    assert "masters" in top()  # the rerun replaced the artifacts, now with masters


def test_the_tag_names_the_phases_that_were_loaded(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    run_cli(
        metrics, tmp_path / "out", store, "--skip-missing",
        phases=("final-1m", "final-9b"), method="sil",
    )  # fmt: skip
    tags = runs_of(store)["training-analysis-1m-sil"].data.tags
    assert tags["analysis/phases"] == "final-1m"


def test_a_full_disk_stops_the_run_before_any_work(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    monkeypatch.setattr(
        training.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(1e8))
    )
    with pytest.raises(SystemExit, match="GB free"):
        run_cli(metrics, tmp_path / "out", store, "--no-log", method="sil")
    assert not (tmp_path / "out").exists()


def test_a_rerun_skips_complete_methods_and_logs_nothing_twice(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, phases=("final-1m",), method="sil")
    first = fake_replay.exported[:]
    ids = {n: r.info.run_id for n, r in runs_of(store).items()}
    stamp = (folder(out, "1m", "sil") / "obstacles.md").stat().st_mtime_ns
    run_cli(metrics, out, store, phases=("final-1m",), method="sil")
    assert fake_replay.exported == first  # nothing replayed again
    assert (folder(out, "1m", "sil") / "obstacles.md").stat().st_mtime_ns == stamp
    assert {n: r.info.run_id for n, r in runs_of(store).items()} == ids


def test_force_recomputes_and_replaces_the_logged_artifacts(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, phases=("final-1m",), method="sil")
    ids = {n: r.info.run_id for n, r in runs_of(store).items()}
    n = len(fake_replay.exported)
    stale = folder(out, "1m", "sil") / "hard" / "stale.csv"
    stale.write_text("x")
    run_cli(metrics, out, store, "--force", phases=("final-1m",), method="sil")
    assert len(fake_replay.exported) == 2 * n
    assert not stale.exists()  # a forced run starts from an empty folder
    assert {
        n: r.info.run_id for n, r in runs_of(store).items()
    } == ids  # the run is reused
    client = mlflow.MlflowClient(store)
    run_id = ids["training-analysis-1m-sil"]
    hard = {a.path for a in client.list_artifacts(run_id, "hard")}
    assert "hard/stale.csv" not in hard
    assert "hard/floor_reach.csv" in hard


def test_a_forced_run_removes_the_old_flat_rollouts_of_the_method(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method="sil")
    flat = out / "visualisation" / "runs"
    old = flat / "case-1m-sil-stall-99.npz"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"x")
    table = folder(out, "1m", "sil") / "cases.csv"
    rows = pd.read_csv(table, keep_default_na=False)
    rows.loc[0, "rollout"] = "visualisation/runs/case-1m-sil-stall-99.npz"
    rows.to_csv(table, index=False)
    other = flat / "case-1m-ppo-death-1.npz"
    other.write_bytes(b"x")
    run_cli(
        metrics, out, store, "--no-log", "--force", phases=("final-1m",), method="sil"
    )
    assert not old.exists()
    assert other.exists()  # another method's rollouts stay


def test_logging_alone_is_retried_when_the_files_are_complete(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method="sil")
    n = len(fake_replay.exported)
    assert (folder(out, "1m", "sil") / ".complete").is_file()
    assert not (folder(out, "1m", "sil") / ".logged").exists()
    run_cli(metrics, out, store, phases=("final-1m",), method="sil")
    assert len(fake_replay.exported) == n  # only the logging ran
    assert "training-analysis-1m-sil" in runs_of(store)
    assert (folder(out, "1m", "sil") / ".logged").is_file()


def test_a_failed_case_fails_the_run_but_not_the_other_methods(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    fake_replay.fail.add("1m-sil-death-1")
    with pytest.raises(SystemExit, match="1m-sil-death-1"):
        run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method="all")
    assert (folder(out, "1m", "ppo") / ".complete").is_file()  # the others finished
    assert not (folder(out, "1m", "sil") / ".complete").exists()  # this one reruns
    assert (folder(out, "1m", "sil") / "obstacles.md").is_file()  # what worked stays
    failed = json.loads((folder(out, "1m", "sil") / "failed.json").read_text())
    assert list(failed) == ["1m-sil-death-1"]
    assert "ValueError" in failed["1m-sil-death-1"]
    # the partial flat rollout of the failed case is gone, the others' stay
    flat = out / "visualisation" / "runs"
    assert not (flat / "case-1m-sil-death-1.npz").exists()
    assert (flat / "case-1m-sil-stall-1.npz").exists()
    fake_replay.fail.clear()
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method="sil")
    assert (folder(out, "1m", "sil") / ".complete").is_file()
    assert not (folder(out, "1m", "sil") / "failed.json").exists()


def test_a_method_without_exported_cases_loses_its_old_rows(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method="all")
    table = out / "analysis" / "1m" / "cases.csv"
    assert "sil" in set(pd.read_csv(table)["method"])
    ids = pd.read_csv(table).query("method == 'sil'")["id"]
    fake_replay.fail.update(ids)  # every case of sil fails now
    with pytest.raises(SystemExit):
        run_cli(
            metrics,
            out,
            store,
            "--no-log",
            "--force",
            phases=("final-1m",),
            method="sil",
        )
    rows = pd.read_csv(table)
    assert "sil" not in set(rows["method"])  # no row names a deleted rollout
    assert "ppo" in set(rows["method"])


def test_a_logging_failure_keeps_the_previous_artifacts(
    store: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wut_murl_craftax.orchestration.tracking import log_analysis_run

    folder_ = tmp_path / "f"
    (folder_ / "d").mkdir(parents=True)
    (folder_ / "d" / "old.txt").write_text("old")
    (folder_ / "keep.txt").write_text("1")
    paths = lambda: sorted(folder_.iterdir())  # noqa: E731
    run_id = log_analysis_run(folder_, paths(), "r", {"analysis/kind": "x"}, store)
    client = mlflow.MlflowClient(store)
    (folder_ / "d" / "old.txt").unlink()
    (folder_ / "d" / "new.txt").write_text("new")

    def boom(*a: Any, **k: Any) -> None:
        msg = "upload failed"
        raise OSError(msg)

    with monkeypatch.context() as m:
        m.setattr(mlflow.MlflowClient, "log_artifacts", boom)
        with pytest.raises(OSError, match="upload failed"):
            log_analysis_run(folder_, paths(), "r", {}, store)
    assert {a.path for a in client.list_artifacts(run_id, "d")} == {"d/old.txt"}
    assert log_analysis_run(folder_, paths(), "r", {}, store) == run_id
    assert {a.path for a in client.list_artifacts(run_id, "d")} == {"d/new.txt"}
    assert client.get_run(run_id).info.status == "FINISHED"


# ------------------------------------------------------------- replay inputs


def status_of(out: Path, method: str, obstacle: str, ref: str = "tuned") -> set[str]:
    rows = obstacle_rows(out, "1m", method)
    return set(
        rows[(rows["obstacle"] == obstacle) & (rows["config_ref"] == ref)]["status"]
    )


def run_ids(metrics: Path, method: str, ref: str = "tuned") -> list[str]:
    runs = pd.read_parquet(metrics / "final-1m" / "runs.parquet")
    return sorted(
        runs[(runs["method"] == method) & (runs["config_ref"] == ref)]["run_id"]
    )


def test_eval_actions_of_the_final_checkpoint_reach_the_detectors(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    method = "count-bonus-ppo-rnn"
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method=method)
    # no eval file in the tmp run folders: the detector says so, never "absent" (R78)
    assert status_of(out, method, "Hoarding or key cycling") == {"needs eval actions"}
    cycle = np.tile([1, 2, 3, 4], 30)
    for rid in run_ids(metrics, method):
        write_eval(fake_replay, rid, cycle)
    run_cli(
        metrics, out, store, "--no-log", "--force", phases=("final-1m",), method=method
    )
    assert status_of(out, method, "Hoarding or key cycling") == {"detected"}
    # the ablation's runs have no file: judged apart, still "needs eval actions"
    assert status_of(out, method, "Hoarding or key cycling", "simhash") == {
        "needs eval actions"
    }


def skill_ends(update: int = FINAL) -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": "x", "update": update, "env_steps": 1_000_000, "mode": "stochastic",
        "eval_seed": [10000, 10000], "start": [0, 30], "end": [30, 60],
        "skill": ["fight", "descend"], "key": [1, 1], "reason": ["success", "timeout"],
    })  # fmt: skip


def write_replay(
    root: Path, run_id: str, update: int, mode: str = "stochastic"
) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    s = skill_ends(update).assign(run_id=run_id, mode=mode)
    s.to_parquet(d / f"skill_ends_{update}_{mode}.parquet")
    inv = pd.DataFrame({
        "run_id": run_id, "update": update, "mode": mode, "eval_seed": [10000, 10001],
        "steps": [100, 100], "capped_steps": [0, 0],
    })  # fmt: skip
    inv.to_parquet(d / f"inventory_{update}_{mode}.parquet")


def test_replay_tables_become_the_optional_frames(tmp_path: Path) -> None:
    runs = pd.DataFrame({"run_id": ["a", "b", "c"]})
    write_replay(tmp_path, "a", 122)
    write_replay(tmp_path, "a", FINAL)
    write_replay(tmp_path, "a", FINAL, "greedy")  # the detectors read stochastic only
    write_replay(tmp_path, "b", 122)  # no final checkpoint: its rows would mislead
    frames = training.replay_frames(
        runs, {"a": FINAL, "b": FINAL, "c": FINAL}, tmp_path
    )
    ends = frames["skill_ends"]
    assert set(ends["run_id"]) == {"a"}
    assert sorted(ends["update"].unique()) == [122, FINAL]
    assert set(ends["mode"]) == {"stochastic"}
    assert set(frames["eval_inventory"]["run_id"]) == {"a"}
    # a run with files for neither stays out, so its detectors read "needs replay"
    assert training.replay_frames(runs.iloc[2:], {"c": FINAL}, tmp_path) == {}


def test_replay_frames_reach_the_skill_detectors(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    method = "smdp-skills"
    out = tmp_path / "out"
    run_cli(metrics, out, store, "--no-log", phases=("final-1m",), method=method)
    assert status_of(out, method, "Key aliasing") == {
        "needs replay (evaluation Task 6)"
    }
    for rid in run_ids(metrics, method):
        write_replay(metrics, rid, FINAL)
    try:
        run_cli(
            metrics,
            out,
            store,
            "--no-log",
            "--force",
            phases=("final-1m",),
            method=method,
        )
        assert status_of(out, method, "Key aliasing") <= {"detected", "absent"}
        # the ablation's runs have none: still needs replay
        assert status_of(out, method, "Key aliasing", "monte-carlo") == {
            "needs replay (evaluation Task 6)"
        }
    finally:
        for rid in run_ids(metrics, method):
            for p in (metrics / rid).glob("*.parquet"):
                p.unlink()


# --------------------------------------------------------------------- store


def test_the_store_is_read_through_short_queries_with_the_query_stripped(
    tmp_path: Path,
) -> None:
    from wut_murl_craftax.analysis.cases import run_dirs

    db = tmp_path / "mlruns.db"
    with closing(sqlite3.connect(db)) as con:
        con.execute("create table runs (run_uuid text, artifact_uri text)")
        con.execute("insert into runs values ('a1', ?)", ((tmp_path / "x").as_uri(),))
        con.commit()
    assert run_dirs(["a1"], f"sqlite:///{db}?timeout=300") == {"a1": tmp_path / "x"}
    # no read transaction is left open: a writer commits at once
    with closing(sqlite3.connect(db, timeout=0)) as w:
        w.execute("insert into runs values ('b2', 'file:///y')")
        w.commit()
    assert not (tmp_path / "mlruns.db?timeout=300").exists()


def test_the_store_is_opened_read_only_also_with_a_query(tmp_path: Path) -> None:
    from wut_murl_craftax.analysis.cases import run_dirs

    db = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        run_dirs(["a1"], f"sqlite:///{db}?timeout=300")
    assert not db.exists()  # a read never creates the store


def test_the_cli_refuses_an_out_folder_not_named_analysis(
    metrics: Path, fake_replay: Replay, store: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        training.main([
            "--phases", "final-1m", "--out", str(tmp_path / "elsewhere"),
            "--metrics-root", str(metrics), "--tracking-uri", store, "--no-log",
        ])  # fmt: skip


def test_the_cli_refuses_a_store_that_is_not_sqlite(
    metrics: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        training.main([
            "--phases", "final-1m", "--out", str(tmp_path / "analysis"),
            "--metrics-root", str(metrics), "--tracking-uri", "http://localhost:5000",
        ])  # fmt: skip


def test_importing_the_cli_imports_neither_jax_nor_mlflow() -> None:
    import subprocess

    code = (
        "import sys; from wut_murl_craftax.analysis import training; "
        "print('mlflow' in sys.modules, 'jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


# ------------------------------------------------------- the real replay path


def real_frames(run_dir: Path, run_id: str) -> dict[str, pd.DataFrame]:
    """The tiny run's eval files as the export tables of one ``ppo`` run."""
    from wut_murl_craftax.evaluation.episode_records import stored_records
    from wut_murl_craftax.visualisation.eval_rollouts import EvalRun, load_eval

    opened = EvalRun.load(run_dir)
    rows = []
    for update in opened.updates:
        for mode in ("stochastic", "greedy"):
            ev = load_eval(run_dir, update, mode == "greedy")
            for seed, rec in zip(
                ev["seeds"], stored_records(ev, "classic"), strict=True
            ):
                rows.append({
                    "run_id": run_id, "method": "ppo", "config_ref": "tuned",
                    "budget": "1m", "seed": 0, "update": update,
                    "env_steps": opened.env_steps(update), "mode": mode,
                    "eval_seed": int(seed), **rec.row(),
                })  # fmt: skip
    runs = pd.DataFrame([{
        "run_id": run_id, "run_name": "ppo-1m-tiny", "method": "ppo",
        "config_ref": "tuned", "budget": "1m", "phases": "final-1m",
        "config_json": json.dumps(opened.cfg),
    }])  # fmt: skip
    curves = pd.DataFrame(columns=["run_id", "env_steps", "metric", "value"])
    return {"runs": runs, "curves": curves, "eval_episodes": pd.DataFrame(rows)}


@needs_ffmpeg
def test_the_real_replay_path_writes_a_case_folder(tmp_path: Path) -> None:
    run_dir = tiny_run()
    db = tmp_path / "mlruns.db"
    with closing(sqlite3.connect(db)) as con:
        con.execute("create table runs (run_uuid text, artifact_uri text)")
        con.execute("insert into runs values ('tiny', ?)", (run_dir.as_uri(),))
        con.commit()
    out = tmp_path / "outputs" / "analysis"
    summary = training.analyse_training(
        real_frames(run_dir, "tiny"),
        methods=["ppo"],
        out=out,
        tracking_uri=f"sqlite:///{db}?timeout=300",
        metrics_root=tmp_path / "none",
        log=False,
    )
    assert summary.failed == []
    d = out / "1m" / "ppo"
    cases = pd.read_csv(d / "cases.csv", keep_default_na=False)
    assert len(cases) >= 2
    root = tmp_path / "outputs"
    for row in cases.itertuples():
        for column in ("rollout", "master", "gif"):
            assert (root / getattr(row, column)).is_file(), (row.id, column)
        facts = json.loads((d / "cases" / f"{row.id}.json").read_text())
        assert facts["id"] == row.id
        assert {"floor", "stats", "inventory", "actions", "last_actions"} <= set(facts)
    assert (d / ".complete").is_file()


def test_logging_an_analysis_run_leaves_the_global_mlflow_state_alone(
    store: str, tmp_path: Path
) -> None:
    import os

    from wut_murl_craftax.orchestration.tracking import log_analysis_run

    keys = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID")
    before = ({k: os.environ.get(k) for k in keys}, mlflow.get_tracking_uri())
    (tmp_path / "f.txt").write_text("1")
    log_analysis_run(tmp_path, [tmp_path / "f.txt"], "r", {}, store)
    assert ({k: os.environ.get(k) for k in keys}, mlflow.get_tracking_uri()) == before
