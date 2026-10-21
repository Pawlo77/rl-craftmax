"""Tests for the resumable runner: lookup, resume, signals, run directory, workers."""

import dataclasses
import functools
import inspect
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
import yaml
from _fake_registry import NO_EVAL, REGISTRY
from _fake_trainer import STEPS_PER_UPDATE, FakeTrainer
from _toy_method import TINY, TOY

from wut_murl_craftax.orchestration import queue, runner
from wut_murl_craftax.orchestration.profiles import Profile, load_profile
from wut_murl_craftax.orchestration.queue import RunSpec, config_hash, expand_phase
from wut_murl_craftax.orchestration.runner import final_score, worker_specs
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training.checkpoints import (
    checkpoint_schedule,
    load_index,
    load_params,
    load_state,
)
from wut_murl_craftax.training.ppo import make_env
from wut_murl_craftax.training.records import LOSS_COLUMNS
from wut_murl_craftax.visualisation import eval_rollouts

HERE = Path(__file__).parent
METHODS = HERE / "fixtures" / "configs" / "methods"
PROFILE = HERE / "fixtures" / "profiles" / "test.yaml"
REPO = HERE.parents[1]
ENV = "Craftax-Classic-Symbolic-v1"
EVAL_MODULE = "wut_murl_craftax.visualisation.eval_rollouts"
WORLD_GEN = "craftax.craftax.world_gen.world_gen_configs"
RUN_FILES = ["ckpt", "config.json", "episodes.npz", "eval", "metrics.npz", "state"]


class Store:
    """A throwaway MLflow store reached only through its experiment.yaml."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
        self.config = tmp_path / "experiment.yaml"
        self.config.write_text(
            yaml.safe_dump({
                "experiment": {"name": "craftax"},
                "mlflow": {"tracking_uri": self.tracking_uri, "experiment_name": "x"},
            })
        )
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self.runtime_calls: list[int | None] = []
        self.evaluated: list[tuple[str, str, str, list[str]]] = []
        self.signal_in_eval = False
        self.stop_seen: list[bool] = []
        monkeypatch.setattr(
            runner,
            "configure_runtime",
            lambda host_devices=None: self.runtime_calls.append(host_devices),
        )
        # the platform choice would change JAX_PLATFORMS of the test process
        monkeypatch.setattr(runner, "select_platforms", lambda mode: None)
        monkeypatch.delenv("MLFLOW_ACTIVE_RUN_ID", raising=False)

    def phase(
        self, name: str, configs: list[str], eval_mode: str = "final"
    ) -> list[RunSpec]:
        path = self.tmp / f"{name}.yaml"
        body = {"budget": "1m", "eval": eval_mode, "env": ENV}
        body["runs"] = {"toy": {"configs": configs, "seeds": 1}}
        path.write_text(yaml.safe_dump(body))
        return expand_phase(path, methods_dir=METHODS)

    def evaluate(
        self, run_dir: Path, spec: RunSpec, should_stop: Callable[[], bool]
    ) -> None:
        if self.signal_in_eval:
            os.kill(os.getpid(), signal.SIGTERM)
        self.stop_seen.append(should_stop())
        status = mlflow.active_run().info.status
        files = sorted(p.name for p in run_dir.iterdir())
        self.evaluated.append((spec.name, spec.phase, status, files))

    def run(self, specs: list[RunSpec], trainer: Any, **kw: Any) -> None:
        runner.main(
            specs,
            PROFILE,
            experiment_config=self.config,
            methods={"toy": trainer},
            evaluate=self.evaluate,
            **kw,
        )

    def runs(self) -> dict[str, Any]:
        """Every run of the store by name."""
        ids = [e.experiment_id for e in self.client.search_experiments()]
        runs = self.client.search_runs(ids)
        assert len({r.info.run_name for r in runs}) == len(runs), "duplicate run"
        return {r.info.run_name: r for r in runs}

    def run_dir(self, name: str) -> Path:
        return artifact_path(self.runs()[name].info.artifact_uri)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    return Store(tmp_path, monkeypatch)


def _state_of(run_dir: Path, spec: RunSpec) -> dict[str, Any]:
    like = FakeTrainer().init(spec.config, jax.random.key(spec.seed))
    return load_state(run_dir, runner.state_template(like, spec))["trainer"]


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _assert_same_run(got: Path, want: Path, spec: RunSpec) -> None:
    """Same params, optimizer, keys, method state, records and checkpoints."""
    a, b = _state_of(got, spec), _state_of(want, spec)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    m_got, m_want = _npz(got / "metrics.npz"), _npz(want / "metrics.npz")
    wall = list(m_want["cols"]).index("wall_s")
    np.testing.assert_array_equal(
        np.delete(m_got["data"], wall, 1), np.delete(m_want["data"], wall, 1)
    )
    e_got, e_want = _npz(got / "episodes.npz"), _npz(want / "episodes.npz")
    assert e_got.keys() == e_want.keys()
    for k in e_want:
        np.testing.assert_array_equal(e_got[k], e_want[k])
    updates = [e["update"] for e in load_index(want)]
    assert [e["update"] for e in load_index(got)] == updates
    assert updates == checkpoint_schedule(len(m_want["data"]), runner.N_CHECKPOINTS)
    for u in updates:
        template = {"w": np.zeros(3, np.float32)}
        np.testing.assert_array_equal(
            load_params(got, u, template)["w"], load_params(want, u, template)["w"]
        )


# --- final score ----------------------------------------------------------------


def test_final_score_widens_the_last_2_percent_to_100_episodes() -> None:
    scores = np.arange(100.0)
    assert final_score(scores, np.full(100, 60)) == 98.5  # 2 updates, 120 episodes
    assert final_score(scores, np.full(100, 10)) == 94.5  # widened to 10 updates
    assert final_score(scores, np.full(100, 1)) == 49.5  # all 100 updates


def test_final_score_weights_updates_by_their_episodes() -> None:
    scores = np.array([np.nan, 10.0, 20.0, np.nan, 40.0])
    counts = np.array([0, 30, 10, 0, 60])
    assert final_score(scores, counts) == (10 * 30 + 20 * 10 + 40 * 60) / 100
    assert math.isnan(final_score(np.full(3, np.nan), np.zeros(3)))


# --- profiles and workers -------------------------------------------------------


def test_fixture_profile_loads() -> None:
    assert load_profile(PROFILE) == Profile(
        host_devices=2,
        device_roles="cpu-actor",
        memory_limit_bytes=1_000_000_000,
        checkpoint_every=2,
    )


def test_two_workers_never_take_the_same_run(store: Store) -> None:
    specs = store.phase(
        "tune-coarse", ["published", "lr-hi", "lr-lo", "lam-lo", "ent-0"]
    )
    first, second = worker_specs(specs, 0, 2), worker_specs(specs, 1, 2)
    assert {s.name for s in first}.isdisjoint(s.name for s in second)
    assert sorted(s.name for s in first + second) == sorted(s.name for s in specs)
    store.run(specs, FakeTrainer(), worker_index=0, num_workers=2)
    names_0 = set(store.runs())
    store.run(specs, FakeTrainer(), worker_index=1, num_workers=2)
    names_1 = set(store.runs()) - names_0
    assert names_0 == {s.name for s in first}
    assert names_1 == {s.name for s in second}
    with pytest.raises(ValueError, match="worker index"):
        worker_specs(specs, 2, 2)


def test_a_tracking_uri_sends_the_runs_to_another_store(
    store: Store, tmp_path: Path
) -> None:
    """A cluster node writes its own database, not the experiment file's."""
    specs = store.phase("tune-coarse", ["published"])
    node_uri = f"sqlite:///{tmp_path / 'node' / 'mlruns-node-1.db'}"
    (tmp_path / "node").mkdir()
    store.run(specs, FakeTrainer(), tracking_uri=node_uri)
    assert store.runs() == {}
    node = mlflow.MlflowClient(tracking_uri=node_uri)
    ids = [e.experiment_id for e in node.search_experiments()]
    (run,) = node.search_runs(ids)
    assert run.info.run_name == specs[0].name
    assert run.info.status == "FINISHED"
    assert artifact_path(run.info.artifact_uri).is_relative_to(tmp_path / "node")


def test_max_runs_counts_the_runs_it_trains(store: Store) -> None:
    specs = store.phase("tune-coarse", ["published", "lr-hi", "lr-lo"])
    store.run(specs[:1], FakeTrainer())
    trainer = FakeTrainer()
    store.run(specs, trainer, max_runs=1)
    assert set(store.runs()) == {specs[0].name, specs[1].name}
    assert trainer.updates == [1, 2, 3]


# --- MLflow bookkeeping ---------------------------------------------------------


def test_the_run_directory_is_the_mlflow_artifact_directory(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    run = store.runs()[spec.name]
    out = store.run_dir(spec.name)
    assert out == Path(run.info.artifact_uri.removeprefix("file://"))
    assert out.is_relative_to(store.tmp / "mlruns")
    assert sorted(p.name for p in out.iterdir() if p.name in RUN_FILES) == RUN_FILES
    assert json.loads((out / queue.CONFIG_ARTIFACT).read_text()) == spec.config
    assert run.info.status == "FINISHED"
    tags = {
        queue.TAG_METHOD: "toy",
        queue.TAG_BUDGET: "1m",
        queue.TAG_SEED: "0",
        queue.TAG_CONFIG_HASH: spec.config_hash,
        queue.TAG_PHASES: "tune-coarse",
    }
    assert {k: run.data.tags.get(k) for k in tags} == tags
    assert queue.SCORE_METRIC in run.data.metrics
    history = store.client.get_metric_history(run.info.run_id, "train/return")
    assert [m.step for m in history] == [STEPS_PER_UPDATE * u for u in (1, 2, 3)]


def test_init_gets_the_budget_steps_outside_the_saved_config(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    trainer = FakeTrainer()
    store.run([spec], trainer)
    assert trainer.init_steps == [spec.steps]
    assert "steps" not in spec.config
    config = json.loads((store.run_dir(spec.name) / queue.CONFIG_ARTIFACT).read_text())
    assert config == spec.config


def test_the_tracker_comes_from_experiment_yaml_named_after_the_phase(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, str]] = []
    real = runner.build_mlflow_tracker

    def spy(experiment_config: Any, **kw: Any) -> Any:
        built.append((experiment_config.name, experiment_config.mlflow.experiment_name))
        return real(experiment_config, **kw)

    monkeypatch.setattr(runner, "build_mlflow_tracker", spy)
    store.run(store.phase("tune-coarse", ["published"]), FakeTrainer())
    assert built == [("craftax", "tune-coarse")]
    (run,) = store.runs().values()
    assert store.client.get_experiment(run.info.experiment_id).name == "tune-coarse"
    default = inspect.signature(runner.main).parameters["experiment_config"].default
    assert default == queue.EXPERIMENT_CONFIG


def test_the_shipped_experiment_yaml_names_the_project() -> None:
    raw = yaml.safe_load(queue.EXPERIMENT_CONFIG.read_text())
    assert raw["experiment"]["name"] == raw["mlflow"]["experiment_name"] == "craftax"
    assert raw["mlflow"]["run_name"] is None


def test_evaluation_runs_after_training_and_before_finished(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    ((name, phase, status, files),) = store.evaluated
    assert (name, phase, status) == (spec.name, "tune-coarse", "RUNNING")
    assert set(RUN_FILES) <= set(files)


def test_a_finished_run_is_skipped_but_evaluated(store: Store) -> None:
    specs = store.phase("tune-coarse", ["published", "lr-hi"])
    store.run(specs, FakeTrainer())
    trainer = FakeTrainer()
    store.run(specs, trainer)
    assert trainer.updates == []
    assert len(store.runs()) == 2
    assert all(r.info.status == "FINISHED" for r in store.runs().values())
    assert [e[0] for e in store.evaluated] == [s.name for s in specs] * 2


def test_a_run_of_another_phase_is_found_not_duplicated(store: Store) -> None:
    coarse = store.phase("tune-coarse", ["published", "lr-hi"])
    store.run(coarse, FakeTrainer())
    (final,) = store.phase("final-1m", ["published"], eval_mode="log_spaced")
    assert final.name == coarse[0].name
    trainer = FakeTrainer()
    store.run([final], trainer)
    assert trainer.updates == []
    runs = store.runs()
    assert len(runs) == 2
    assert runs[final.name].data.tags[queue.TAG_PHASES] == "tune-coarse,final-1m"
    assert store.evaluated[-1][:3] == (final.name, "final-1m", "RUNNING")
    store.run([final], FakeTrainer())
    assert store.runs()[final.name].data.tags[queue.TAG_PHASES] == (
        "tune-coarse,final-1m"
    )


# --- interruption and resume ----------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_saves_the_state_and_marks_the_run_killed(
    store: Store, sig: int
) -> None:
    specs = store.phase("tune-coarse", ["published", "lr-hi"])
    trainer = FakeTrainer(signal_at=(2, sig))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(SystemExit) as exit_info:
        store.run(specs, trainer)
    assert exit_info.value.code == 128 + sig
    assert trainer.updates == [1, 2]  # the chunk ends, the next run never starts
    runs = store.runs()
    assert list(runs) == [specs[0].name]
    assert runs[specs[0].name].info.status == "KILLED"
    out = store.run_dir(specs[0].name)
    assert (out / "state" / "state.npz").is_file()
    assert _npz(out / "metrics.npz")["data"].shape[0] == 2
    assert {s: signal.getsignal(s) for s in handlers} == handlers


def test_each_training_segment_records_its_code_version(
    store: Store, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A resume under another commit is tagged and logged, never refused."""
    versions = iter([("a" * 40, False), ("b" * 40, True)])
    monkeypatch.setattr(runner, "code_version", lambda: next(versions))
    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(SystemExit):
        store.run([spec], FakeTrainer((2, signal.SIGINT)))
    tags = store.runs()[spec.name].data.tags
    assert (tags["code/commit"], tags["code/dirty"]) == ("a" * 40, "false")
    assert tags["code/segments"] == "a" * 12
    assert "code/changed" not in tags
    store.run([spec], FakeTrainer())
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    tags = run.data.tags
    assert (tags["code/commit"], tags["code/dirty"]) == ("a" * 40, "false")
    assert tags["code/segments"] == f"{'a' * 12},{'b' * 12}+dirty"
    assert tags["code/changed"] == "true"
    assert f"{spec.name} resumes under commit {'b' * 12}" in caplog.text


def test_a_resume_into_another_state_layout_fails_the_run_clearly(store: Store) -> None:
    """New code with another state shape never restores a run's old state."""

    class Wider(FakeTrainer):
        def init(self, config: Any, key: jax.Array) -> dict[str, Any]:
            state = super().init(config, key)
            return {**state, "params": {"w": jnp.zeros(4)}}

    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(SystemExit):
        store.run([spec], FakeTrainer((2, signal.SIGINT)))
    trainer = Wider()
    with pytest.raises(RuntimeError, match=r"\['w'\].*\(3,\).*\(4,\)"):
        store.run([spec], trainer)
    run = store.runs()[spec.name]
    assert run.info.status == "FAILED"
    assert "comes from other code" in run.data.tags["fail/error"]
    assert trainer.updates == []


@pytest.mark.parametrize("left", ["KILLED", "RUNNING"])
@pytest.mark.parametrize(
    ("spu", "cut_at", "saved_at"),
    [
        (STEPS_PER_UPDATE, 2, 2),  # 3 updates, chunks of 1
        (25_000, 11, 12),  # 40 updates: the chunk 10-12 ends before the stop
    ],
)
def test_an_interrupted_run_resumes_to_the_uninterrupted_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    left: str,
    spu: int,
    cut_at: int,
    saved_at: int,
) -> None:
    whole = Store(tmp_path / "whole", monkeypatch)
    (spec,) = whole.phase("tune-coarse", ["published"])
    whole.run([spec], FakeTrainer(steps_per_update=spu))

    cut = Store(tmp_path / "cut", monkeypatch)
    first = FakeTrainer((cut_at, signal.SIGINT), steps_per_update=spu)
    with pytest.raises(SystemExit):
        cut.run([spec], first)
    assert first.updates == list(range(1, saved_at + 1))
    rid = cut.runs()[spec.name].info.run_id
    if left == "RUNNING":  # a hard kill leaves the run RUNNING
        cut.client.update_run(rid, status="RUNNING")
    second = FakeTrainer(steps_per_update=spu)
    cut.run([spec], second)
    n_updates = 1_000_000 // spu
    assert second.updates == list(range(saved_at + 1, n_updates + 1))
    run = cut.runs()[spec.name]
    assert (len(cut.runs()), run.info.run_id, run.info.status) == (1, rid, "FINISHED")
    _assert_same_run(cut.run_dir(spec.name), whole.run_dir(spec.name), spec)
    history = cut.client.get_metric_history(rid, "train/return")
    assert [m.step for m in history] == [spu * u for u in range(1, n_updates + 1)]
    score = cut.client.get_run(rid).data.metrics[queue.SCORE_METRIC]
    assert score == whole.runs()[spec.name].data.metrics[queue.SCORE_METRIC]


def test_a_signal_in_the_last_chunk_skips_the_evaluation(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(SystemExit):
        store.run([spec], FakeTrainer((3, signal.SIGINT)))
    run = store.runs()[spec.name]
    assert (run.info.status, store.evaluated) == ("KILLED", [])
    assert queue.SCORE_METRIC not in run.data.metrics
    trainer = FakeTrainer()
    store.run([spec], trainer)
    run = store.runs()[spec.name]
    assert (trainer.updates, run.info.status) == ([], "FINISHED")
    assert [e[0] for e in store.evaluated] == [spec.name]
    assert queue.SCORE_METRIC in run.data.metrics


def test_evaluate_sees_a_signal_and_the_run_stays_killed(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.signal_in_eval = True
    with pytest.raises(SystemExit) as exit_info:
        store.run([spec], FakeTrainer())
    assert exit_info.value.code == 128 + signal.SIGTERM
    assert store.stop_seen == [True]
    assert store.runs()[spec.name].info.status == "KILLED"
    store.signal_in_eval = False
    store.run([spec], FakeTrainer())
    assert store.stop_seen == [True, False]
    assert store.runs()[spec.name].info.status == "FINISHED"


def test_a_run_that_cannot_be_reopened_fails_loudly_without_a_duplicate(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    rid = store.runs()[spec.name].info.run_id
    real = mlflow.start_run

    def locked(*a: Any, run_id: str | None = None, **kw: Any) -> Any:
        if run_id is not None:
            raise mlflow.exceptions.MlflowException.invalid_parameter_value("locked")
        return real(*a, **kw)

    monkeypatch.setattr(mlflow, "start_run", locked)
    for trainer in (FakeTrainer(), FakeTrainer()):
        with pytest.raises(RuntimeError, match=f"{spec.name}.*{rid}"):
            store.run([spec], trainer)
        assert trainer.updates == []
    assert [r.info.run_id for r in store.runs().values()] == [rid]


@pytest.mark.parametrize("variable", ["MLFLOW_ACTIVE_RUN_ID", "MLFLOW_RUN_ID"])
def test_a_run_id_in_the_environment_never_hijacks_a_new_run(
    store: Store, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    first, second = store.phase("tune-coarse", ["published", "lr-hi"])
    store.run([first], FakeTrainer())
    foreign = store.runs()[first.name]
    monkeypatch.setenv(variable, foreign.info.run_id)
    store.run([second], FakeTrainer())
    assert variable not in os.environ
    assert store.runs()[second.name].info.run_id != foreign.info.run_id
    assert store.runs()[first.name].data.tags == foreign.data.tags
    assert store.runs()[first.name].info.status == "FINISHED"


def test_two_workers_creating_one_phase_experiment_share_it(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = store.phase("tune-coarse", ["published", "lr-hi"])
    store.run([first], FakeTrainer())
    real = mlflow.get_experiment_by_name
    calls: list[str] = []

    def stale_once(name: str) -> Any:  # the other worker created it meanwhile
        calls.append(name)
        return None if len(calls) == 1 else real(name)

    monkeypatch.setattr(mlflow, "get_experiment_by_name", stale_once)
    store.run([second], FakeTrainer())
    runs = store.runs()
    assert runs[second.name].info.status == "FINISHED"
    assert runs[second.name].info.experiment_id == runs[first.name].info.experiment_id


def test_a_trainer_error_marks_the_run_failed(store: Store) -> None:
    class Broken(FakeTrainer):
        def train_chunk(self, state: Any, n_updates: int) -> Any:
            raise FloatingPointError

    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(RuntimeError, match=r"1 of 1 runs failed.*FloatingPointError"):
        store.run([spec], Broken())
    run = store.runs()[spec.name]
    assert run.info.status == "FAILED"
    assert run.data.tags["fail/error"].startswith("FloatingPointError")
    assert run.data.tags["fail/count"] == "1"
    trainer = FakeTrainer()
    store.run([spec], trainer)
    assert trainer.updates == [1, 2, 3]
    assert store.runs()[spec.name].info.status == "FINISHED"


def test_a_failing_run_does_not_stop_the_queue(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """The error is recorded on the run and logged; the next spec still runs."""

    class FailsFirst(FakeTrainer):
        def init(self, config: Any, key: jax.Array) -> Any:
            if not self.init_steps:
                self.init_steps.append(None)
                msg = "out of memory"
                raise MemoryError(msg)
            return super().init(config, key)

    first, second = store.phase("tune-coarse", ["published", "lr-hi"])
    trainer = FailsFirst()
    with pytest.raises(RuntimeError, match=rf"1 of 2 runs failed: {first.name}"):
        store.run([first, second], trainer)
    runs = store.runs()
    assert runs[first.name].info.status == "FAILED"
    assert runs[first.name].data.tags["fail/error"] == "MemoryError: out of memory"
    assert runs[second.name].info.status == "FINISHED"
    assert trainer.updates == [1, 2, 3]
    assert [e[0] for e in store.evaluated] == [second.name]
    assert f"{first.name} failed" in caplog.text


# --- runtime --------------------------------------------------------------------


def test_main_configures_the_runtime_once_before_the_trainers(store: Store) -> None:
    trainer = FakeTrainer()
    store.run(store.phase("tune-coarse", ["published", "lr-hi"]), trainer)
    assert store.runtime_calls == [2]


def test_eval_metrics_of_files_written_before_a_crash_are_logged_once(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    real, calls = eval_rollouts.evaluate, []

    def crash_after_one_file(*args: Any, **kw: Any) -> Any:
        calls.append(args[2])
        if len(calls) > 1:
            msg = "crash while evaluating"
            raise RuntimeError(msg)
        return real(*args, **kw)

    monkeypatch.setattr(eval_rollouts, "evaluate", crash_after_one_file)
    run_main = functools.partial(
        runner.main,
        [spec],
        PROFILE,
        experiment_config=store.config,
        methods={"toy": FakeTrainer()},
    )
    with pytest.raises(RuntimeError, match="crash while evaluating"):
        run_main()
    assert store.runs()[spec.name].info.status == "FAILED"
    monkeypatch.setattr(eval_rollouts, "evaluate", real)
    run_main()
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    for mode in ("stochastic", "greedy"):
        for name in ("return", "score_pct"):
            history = store.client.get_metric_history(
                run.info.run_id, f"eval/{mode}/{name}"
            )
            assert len(history) == 1, (mode, name)


def test_main_splits_the_cpu_into_the_profiles_host_devices(store: Store) -> None:
    """With the real evaluation wired: the runner imports it after configure_runtime."""
    (spec,) = store.phase("tune-coarse", ["published"])
    code = (
        f"import sys; sys.path.insert(0, {str(HERE)!r})\n"
        "from pathlib import Path\n"
        "from _fake_trainer import FakeTrainer\n"
        "from wut_murl_craftax.orchestration import runner\n"
        f"assert {EVAL_MODULE!r} not in sys.modules\n"
        f"assert {WORLD_GEN!r} not in sys.modules\n"
        "from wut_murl_craftax.orchestration.queue import expand_phase\n"
        f"specs = expand_phase(Path({str(store.tmp / 'tune-coarse.yaml')!r}),"
        f" methods_dir=Path({str(METHODS)!r}))\n"
        "trainer = FakeTrainer()\n"
        f"runner.main(specs, {str(PROFILE)!r},"
        f" experiment_config={str(store.config)!r}, methods={{'toy': trainer}})\n"
        f"print({WORLD_GEN!r} in sys.modules, trainer.cpu_devices)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "XLA_FLAGS": ""},
        cwd=store.tmp,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    # Craftax imported before any trace: a Classic-only queue never builds Craftax
    assert out.stdout.split()[-2:] == ["True", str(load_profile(PROFILE).host_devices)]
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    last = load_index(store.run_dir(spec.name))[-1]
    for mode in ("stochastic", "greedy"):
        assert (store.run_dir(spec.name) / "eval" / mode / "00003.npz").exists()
        for name in ("return", "score_pct"):
            (point,) = store.client.get_metric_history(
                run.info.run_id, f"eval/{mode}/{name}"
            )
            assert point.step == last["env_steps"]
    ret = store.client.get_metric_history(run.info.run_id, "eval/greedy/return")
    pct = store.client.get_metric_history(run.info.run_id, "eval/greedy/score_pct")
    assert pct[0].value == pytest.approx(ret[0].value / 22 * 100)


# --- abort rules and flags ------------------------------------------------------


def nan_at(update: int) -> Callable[[int], dict[str, float]]:
    return lambda u: {"ppo/value_loss": float("nan") if u == update else 0.5}


def tags_of(store: Store, name: str, prefix: str) -> dict[str, str]:
    tags = store.runs()[name].data.tags
    return {k: v for k, v in tags.items() if k.startswith(prefix)}


def test_abort_metrics_the_trainer_never_reports_are_tagged_once(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo in ``abort.finite`` or a flag metric would disable its guard silently."""
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer(extra=lambda u: {"ppo/entropy": 0.5}))
    unreported = "ppo/approx_kl,ppo/explained_variance,ppo/value_loss"
    tags = tags_of(store, spec.name, "warn/")
    assert tags == {"warn/abort_unreported": unreported}
    assert caplog.text.count("never reports") == 1
    (other,) = store.phase("tune-coarse", ["lr-hi"])
    every = {"ppo/entropy": 0.5, "ppo/approx_kl": 0.0}
    every |= {"ppo/explained_variance": 1.0, "ppo/value_loss": 0.1}
    store.run([other], FakeTrainer(extra=lambda u: every))
    assert tags_of(store, other.name, "warn/") == {}


def test_a_non_finite_loss_fails_the_run_and_keeps_its_last_state(store: Store) -> None:
    specs = store.phase("tune-coarse", ["published", "lr-hi"])
    trainer = FakeTrainer(extra=nan_at(3))
    store.run(specs, trainer)  # the queue goes on with the next run
    assert trainer.updates == [1, 2, 3, 1, 2, 3]
    for spec in specs:
        run = store.runs()[spec.name]
        assert run.info.status == "FAILED"
        assert queue.SCORE_METRIC not in run.data.metrics
        out = store.run_dir(spec.name)
        saved = load_state(out, runner.state_template(_state_of(out, spec), spec))
        assert saved["update"] == 2  # the chunk with the NaN was not saved
        assert _npz(out / "metrics.npz")["data"].shape[0] == 2
        assert tags_of(store, spec.name, "abort/") == {
            "abort/count": "1",
            "abort/metric": "ppo/value_loss",
            "abort/update": "3",
        }
        steps = [
            m.step
            for m in store.client.get_metric_history(run.info.run_id, "ppo/value_loss")
        ]
        assert steps == [STEPS_PER_UPDATE * u for u in (1, 2)]
    assert store.evaluated == []


def test_the_next_call_resumes_it_once_and_a_second_nan_marks_a_bug(
    store: Store,
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer(extra=nan_at(3)))
    resumed = FakeTrainer(extra=nan_at(3))
    store.run([spec], resumed)
    assert resumed.updates == [3]  # from the state after update 2
    assert tags_of(store, spec.name, "abort/") == {
        "abort/count": "2",
        "abort/metric": "ppo/value_loss",
        "abort/update": "3",
        "abort/bug": "true",
    }
    assert store.runs()[spec.name].info.status == "FAILED"
    skipped = FakeTrainer()
    store.run([spec], skipped, max_runs=1)
    assert skipped.updates == []
    assert [r.info.status for r in store.runs().values()] == ["FAILED"]
    assert store.evaluated == []


def test_a_rerun_that_stays_finite_equals_the_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (spec,) = (whole := Store(tmp_path / "whole", monkeypatch)).phase(
        "tune-coarse", ["published"]
    )
    clean = lambda u: {"ppo/value_loss": 0.5 / u}  # noqa: E731
    whole.run([spec], FakeTrainer(extra=clean))
    cut = Store(tmp_path / "cut", monkeypatch)
    cut.run(
        [spec],
        FakeTrainer(
            extra=lambda u: {"ppo/value_loss": float("nan") if u == 3 else 0.5 / u}
        ),
    )
    second = FakeTrainer(extra=clean)
    cut.run([spec], second)
    assert second.updates == [3]
    assert cut.runs()[spec.name].info.status == "FINISHED"
    _assert_same_run(cut.run_dir(spec.name), whole.run_dir(spec.name), spec)
    np.testing.assert_array_equal(
        _npz(cut.run_dir(spec.name) / "extras.npz")["ppo/value_loss"],
        _npz(whole.run_dir(spec.name) / "extras.npz")["ppo/value_loss"],
    )
    rid = cut.runs()[spec.name].info.run_id
    history = cut.client.get_metric_history(rid, "ppo/value_loss")
    assert [(m.step, m.value) for m in history] == [
        (STEPS_PER_UPDATE * u, pytest.approx(0.5 / u)) for u in (1, 2, 3)
    ]


def test_an_unlisted_nan_method_metric_does_not_abort_a_listed_one_does(
    store: Store,
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    unlisted = FakeTrainer(extra=lambda u: {"sil/loss_mean": float("nan")})
    store.run([spec], unlisted)  # `finite` of the toy method lists only ppo/value_loss
    run = store.runs()[spec.name]
    assert (run.info.status, unlisted.updates) == ("FINISHED", [1, 2, 3])
    assert tags_of(store, spec.name, "abort/") == {}
    (other,) = store.phase("tune-coarse", ["lr-hi"])
    store.run([other], FakeTrainer(extra=nan_at(2)))
    assert store.runs()[other.name].info.status == "FAILED"
    assert tags_of(store, other.name, "abort/")["abort/metric"] == "ppo/value_loss"


def test_a_non_finite_reported_loss_column_aborts(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer(loss_nan_at=2))
    assert store.runs()[spec.name].info.status == "FAILED"
    assert tags_of(store, spec.name, "abort/")["abort/update"] == "2"


def test_unreported_loss_columns_are_neither_checked_nor_logged(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    reported = ("value_loss", "grad_norm")
    store.run([spec], FakeTrainer(loss_columns=reported))
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    logged = {k for k in run.data.metrics if k.startswith("train/")}
    assert {"train/value_loss", "train/grad_norm"} <= logged
    assert not logged & {f"train/{c}" for c in LOSS_COLUMNS if c not in reported}
    table = _npz(store.run_dir(spec.name) / "metrics.npz")
    unreported = table["data"][:, list(table["cols"]).index("policy_loss")]
    assert np.isnan(unreported).all()  # the fixed layout keeps NaN for them


def test_flags_tag_the_run_without_stopping_it(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])

    def metrics(u: int) -> dict[str, float]:
        return {
            "ppo/approx_kl": 0.2,  # above 0.1 at all 3 updates: flagged at the 3rd
            "ppo/entropy": 0.1 if u == 1 else 3.0,  # collapse in the first half only
            "ppo/explained_variance": -1.0 if u == 1 else 0.5,  # before the 2nd half
        }

    trainer = FakeTrainer(extra=metrics)
    store.run([spec], trainer)
    assert trainer.updates == [1, 2, 3]
    assert store.runs()[spec.name].info.status == "FINISHED"
    assert tags_of(store, spec.name, "flag/") == {
        "flag/kl_high": "3",
        "flag/entropy_collapse": "1",
    }


def test_a_flag_needs_consecutive_updates_and_a_later_value_failure_counts(
    store: Store,
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    kl = {1: 0.2, 2: 0.05, 3: 0.2}
    ev = {1: 0.5, 2: -0.1, 3: 0.5}
    store.run(
        [spec],
        FakeTrainer(
            extra=lambda u: {"ppo/approx_kl": kl[u], "ppo/explained_variance": ev[u]}
        ),
    )
    assert tags_of(store, spec.name, "flag/") == {"flag/value_failure": "2"}


def test_method_metrics_are_logged_at_their_env_step_and_survive_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (spec,) = (whole := Store(tmp_path / "whole", monkeypatch)).phase(
        "tune-coarse", ["published"]
    )
    extra = lambda u: {"ppo/lr": 1e-3 / u}  # noqa: E731
    whole.run([spec], FakeTrainer(extra=extra))
    cut = Store(tmp_path / "cut", monkeypatch)
    with pytest.raises(SystemExit):
        cut.run([spec], FakeTrainer((2, signal.SIGINT), extra=extra))
    cut.run([spec], FakeTrainer(extra=extra))
    for store in (whole, cut):
        rid = store.runs()[spec.name].info.run_id
        history = store.client.get_metric_history(rid, "ppo/lr")
        assert [(m.step, m.value) for m in history] == [
            (STEPS_PER_UPDATE * u, pytest.approx(1e-3 / u)) for u in (1, 2, 3)
        ]


def test_saved_method_metrics_are_cut_or_padded_to_the_resumed_update(
    tmp_path: Path,
) -> None:
    extras = runner._Extras(tmp_path / runner.EXTRAS_FILE)
    extras.extend(0, {"ppo/lr": np.array([1.0, 2.0, 3.0])})
    extras.save()  # a crash after this and before the state save leaves it ahead
    assert runner._Extras.load(tmp_path, 2).series == {"ppo/lr": [1.0, 2.0]}
    short = runner._Extras.load(tmp_path, 5).series["ppo/lr"]
    assert short[:3] == [1.0, 2.0, 3.0]
    assert len(short) == 5
    assert all(math.isnan(v) for v in short[3:])
    assert runner._Extras.load(tmp_path / "none", 2).series == {}


# --- lost checkpoint files ------------------------------------------------------


def _ckpt(store: Store, name: str, update: int) -> Path:
    return store.run_dir(name) / "ckpt" / f"{update:05d}.msgpack"


def test_a_lost_final_checkpoint_is_rebuilt_from_the_state(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    final = _ckpt(store, spec.name, 3)
    saved = final.read_bytes()
    final.unlink()
    (_ckpt(store, spec.name, 0)).unlink()  # the initial one comes back from the seed
    trainer = FakeTrainer()
    store.run([spec], trainer)
    assert trainer.updates == []  # nothing is retrained
    assert final.read_bytes() == saved
    assert _ckpt(store, spec.name, 0).is_file()
    index = load_index(store.run_dir(spec.name))
    assert [e["update"] for e in index] == [0, 1, 2, 3]
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    assert "ckpt/missing" not in run.data.tags
    assert [e[0] for e in store.evaluated] == [spec.name] * 2


def test_an_earlier_lost_checkpoint_fails_a_complete_run_then_skips_it(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    _ckpt(store, spec.name, 1).unlink()
    _ckpt(store, spec.name, 2).unlink()
    with pytest.raises(RuntimeError, match=r"\[1, 2\] are lost"):
        store.run([spec], FakeTrainer())
    for _ in range(2):  # later calls skip it with a warning, they never retrain it
        trainer = FakeTrainer()
        store.run([spec], trainer)
        assert trainer.updates == []
        run = store.runs()[spec.name]
        assert run.info.status == "FAILED"
        assert run.data.tags["ckpt/missing"] == "1,2"
    assert [e["update"] for e in load_index(store.run_dir(spec.name))] == [0, 3]
    assert len(store.evaluated) == 1  # only the first call evaluated
    assert f"{spec.name} lost checkpoints 1,2 for good, skipped" in caplog.text


def test_a_lost_checkpoint_of_a_run_in_progress_stays_tagged_and_fails_at_the_end(
    store: Store,
) -> None:
    spu = 25_000  # 40 updates: the schedule has checkpoints at 1, 2, ...
    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(SystemExit):
        store.run([spec], FakeTrainer((11, signal.SIGINT), steps_per_update=spu))
    _ckpt(store, spec.name, 1).unlink()
    with pytest.raises(SystemExit):  # tagged at once, interrupted again after it
        store.run([spec], FakeTrainer((20, signal.SIGINT), steps_per_update=spu))
    run = store.runs()[spec.name]
    assert (run.info.status, run.data.tags["ckpt/missing"]) == ("KILLED", "1")
    with pytest.raises(RuntimeError, match=r"\[1\] are lost"):  # trains on, fails
        store.run([spec], FakeTrainer(steps_per_update=spu))
    run = store.runs()[spec.name]
    assert (run.info.status, run.data.tags["ckpt/missing"]) == ("FAILED", "1")
    assert _ckpt(store, spec.name, 40).is_file()
    trainer = FakeTrainer(steps_per_update=spu)
    store.run([spec], trainer)  # a rerun skips it
    assert trainer.updates == []


def test_a_methods_knobs_reach_its_hooks_in_training_and_evaluation(
    store: Store,
) -> None:
    """Two grid entries of a hook method: the runner trains each entry's bonus and
    evaluation opens each run with its own env wrapper (2 PPO-RNN updates); the hooks
    read the method keys and the resolved base config (``gamma``)."""
    knobs = [
        {"toy_bonus": 0.0, "toy_features": 0, "gamma": 0.99},
        {"toy_bonus": 0.5, "toy_features": 3, "gamma": 0.5},
    ]
    specs = []
    for k in knobs:
        cfg = {"method": "toy", **TINY, **k}
        specs.append(
            RunSpec("toy", "a", "0.128k", ENV, 0, cfg, config_hash(cfg), "x", "final")
        )
    store.run(specs, TOY)
    plain, params = make_env(ENV)
    width = plain.observation_space(params).shape[0]
    for k, spec in zip(knobs, specs, strict=True):
        out = store.run_dir(spec.name)
        with np.load(out / runner.EXTRAS_FILE) as z:
            per_update = 64 * k["toy_bonus"] * k["gamma"]  # 8 envs x 8 steps
            assert z["toy/bonus_sum"].tolist() == [per_update, 2 * per_update]
        run = eval_rollouts.EvalRun.load(out, {"toy": TOY})
        obs, _ = run.env.reset(jax.random.key(0), run.env_params)
        assert obs.shape == (width + k["toy_features"],)
        assert (obs[width:] == k["gamma"]).all()  # the builder saw the base gamma
        policy, p = run.policy, run.params(run.updates[-1])
        act, _ = policy.step(
            p, policy.init_carry(1), obs[None], jax.random.key(1), True
        )
        assert act.shape == (1,)


# --- one child process per run ----------------------------------------------------


def _in_children(store: Store, specs: list[RunSpec]) -> None:
    """``runner.main`` as the queue calls it: no live trainers, a child per run."""
    runner.main(
        specs,
        PROFILE,
        experiment_config=store.config,
        registry=REGISTRY,
        evaluate=NO_EVAL,
    )


def _as(spec: RunSpec, registry: str) -> RunSpec:
    return dataclasses.replace(spec, registry_name=registry)


def test_each_run_trains_in_its_own_child_process(store: Store) -> None:
    specs = store.phase("tune-coarse", ["published", "lr-hi"])
    _in_children(store, specs)
    runs = store.runs()
    assert {runs[s.name].info.status for s in specs} == {"FINISHED"}
    pids = {runs[s.name].data.tags["pid"] for s in specs}
    assert len(pids) == 2
    assert str(os.getpid()) not in pids
    assert store.runtime_calls == []  # the parent never sets up JAX


def test_a_crashed_child_fails_its_run_and_the_queue_continues(store: Store) -> None:
    published, hi = store.phase("tune-coarse", ["published", "lr-hi"])
    crash = _as(published, "crash")
    with pytest.raises(RuntimeError, match=rf"1 of 2 runs failed: {crash.name}.*-9"):
        _in_children(store, [crash, hi])
    runs = store.runs()
    assert runs[crash.name].info.status == "FAILED"
    assert "exited with -9" in runs[crash.name].data.tags["fail/error"]
    assert runs[hi.name].info.status == "FINISHED"


def test_a_signal_to_the_parent_stops_the_child_with_its_state_saved(
    store: Store,
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    stop = _as(spec, "stop")
    with pytest.raises(SystemExit) as exit_info:
        _in_children(store, [stop])
    assert exit_info.value.code == 128 + signal.SIGINT
    run = store.runs()[stop.name]
    assert run.info.status == "KILLED"
    saved = load_state(
        store.run_dir(stop.name),
        runner.state_template(FakeTrainer().init(stop.config, jax.random.key(0)), stop),
    )
    assert saved["update"] == 2
    _in_children(store, [stop])  # the rerun resumes at update 3 and finishes
    assert store.runs()[stop.name].info.status == "FINISHED"


_QUEUE = """
import json, sys
from wut_murl_craftax.orchestration import runner
from wut_murl_craftax.orchestration.queue import RunSpec
spec = RunSpec(**json.loads(sys.argv[1]))
runner.main([spec], sys.argv[2], experiment_config=sys.argv[3],
            registry="_fake_registry:METHODS", evaluate="_fake_registry:no_eval")
"""
"""A queue process of one spec, as ``python -m ...queue`` would run it."""


def _start_queue(
    store: Store,
    spec: RunSpec,
    launcher: tuple[str, ...] = (sys.executable,),
    *,
    nohup: bool = False,
) -> subprocess.Popen[str]:
    """The queue in its own session, as a terminal's foreground job (under ``nohup``:
    SIGHUP ignored, as the queue inherits it)."""
    path = os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])
    argv = [json.dumps(dataclasses.asdict(spec)), str(PROFILE), str(store.config)]
    return subprocess.Popen(
        [*launcher, "-c", _QUEUE, *argv],
        env={**os.environ, "PYTHONPATH": path},
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=(lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
        if nohup
        else None,
    )


def _until(cond: Callable[[], Any], what: str, timeout: float = 120.0) -> Any:
    deadline = time.monotonic() + timeout
    while not (got := cond()):
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.2)
    return got


def _training_pid(store: Store, name: str) -> int:
    """The PID of the child once it trains the run."""

    def pid() -> int | None:
        run = store.runs().get(name)
        return int(run.data.tags["pid"]) if run and "pid" in run.data.tags else None

    return int(_until(pid, f"{name} to start training"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_child_whose_queue_dies_saves_its_state_and_stops(store: Store) -> None:
    """kill -9 of the queue process: the orphan stops as if signalled."""
    (spec,) = store.phase("tune-coarse", ["published"])
    slow = _as(spec, "slow")
    queue_process = _start_queue(store, slow)
    child = _training_pid(store, slow.name)
    os.kill(queue_process.pid, signal.SIGKILL)
    queue_process.communicate(timeout=120)
    _until(lambda: not _alive(child), "the orphaned child to exit", 60)
    run = store.runs()[slow.name]
    assert run.info.status == "KILLED"
    out = store.run_dir(slow.name)
    assert (out / "state" / "state.npz").is_file()
    assert not (out / runner.OWNER_FILE).exists()


def test_a_run_with_a_live_child_is_never_trained_twice(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    slow = _as(spec, "slow")
    queue_process = _start_queue(store, slow)
    child = _training_pid(store, slow.name)
    _in_children(store, [slow])  # a second queue over the same run
    assert f"{slow.name} is trained by live process {child}" in caplog.text
    run = store.runs()[slow.name]
    assert (run.info.status, run.data.tags["pid"]) == ("RUNNING", str(child))
    os.kill(queue_process.pid, signal.SIGINT)
    queue_process.communicate(timeout=120)
    assert queue_process.returncode == 128 + signal.SIGINT
    assert store.runs()[slow.name].info.status == "KILLED"


def test_a_hangup_of_the_queue_stops_the_child_like_a_signal(store: Store) -> None:
    """SIGHUP (the terminal or SSH session closed) is a stop, not a death."""
    (spec,) = store.phase("tune-coarse", ["published"])
    hangup = _as(spec, "hangup")
    with pytest.raises(SystemExit) as exit_info:
        _in_children(store, [hangup])
    assert exit_info.value.code == 128 + signal.SIGHUP
    assert store.runs()[hangup.name].info.status == "KILLED"


def test_a_signal_before_the_child_starts_spawns_none(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def signalled(found: Any) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(runner, "_skip_reason", signalled)
    (spec,) = store.phase("tune-coarse", ["published"])
    with pytest.raises(SystemExit) as exit_info:
        _in_children(store, [spec])
    assert exit_info.value.code == 128 + signal.SIGTERM
    assert store.runs() == {}


def test_a_crash_at_exit_after_the_run_finished_is_only_a_warning(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    crash = _as(spec, "exit-crash")
    _in_children(store, [crash])  # no failure is raised
    assert store.runs()[crash.name].info.status == "FINISHED"
    assert "exited with 3 after closing its run" in caplog.text


def test_a_repeated_signal_within_a_second_is_one_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``uv run`` forwards the terminal's SIGINT that python also got: ~0.2 s apart."""
    now = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    stop = runner._Stop()
    stop(signal.SIGINT, None)
    now[0] += 0.5
    stop(signal.SIGINT, None)  # the echo: ignored
    assert stop.signum == signal.SIGINT
    now[0] += 1.0
    with pytest.raises(KeyboardInterrupt):  # a deliberate second Ctrl-C
        stop(signal.SIGINT, None)


def test_one_ctrl_c_under_uv_run_stops_the_child_cleanly(store: Store) -> None:
    """A terminal's Ctrl-C reaches uv and python; uv forwards it a second time."""
    (spec,) = store.phase("tune-coarse", ["published"])
    slow = _as(spec, "slow")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    queue_process = _start_queue(store, slow, (uv, "run", "--no-sync", "python"))
    _training_pid(store, slow.name)
    os.killpg(queue_process.pid, signal.SIGINT)  # the whole foreground job
    _, stderr = queue_process.communicate(timeout=120)
    assert queue_process.returncode == 128 + signal.SIGINT, stderr[-2000:]
    assert "KeyboardInterrupt" not in stderr
    assert stderr.count("signal 2: stopping") == 1
    run = store.runs()[slow.name]
    assert run.info.status == "KILLED"
    assert (store.run_dir(slow.name) / "state" / "state.npz").is_file()


def test_a_queue_under_nohup_keeps_ignoring_hangups(store: Store) -> None:
    """``nohup make queue``: the inherited SIG_IGN for SIGHUP stays in force."""
    (spec,) = store.phase("tune-coarse", ["published"])
    slow = _as(spec, "slow")
    queue_process = _start_queue(store, slow, nohup=True)
    _training_pid(store, slow.name)
    os.kill(queue_process.pid, signal.SIGHUP)
    time.sleep(2.0)  # a stop would end the run within a chunk (0.5 s)
    assert queue_process.poll() is None
    assert store.runs()[slow.name].info.status == "RUNNING"
    os.kill(queue_process.pid, signal.SIGINT)
    queue_process.communicate(timeout=120)
    assert queue_process.returncode == 128 + signal.SIGINT
    assert store.runs()[slow.name].info.status == "KILLED"


# --- checkpoint artifacts -------------------------------------------------------


class ArtifactTrainer(FakeTrainer):
    """Records `checkpoint_artifacts` calls and writes a file; fails at ``fail_at``."""

    def __init__(self, fail_at: int | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.fail_at = fail_at
        self.artifact_calls: list[tuple[int, Path, int]] = []

    def checkpoint_artifacts(self, state: Any, run_dir: Path, update: int) -> None:
        # the runner passes the live state: the method counter is one step per update
        self.artifact_calls.append((update, run_dir, int(state["method"][0])))
        if update == self.fail_at:
            msg = "disk full"
            raise OSError(msg)
        (run_dir / "eval").mkdir(exist_ok=True)
        (run_dir / "eval" / f"art_{update}.txt").write_text(str(update))


def test_checkpoint_artifacts_are_written_once_per_checkpoint_after_its_save(
    store: Store,
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    trainer = ArtifactTrainer()
    store.run([spec], trainer)
    run_dir = store.run_dir(spec.name)
    # updates 1, 2, 3 of the 3-update run; the initial checkpoint (0) has none
    assert [(u, d, n) for u, d, n in trainer.artifact_calls] == [
        (1, run_dir, 1),
        (2, run_dir, 2),
        (3, run_dir, 3),
    ]
    assert sorted(p.name for p in (run_dir / "eval").glob("art_*")) == [
        "art_1.txt",
        "art_2.txt",
        "art_3.txt",
    ]
    run = store.runs()[spec.name]
    assert "warn/artifacts" not in run.data.tags
    assert run.info.status == "FINISHED"


def test_a_failing_checkpoint_artifact_warns_and_the_run_finishes(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    trainer = ArtifactTrainer(fail_at=2)
    store.run([spec], trainer)
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    assert run.data.tags["warn/artifacts"] == "update 2: OSError: disk full"
    assert [u for u, _, _ in trainer.artifact_calls] == [1, 2, 3]  # training went on
    assert "fail/error" not in run.data.tags
    assert "checkpoint artifacts of update 2 failed" in caplog.text
    assert [e["update"] for e in load_index(store.run_dir(spec.name))] == [0, 1, 2, 3]
    assert math.isfinite(run.data.metrics["final_score"])


def test_a_trainer_without_checkpoint_artifacts_is_unaffected(store: Store) -> None:
    (spec,) = store.phase("tune-coarse", ["published"])
    store.run([spec], FakeTrainer())
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    assert "warn/artifacts" not in run.data.tags
