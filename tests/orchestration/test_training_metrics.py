"""Tests for the training metrics the runner logs to MLflow: windows, facts, e2e."""

import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from _fake_trainer import STEPS_PER_UPDATE, FakeTrainer
from test_runner import PROFILE, Store

from wut_murl_craftax.config import MLflowTrackingConfig, load_experiment_config
from wut_murl_craftax.orchestration import queue, runner, train_metrics
from wut_murl_craftax.orchestration.queue import RunSpec
from wut_murl_craftax.orchestration.runner import final_score
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.records import (
    LOSS_COLUMNS,
    achievement_names,
    load_episodes,
    load_metrics,
)

CLASSIC = "Craftax-Classic-Symbolic-v1"
NAMES = achievement_names(CLASSIC)
TINY = {
    "method": "ppo",
    "env": CLASSIC,
    "envs": 32,
    "num_steps": 4,
    "width": 16,
    "minibatches": 2,
    "update_epochs": 1,
}
BUDGET = "4k"
"""31 updates of 128 steps."""
STEP = 128


def _row(u: int, n: int, ret: float, length: float, ach: list[float]) -> list[float]:
    """A metrics.npz row of the two achievements a, b: wall 2 s and 100 steps/update."""
    losses = [0.1 * u] * len(LOSS_COLUMNS)
    return [u, 100 * u, 2.0 * u, n, ret, length, *losses, *ach]


NAN = float("nan")
ROWS = np.array([
    _row(1, 2, 1.0, 10, [50.0, 0.0]),
    _row(2, 0, NAN, NAN, [NAN, NAN]),
    _row(3, 4, 2.0, 20, [100.0, 25.0]),
    _row(4, 1, 4.0, 30, [0.0, 100.0]),
    _row(5, 3, 0.5, 40, [0.0, 0.0]),
])


def _points(first: int, last: int, every: int = 2) -> dict[int, dict[str, float]]:
    got = train_metrics.metric_points(ROWS, ["a", "b"], first, last, every, 5, 22.0)
    return dict(got)


def test_windows_average_updates_weighted_by_their_episodes() -> None:
    points = _points(0, 5)
    assert sorted(points) == [200, 400, 500]  # window ends: 2, 4 and the budget
    first, second, last = points[200], points[400], points[500]
    assert first["train/return"] == 1.0  # the empty update does not dilute it
    assert first["train/ach/a"] == 50.0
    assert second["train/return"] == pytest.approx((4 * 2.0 + 4.0) / 5)
    assert second["train/length"] == pytest.approx((4 * 20 + 30) / 5)
    assert second["train/ach/b"] == pytest.approx((4 * 25 + 100) / 5)
    assert last["train/return"] == 0.5  # the partial last window is its one update
    assert second["train/score_pct"] == pytest.approx(second["train/return"] / 22 * 100)
    assert (first["train/n_episodes"], second["train/n_episodes"]) == (2, 5)


def test_windows_report_the_last_wall_clock_and_steps_per_second() -> None:
    points = _points(0, 5)
    assert points[400]["train/wall_s"] == 8.0
    assert points[400]["train/env_steps"] == 400
    assert points[200]["train/sps"] == points[400]["train/sps"] == 50.0
    assert points[500]["train/sps"] == 50.0  # 100 steps in 2 s


def test_losses_are_plain_means_of_the_window() -> None:
    losses = _points(0, 5)[400]
    for name in LOSS_COLUMNS:
        assert losses[f"train/{name}"] == pytest.approx(0.35)  # (0.3 + 0.4) / 2


def test_a_window_without_episodes_logs_no_score_and_no_nan() -> None:
    empty = _points(1, 2, every=1)[200]
    assert not any(k in empty for k in ("train/return", "train/score_pct"))
    assert not any(k.startswith("train/ach/") for k in empty)
    assert not any(math.isnan(v) for v in empty.values())


def test_chunks_of_a_resumed_run_give_the_same_points() -> None:
    """Windows depend on the update number, not on where a chunk or a resume cut."""
    whole = _points(0, 5)
    parts = {**_points(0, 1), **_points(1, 3), **_points(3, 5)}
    assert parts == whole


@pytest.mark.parametrize(
    ("updates", "every"),
    [(1, 1), (61, 1), (2000, 1), (2001, 2), (15_258, 8), (244_140, 123)],
)
def test_the_log_interval_keeps_a_run_under_the_point_budget(
    updates: int, every: int
) -> None:
    assert train_metrics.log_every(updates) == every
    assert math.ceil(updates / every) <= train_metrics.MAX_POINTS


def test_compile_time_is_the_first_chunk_beyond_the_steady_updates() -> None:
    chunks = [(1, 30.0), (2, 2.0), (2, 2.2)]
    assert train_metrics.compile_seconds(chunks) == pytest.approx(30 - 1.05)
    assert train_metrics.compile_seconds(chunks[:1]) is None
    assert train_metrics.compile_seconds([(1, 0.5), (1, 1.0)]) == 0.0


# --- a tiny Classic run through the runner and METHODS["ppo"] -----------------------


@pytest.fixture(scope="module")
def tiny(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Store, RunSpec, Any]]:
    """One PPO run through the runner, with MLflow system metrics sampled fast."""
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "0.2")
    mp.setattr(train_metrics, "MAX_POINTS", 8)  # 31 updates -> every 4th
    store = Store(tmp_path_factory.mktemp("store"), mp)
    raw = yaml.safe_load(store.config.read_text())
    raw["mlflow"]["log_system_metrics"] = True
    store.config.write_text(yaml.safe_dump(raw))
    spec = RunSpec(
        "ppo", "tiny", BUDGET, CLASSIC, 0, TINY, queue.config_hash(TINY), "smoke",
        "final",
    )  # fmt: skip
    t0 = time.monotonic()
    runner.main([spec], PROFILE, experiment_config=store.config, methods=METHODS)
    store.elapsed = time.monotonic() - t0
    yield store, spec, store.runs()[spec.name]
    mp.undo()


def _history(store: Store, run: Any, key: str) -> list[Any]:
    return store.client.get_metric_history(run.info.run_id, key)


def test_the_run_finishes_with_the_score_of_the_logged_episodes(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    store, spec, run = tiny
    assert run.info.status == "FINISHED"
    out = store.run_dir(spec.name)
    m, eps = load_metrics(out), load_episodes(out)
    n = m.col("n_episodes")
    assert n.sum() == len(eps.ret) > 0
    score = run.data.metrics[queue.SCORE_METRIC]
    assert score == pytest.approx(final_score(m.score, n))
    if n.sum() < runner.MIN_FINAL_EPISODES:  # the window widens to every episode
        assert score == pytest.approx(eps.ret.mean() / 22 * 100, rel=1e-5)
    assert math.isfinite(score)


def test_updates_are_logged_in_windows_at_their_env_steps(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    store, _, run = tiny
    steps = [m.step for m in _history(store, run, "train/score_pct")]
    assert set(steps) <= {STEP * u for u in (*range(4, 31, 4), 31)}
    assert len(steps) == len(set(steps)) > 1
    assert STEP * 31 in {m.step for m in _history(store, run, "train/entropy")}


def test_score_and_success_rates_follow_the_records(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    store, spec, run = tiny
    m = load_metrics(store.run_dir(spec.name))
    first = _history(store, run, "train/score_pct")[0]
    window = slice(0, first.step // STEP)
    w = m.col("n_episodes")[window]
    ret = np.where(w > 0, m.col("return")[window], 0.0)
    assert first.value == pytest.approx((ret @ w) / w.sum() / 22 * 100, rel=1e-5)
    got = {
        k[len("train/ach/") :] for k in run.data.metrics if k.startswith("train/ach/")
    }
    assert got == set(NAMES)
    rates = [run.data.metrics[f"train/ach/{n}"] for n in NAMES]
    assert all(0 <= r <= 100 for r in rates)


def test_hardware_facts_are_logged_per_run(tiny: tuple[Store, RunSpec, Any]) -> None:
    store, _, run = tiny
    tags, metrics = run.data.tags, run.data.metrics
    roles = runner.device_roles()
    assert tags["actor_device"] == str(roles.actor)
    assert tags["learner_device"] == str(roles.learner)
    assert tags["device_kind"] == roles.learner.device_kind
    assert tags["host_name"]
    assert metrics["peak_rss_mb"] > 50
    assert metrics["compile_s"] >= 0
    assert all(m.value > 0 for m in _history(store, run, "train/sps"))


def test_system_metrics_come_from_mlflow(tiny: tuple[Store, RunSpec, Any]) -> None:
    run = tiny[2]
    assert "system/cpu_utilization_percentage" in run.data.metrics
    assert "sys.cpu_cores" in run.data.tags  # mlflow-sysmetrics run tags


def test_experiment_yaml_turns_system_metrics_on() -> None:
    shipped = load_experiment_config(queue.EXPERIMENT_CONFIG)
    assert shipped.mlflow.log_system_metrics is True
    assert MLflowTrackingConfig().log_system_metrics is False


# --- wall clock and steps per second ---------------------------------------------


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    return Store(tmp_path, monkeypatch)


@jax.jit
def _work(a: jax.Array) -> jax.Array:
    """About 0.03 s of device work that the host does not wait for."""
    return jax.lax.fori_loop(0, 150, lambda _, x: jnp.tanh(x @ a), a)


class SlowTrainer(FakeTrainer):
    """Returns its metrics before they exist: each update queues `_work`."""

    def train_chunk(self, state: Any, n_updates: int) -> tuple[Any, Any]:
        state, metrics = super().train_chunk(state, n_updates)
        a = jnp.full((512, 512), 1e-3)
        for _ in range(n_updates):
            a = _work(a)
        return state, {**metrics, "ret": metrics["ret"] + 0 * a.sum()}


def test_wall_clock_and_sps_wait_for_the_chunks_computation(
    store: Store, tmp_path: Path
) -> None:
    """40 updates of `_work`, in chunks of up to 13: wall_s counts all of it."""
    a = jnp.full((512, 512), 1e-3)
    _work(a).block_until_ready()
    t0 = time.monotonic()
    _work(a).block_until_ready()
    nap = time.monotonic() - t0
    profile = tmp_path / "long-chunks.yaml"
    profile.write_text(
        PROFILE.read_text().replace("checkpoint_every: 2", "checkpoint_every: 100")
    )
    spu = STEPS_PER_UPDATE // 13  # 1m budget: 39 updates
    (spec,) = store.phase("tune-coarse", ["published"])
    trainer = SlowTrainer(steps_per_update=spu)
    runner.main(
        [spec],
        profile,
        experiment_config=store.config,
        methods={"toy": trainer},
        evaluate=store.evaluate,
    )
    n = 1_000_000 // spu
    wall = load_metrics(store.run_dir(spec.name)).col("wall_s")
    assert wall[-1] >= 0.95 * n * nap
    sps = [p.value for p in _hist(store, spec, "train/sps")]
    assert sps
    assert max(sps) <= spu / (0.9 * nap)


def _hist(store: Store, spec: RunSpec, key: str) -> list[Any]:
    rid = store.runs()[spec.name].info.run_id
    return store.client.get_metric_history(rid, key)


def test_a_real_runs_wall_clock_covers_its_training(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    """wall_s ends close to the time the whole runner call took, not a dispatch time."""
    store, spec, _ = tiny
    wall = load_metrics(store.run_dir(spec.name)).col("wall_s")[-1]
    assert 0.4 * store.elapsed < wall <= store.elapsed
