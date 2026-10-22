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
from flax import struct
from test_runner import PROFILE, Store

from wut_murl_craftax.config import MLflowTrackingConfig, load_experiment_config
from wut_murl_craftax.evaluation.episode_metrics import ONLINE_LAYOUT, metric_keys
from wut_murl_craftax.orchestration import queue, runner, train_metrics
from wut_murl_craftax.orchestration.queue import RunSpec
from wut_murl_craftax.orchestration.runner import final_score
from wut_murl_craftax.training.checkpoints import save_state
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.records import (
    LOSS_COLUMNS,
    RunLog,
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


class _Timed:
    """A trainer that adds up the time of each chunk, until its results exist."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.busy = inner, 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def train_chunk(self, state: Any, n_updates: int) -> Any:
        start = time.monotonic()
        out = self.inner.train_chunk(state, n_updates)
        jax.block_until_ready(out)
        self.busy += time.monotonic() - start
        return out


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
    timed = _Timed(METHODS["ppo"])
    methods = {**METHODS, "ppo": timed}
    runner.main([spec], PROFILE, experiment_config=store.config, methods=methods)
    store.elapsed, store.busy = time.monotonic() - t0, timed.busy
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


def test_the_online_families_reach_the_extras_and_mlflow_windows(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    store, spec, run = tiny
    with np.load(store.run_dir(spec.name) / runner.EXTRAS_FILE) as z:
        assert set(metric_keys("classic")) <= set(z.files)
        assert len(z["train/end/death"]) == 31
    logged = run.data.metrics
    for key in ("train/funnel/floor_0", "train/action_entropy", "train/end/death"):
        assert key in logged
    assert all(math.isfinite(v) for k, v in logged.items() if k.startswith("train/"))
    score = [m.step for m in _history(store, run, "train/score_pct")]
    assert [m.step for m in _history(store, run, "train/funnel/floor_0")] == score


def test_first_unlocks_are_logged_once_at_their_env_step(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    store, _, run = tiny
    keys = [k for k in run.data.metrics if k.startswith("first_unlock/")]
    assert keys, "a random policy collects something in 31 updates"
    for key in keys:
        (point,) = _history(store, run, key)
        assert 0 < point.value == point.step <= 31 * STEP
        name = key.removeprefix("first_unlock/")
        (wall,) = _history(store, run, f"first_unlock_wall/{name}")
        assert wall.step == point.step
        assert wall.value > 0


@pytest.fixture
def active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """An active MLflow run in a throwaway store; yields ``(store, run)``."""
    store = Store(tmp_path, monkeypatch)
    mlflow, previous = runner.mlflow, runner.mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(store.tracking_uri)
    try:
        exp = store.client.create_experiment("unit")
        with mlflow.start_run(experiment_id=exp) as run:
            yield store, run
    finally:
        mlflow.set_tracking_uri(previous)


def test_a_replayed_chunk_logs_a_first_unlock_once(active: Any) -> None:
    store, run = active
    stamps = np.array([[-1, 150], [250, 150]])
    for _ in range(2):  # a chunk replayed after a crash before its state save
        runner._log_first_unlocks(stamps, ["a", "b"], 1, 100, np.array([5.0, 6.0]))
    history = store.client.get_metric_history(run.info.run_id, "first_unlock/b")
    assert [(m.step, m.value) for m in history] == [(150, 150.0)]
    wall = store.client.get_metric_history(run.info.run_id, "first_unlock_wall/a")
    assert [(m.step, m.value) for m in wall] == [(250, 6.0)]


@struct.dataclass
class _OldAcc:
    """``EpisodeAcc`` before the online families (a5352cf): the live queue's states."""

    ret: jax.Array
    length: jax.Array
    ach: jax.Array | None = None
    """None: a state that lacks it too (no leaf)."""


def _old_state(tmp_path: Path, fields: tuple[str, ...]) -> tuple[Any, int]:
    """A tiny PPO run's fresh template, and its state at update 4 saved with only
    ``fields`` of the episode accumulator (returns 1.5, lengths 3)."""
    trainer, envs = METHODS["ppo"], TINY["envs"]
    spec = RunSpec("ppo", "old", BUDGET, CLASSIC, 0, TINY, "h", "smoke", "final")
    fresh = runner.state_template(
        trainer.init(TINY | {"steps": 4000}, jax.random.key(0)), spec
    )
    ep = fresh["trainer"].episode
    kept = {"ret": ep.ret + 1.5, "length": ep.length + 3, "ach": ep.ach}
    old = fresh["trainer"].replace(episode=_OldAcc(**{k: kept[k] for k in fields}))
    save_state(tmp_path, {**fresh, "trainer": old, "update": 4, "wall_s": 9.0})
    return fresh, envs


def test_a_state_saved_before_the_online_families_resumes_with_them(
    active: Any, tmp_path: Path
) -> None:
    store, run = active
    fresh, envs = _old_state(tmp_path, ("ret", "length", "ach"))
    saved = runner._resume(tmp_path, fresh, STEP)
    tags = store.client.get_run(run.info.run_id).data.tags
    assert tags[runner.TAG_ONLINE_FROM] == "4"
    ep = saved["trainer"].episode
    assert (np.asarray(ep.ret) == 1.5).all()
    assert (np.asarray(ep.length) == 3).all()
    assert not np.asarray(ep.counts).any()
    assert (np.asarray(ep.first) == -1).all()
    # env steps go on from the saved update, so first unlocks land after it
    assert (np.asarray(ep.t) * envs == 4 * STEP).all()
    state, m = METHODS["ppo"].train_chunk(saved["trainer"], 1)
    stamps = np.asarray(m["first_unlock"])
    assert ((stamps == -1) | (stamps > 4 * STEP)).all()
    assert int(state.episode.t[0]) * envs == 5 * STEP
    # the run's extras end at update 4 without the families: they start at update 5
    np.savez(tmp_path / runner.EXTRAS_FILE, **{"ppo/lr": np.full(4, 0.1)})
    extras = runner._Extras.load(tmp_path, 4)
    log = RunLog(tmp_path, CLASSIC, NAMES)
    done, online = np.zeros_like(m["done"]), np.zeros_like(m["online"])
    done[0, -1, 0] = True  # one episode of 10 steps ends, alive, in the chunk
    online[0, ONLINE_LAYOUT["episodes"]], online[0, ONLINE_LAYOUT["steps"]] = 1, 10
    chunk = {**m, "done": done, "online": online}
    runner._record_chunk(log, extras, 4, STEP, np.array([10.0]), chunk)
    timeout = extras.series["train/end/timeout"]
    assert len(timeout) == 5
    assert np.isnan(timeout[:4]).all()
    assert timeout[4] == 100.0


def test_a_state_missing_more_than_the_online_leaves_is_refused(
    active: Any, tmp_path: Path
) -> None:
    store, run = active
    fresh, _ = _old_state(tmp_path, ("ret", "length"))
    with pytest.raises(ValueError, match="saved state has"):
        runner._resume(tmp_path, fresh, STEP)
    assert runner.TAG_ONLINE_FROM not in store.client.get_run(run.info.run_id).data.tags


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
    naps = []
    for _ in range(7):  # the fastest sample: load only ever makes `_work` slower
        t0 = time.monotonic()
        _work(a).block_until_ready()
        naps.append(time.monotonic() - t0)
    nap = min(naps)
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
    # waiting for every chunk gives >= n * nap; counting only dispatch leaves out the
    # last chunk (1/3 of the updates), so 0.85 still tells them apart under load
    assert wall[-1] >= 0.85 * n * nap
    sps = [p.value for p in _hist(store, spec, "train/sps")]
    assert sps
    assert max(sps) <= spu / (0.9 * nap)


def _hist(store: Store, spec: RunSpec, key: str) -> list[Any]:
    rid = store.runs()[spec.name].info.run_id
    return store.client.get_metric_history(rid, key)


def test_a_real_runs_wall_clock_covers_its_training(
    tiny: tuple[Store, RunSpec, Any],
) -> None:
    """wall_s spans the chunks' computation and never exceeds the runner call.

    Bounds that hold however loaded the machine is (evaluation and MLflow time around
    the training vary): the chunks are timed until their results exist, and wall_s,
    measured from before the first chunk, must contain all of that. The long-chunk
    test below is the one that tells a dispatch time from a computation time.
    """
    store, spec, _ = tiny
    wall = load_metrics(store.run_dir(spec.name)).col("wall_s")[-1]
    assert 0.99 * store.busy <= wall <= store.elapsed


# --- a tiny Classic run through the runner and METHODS["pqn-rnn"] -------------------

PQN_TINY = {"method": "pqn-rnn", "env": CLASSIC, "envs": 16, "num_steps": 8}
PQN_TINY |= {"hidden": 16, "minibatches": 2, "epochs": 1}


def test_a_pqn_rnn_run_logs_only_the_loss_columns_it_fills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TD loss (``value_loss``) and ``grad_norm`` are logged and finite; the
    PPO-only columns, NaN in ``metrics.npz``, are neither logged nor abort-checked."""
    store = Store(tmp_path, monkeypatch)
    spec = RunSpec(
        "pqn-rnn", "tiny", BUDGET, CLASSIC, 0, PQN_TINY, queue.config_hash(PQN_TINY),
        "smoke", "final",
    )  # fmt: skip
    runner.main(
        [spec],
        PROFILE,
        experiment_config=store.config,
        methods=METHODS,
        evaluate=store.evaluate,
    )
    run = store.runs()[spec.name]
    assert run.info.status == "FINISHED"
    assert not any(k.startswith("abort/") for k in run.data.tags)
    filled = set(METHODS["pqn-rnn"].loss_columns)  # type: ignore[attr-defined]
    assert filled == {"value_loss", "grad_norm"}
    logged = {k for k in run.data.metrics if k[len("train/") :] in LOSS_COLUMNS}
    assert logged == {f"train/{c}" for c in filled}
    for key in logged:
        assert all(math.isfinite(m.value) for m in _history(store, run, key))
    m = load_metrics(store.run_dir(spec.name))
    assert np.isnan(m.col("policy_loss")).all()  # the npz keeps the full layout
    assert np.isfinite(m.col("value_loss")).all()
