"""Resumable runner: trains, evaluates and logs every RunSpec of this worker.

A run's directory is its MLflow artifact directory: ``config.json``, ``metrics.npz``,
``episodes.npz``, ``ckpt/`` (log-spaced params), ``state/`` (full resume state) and
``eval/``. A run name is looked up across the whole store: a FINISHED run only joins
the phase (tag ``phases``) and gets its missing evaluations; any other run resumes
from ``state/`` (or restarts in place without it); a new name starts in the phase's
experiment. SIGINT or SIGTERM end the current chunk, save the state and mark the run
KILLED. Trainers come from ``training.interface.METHODS``; see `Trainer`. Evaluation:
``visualisation.eval_rollouts.evaluate_spec``, imported after ``configure_runtime``.
"""

import functools
import importlib
import json
import logging
import math
import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, TypedDict, cast

import jax
import mlflow
import numpy as np

from ..config import ExperimentConfig, load_experiment_config
from ..evaluation.episode_metrics import score_pct
from ..training.checkpoints import (
    STATE_FILE,
    checkpoint_schedule,
    drop_missing_checkpoints,
    load_index,
    load_state,
    missing_checkpoints,
    save_checkpoint,
    save_state,
)
from ..training.records import (
    BASE_COLUMNS,
    LOSS_COLUMNS,
    RunLog,
    achievement_names,
    achievement_weights,
)
from .abort import (
    FLAG_PREFIX,
    MAX_ABORTS,
    TAG_ABORT_BUG,
    TAG_ABORT_COUNT,
    TAG_ABORT_METRIC,
    TAG_ABORT_UPDATE,
    first_nonfinite,
    flag_hits,
)
from .profiles import Profile, load_profile
from .queue import (
    CONFIG_ARTIFACT,
    EXPERIMENT_CONFIG,
    SCORE_METRIC,
    TAG_BUDGET,
    TAG_CONFIG_HASH,
    TAG_METHOD,
    TAG_PHASES,
    TAG_SEED,
    RunSpec,
)
from .runtime import (
    DeviceRoles,
    configure_runtime,
    device_roles,
    import_craftax,
    select_platforms,
)
from .tracking import _resolve_tracking_uri, build_mlflow_tracker
from .train_metrics import (
    compile_seconds,
    log_every,
    log_peak_memory,
    log_points,
    metric_points,
    run_tags,
)

logger = logging.getLogger(__name__)

N_CHECKPOINTS = 12
"""Log-spaced params checkpoints per run (`checkpoint_schedule`)."""
MIN_FINAL_EPISODES = 100
"""Episodes the final-score window is widened backwards to."""
TAG_CKPT_MISSING = "ckpt/missing"
"""Run tag: comma-joined updates whose checkpoint file is lost and cannot be rebuilt."""
EXTRAS_FILE = "extras.npz"
"""Per-update method metrics of a run, name -> floats, in the run directory."""
_RUN_ID_VARIABLES = ("MLFLOW_RUN_ID", "MLFLOW_ACTIVE_RUN_ID")
"""Environment variables that make MLflow or the tracker reopen a given run."""


class ChunkMetrics(TypedDict):
    """What `Trainer.train_chunk` reports; every array has a leading update axis.

    A trainer may add method metrics under ``<group>/<name>`` keys (e.g.
    ``ppo/entropy``): ``(updates,)`` floats that the runner logs unchanged.

    ``ach``, ``ret`` and ``length`` are read on done steps only, and there they are
    the WHOLE finished episode, the done step included: its reward, its step and
    its unlocks, taken from ``StepEvents.unlocked`` (``info["events"]``, the
    pre-reset state). Never rebuild them from env-state comparisons or Craftax's
    ``Achievements/*`` info. Every trainer accumulates them with
    ``evaluation.episode_metrics.episode_step`` and ``events_of``, and its test
    calls ``tests/training/_contract.assert_done_step_unlock_counted``.
    """

    losses: jax.Array | np.ndarray
    """(updates, len(LOSS_COLUMNS)) floats in `LOSS_COLUMNS` order; the columns the
    trainer does not report (see `Trainer`) are NaN."""
    done: jax.Array | np.ndarray
    """(updates, steps, envs) bools: the episode ended at this step."""
    ach: jax.Array | np.ndarray
    """(updates, steps, envs, achievements) bools: the episode flags, by ``.value``."""
    ret: jax.Array | np.ndarray
    """(updates, steps, envs) episode return so far, incl. this step."""
    length: jax.Array | np.ndarray
    """(updates, steps, envs) episode length so far, incl. this step."""


class Trainer(Protocol):
    """A registry entry: one method's training, as the runner drives it.

    The state is one pytree (`save_state` round-trips it); ``train_chunk`` called
    with n then m updates must equal one call with n + m. Its `ChunkMetrics` must
    follow that class's episode rules; the runner times the chunk after
    ``jax.block_until_ready`` on the returned state and metrics.

    A trainer may set the attribute ``loss_columns: tuple[str, ...]``, the
    `LOSS_COLUMNS` it fills (default: all; PQN-RNN leaves the PPO-only ones NaN). The
    runner logs and abort-checks only those; ``metrics.npz`` keeps the full layout.
    """

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Environment steps one update consumes under ``config``."""
        ...

    def init(self, config: Mapping[str, Any], key: jax.Array) -> Any:
        """Initial state: params, optimizer, env states, keys, method state.

        ``config`` is the run's config plus ``steps``, the budget in env steps: the
        one runtime key the runner injects (it is not hashed or saved in
        ``config.json``); later runtime inputs, such as the profile, follow suit.
        """
        ...

    def train_chunk(self, state: Any, n_updates: int) -> tuple[Any, ChunkMetrics]:
        """Run ``n_updates`` updates; return the new state and their metrics."""
        ...

    def params(self, state: Any) -> Any:
        """The params pytree saved in the log-spaced ``ckpt/``."""
        ...

    def policy(self, state: Any) -> Any:
        """The acting policy (``Policy.init_carry``, ``Policy.step``) for evaluation.

        Evaluation passes the abstract state of ``jax.eval_shape(init)``: read its
        structure and static fields only; acting reads values from the params alone.
        """
        ...

    def env_wrapper(self, state: Any) -> Callable[[Any, Any], tuple[Any, Any]] | None:
        """The method's ``env_wrapper_fn``, None without one (abstract state as above).

        Evaluation applies it inside ``LogWrapper`` as training does
        (``training.ppo.make_env``).
        """
        ...


Evaluate = Callable[[Path, RunSpec, Callable[[], bool]], None]
"""Evaluates a run directory's checkpoints for its spec, skipping finished files;
returns early once the third argument, ``should_stop()``, is true (SIGINT, SIGTERM)."""


def evaluate_fixed_seeds(
    trainers: Mapping[str, Trainer],
    run_dir: Path,
    spec: RunSpec,
    should_stop: Callable[[], bool],
) -> None:
    """Default `Evaluate` (with ``trainers`` bound): ``eval/<mode>/<update>.npz``.

    Logs ``eval/<mode>/return`` and ``eval/<mode>/score_pct`` (% of the maximum
    reward) at the checkpoint's env step as each file is written, and once for files
    a crashed call left unlogged. The module imports Craftax, which starts the JAX
    backends, so it is imported here, after ``configure_runtime``.
    """
    module = importlib.import_module("..visualisation.eval_rollouts", __package__)
    run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]  # the runner's run
    client = mlflow.MlflowClient()
    logged: dict[str, set[int]] = {}
    for s in module.evaluate_spec(run_dir, spec, should_stop, methods=trainers):
        for name, value in (("return", s.ret), ("score_pct", s.score_pct)):
            key = f"eval/{s.mode}/{name}"
            if key not in logged:  # files of a crashed call exist but were not logged
                history = client.get_metric_history(run_id, key)
                logged[key] = {m.step for m in history}
            if s.env_steps not in logged[key]:
                mlflow.log_metric(key, value, step=s.env_steps)


def final_score(update_scores: np.ndarray, episode_counts: np.ndarray) -> float:
    """Mean training-episode score over the last 2% of the updates (NaN if none).

    The window is widened backwards, whole updates at a time, until it holds at least
    `MIN_FINAL_EPISODES` episodes. ``update_scores`` are per-update episode means (NaN
    without episodes), so weighting them by ``episode_counts`` gives the episode mean.
    """
    counts = np.asarray(episode_counts, float)
    n = len(counts)
    reach = np.cumsum(counts[::-1])
    enough = int(np.searchsorted(reach, MIN_FINAL_EPISODES)) + 1
    k = max(math.ceil(n / 50), min(enough, n))
    w, s = counts[n - k :], np.asarray(update_scores, float)[n - k :]
    if not w.sum():
        return float("nan")
    return float(np.where(w > 0, s, 0.0) @ w / w.sum())


def worker_specs(
    specs: Sequence[RunSpec], worker_index: int, num_workers: int
) -> list[RunSpec]:
    """The specs of worker ``worker_index``: every ``num_workers``-th from it."""
    if not 0 <= worker_index < num_workers:
        msg = f"worker index {worker_index} is not in [0, {num_workers})"
        raise ValueError(msg)
    return list(specs[worker_index::num_workers])


def state_template(trainer_state: Any, spec: RunSpec) -> dict[str, Any]:
    """The pytree saved under ``state/``: trainer state, update, wall time, run log."""
    names = achievement_names(spec.env)
    episodes = {
        "update": np.zeros(0, np.int32),
        "ret": np.zeros(0, np.float32),
        "length": np.zeros(0, np.int32),
        "ach": np.zeros((0, math.ceil(len(names) / 8)), np.uint8),
    }
    width = len(BASE_COLUMNS) + len(LOSS_COLUMNS) + len(names)
    rows = np.zeros((0, width))
    return {
        "trainer": trainer_state,
        "update": 0,
        "wall_s": 0.0,
        "log": (rows, episodes),
    }


def _snapshot(
    fresh: dict[str, Any], trainer_state: Any, update: int, wall: float, log: RunLog
) -> dict[str, Any]:
    """The state to save after ``update``; compacts ``log.episodes`` into one block."""
    rows, empty = fresh["log"]
    episodes = {
        k: np.concatenate([v, *(e[k] for e in log.episodes)]) for k, v in empty.items()
    }
    log.episodes = [episodes]
    table = np.array(log.rows, np.float64).reshape(-1, rows.shape[1])
    return {
        **fresh,
        "trainer": trainer_state,
        "update": update,
        "wall_s": wall,
        "log": (table, episodes),
    }


@dataclass
class _Stop:
    """Signal handler that records the first SIGINT or SIGTERM."""

    signum: int | None = None

    def __call__(self, signum: int, _frame: FrameType | None) -> None:
        """Record the signal; a second one stops at once."""
        if self.signum is not None:
            raise KeyboardInterrupt
        self.signum = signum
        logger.warning("signal %d: stopping after the current chunk", signum)


@contextmanager
def _stop_on_signals() -> Iterator[_Stop]:
    stop = _Stop()
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield stop
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def find_run(name: str) -> Any:
    """The run named ``name`` in any experiment: the finished one, else the latest."""
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.mlflow.runName = '{name}'",
        output_format="list",
    )
    if len(runs) > 1:
        logger.warning("%d runs are named %s; using one", len(runs), name)
    return max(
        runs,
        key=lambda r: (r.info.status == "FINISHED", r.info.start_time),
        default=None,
    )


class _Aborted(Exception):  # noqa: N818 - control flow, not an error of the caller
    """A chunk reported a non-finite loss: ``update`` (1-based) and ``metric``."""

    def __init__(self, update: int, metric: str) -> None:
        super().__init__(f"{metric} is not finite at update {update}")
        self.update, self.metric = update, metric


def _loss_columns(trainer: Trainer) -> tuple[str, ...]:
    """The `LOSS_COLUMNS` the trainer reports (its ``loss_columns``, default all)."""
    columns = tuple(getattr(trainer, "loss_columns", LOSS_COLUMNS))
    if unknown := set(columns) - set(LOSS_COLUMNS):
        msg = f"loss_columns {sorted(unknown)} are not in {LOSS_COLUMNS}"
        raise ValueError(msg)
    return columns


def _method_metrics(metrics: ChunkMetrics) -> dict[str, np.ndarray]:
    """The chunk's method metrics: every key beyond `ChunkMetrics`, as float arrays."""
    return {
        k: np.asarray(v, float)
        for k, v in metrics.items()
        if k not in ChunkMetrics.__annotations__
    }


@dataclass
class _Extras:
    """Method metrics per update: ``series[name][update - 1]``, NaN where absent."""

    path: Path
    series: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, out: Path, update: int) -> "_Extras":
        """The saved series cut to ``update`` (the file may run ahead of ``state/``)."""
        extras = cls(out / EXTRAS_FILE)
        if extras.path.is_file():
            with np.load(extras.path) as z:
                extras.series = {k: z[k][:update].tolist() for k in z.files}
        extras.pad(update)  # a run saved before it reported these has no history
        return extras

    def pad(self, n: int) -> None:
        """Fill every series with NaN up to ``n`` updates."""
        for values in self.series.values():
            values += [math.nan] * (n - len(values))

    def extend(self, first: int, new: Mapping[str, np.ndarray]) -> None:
        """Append a chunk that starts after update ``first``."""
        self.pad(first)
        for name, values in new.items():
            self.series.setdefault(name, []).extend(values.tolist())
        self.pad(first + max((len(v) for v in new.values()), default=0))

    def save(self) -> None:
        """Write the file atomically; it goes before ``state/`` so it never lags it."""
        tmp = self.path.with_name("extras.tmp.npz")
        arrays: dict[str, Any] = {k: np.array(v) for k, v in self.series.items()}
        np.savez(tmp, **arrays)
        tmp.replace(self.path)


def _repair_checkpoints(
    trainer: Trainer,
    out: Path,
    fresh: dict[str, Any],
    saved: dict[str, Any],
    ckpts: Sequence[int],
    spu: int,
) -> list[int]:
    """Rebuild lost checkpoint files that can be rebuilt; return the ones that cannot.

    Lost means: in the log-spaced schedule, before the saved update, no file on disk
    (not what ``index.json`` says: it is cleaned of missing files here, so a later call
    must not take the index's word). Update 0 is rebuilt from the same-seed initial
    state, the saved update from the saved state. Earlier ones cannot be rebuilt and
    stay lost: they are tagged `TAG_CKPT_MISSING` (also when an earlier call tagged
    them) and returned.
    """
    drop_missing_checkpoints(out)
    update, ckpt_dir = saved["update"], out / "ckpt"
    on_disk = {u for u in ckpts if (ckpt_dir / f"{u:05d}.msgpack").is_file()}
    if 0 not in on_disk:
        save_checkpoint(out, 0, 0, 0.0, trainer.params(fresh["trainer"]))
    if update in ckpts and update and update not in on_disk:
        wall, params = saved["wall_s"], trainer.params(saved["trainer"])
        save_checkpoint(out, update, update * spu, wall, params)
    gone = {u for u in ckpts if 0 < u < update and u not in on_disk}
    run = mlflow.get_run(mlflow.active_run().info.run_id)  # type: ignore[union-attr]  # the runner's run
    tagged = run.data.tags.get(TAG_CKPT_MISSING, "")
    lost = sorted(gone | {int(u) for u in tagged.split(",") if u})
    if lost:
        mlflow.set_tag(TAG_CKPT_MISSING, ",".join(map(str, lost)))
        logger.warning("checkpoints %s of %s are lost for good", lost, out)
    return lost


def _check_finite(
    metrics: ChunkMetrics, columns: Sequence[str], finite: Sequence[str], update: int
) -> None:
    """Raise `_Aborted` on the first non-finite logged loss of the chunk."""
    hit = first_nonfinite(
        np.asarray(metrics["losses"], float), columns, _method_metrics(metrics), finite
    )
    if hit is not None:
        raise _Aborted(update + hit[0] + 1, hit[1])


def _registry() -> Mapping[str, Trainer]:
    """``training.interface.METHODS``, imported after ``configure_runtime``.

    The trainers import Craftax, which starts the JAX backends on import.
    """
    module = importlib.import_module("..training.interface", __package__)
    return cast(Mapping[str, Trainer], module.METHODS)


def _chunk_size(update: int, n_updates: int, ckpts: list[int], every: int) -> int:
    """Updates to the next log-spaced checkpoint, multiple of ``every`` or the end."""
    stops = [u for u in ckpts if u > update]
    return min([*stops, (update // every + 1) * every, n_updates]) - update


def _add_updates(
    log: RunLog, first: int, spu: int, walls: np.ndarray, metrics: ChunkMetrics
) -> None:
    """Append the chunk's updates ``first + 1 ...`` to the run log."""
    m = {k: np.asarray(v) for k, v in metrics.items()}
    for i, wall in enumerate(walls.tolist()):
        u = first + i + 1
        log.add_update(
            u,
            u * spu,
            wall,
            m["losses"][i],
            m["done"][i],
            m["ach"][i],
            m["ret"][i],
            m["length"][i],
        )


def _train(
    trainer: Trainer,
    spec: RunSpec,
    out: Path,
    profile: Profile,
    roles: DeviceRoles,
    stop: _Stop,
) -> RunLog | None:
    """Train from ``state/`` or from scratch; the log at the budget, None on a signal.

    Every chunk ends at a log-spaced checkpoint, a multiple of ``checkpoint_every``
    or the budget, and is followed by a state save, so chunks are the same on resume.
    MLflow gets windows of `log_every` updates (`train_metrics`).

    Raises:
        _Aborted: A chunk reported a non-finite loss; nothing of it is saved.
        RuntimeError: The run is complete but checkpoints are lost for good (a lost
            final one is rebuilt from ``state/``).
    """
    spu = trainer.steps_per_update(spec.config)
    n_updates = spec.steps // spu
    if n_updates < 1:
        msg = f"{spec.name}: {spu} steps per update exceed the budget {spec.budget}"
        raise ValueError(msg)
    ckpts = checkpoint_schedule(n_updates, N_CHECKPOINTS)
    # a resume needs init too: it gives the tree, key implementations and placement
    config = {**spec.config, "steps": spec.steps}
    fresh = state_template(trainer.init(config, jax.random.key(spec.seed)), spec)
    saved = load_state(out, fresh) if (out / STATE_FILE).exists() else fresh
    state, update = saved["trainer"], saved["update"]
    log = RunLog(out, spec.env, achievement_names(spec.env))
    rows, episodes = saved["log"]
    log.rows, log.episodes = rows.tolist(), [episodes]
    lost = _repair_checkpoints(trainer, out, fresh, saved, ckpts, spu)
    written = {e["update"] for e in load_index(out)}
    columns, extras = _loss_columns(trainer), _Extras.load(out, update)
    abort = spec.abort or {}
    rules, finite, flagged = abort.get("flags", []), abort.get("finite", []), set[str]()
    t0 = time.monotonic() - saved["wall_s"]
    every, max_reward = log_every(n_updates), float(achievement_weights(spec.env).sum())
    chunks: list[tuple[int, float]] = []
    while update < n_updates and stop.signum is None:
        n = _chunk_size(update, n_updates, ckpts, profile.checkpoint_every)
        start = time.monotonic() - t0
        state, metrics = trainer.train_chunk(state, n)
        # dispatch is asynchronous: without waiting, wall_s and SPS time the dispatch
        jax.block_until_ready((state, metrics))  # type: ignore[no-untyped-call]  # jax API
        wall = time.monotonic() - t0
        _check_finite(metrics, columns, finite, update)
        chunks.append((n, wall - start))
        _add_updates(log, update, spu, np.linspace(start, wall, n + 1)[1:], metrics)
        extras.extend(update, _method_metrics(metrics))
        update += n
        for name, at in flag_hits(rules, extras.series, n_updates).items():
            if name not in flagged:  # the first update the rule held at
                mlflow.set_tag(f"{FLAG_PREFIX}{name}", str(at))
                flagged.add(name)
        if update in ckpts and update not in written:
            save_checkpoint(out, update, update * spu, wall, trainer.params(state))
        extras.save()
        save_state(out, _snapshot(fresh, state, update, wall, log))
        log.flush()
        points = metric_points(
            log.rows,
            log.names,
            update - n,
            update,
            every,
            n_updates,
            max_reward,
            loss_columns=columns,
            extras=extras.series,
        )
        log_points(points)
        log_peak_memory(roles, update * spu)
    if (compile_s := compile_seconds(chunks)) is not None:
        mlflow.log_metric("compile_s", compile_s, step=update * spu)
    if lost and update == n_updates:
        msg = f"{spec.name}: checkpoints {lost} are lost and cannot be rebuilt"
        raise RuntimeError(msg)
    return log if update == n_updates else None


def _log_final_score(log: RunLog, env: str) -> None:
    """Log `final_score` of the per-update returns as % of the maximum reward."""
    rows = np.array(log.rows)
    cols = list(BASE_COLUMNS)
    scores = score_pct(rows[:, cols.index("return")], achievement_weights(env).sum())
    score = final_score(scores, rows[:, cols.index("n_episodes")])
    mlflow.log_metric(SCORE_METRIC, score)


def _mark_abort(previous: Mapping[str, str], abort: _Aborted) -> None:
    """Tag the active run with the abort; the `MAX_ABORTS`-th one marks it a bug."""
    count = int(previous.get(TAG_ABORT_COUNT, 0)) + 1
    tags = {
        TAG_ABORT_COUNT: str(count),
        TAG_ABORT_METRIC: abort.metric,
        TAG_ABORT_UPDATE: str(abort.update),
    }
    if count >= MAX_ABORTS:
        tags[TAG_ABORT_BUG] = "true"
    mlflow.set_tags(tags)
    logger.warning("run aborted (%d of %d allowed): %s", count, MAX_ABORTS, abort)


def _run_spec(
    spec: RunSpec,
    trainer: Trainer,
    experiment: ExperimentConfig,
    profile: Profile,
    evaluate: Evaluate,
    stop: _Stop,
) -> bool:
    """Bring one run to FINISHED, or to KILLED on a signal; return if it trained."""
    found = find_run(spec.name)
    if found and found.data.tags.get(TAG_ABORT_BUG) == "true":
        logger.warning("%s is marked a bug (%s), skipped", spec.name, TAG_ABORT_BUG)
        return False
    phases = found.data.tags.get(TAG_PHASES, "").split(",") if found else []
    tags = {
        TAG_METHOD: spec.registry_name,
        TAG_BUDGET: spec.budget,
        TAG_SEED: str(spec.seed),
        TAG_CONFIG_HASH: spec.config_hash,
        TAG_PHASES: ",".join(dict.fromkeys([*filter(None, phases), spec.phase])),
    }
    cfg = replace(
        experiment, mlflow=replace(experiment.mlflow, experiment_name=spec.phase)
    )
    run_id = found.info.run_id if found else None
    tracker = build_mlflow_tracker(
        cfg, run_name=spec.name, run_id=run_id, tags=tags, strict=True
    )
    tracker.start()
    active = mlflow.get_run(tracker.run_id or "").info
    if active.run_name != spec.name or run_id not in (None, active.run_id):
        tracker.close("KILLED")
        msg = f"{spec.name}: MLflow opened run {active.run_name} ({active.run_id})"
        raise RuntimeError(msg)  # e.g. $MLFLOW_RUN_ID set in the environment
    out = tracker.artifact_dir
    # a FINISHED run with a lost checkpoint file goes through `_train` to repair it
    trains = (
        found is None
        or found.info.status != "FINISHED"
        or bool(missing_checkpoints(out))
    )
    try:
        if trains:
            roles = device_roles()
            mlflow.set_tags(run_tags(roles))  # where it trains now, not where it began
            (out / CONFIG_ARTIFACT).write_text(json.dumps(spec.config, indent=1))
            log = _train(trainer, spec, out, profile, roles, stop)
            if log is None or stop.signum is not None:
                tracker.close("KILLED")  # the next call logs the score and evaluates
                return True
            _log_final_score(log, spec.env)
        (out / "eval").mkdir(exist_ok=True)
        evaluate(out, spec, lambda: stop.signum is not None)
    except KeyboardInterrupt:
        tracker.close("KILLED")
        raise
    except _Aborted as abort:
        _mark_abort(found.data.tags if found else {}, abort)
        tracker.close("FAILED")  # its last state/ stays for the one retry
        return True
    except Exception:
        tracker.close("FAILED")
        raise
    if stop.signum is not None:
        tracker.close("KILLED")  # evaluation may have stopped early
        return trains
    tracker.close("FINISHED")
    return trains


def main(
    specs: Sequence[RunSpec],
    profile: Path | str,
    worker_index: int = 0,
    num_workers: int = 1,
    max_runs: int | None = None,
    *,
    experiment_config: Path = EXPERIMENT_CONFIG,
    tracking_uri: str | None = None,
    methods: Mapping[str, Trainer] | None = None,
    evaluate: Evaluate | None = None,
) -> None:
    """Train, evaluate and log this worker's specs, at most ``max_runs`` trained ones.

    ``select_platforms``, ``configure_runtime`` and ``import_craftax`` run first,
    before any JAX computation (the profile decides the backends and the CPU split)
    or trace; runs reused as FINISHED do not count towards ``max_runs``. MLflow
    settings come from ``experiment_config``, the experiment name replaced by each
    spec's phase and the tracking URI by ``tracking_uri`` when given (a cluster
    node's own store).
    ``evaluate`` defaults to `evaluate_fixed_seeds` with this worker's trainers.

    Raises:
        SystemExit: 128 + the signal number after SIGINT or SIGTERM.
    """
    machine = load_profile(profile)
    select_platforms(machine.device_roles)
    configure_runtime(host_devices=machine.host_devices)
    import_craftax()  # before evaluation traces ``init``, which builds the env
    mine = worker_specs(specs, worker_index, num_workers)
    trainers = _registry() if methods is None else methods
    missing = sorted({s.registry_name for s in mine} - set(trainers))
    if missing:
        msg = f"no trainer registered for {missing}"
        raise ValueError(msg)
    if evaluate is None:
        evaluate = functools.partial(evaluate_fixed_seeds, trainers)
    for variable in _RUN_ID_VARIABLES:
        if os.environ.pop(variable, None):
            logger.warning("ignoring %s: the runner picks runs by name", variable)
    experiment = load_experiment_config(experiment_config)
    if tracking_uri is not None:
        experiment = replace(
            experiment, mlflow=replace(experiment.mlflow, tracking_uri=tracking_uri)
        )
    mlflow.set_tracking_uri(_resolve_tracking_uri(experiment.mlflow.tracking_uri))
    trained = 0
    with _stop_on_signals() as stop:
        for spec in mine:
            if max_runs is not None and trained >= max_runs:
                break
            trainer = trainers[spec.registry_name]
            trained += _run_spec(spec, trainer, experiment, machine, evaluate, stop)
            if stop.signum is not None:
                raise SystemExit(128 + stop.signum)
