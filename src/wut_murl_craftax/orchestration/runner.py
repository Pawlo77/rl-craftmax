"""Resumable runner: trains, evaluates and logs every RunSpec of this worker.

A run's directory is its MLflow artifact directory: ``config.json``, ``metrics.npz``,
``episodes.npz``, ``ckpt/`` (log-spaced params), ``state/`` (full resume state) and
``eval/``. A run name is looked up across the whole store: a FINISHED run only joins
the phase (tag ``phases``) and gets its missing evaluations; any other run resumes
from ``state/`` (or restarts in place without it); a new name starts in the phase's
experiment. SIGINT or SIGTERM end the current chunk, save the state and mark the run
KILLED. Trainers come from ``training.interface.METHODS``; see `Trainer`.
"""

import importlib
import json
import logging
import math
import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
    load_index,
    load_state,
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
from .runtime import DeviceRoles, configure_runtime, device_roles, select_platforms
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
_RUN_ID_VARIABLES = ("MLFLOW_RUN_ID", "MLFLOW_ACTIVE_RUN_ID")
"""Environment variables that make MLflow or the tracker reopen a given run."""


class ChunkMetrics(TypedDict):
    """What `Trainer.train_chunk` reports; every array has a leading update axis."""

    losses: jax.Array | np.ndarray
    """(updates, len(LOSS_COLUMNS)) floats in `LOSS_COLUMNS` order."""
    done: jax.Array | np.ndarray
    """(updates, steps, envs) bools: the episode ended at this step."""
    ach: jax.Array | np.ndarray
    """(updates, steps, envs, achievements) bools, read on done steps only."""
    ret: jax.Array | np.ndarray
    """(updates, steps, envs) episode return so far, read on done steps only."""
    length: jax.Array | np.ndarray
    """(updates, steps, envs) episode length so far, read on done steps only."""


class Trainer(Protocol):
    """A registry entry: one method's training, as the runner drives it.

    The state is one pytree (`save_state` round-trips it); ``train_chunk`` called
    with n then m updates must equal one call with n + m.
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
        """The acting policy (``Policy.init_carry``, ``Policy.step``) for evaluation."""
        ...


Evaluate = Callable[[Path, RunSpec, Callable[[], bool]], None]
"""Evaluates a run directory's checkpoints for its spec, skipping finished files;
returns early once the third argument, ``should_stop()``, is true (SIGINT, SIGTERM)."""


def no_evaluation(
    run_dir: Path, spec: RunSpec, should_stop: Callable[[], bool]
) -> None:
    """Default `Evaluate`: fixed-seed evaluation arrives in infrastructure Task 5."""
    del run_dir, spec, should_stop


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
    written = {e["update"] for e in load_index(out)}
    if 0 not in written:
        save_checkpoint(out, 0, 0, 0.0, trainer.params(state))
    t0 = time.monotonic() - saved["wall_s"]
    every, max_reward = log_every(n_updates), float(achievement_weights(spec.env).sum())
    chunks: list[tuple[int, float]] = []
    while update < n_updates and stop.signum is None:
        n = _chunk_size(update, n_updates, ckpts, profile.checkpoint_every)
        start = time.monotonic() - t0
        state, metrics = trainer.train_chunk(state, n)
        wall = time.monotonic() - t0
        chunks.append((n, wall - start))
        _add_updates(log, update, spu, np.linspace(start, wall, n + 1)[1:], metrics)
        update += n
        if update in ckpts and update not in written:
            save_checkpoint(out, update, update * spu, wall, trainer.params(state))
        save_state(out, _snapshot(fresh, state, update, wall, log))
        log.flush()
        points = metric_points(
            log.rows, log.names, update - n, update, every, n_updates, max_reward
        )
        log_points(points)
        log_peak_memory(roles, update * spu)
    if (compile_s := compile_seconds(chunks)) is not None:
        mlflow.log_metric("compile_s", compile_s, step=update * spu)
    return log if update == n_updates else None


def _log_final_score(log: RunLog, env: str) -> None:
    """Log `final_score` of the per-update returns as % of the maximum reward."""
    rows = np.array(log.rows)
    cols = list(BASE_COLUMNS)
    scores = score_pct(rows[:, cols.index("return")], achievement_weights(env).sum())
    score = final_score(scores, rows[:, cols.index("n_episodes")])
    mlflow.log_metric(SCORE_METRIC, score)


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
    trains = found is None or found.info.status != "FINISHED"
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
    evaluate: Evaluate = no_evaluation,
) -> None:
    """Train, evaluate and log this worker's specs, at most ``max_runs`` trained ones.

    ``select_platforms`` and ``configure_runtime`` run first, before any JAX
    computation (the profile decides the backends and the CPU split); runs reused as
    FINISHED do not count towards ``max_runs``. MLflow settings come from
    ``experiment_config``, the experiment name replaced by each spec's phase and the
    tracking URI by ``tracking_uri`` when given (a cluster node's own store).

    Raises:
        SystemExit: 128 + the signal number after SIGINT or SIGTERM.
    """
    machine = load_profile(profile)
    select_platforms(machine.device_roles)
    configure_runtime(host_devices=machine.host_devices)
    mine = worker_specs(specs, worker_index, num_workers)
    trainers = _registry() if methods is None else methods
    missing = sorted({s.registry_name for s in mine} - set(trainers))
    if missing:
        msg = f"no trainer registered for {missing}"
        raise ValueError(msg)
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
