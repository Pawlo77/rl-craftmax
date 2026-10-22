"""Resumable runner: trains, evaluates and logs every RunSpec of this worker.

A run's directory is its MLflow artifact directory: ``config.json``, ``metrics.npz``,
``episodes.npz``, ``ckpt/`` (log-spaced params), ``state/`` (full resume state) and
``eval/``. A run name is looked up across the whole store: a FINISHED run only joins
the phase (tag ``phases``) and gets its missing evaluations; any other run resumes
from ``state/`` (or restarts in place without it); a new name starts in the phase's
experiment. Each run trains in a fresh child process that builds its trainer from the
registry name and config alone (`main`); the parent forwards SIGINT and SIGTERM, which
end the chunk, save the state and mark the run KILLED. Trainers: `Trainer`, registered
in ``training.interface.METHODS``; evaluation: ``eval_rollouts.evaluate_spec``.
"""

import functools
import importlib
import json
import logging
import math
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, TypedDict

import jax
import mlflow
import numpy as np

from ..config import ExperimentConfig, load_experiment_config
from ..evaluation.episode_metrics import (
    added_paths,
    chunk_sums,
    new_first_unlocks,
    resume_from,
    score_pct,
)
from ..evaluation.milestones import game_of
from ..training.checkpoints import (
    STATE_FILE,
    checkpoint_schedule,
    drop_missing_checkpoints,
    load_index,
    load_state_filling,
    missing_checkpoints,
    save_checkpoint,
    save_state,
    saved_update,
)
from ..training.hooks import MEMORY_LIMIT_KEY, MemoryLimitError
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
    TAG_ABORT_GUARD,
    TAG_ABORT_MEMORY,
    TAG_ABORT_METRIC,
    TAG_ABORT_UNMEASURED,
    TAG_ABORT_UNREPORTED,
    TAG_ABORT_UPDATE,
    first_nonfinite,
    first_out_of_bounds,
    flag_hits,
    unreported,
)
from .profiles import Profile, load_profile
from .queue import (
    CONFIG_ARTIFACT,
    EXPERIMENT_CONFIG,
    SCORE_METRIC,
    TAG_BUDGET,
    TAG_CONFIG_HASH,
    TAG_EVAL_PENDING,
    TAG_FAIL_ERROR,
    TAG_METHOD,
    TAG_PHASES,
    TAG_SEED,
    RunSpec,
)
from .reproducibility import code_version
from .runtime import (
    BackendError,
    DeviceRoles,
    check_backend,
    configure_runtime,
    device_roles,
    import_craftax,
    select_platforms,
)
from .store import (
    TAG_STATE_UNREADABLE,
    claim_node_store,
    is_partial,
    merge_stores,
)
from .tracking import _resolve_tracking_uri, artifact_path, build_mlflow_tracker
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
TAG_WARN_ARTIFACTS = "warn/artifacts"
"""Run tag: the latest `Trainer.checkpoint_artifacts` failure (update, error)."""
TAG_CODE_COMMIT = "code/commit"
"""Run tag: the git commit the run started training under (``code/dirty`` beside)."""
TAG_CODE_SEGMENTS = "code/segments"
"""Run tag: comma-joined 12-hex commits (``+dirty``) of each training segment."""
TAG_CODE_CHANGED = "code/changed"
"""Run tag: ``true`` once a segment resumed under another commit than the last one."""
TAG_ONLINE_FROM = "metrics/online_from_update"
"""Run tag: the update a state saved before the online families resumed at; their
sums start from zero there (the episodes running then are partly counted)."""
TAG_FAIL_COUNT = "fail/count"
"""Run tag: how many calls failed on the run (errors and crashes, not aborts)."""
EXTRAS_FILE = "extras.npz"
"""Per-update method metrics and online-family sums (``online/<column>``) of a run,
name -> floats, in the run directory."""
OWNER_FILE = "owner.json"
"""Run-directory file of the process that trains the run (pid, host); gone once the
run is closed, so a RUNNING run with a live owner is never trained twice."""
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
"""Signals that stop the queue after the chunk; SIGHUP: the terminal or SSH closed
(unless the queue inherited it ignored, as under ``nohup``)."""
ECHO_S = 1.0
"""Any stop signal within this many seconds of the first is the same stop: ``uv run``
forwards the terminal's Ctrl-C, which python also got (measured 0.2 s apart), and
``kill <uv>`` (SIGTERM, forwarded) beside ``kill -INT <python>`` is one stop too."""
REGISTRY = "wut_murl_craftax.training.interface:METHODS"
"""``module:attribute`` of the trainer registry a run's child process imports."""
_RUN_ID_VARIABLES = ("MLFLOW_RUN_ID", "MLFLOW_ACTIVE_RUN_ID")
"""Environment variables that make MLflow or the tracker reopen a given run."""
EXIT_NO_BACKEND = 78
"""Exit code of a child whose machine lacks the profile's backend (EX_CONFIG)."""


class ChunkMetrics(TypedDict):
    """What `Trainer.train_chunk` reports; every array has a leading update axis.

    A trainer may add method metrics under ``<group>/<name>`` keys (e.g.
    ``ppo/entropy``): ``(updates,)`` floats that the runner logs unchanged.

    ``ach``, ``ret`` and ``length`` are read on done steps only, and there they are
    the WHOLE finished episode, the done step included: its reward, its step and
    its unlocks, taken from ``StepEvents.unlocked`` (``info["events"]``, the
    pre-reset state). Never rebuild them from env-state comparisons or Craftax's
    ``Achievements/*`` info. Every trainer accumulates them with
    ``evaluation.episode_metrics.episode_step`` and ``events_of``, closes each
    update with ``close_update`` (``online``, ``first_unlock``), and its test calls
    ``tests/training/_contract.assert_done_step_unlock_counted``.
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
    online: jax.Array | np.ndarray
    """(updates, columns) sums of the episodes finished in the update, by
    ``episode_metrics.ONLINE_LAYOUT``; the runner keeps them as `chunk_sums`."""
    first_unlock: jax.Array | np.ndarray
    """(updates, achievements) ints: the run's first env step of each unlock, -1."""


class Trainer(Protocol):
    """A registry entry: one method's training, as the runner drives it.

    The state is one pytree (`save_state` round-trips it); ``train_chunk`` called
    with n then m updates must equal one call with n + m. Its `ChunkMetrics` must
    follow that class's episode rules; the runner times the chunk after
    ``jax.block_until_ready`` on the returned state and metrics.

    A trainer may set the attribute ``loss_columns: tuple[str, ...]``, the
    `LOSS_COLUMNS` it fills (default: all; PQN-RNN leaves the PPO-only ones NaN). The
    runner logs and abort-checks only those; ``metrics.npz`` keeps the full layout.

    A trainer may also define ``checkpoint_artifacts(state, run_dir, update) -> None``:
    the runner calls it in the training process right after each ``ckpt/`` save with
    the live state, the run directory and the checkpoint's update (also for the final
    one), and the method writes its per-checkpoint files there (by convention under
    ``eval/``; writers replace the rows of an update, so a repeated call is safe). It
    is not called for the initial checkpoint nor for one rebuilt from ``state/``; a
    chunk replayed after a crash calls it again for its checkpoint. It never fails a
    run: an exception is logged and tagged `TAG_WARN_ARTIFACTS`.
    """

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Environment steps one update consumes under ``config``."""
        ...

    def init(self, config: Mapping[str, Any], key: jax.Array) -> Any:
        """Initial state: params, optimizer, env states, keys, method state.

        ``config`` is the run's config plus the runtime keys the runner injects (not
        hashed or saved in ``config.json``): ``steps``, the budget in env steps, and
        `MEMORY_LIMIT_KEY`, the profile's ``memory_limit_bytes`` (evaluation leaves it
        out, so nothing is checked there).
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
        ``Policy.step`` also gets the previous env step's ``reward`` and ``done``.
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
    """Signal handler that records the first stop signal (`STOP_SIGNALS`)."""

    signum: int | None = None
    at: float = 0.0
    """`time.monotonic` of the first signal."""

    def __call__(self, signum: int, _frame: FrameType | None) -> None:
        """Record the signal; a second one stops at once, unless it is an echo."""
        if self.signum is not None:
            if time.monotonic() - self.at < ECHO_S:
                return
            raise KeyboardInterrupt
        self.signum, self.at = signum, time.monotonic()
        logger.warning("signal %d: stopping after the current chunk", signum)


@contextmanager
def _stop_on_signals() -> Iterator[_Stop]:
    stop = _Stop()
    # an inherited SIG_IGN for SIGHUP (``nohup``) stays: the queue outlives a hangup
    kept = signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    old = {
        s: signal.signal(s, stop)
        for s in STOP_SIGNALS
        if not (kept and s == signal.SIGHUP)
    }
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
    runs = [r for r in runs if not is_partial(r)]  # a merge a kill cut
    if len(runs) > 1:
        logger.warning("%d runs are named %s; using one", len(runs), name)
    return max(
        runs,
        key=lambda r: (r.info.status == "FINISHED", r.info.start_time),
        default=None,
    )


class _Aborted(Exception):  # noqa: N818 - control flow, not an error of the caller
    """A chunk broke an abort rule: a non-finite loss, or a metric out of its bounds.

    ``update`` is 1-based; ``bound`` marks an ``abort.limits`` hit (permanent).
    """

    def __init__(self, update: int, metric: str, bound: bool = False) -> None:
        what = "is out of its bounds" if bound else "is not finite"
        super().__init__(f"{metric} {what} at update {update}")
        self.update, self.metric, self.bound = update, metric, bound


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
        """Append a chunk after update ``first``; a new name is NaN before it."""
        self.pad(first)
        for name, values in new.items():
            self.series.setdefault(name, [math.nan] * first).extend(values.tolist())
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


def _check_aborts(
    metrics: ChunkMetrics, columns: Sequence[str], abort: Mapping[str, Any], update: int
) -> None:
    """Raise `_Aborted` on the first non-finite logged loss or out-of-bounds metric.

    A non-finite loss comes first, as it ends the run for a retry; a metric out of its
    ``abort.limits`` bounds is checked only on a finite chunk.
    """
    method = _method_metrics(metrics)
    losses = np.asarray(metrics["losses"], float)
    if hit := first_nonfinite(losses, columns, method, abort.get("finite", [])):
        raise _Aborted(update + hit[0] + 1, hit[1])
    if hit := first_out_of_bounds(method, abort.get("limits", [])):
        raise _Aborted(update + hit[0] + 1, hit[1], bound=True)


def _imported(ref: str) -> Any:
    """The object a ``module:attribute`` reference names."""
    module, _, name = ref.partition(":")
    return getattr(importlib.import_module(module), name)


def _set_up_jax(
    machine: Profile,
    methods: Mapping[str, Trainer] | str,
    evaluate: Evaluate | str | None,
) -> tuple[Mapping[str, Trainer], Evaluate]:
    """Platforms, runtime and Craftax before any JAX use, then trainers and evaluation.

    A registry reference is imported after ``configure_runtime``: the trainers import
    Craftax, which starts the JAX backends on import.

    Raises:
        BackendError: The profile's mode needs a GPU this machine lacks.
    """
    select_platforms(machine.device_roles)
    configure_runtime(host_devices=machine.host_devices)
    check_backend(machine.device_roles)
    import_craftax()  # before evaluation traces ``init``, which builds the env
    trainers = _imported(methods) if isinstance(methods, str) else methods
    if isinstance(evaluate, str):
        return trainers, _imported(evaluate)
    return trainers, evaluate or functools.partial(evaluate_fixed_seeds, trainers)


def _tag_flags(hits: Mapping[str, int], flagged: set[str]) -> None:
    """Tag each flag at the first update its rule held at, once per call."""
    for name, at in hits.items():
        if name not in flagged:
            mlflow.set_tag(f"{FLAG_PREFIX}{name}", str(at))
            flagged.add(name)


def _tag_unreported(names: Sequence[str]) -> None:
    """Tag and log abort metrics the trainer never reports: their guard is off."""
    if names:
        mlflow.set_tag(TAG_ABORT_UNREPORTED, ",".join(names))
        logger.warning("the trainer never reports abort metrics %s", names)


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


def _checkpoint_artifacts(trainer: Trainer, state: Any, out: Path, update: int) -> None:
    """Call the trainer's optional `Trainer.checkpoint_artifacts`; warn, never fail."""
    write = getattr(trainer, "checkpoint_artifacts", None)
    if write is None:
        return
    try:
        write(state, out, update)
    except Exception as error:
        logger.warning("checkpoint artifacts of update %d failed: %r", update, error)
        text = f"update {update}: {type(error).__name__}: {error}"
        mlflow.set_tag(TAG_WARN_ARTIFACTS, text[:500])


def _log_first_unlocks(
    stamps: np.ndarray, names: Sequence[str], first: int, spu: int, walls: np.ndarray
) -> None:
    """Log ``first_unlock/<name>`` (env step) and ``first_unlock_wall/<name>`` once.

    Both at the unlock's env step; the wall time is that of the update's end. A chunk
    replayed after a crash finds the point logged and skips it.
    """
    found = new_first_unlocks(stamps, first, spu)
    if not found:
        return
    run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]  # the runner's run
    client = mlflow.MlflowClient()
    for a, step, i in found:
        key = f"first_unlock/{names[a]}"
        if any(m.step == step for m in client.get_metric_history(run_id, key)):
            continue
        wall = float(walls[i])
        mlflow.log_metrics(
            {key: step, f"first_unlock_wall/{names[a]}": wall}, step=step
        )


def _record_chunk(
    log: RunLog,
    extras: "_Extras",
    first: int,
    spu: int,
    walls: np.ndarray,
    metrics: ChunkMetrics,
) -> None:
    """Add a chunk after update ``first`` to the run log and the per-update extras.

    The extras get the method metrics and the online families' sums (`chunk_sums`); the
    first unlocks go to MLflow at once (`_log_first_unlocks`).
    """
    _add_updates(log, first, spu, walls, metrics)
    families = chunk_sums(metrics, game_of(log.env_name))
    extras.extend(first, _method_metrics(metrics) | families)
    stamps = np.asarray(metrics["first_unlock"])
    _log_first_unlocks(stamps, log.names, first, spu, walls)


def _checkpoint(
    trainer: Trainer,
    state: Any,
    out: Path,
    update: int,
    spu: int,
    wall: float,
    written: set[int],
) -> None:
    """Save checkpoint ``update`` unless ``written``, then its artifacts.

    A chunk replayed after a crash finds its checkpoint written and only rewrites
    the artifacts (`_checkpoint_artifacts`), so none is lost.
    """
    if update not in written:
        save_checkpoint(out, update, update * spu, wall, trainer.params(state))
    _checkpoint_artifacts(trainer, state, out, update)


def _move_aside(path: Path) -> None:
    """Rename ``path`` (if any) to ``<name>.unreadable``, or ``.unreadable.<n>``."""
    if not path.exists():
        return
    target, n = path.with_name(f"{path.name}.unreadable"), 1
    while target.exists():
        target, n = path.with_name(f"{path.name}.unreadable.{n}"), n + 1
    path.replace(target)


def _resume(out: Path, fresh: dict[str, Any], spu: int) -> dict[str, Any]:
    """The saved state of ``out``, else ``fresh``.

    An unreadable state (a cut copy) is moved aside with ``ckpt/`` and ``eval/``
    (`_move_aside`), so the run restarts as one lineage; tag `TAG_STATE_UNREADABLE`.

    A state saved before the online families gets fresh accumulators that count
    env steps from its update on, and the run is tagged `TAG_ONLINE_FROM`.
    """
    if not (out / STATE_FILE).exists():
        return fresh
    if saved_update(out) < 0:  # a cut copy: its progress is lost, start afresh
        # its checkpoints and evaluations go too: the restart is a new lineage
        for path in (out / STATE_FILE, out / "ckpt", out / "eval"):
            _move_aside(path)
        mlflow.set_tag(TAG_STATE_UNREADABLE, "true")
        return fresh
    saved, filled = load_state_filling(out, fresh, added_paths(fresh))
    if filled:
        update = saved["update"]
        saved = {**saved, "trainer": resume_from(saved["trainer"], update * spu)}
        mlflow.set_tag(TAG_ONLINE_FROM, str(update))
        logger.warning("online metrics of %s start at update %d", out, update)
    return saved


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
        _Aborted: A chunk reported a non-finite loss or a metric out of its bounds;
            nothing of it is saved.
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
    config = {
        **spec.config,
        "steps": spec.steps,
        MEMORY_LIMIT_KEY: profile.memory_limit_bytes,
    }
    fresh = state_template(trainer.init(config, jax.random.key(spec.seed)), spec)
    saved = _resume(out, fresh, spu)
    state, update = saved["trainer"], saved["update"]
    log = RunLog(out, spec.env, achievement_names(spec.env))
    rows, episodes = saved["log"]
    log.rows, log.episodes = rows.tolist(), [episodes]
    lost = _repair_checkpoints(trainer, out, fresh, saved, ckpts, spu)
    written = {e["update"] for e in load_index(out)}
    columns, extras = _loss_columns(trainer), _Extras.load(out, update)
    abort = spec.abort or {}
    rules, flagged = abort.get("flags", []), set[str]()
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
        _check_aborts(metrics, columns, abort, update)
        chunks.append((n, wall - start))
        _record_chunk(
            log, extras, update, spu, np.linspace(start, wall, n + 1)[1:], metrics
        )
        if not chunks[
            1:
        ]:  # the first chunk of this call shows what the trainer reports
            names = unreported(
                rules, abort.get("finite", []), extras.series, abort.get("limits", [])
            )
            _tag_unreported(names)
        update += n
        _tag_flags(flag_hits(rules, extras.series, n_updates), flagged)
        if update in ckpts:
            _checkpoint(trainer, state, out, update, spu, wall, written)
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
            game=game_of(spec.env),
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


def _mark_abort(
    previous: Mapping[str, str],
    abort: _Aborted,
    rules: Mapping[str, Any] | None,
    unmeasured: Sequence[str] = (),
) -> None:
    """Tag the active run with the abort; the `MAX_ABORTS`-th one marks it a bug.

    A guard metric in the method's ``abort.permanent`` (``rules``; a retry would
    break it again) or a metric out of its ``abort.limits`` bounds is tagged
    `TAG_ABORT_GUARD` at once instead, and the queue skips the run from then on;
    unless the profile has not measured that bound on its devices (``unmeasured``,
    `Profile.unmeasured_limits`): then it counts as a retried abort, tagged
    `TAG_ABORT_UNMEASURED`.
    """
    soft = abort.bound and abort.metric in unmeasured
    if not soft and (abort.bound or abort.metric in (rules or {}).get("permanent", [])):
        mlflow.set_tags({
            TAG_ABORT_GUARD: abort.metric,
            TAG_ABORT_METRIC: abort.metric,
            TAG_ABORT_UPDATE: str(abort.update),
        })
        logger.warning("run aborted for good (%s): %s", TAG_ABORT_GUARD, abort)
        return
    count = int(previous.get(TAG_ABORT_COUNT, 0)) + 1
    tags = {
        TAG_ABORT_COUNT: str(count),
        TAG_ABORT_METRIC: abort.metric,
        TAG_ABORT_UPDATE: str(abort.update),
    }
    if soft:
        tags[TAG_ABORT_UNMEASURED] = abort.metric
    if count >= MAX_ABORTS:
        tags[TAG_ABORT_BUG] = "true"
    mlflow.set_tags(tags)
    logger.warning("run aborted (%d of %d allowed): %s", count, MAX_ABORTS, abort)


def _mark_memory(previous: Mapping[str, str], error: MemoryLimitError) -> None:
    """Tag the active run a permanent abort: its config does not fit this machine."""
    mlflow.set_tags(_fail_tags(previous, error) | {TAG_ABORT_MEMORY: "true"})
    logger.warning("run aborted for good (%s): %s", TAG_ABORT_MEMORY, error)


def _live_owner(found: Any) -> int | None:
    """The PID of a live process on this host that trains ``found`` (RUNNING)."""
    if found is None or found.info.status != "RUNNING":
        return None
    try:
        path = artifact_path(found.info.artifact_uri) / OWNER_FILE
        owner = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    pid = int(owner["pid"])
    if owner["host"] != socket.gethostname() or pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:  # alive, another user's process
        pass
    return pid


def _skip_reason(found: Any) -> str | None:
    """Why the queue skips a run: live owner, bug, guard or memory abort, lost files."""
    if (pid := _live_owner(found)) is not None:
        return f"is trained by live process {pid} on this host"
    tags = found.data.tags if found else {}
    if tags.get(TAG_ABORT_BUG) == "true":
        return f"is marked a bug ({TAG_ABORT_BUG})"
    if guard := tags.get(TAG_ABORT_GUARD):
        return f"aborted for good on guard {guard} ({TAG_ABORT_GUARD})"
    if tags.get(TAG_ABORT_MEMORY) == "true":
        return f"is over the machine's memory limit ({TAG_ABORT_MEMORY})"
    if found and found.info.status == "FAILED" and tags.get(TAG_CKPT_MISSING):
        return f"lost checkpoints {tags[TAG_CKPT_MISSING]} for good"
    return None


def _fail_tags(
    previous: Mapping[str, str], error: BaseException | str
) -> dict[str, str]:
    """`TAG_FAIL_ERROR` and the next `TAG_FAIL_COUNT` of a failed call."""
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    count = int(previous.get(TAG_FAIL_COUNT, 0)) + 1
    return {TAG_FAIL_ERROR: text[:500], TAG_FAIL_COUNT: str(count)}


def _code_tags(name: str, previous: Mapping[str, str]) -> dict[str, str]:
    """Code version tags of a training segment; a commit change is tagged and logged.

    A resume under another commit is allowed (fixes must reach running runs).
    """
    commit, dirty = code_version()
    segment = commit[:12] + ("+dirty" if dirty else "")
    segments = [s for s in previous.get(TAG_CODE_SEGMENTS, "").split(",") if s]
    tags = {TAG_CODE_SEGMENTS: ",".join([*segments, segment])}
    if TAG_CODE_COMMIT not in previous:
        tags |= {TAG_CODE_COMMIT: commit, "code/dirty": str(dirty).lower()}
    if segments and segments[-1].split("+")[0] != commit[:12]:
        tags[TAG_CODE_CHANGED] = "true"
        logger.warning(
            "%s resumes under commit %s, not %s", name, commit[:12], segments[-1]
        )
    return tags


def _close(tracker: Any, status: str, trained: bool) -> None:
    """Close the active run ``status``; one that did not train this call stays FINISHED.

    Phase 0 review I6: a cut evaluation of a reused FINISHED run must not send it
    back through `_train`; it keeps `TAG_EVAL_PENDING`, which a FINISHED close clears
    with the error of the cut call (`TAG_FAIL_ERROR`; `TAG_FAIL_COUNT` stays).
    """
    if status == "FINISHED":
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]  # the runner's run
        tags = mlflow.get_run(run_id).data.tags
        if TAG_EVAL_PENDING in tags:
            for key in (TAG_EVAL_PENDING, TAG_FAIL_ERROR):
                if key in tags:
                    mlflow.delete_tag(key)
    elif not trained:
        status = "FINISHED"
    tracker.close(status)


def _train_to_score(
    trainer: Trainer,
    spec: RunSpec,
    out: Path,
    profile: Profile,
    stop: _Stop,
    found: Any,
) -> bool:
    """Train the active run (`_train`) and log its final score; False on a stop."""
    roles = device_roles()
    mlflow.set_tags(run_tags(roles))  # where it trains now, not where it began
    mlflow.set_tags(_code_tags(spec.name, found.data.tags if found else {}))
    (out / CONFIG_ARTIFACT).write_text(json.dumps(spec.config, indent=1))
    log = _train(trainer, spec, out, profile, roles, stop)
    if log is None or stop.signum is not None:
        return False
    _log_final_score(log, spec.env)
    return True


def _run_spec(
    spec: RunSpec,
    trainer: Trainer,
    experiment: ExperimentConfig,
    profile: Profile,
    evaluate: Evaluate,
    stop: _Stop,
) -> bool:
    """Bring one run to FINISHED, or to KILLED on a signal; return if it trained.

    A stop that came before the run was opened leaves the run untouched.

    Raises:
        Exception: Any error of the run, after it is closed FAILED with the
            `TAG_FAIL_ERROR` and `TAG_FAIL_COUNT` tags.
    """
    if stop.signum is not None:
        return False
    found = find_run(spec.name)
    if (reason := _skip_reason(found)) is not None:
        logger.warning("%s %s, skipped", spec.name, reason)
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
    owner = {"pid": os.getpid(), "host": socket.gethostname()}
    (out / OWNER_FILE).write_text(json.dumps(owner))
    # a FINISHED run with a lost checkpoint file goes through `_train` to repair it
    trains = (
        found is None
        or (found.info.status != "FINISHED" and TAG_EVAL_PENDING not in found.data.tags)
        or bool(missing_checkpoints(out))
    )
    try:
        if trains and not _train_to_score(trainer, spec, out, profile, stop, found):
            tracker.close("KILLED")  # the next call logs the score and evaluates
            return True
        (out / "eval").mkdir(exist_ok=True)
        if not trains:  # cleared by a complete evaluation (`_close`)
            mlflow.set_tag(TAG_EVAL_PENDING, "true")
        evaluate(out, spec, lambda: stop.signum is not None)
    except KeyboardInterrupt:
        _close(tracker, "KILLED", trains)
        raise
    except _Aborted as abort:
        previous = found.data.tags if found else {}
        _mark_abort(previous, abort, spec.abort, profile.unmeasured_limits)
        tracker.close("FAILED")  # its last state/ stays for the one retry, if any
        return True
    except MemoryLimitError as error:  # no retry here can fit: skipped from now on
        _mark_memory(found.data.tags if found else {}, error)
        tracker.close("FAILED")
        return True
    except Exception as error:
        mlflow.set_tags(_fail_tags(found.data.tags if found else {}, error))
        _close(tracker, "FAILED", trains)
        raise
    else:
        # evaluation may have stopped early
        _close(tracker, "KILLED" if stop.signum is not None else "FINISHED", trains)
        return trains
    finally:
        (out / OWNER_FILE).unlink(missing_ok=True)


@dataclass(frozen=True)
class _Job:
    """What a run's child process gets: picklable values only, no live objects."""

    spec: RunSpec
    experiment: ExperimentConfig
    machine: Profile
    registry: str
    evaluate: str | None


def _child(job: _Job, stop_set: Any, stop_signum: Any, trained: Any, done: Any) -> None:
    """One run in a fresh process; the parent sets ``stop_set`` to forward a signal.

    The child also stops (saves the state, closes the run KILLED) when its parent
    dies, so an orphan never trains on beside a later queue call.
    """
    os.setpgrp()  # a terminal's Ctrl-C reaches the parent only, which forwards it
    for s in STOP_SIGNALS:  # also a scheduler's, sent to every PID
        signal.signal(s, signal.SIG_IGN)
    parent = multiprocessing.parent_process()  # its sentinel sees a death at once
    stop = _Stop()

    def forwarded() -> None:
        while not stop_set.wait(1.0):
            if parent is not None and not parent.is_alive():  # the queue died
                logger.warning("queue process %d is gone: stopping", parent.pid)
                stop.signum = signal.SIGHUP
                return
        stop.signum = stop_signum.value

    # before the setup (imports, backends: seconds), so a stop then opens no run
    threading.Thread(target=forwarded, daemon=True).start()
    try:
        trainers, evaluate = _set_up_jax(job.machine, job.registry, job.evaluate)
    except BackendError:
        logger.exception("no backend for profile mode %s", job.machine.device_roles)
        sys.exit(EXIT_NO_BACKEND)  # every later run would fail the same way
    if job.spec.registry_name not in trainers:
        msg = f"no trainer registered for {job.spec.registry_name!r} in {job.registry}"
        raise ValueError(msg)
    mlflow.set_tracking_uri(_resolve_tracking_uri(job.experiment.mlflow.tracking_uri))
    trainer = trainers[job.spec.registry_name]
    trained.value = _run_spec(
        job.spec, trainer, job.experiment, job.machine, evaluate, stop
    )
    done.value = True  # the run is closed: a crash from here on is only a warning


def _close_orphan(name: str, status: str, error: str) -> str:
    """Close a run its dead child left RUNNING; return the run's error text.

    A reused run cut in its evaluation (`TAG_EVAL_PENDING`) is closed FINISHED.
    """
    found = find_run(name)
    if found is None or found.info.status != "RUNNING":
        return str(found.data.tags.get(TAG_FAIL_ERROR, error)) if found else error
    client = mlflow.MlflowClient()
    if status == "FAILED":
        for key, value in _fail_tags(found.data.tags, error).items():
            client.set_tag(found.info.run_id, key, value)
    if found.data.tags.get(TAG_EVAL_PENDING) == "true":
        status = "FINISHED"
    client.set_terminated(found.info.run_id, status)
    (artifact_path(found.info.artifact_uri) / OWNER_FILE).unlink(missing_ok=True)
    return error


def _in_child(job: _Job, stop: _Stop) -> bool:
    """`_run_spec` in a spawned child; return if it trained.

    A signal to this process is forwarded once; a second one kills the child.

    Raises:
        BackendError: The child found no backend for the profile's mode.
        RuntimeError: The child exited non-zero without a signal (its error, or a
            crash such as an OOM kill: the run is then closed FAILED here).
    """
    ctx = multiprocessing.get_context("spawn")
    stop_set, signum = ctx.Event(), ctx.Value("i", 0)
    trained, done = ctx.Value("b", False), ctx.Value("b", False)
    args = (job, stop_set, signum, trained, done)
    child = ctx.Process(target=_child, args=args, name=job.spec.name)
    try:
        child.start()
        while child.exitcode is None:
            child.join(0.1)
            if stop.signum is not None and not stop_set.is_set():
                signum.value = stop.signum
                stop_set.set()
    except KeyboardInterrupt:  # a second signal: stop at once
        if child.is_alive():  # false while `start` had not returned
            child.kill()
            child.join()
        _close_orphan(job.spec.name, "KILLED", "killed by a second signal")
        raise
    if child.exitcode == EXIT_NO_BACKEND:
        msg = "the child found no GPU backend for profile mode gpu (see its log)"
        raise BackendError(msg)
    if child.exitcode and done.value:
        logger.warning(
            "%s: child exited with %d after closing its run",
            job.spec.name,
            child.exitcode,
        )
    elif child.exitcode:
        crashed = f"child process exited with {child.exitcode}"
        if stop.signum is not None:
            _close_orphan(job.spec.name, "KILLED", crashed)
        else:
            msg = f"{_close_orphan(job.spec.name, 'FAILED', crashed)} ({crashed})"
            raise RuntimeError(msg)
    return bool(trained.value)


type RunOne = Callable[[RunSpec, _Stop], bool]
"""Brings one spec's run to FINISHED (or KILLED on a stop); returns if it trained."""


def _runs_here(
    machine: Profile,
    experiment: ExperimentConfig,
    methods: Mapping[str, Trainer],
    evaluate: Evaluate | str | None,
    specs: Sequence[RunSpec],
) -> RunOne:
    """Runs in this process with live trainers (tests).

    Raises:
        ValueError: If a spec's registry name has no trainer.
    """
    trainers, evaluate_here = _set_up_jax(machine, methods, evaluate)
    if missing := sorted({s.registry_name for s in specs} - set(trainers)):
        msg = f"no trainer registered for {missing}"
        raise ValueError(msg)

    def run(spec: RunSpec, stop: _Stop) -> bool:
        trainer = trainers[spec.registry_name]
        return _run_spec(spec, trainer, experiment, machine, evaluate_here, stop)

    return run


def _runs_in_children(
    machine: Profile,
    experiment: ExperimentConfig,
    registry: str,
    evaluate: Evaluate | str | None,
) -> RunOne:
    """Runs in one spawned child each; skipped runs start no child.

    Raises:
        TypeError: If ``evaluate`` is a callable, which a child cannot import.
    """
    if evaluate is not None and not isinstance(evaluate, str):
        msg = "a child process imports `evaluate`: pass a module:attribute reference"
        raise TypeError(msg)

    def run(spec: RunSpec, stop: _Stop) -> bool:
        if (reason := _skip_reason(find_run(spec.name))) is not None:
            logger.warning("%s %s, skipped", spec.name, reason)
            return False
        if stop.signum is not None:  # stopped before the child: start none
            return False
        return _in_child(_Job(spec, experiment, machine, registry, evaluate), stop)

    return run


def _check_node_store(
    seed_from: str | None, tracking_uri: str | None, num_workers: int, node: bool
) -> None:
    """Refuse a seed or node without a node store, and one of another array size.

    Only an array node's store (``node`` or a seed) records its array size: the
    main store, even reached by ``tracking_uri``, takes any ``num_workers``. Runs
    before the seed, so a refused store is not touched (`claim_node_store`).
    """
    if not (node or seed_from is not None):
        return
    if tracking_uri is None:
        msg = "an array node (node, seed_from) needs tracking_uri: its own store"
        raise ValueError(msg)
    claim_node_store(mlflow.get_tracking_uri(), num_workers)


def _seed(seed_from: str, node: str, specs: Sequence[RunSpec], stop: _Stop) -> None:
    """Merge the specs' runs of ``seed_from`` into ``node``; a stop then ends `main`.

    It runs under `main`'s signal handlers: a stop lands after the seed (the merge
    leaves no half copy, see `merge_stores`) and opens no run.
    """
    counts = merge_stores(seed_from, node, names={s.name for s in specs})
    logger.info("seeded from %s: %s", seed_from, counts)
    if stop.signum is not None:
        raise SystemExit(128 + stop.signum)


def main(
    specs: Sequence[RunSpec],
    profile: Path | str,
    worker_index: int = 0,
    num_workers: int = 1,
    max_runs: int | None = None,
    *,
    experiment_config: Path = EXPERIMENT_CONFIG,
    tracking_uri: str | None = None,
    seed_from: str | None = None,
    node: bool = False,
    methods: Mapping[str, Trainer] | None = None,
    evaluate: Evaluate | str | None = None,
    registry: str = REGISTRY,
) -> None:
    """Train, evaluate and log this worker's specs, at most ``max_runs`` trained ones.

    Each run trains in a fresh spawned child that imports ``registry`` (a
    ``module:attribute`` reference) and ``evaluate`` (a reference, or None for
    `evaluate_fixed_seeds`). Live ``methods`` (tests) run in this process instead,
    with ``evaluate`` a callable. ``select_platforms``, ``configure_runtime`` and
    ``import_craftax`` run first wherever runs train, before any JAX computation or
    trace. Runs reused as FINISHED do not count towards ``max_runs``. MLflow settings
    come from ``experiment_config``, the experiment name replaced by each spec's
    phase and the tracking URI by ``tracking_uri`` when given (a node's own store).
    ``seed_from``, a read-only copy of the main store, first merges this worker's runs
    into the node's (`merge_stores`): the node reuses and resumes them as the main
    store would, and the merge back returns their new evaluations (review I5).
    ``node`` (or a seed) marks this worker as one node of an array: its store records
    ``num_workers`` and refuses another array size until merged (`claim_node_store`).

    A run that fails is closed FAILED with its error tagged (`TAG_FAIL_ERROR`) and
    logged, and the next spec runs; runs marked a bug, or failed for lost
    checkpoints, are skipped with a warning.

    Raises:
        SystemExit: 128 + the signal number after SIGINT or SIGTERM.
        RuntimeError: At the end, naming every run of this call that failed.
        ValueError: ``seed_from`` or ``node`` without ``tracking_uri``, or a node
            store of another array size that is not merged (`claim_node_store`).
        BackendError: The profile's mode needs a GPU this machine lacks (at once).
    """
    machine = load_profile(profile)
    mine = worker_specs(specs, worker_index, num_workers)
    for variable in _RUN_ID_VARIABLES:  # before a child inherits the environment
        if os.environ.pop(variable, None):
            logger.warning("ignoring %s: the runner picks runs by name", variable)
    experiment = load_experiment_config(experiment_config)
    if tracking_uri is not None:
        experiment = replace(
            experiment, mlflow=replace(experiment.mlflow, tracking_uri=tracking_uri)
        )
    mlflow.set_tracking_uri(_resolve_tracking_uri(experiment.mlflow.tracking_uri))
    _check_node_store(seed_from, tracking_uri, num_workers, node)
    run_one = (
        _runs_in_children(machine, experiment, registry, evaluate)
        if methods is None
        else _runs_here(machine, experiment, methods, evaluate, mine)
    )
    trained, failed = 0, []
    with _stop_on_signals() as stop:
        if seed_from is not None:
            _seed(seed_from, str(tracking_uri), mine, stop)
        for spec in mine:
            if max_runs is not None and trained >= max_runs:
                break
            try:
                trained += run_one(spec, stop)
            except BackendError:  # every run of this machine would fail the same way
                raise
            except Exception as error:  # one run must not stop the queue
                logger.exception("%s failed, next run", spec.name)
                failed.append(f"{spec.name}: {type(error).__name__}: {error}")
            if stop.signum is not None:
                raise SystemExit(128 + stop.signum)
    if failed:
        msg = f"{len(failed)} of {len(mine)} runs failed: " + "; ".join(failed)
        raise RuntimeError(msg)
