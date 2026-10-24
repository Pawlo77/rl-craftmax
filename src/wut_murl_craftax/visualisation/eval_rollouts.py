"""Fixed-seed evaluation of checkpoints, stored as replayable episodes.

`evaluate` plays `EVAL_SEEDS` with a registry method's ``Policy`` (stochastic or greedy)
(each step handed the previous step's reward and done, zeros at the start)
in jitted chunks of `CHUNK_STEPS` until every episode has ended and writes
``eval/<mode>/<update>.npz``: ``seeds``, ``actions`` (-1 after the end), ``length``,
``method``, ``events/<StepEvents field>`` per step and the policy's ``extras/<name>``.
Observations, rewards, states and frames are rebuilt by replaying seed + actions
(`replay_episode`); `evaluate_spec` is the runner's schedule. Keys: [[Recording training
data]]. Usage: ``python -m wut_murl_craftax.visualisation.eval_rollouts RUN``
"""

import argparse
import functools
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from ..evaluation.episode_metrics import score_pct
from ..evaluation.events import NO_SOURCE, StepEvents, step_events
from ..evaluation.milestones import Game, game_of
from ..orchestration import (
    configure_runtime,
    device_roles,
    env_mesh,
    import_craftax,
    shard_envs,
)
from ..orchestration.queue import COMBINE_KEY, CONFIG_ARTIFACT, RunSpec
from ..orchestration.runner import Trainer
from ..training.checkpoints import CheckpointEntry, load_index, load_params
from ..training.interface import Policy, run_trainer
from ..training.ppo import PPOPolicy, make_env, policy_value
from ..training.records import achievement_weights, is_classic
from ..training.wrappers import unwrap_state
from .rollout import FEATURES, GameKind, Rollout, game_for, keys

logger = logging.getLogger(__name__)

type Mode = Literal["stochastic", "greedy"]
EVAL_SEEDS: tuple[int, ...] = tuple(range(10000, 10064))
"""The 64 eval seeds of every checkpoint, method and benchmark."""
MODES: tuple[Mode, ...] = ("stochastic", "greedy")
"""Sampled actions (key ``fold_in(k_act, t)``) and the policy's argmax."""
CHUNK_STEPS = 1000
"""Steps per jitted scan; scans repeat until every seed's episode has ended."""


def _never() -> bool:
    return False


class EvalSummary(NamedTuple):
    """Mean return of one written eval file."""

    update: int
    env_steps: int
    mode: Mode
    ret: float
    score_pct: float
    """``ret`` as % of the maximum reward (22 Classic, 226 Craftax)."""


class Replay(NamedTuple):
    """One episode rebuilt from seed + actions; arrays lead with the step."""

    obs: np.ndarray
    """The observation each action was chosen on."""
    reward: np.ndarray
    done: np.ndarray
    events: StepEvents
    states: Any
    """Craftax states before each action plus the final one: ``n + 1`` rows."""


def _where(live: jax.Array, new: Any, old: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(live, x, y), new, old)


@dataclass(frozen=True)
class EvalRun:
    """A runner run opened for evaluation: config, checkpoints, policy, env."""

    dir: Path
    cfg: dict[str, Any]
    index: list[CheckpointEntry]
    policy: Policy
    template: Any
    """Params structure of the checkpoints (abstract leaves)."""
    env: Any
    """The training env chain without resets: the bare env (``auto_reset=False``)
    with the method's ``env_wrapper_fn`` inside `LogWrapper` (`make_env`)."""
    env_params: Any

    @classmethod
    def load(
        cls, run_dir: Path, methods: Mapping[str, Trainer] | None = None
    ) -> "EvalRun":
        """Open ``run_dir`` with ``methods[cfg["method"]]`` (`METHODS`, `run_trainer`).

        The trainer is set up from ``config.json`` as the runner trained it: the
        method's own keys build its hooks, its ``env_wrapper_fn`` wraps the env; a
        combo run gets its own pair, not the live combo.yaml's.
        """
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / CONFIG_ARTIFACT).read_text())
        trainer = run_trainer(cfg, methods)
        index = load_index(run_dir)
        steps = max((e["env_steps"] for e in index), default=0)
        config = {k: v for k, v in cfg.items() if k != COMBINE_KEY} | {"steps": steps}
        import_craftax()  # not inside the trace below: Craftax would keep its tracers
        # shapes only: acting needs the params structure, not an env batch or values
        state = jax.eval_shape(lambda: trainer.init(config, jax.random.key(0)))
        env, env_params = make_env(cfg["env"], trainer.env_wrapper(state))
        return cls(
            run_dir,
            cfg,
            index,
            trainer.policy(state),
            trainer.params(state),
            env,
            env_params,
        )

    @property
    def method(self) -> str:
        """Registry name of the run's method."""
        return str(self.cfg["method"])

    @property
    def game(self) -> Game:
        """``classic`` or ``craftax``."""
        return game_of(self.cfg["env"])

    @property
    def classic(self) -> bool:
        """Whether the run is on Craftax-Classic."""
        return is_classic(self.cfg["env"])

    @property
    def max_reward(self) -> float:
        """22 for Classic, 226 for Craftax."""
        return float(achievement_weights(self.cfg["env"]).sum())

    @property
    def max_steps(self) -> int:
        """Episode cap the env enforces: 10,000 on Classic, 100,000 on Craftax."""
        return int(self.env_params.max_timesteps)

    @property
    def updates(self) -> list[int]:
        """Checkpointed updates, ascending."""
        return sorted(e["update"] for e in self.index)

    def env_steps(self, update: int) -> int:
        """Environment steps trained at a checkpoint."""
        return next(e["env_steps"] for e in self.index if e["update"] == update)

    def params(self, update: int) -> Any:
        """Params of a checkpoint."""
        return load_params(self.dir, update, self.template)

    def env_step(
        self, state: Any, action: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, StepEvents]:
        """``(obs, state, reward, done, events)``; events as the training wrapper's."""
        obs, state2, r, d, _ = self.env.step(key, state, action, self.env_params)
        ev = step_events(self.game, unwrap_state(state), action, unwrap_state(state2))
        return obs, state2, r, d, ev

    @functools.cached_property
    def _reset(self) -> Callable[[jax.Array], Any]:
        """Jitted ``seeds -> (obs, state, policy carry, done, reward)``."""

        def reset(seeds: jax.Array) -> Any:
            k_reset = jax.vmap(lambda s: keys(s)[0])(seeds)
            obs, st = jax.vmap(self.env.reset, (0, None))(k_reset, self.env_params)
            n = seeds.shape[0]
            carry = self.policy.init_carry(n)
            return obs, st, carry, jnp.zeros(n, bool), jnp.zeros(n, jnp.float32)

        return cast(Callable[[jax.Array], Any], jax.jit(reset))

    @functools.cached_property
    def _chunk(self) -> Callable[..., Any]:
        """Jitted ``(params, carry, seeds, t0, greedy, n) -> (carry, records)``.

        Each seed has its own keys and a batch of one, so a seed plays the same in
        any seed list; a done seed keeps its final state and records padding.
        """
        extras = getattr(self.policy, "extras", None)

        def seed_chunk(
            params: Any, c: Any, seed: jax.Array, t0: jax.Array, greedy: Any, n: int
        ) -> Any:
            k_reset, k_env = keys(seed)  # type: ignore[arg-type]  # traced seed
            k_act = jax.random.fold_in(k_reset, 2)

            def step(c: Any, t: jax.Array) -> tuple[Any, dict[str, Any]]:
                obs, st, pc, done, r = c  # r, done: the step that produced obs
                one = jax.tree.map(lambda x: x[None], pc)
                k = jax.random.fold_in(k_act, t)
                a, pc2 = self.policy.step(
                    params, one, obs[None], k, greedy, r[None], done[None]
                )
                a, pc2 = a[0], jax.tree.map(lambda x: x[0], pc2)
                obs2, st2, r2, d, ev = self.env_step(
                    st, a, jax.random.fold_in(k_env, t)
                )
                live = ~done
                pad = jax.tree.map(jnp.zeros_like, ev)
                pad = pad.replace(
                    damage_source=jnp.full_like(pad.damage_source, NO_SOURCE)
                )
                rec = {"actions": jnp.where(live, a, -1).astype(jnp.int8)}
                rec["events"] = _where(live, ev, pad)
                if extras is not None:
                    rec["extras"] = jax.tree.map(
                        lambda x: jnp.where(live, x, 0), extras(pc2)
                    )
                kept = _where(live, (obs2, st2, pc2), (obs, st, pc))
                return (*kept, done | d, jnp.where(live, r2, r)), rec

            return jax.lax.scan(step, c, t0 + jnp.arange(n))

        @functools.partial(jax.jit, static_argnames="n")
        def chunk(
            params: Any,
            carry: Any,
            seeds: jax.Array,
            t0: jax.Array,
            greedy: Any,
            n: int,
        ) -> Any:
            return jax.vmap(lambda c, s: seed_chunk(params, c, s, t0, greedy, n))(
                carry, seeds
            )

        return cast(Callable[..., Any], chunk)

    @functools.cached_property
    def _replay(self) -> Callable[[jax.Array, jax.Array], Any]:
        """Jitted ``(seed, actions) -> (obs, reward, done, events, states)``."""

        def replay(seed: jax.Array, actions: jax.Array) -> Any:
            k_reset, k_env = keys(seed)  # type: ignore[arg-type]  # traced seed
            obs, st = self.env.reset(k_reset, self.env_params)

            def step(c: Any, x: tuple[jax.Array, jax.Array]) -> Any:
                obs, st = c
                t, a = x
                obs2, st2, r, d, ev = self.env_step(st, a, jax.random.fold_in(k_env, t))
                return (obs2, st2), (obs, unwrap_state(st), r, d, ev)

            steps = (jnp.arange(actions.shape[0]), actions)
            (_, last), (o, s, r, d, ev) = jax.lax.scan(step, (obs, st), steps)
            final = unwrap_state(last)
            states = jax.tree.map(
                lambda xs, x: jnp.concatenate([xs, x[None]]), s, final
            )
            return o, r, d, ev, states

        return cast(Callable[[jax.Array, jax.Array], Any], jax.jit(replay))


def rollout_seeds(
    run: EvalRun,
    params: Any,
    seeds: Sequence[int],
    greedy: bool,
    chunk: int = CHUNK_STEPS,
    should_stop: Callable[[], bool] = _never,
) -> dict[str, np.ndarray] | None:
    """Play every seed's episode to its end: the eval file's arrays, None if stopped.

    Chunks of ``chunk`` steps repeat until every episode has ended; the env ends
    each one by `EvalRun.max_steps`, so no episode is cut.
    """
    roles, recs, shards = device_roles(), [], 1
    actor = roles.actor
    with jax.default_device(actor):  # Metal steps Craftax slowly: rollouts on the CPU
        seed_arr = jnp.asarray(seeds, jnp.int32)
        carry = run._reset(seed_arr)
        cpus = jax.devices("cpu")
        if actor.platform == "cpu" and len(cpus) > 1 and len(seeds) % len(cpus) == 0:
            mesh = env_mesh()  # the seeds split over the CPU devices, as training envs
            shards = len(cpus)
            seed_arr, carry = shard_envs((seed_arr, carry), mesh)
            params = jax.device_put(params, NamedSharding(mesh, PartitionSpec()))  # type: ignore[no-untyped-call]  # jax API
        flag = jnp.bool_(greedy)
        for t0 in range(0, run.max_steps, chunk):
            carry, rec = run._chunk(params, carry, seed_arr, t0, flag, n=chunk)
            recs.append(jax.device_get(rec))
            if bool(carry[3].all()):
                break
            if should_stop():
                return None
    tr = jax.tree.map(lambda *xs: np.concatenate(xs, 1), *recs)
    actions = tr["actions"]
    out = {
        "seeds": np.asarray(seeds),
        "actions": actions,
        "length": (actions >= 0).sum(1).astype(np.int32),
        # results are bit-exact per device setup; other hardware may round differently
        "devices": np.array(
            json.dumps({
                "actor": str(actor),
                "learner": str(roles.learner),
                "shards": shards,
            })
        ),
    }
    out |= {
        f"events/{f.name}": getattr(tr["events"], f.name) for f in fields(StepEvents)
    }
    out |= {f"extras/{k}": v for k, v in tr.get("extras", {}).items()}
    return out


def eval_returns(ev: Mapping[str, np.ndarray], env: str) -> np.ndarray:
    """Episode return per seed from the stored events (unlock weights + 0.1 health)."""
    unlocks = ev["events/unlocked"].sum(1, dtype=np.float64)
    health = ev["events/health_delta"].sum(1, dtype=np.float64)
    return cast(np.ndarray, unlocks @ achievement_weights(env) + 0.1 * health)


def _summary(
    run: EvalRun, update: int, mode: Mode, ev: Mapping[str, np.ndarray]
) -> EvalSummary:
    ret = float(eval_returns(ev, run.cfg["env"]).mean())
    return EvalSummary(
        update, run.env_steps(update), mode, ret, score_pct(ret, run.max_reward)
    )


def _path(run_dir: Path, update: int, mode: Mode) -> Path:
    return Path(run_dir) / "eval" / mode / f"{update:05d}.npz"


def evaluate(
    run: EvalRun,
    updates: Sequence[int],
    mode: Mode,
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    chunk: int = CHUNK_STEPS,
    should_stop: Callable[[], bool] = _never,
) -> list[EvalSummary]:
    """Write ``eval/<mode>/<update>.npz`` for each update without one.

    Returns early once ``should_stop()`` is true (checked before every checkpoint and
    after every chunk), never leaving a partial file.

    Returns:
        A summary per file written now.
    """
    if mode not in MODES:
        msg = f"mode must be one of {MODES}, got {mode!r}"
        raise ValueError(msg)
    written = []
    for u in updates:
        path = _path(run.dir, u, mode)
        if path.exists():
            continue
        if should_stop():
            break
        t0 = time.time()
        params = run.params(u)
        arrays = rollout_seeds(run, params, seeds, mode == "greedy", chunk, should_stop)
        if arrays is None:
            break
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        payload: dict[str, Any] = {"method": np.array(run.method), **arrays}
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
        tmp.replace(path)  # a file that exists is complete
        written.append(summary := _summary(run, u, mode, arrays))
        logger.info(
            "eval %s update %d: %.1fs, mean return %.2f",
            mode,
            u,
            time.time() - t0,
            summary.ret,
        )
    return written


def evaluate_spec(
    run_dir: Path,
    spec: RunSpec,
    should_stop: Callable[[], bool],
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    chunk: int = CHUNK_STEPS,
    methods: Mapping[str, Trainer] | None = None,
) -> Iterator[EvalSummary]:
    """Both modes of the final checkpoint, or of every one for ``log_spaced`` specs.

    Yields a summary per scheduled file: a new one as soon as it is written, an
    existing one read back (so a caller can log what a crashed call did not). Stops
    early once ``should_stop()`` is true.
    """
    run = EvalRun.load(run_dir, methods)
    updates = run.updates if spec.eval_mode == "log_spaced" else run.updates[-1:]
    for u in updates:
        for mode in MODES:
            if _path(run.dir, u, mode).exists():
                yield _summary(run, u, mode, load_eval(run.dir, u, mode == "greedy"))
                continue
            new = evaluate(
                run, [u], mode, seeds=seeds, chunk=chunk, should_stop=should_stop
            )
            if not new:  # stopped
                return
            yield new[0]


def load_eval(
    run_dir: Path, update: int, greedy: bool = False
) -> dict[str, np.ndarray]:
    """Arrays of ``eval/<mode>/<update>.npz``."""
    with np.load(_path(run_dir, update, "greedy" if greedy else "stochastic")) as z:
        return {k: z[k] for k in z.files}


def replay_episode(run: EvalRun, seed: int, actions: np.ndarray) -> Replay:
    """Observations, rewards, events and states of seed + the episode's actions.

    ``actions`` may keep the eval file's -1 padding after the episode. The replay
    runs at the episode length rounded up to `CHUNK_STEPS` (NOOP past the end, then
    sliced off), so one compile serves every episode of a 1,000-step bucket, and a
    short episode never replays the width of the file's longest one.
    """
    actions = np.asarray(actions)
    n = int((actions >= 0).sum())
    padded = np.zeros(-(-max(n, 1) // CHUNK_STEPS) * CHUNK_STEPS, np.int32)
    padded[:n] = actions[:n]
    with jax.default_device(device_roles().actor):
        o, r, d, ev, states = run._replay(jnp.int32(seed), jnp.asarray(padded))
        states = jax.tree.map(lambda x: x[: n + 1], states)
    o, r, d, ev = jax.tree.map(lambda x: np.asarray(x)[:n], (o, r, d, ev))
    return Replay(o, r, d, ev, states)


def stored_episode(
    run: EvalRun, update: int, seed: int, greedy: bool = False
) -> tuple[Replay, np.ndarray]:
    """The replayed episode of ``seed`` in an eval file, and its actions."""
    ev = load_eval(run.dir, update, greedy)
    i = int(np.flatnonzero(ev["seeds"] == seed)[0])
    row = ev["actions"][i]
    return replay_episode(run, seed, row), row[: ev["length"][i]]


def network_outputs(
    run: EvalRun,
    update: int,
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    greedy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Action probabilities and values of the checkpoint's network at ``obs``.

    ``actions``, ``dones`` and ``greedy`` are for recurrent and two-level policies.

    Raises:
        NotImplementedError: For a method other than PPO.
    """
    del actions, dones, greedy
    if not isinstance(run.policy, PPOPolicy):
        msg = f"network outputs of {run.method!r} are not implemented yet"
        raise NotImplementedError(msg)
    logits, value = policy_value(run.policy.net, run.params(update), jnp.asarray(obs))
    return np.asarray(jax.nn.softmax(logits)), np.asarray(value)


def game_kind(run: EvalRun) -> GameKind:
    """``classic`` or ``craftax``, as the gameplay ``rollout`` module names games."""
    return "classic" if run.classic else "craftax"


@functools.cache
def _batch_renderer(kind: GameKind, bps: int) -> Callable[[Any], jax.Array]:
    return cast(
        Callable[[Any], jax.Array], jax.jit(jax.vmap(game_for(kind).renderer(bps)))
    )


def episode_frames(
    run: EvalRun,
    update: int,
    seed: int = EVAL_SEEDS[0],
    bps: int = 16,
    greedy: bool = False,
) -> np.ndarray:
    """Frames ``(n + 1, H, W, 3)`` of a stored episode at ``bps`` px per tile.

    Rebuilt by replaying the stored actions; the last frame shows the final state.
    """
    states = stored_episode(run, update, seed, greedy)[0].states
    render = _batch_renderer(game_kind(run), bps)
    n = jax.tree.leaves(states)[0].shape[0]
    chunks = [
        np.asarray(render(jax.tree.map(lambda x, i=i: x[i : i + 128], states)))
        for i in range(0, n, 128)
    ]
    return np.concatenate(chunks)


def as_rollout(
    run: EvalRun, update: int, seed: int = EVAL_SEEDS[0], greedy: bool = False
) -> Rollout:
    """The stored episode as a gameplay ``Rollout`` (highlights, masters, cuts)."""
    ep, actions = stored_episode(run, update, seed, greedy)
    game = game_for(game_kind(run))
    feats, ach = jax.tree.map(np.asarray, jax.vmap(game.features)(ep.states))
    label = f"{run.method} after {run.env_steps(update) / 1e6:.1f}M steps"
    return Rollout(
        game=game.kind,
        policy=run.method,
        label=label,
        seed=seed,
        key_scheme="fold_in",
        god=False,
        actions=actions.astype(np.int16),
        rewards=ep.reward,
        features=feats.astype(np.float32),
        feature_names=FEATURES,
        achievements=ach.astype(bool),
        achievement_names=game.achievement_names,
    )


def main(argv: list[str] | None = None) -> None:
    """Command line: ``RUN [--updates 0,9] [--mode greedy] [--seeds 10000-10063]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--seeds", default="10000-10063", help="inclusive range lo-hi")
    p.add_argument("--updates", help="comma list; default every checkpoint")
    p.add_argument("--mode", choices=MODES, default="stochastic")
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split("-"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_runtime()
    run = EvalRun.load(a.run)
    updates = [int(u) for u in a.updates.split(",")] if a.updates else run.updates
    evaluate(run, updates, a.mode, seeds=range(lo, hi + 1))
    print(f"eval written to {run.dir / 'eval' / a.mode}")


if __name__ == "__main__":
    main()
