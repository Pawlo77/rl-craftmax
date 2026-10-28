"""Fixed-seed evaluation of checkpoints, stored as replayable episodes.

`evaluate` plays `EVAL_SEEDS` with a registry method's ``Policy`` (stochastic or greedy)
(each step handed the previous step's reward and done, zeros at the start)
in jitted chunks of `CHUNK_STEPS` until every episode has ended and writes
``eval/<mode>/<update>.npz``: ``seeds``, ``actions`` (-1 after the end), ``length``,
``method``, ``events/<StepEvents field>`` per step and the policy's ``extras/<name>``.
Observations, rewards, states and frames are rebuilt by replaying seed + actions
(`replay_episode`); `evaluate_spec` is the runner's schedule; `network_outputs` and
`episode_outputs` re-run any method's network over an episode. Keys: [[Recording
training data]]. Usage: ``python -m wut_murl_craftax.visualisation.eval_rollouts RUN``
"""

import argparse
import functools
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, NamedTuple, assert_never, cast

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
from ..training import ppo_rnn, pqn_rnn
from ..training.checkpoints import (
    CheckpointEntry,
    load_index,
    load_params,
    pruned_updates,
)
from ..training.interface import Policy, run_trainer
from ..training.ppo import PPOPolicy, make_env, policy_value
from ..training.records import achievement_weights, is_classic
from ..training.skills import NUM_SKILLS, PRIMITIVE, step_limit
from ..training.smdp import SMDPPolicy
from ..training.wrappers import unwrap_state
from .rollout import FEATURES, GameKind, Rollout, game_for, keys

logger = logging.getLogger(__name__)

type Mode = Literal["stochastic", "greedy"]
type Family = Literal["ppo", "ppo-rnn", "pqn-rnn", "skills"]
EVAL_SEEDS: tuple[int, ...] = tuple(range(10000, 10064))
"""The 64 eval seeds of every checkpoint, method and benchmark."""
MODES: tuple[Mode, ...] = ("stochastic", "greedy")
"""Sampled actions (key ``fold_in(k_act, t)``) and the policy's argmax."""
CHUNK_STEPS = 1000
"""Steps per jitted scan; scans repeat until every seed's episode has ended."""
Q_FAMILIES: frozenset[Family] = frozenset({"pqn-rnn", "skills"})
"""Value-based families: value max Q, probabilities epsilon-greedy."""
_FAMILIES: tuple[tuple[type, Family], ...] = (
    (PPOPolicy, "ppo"),
    (ppo_rnn.PPORNNPolicy, "ppo-rnn"),
    (pqn_rnn.PQNRNNPolicy, "pqn-rnn"),
    (SMDPPolicy, "skills"),
)
"""Policy class -> network family; hook methods and combo act through their base's
policy, dyna-q through the skills method's."""


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


class _Inputs(NamedTuple):
    """One episode's network inputs, padded to a `CHUNK_STEPS` multiple."""

    obs: jax.Array
    actions: jax.Array
    """The stored actions: the next step's last-action input, primitive one-hots."""
    resets: jax.Array
    """Step t starts an episode: zero carry, no last action."""
    skill: jax.Array
    steps: jax.Array
    eps: jax.Array
    """Exploration of value-based methods, 0 when greedy."""


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

    @functools.cached_property
    def _last_params(self) -> dict[int, Any]:
        """The last checkpoint `network_outputs` loaded, at most one entry."""
        return {}

    def _cached_params(self, update: int) -> Any:
        """`params`, kept for the last checkpoint only.

        Bulk callers stay on one checkpoint, and skill tables hold up to 2^20 rows.
        """
        memo = self._last_params
        if update not in memo:
            memo.clear()
            memo[update] = self.params(update)
        return memo[update]

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

    @functools.cached_property
    def _network(self) -> Callable[[Any, _Inputs], tuple[jax.Array, jax.Array]]:
        """Jitted ``(params, inputs) -> (probs, value)`` of the policy's network."""
        fn = jax.jit(functools.partial(_outputs, self.policy))
        return cast(Callable[[Any, _Inputs], tuple[jax.Array, jax.Array]], fn)


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
    skip_pruned: bool = False,
) -> list[EvalSummary]:
    """Write ``eval/<mode>/<update>.npz`` for each update without one.

    Returns early once ``should_stop()`` is true (checked before every checkpoint and
    after every chunk), never leaving a partial file. A pruned checkpoint without an
    eval file (`pruned_updates`) raises `PrunedCheckpointError` (its params are
    gone), or is skipped with a warning when ``skip_pruned``.

    Returns:
        A summary per file written now.
    """
    if mode not in MODES:
        msg = f"mode must be one of {MODES}, got {mode!r}"
        raise ValueError(msg)
    written, pruned = [], pruned_updates(run.dir)
    for u in updates:
        path = _path(run.dir, u, mode)
        if path.exists():
            continue
        if u in pruned and skip_pruned:
            logger.warning("%s: checkpoint %d was pruned, not evaluated", run.dir, u)
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
    skip_pruned: bool = False,
) -> Iterator[EvalSummary]:
    """Both modes of the final checkpoint, or of every one for ``log_spaced`` specs.

    Yields a summary per scheduled file: a new one as soon as it is written, an
    existing one read back (so a caller can log what a crashed call did not). Stops
    early once ``should_stop()`` is true. A pruned checkpoint without its file
    raises, or is skipped with ``skip_pruned`` (`evaluate`).
    """
    run = EvalRun.load(run_dir, methods)
    updates = run.updates if spec.eval_mode == "log_spaced" else run.updates[-1:]
    pruned = pruned_updates(run.dir)
    for u in updates:
        for mode in MODES:
            if _path(run.dir, u, mode).exists():
                yield _summary(run, u, mode, load_eval(run.dir, u, mode == "greedy"))
                continue
            if u in pruned and skip_pruned:
                logger.warning(
                    "%s: checkpoint %d was pruned, not evaluated", run.dir, u
                )
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


def _seed_row(ev: Mapping[str, np.ndarray], seed: int) -> int:
    """Row of ``seed`` in an eval file.

    Raises:
        ValueError: If the file does not hold the seed.
    """
    rows = np.flatnonzero(ev["seeds"] == seed)
    if not len(rows):
        msg = f"seed {seed} is not in the eval file (seeds {ev['seeds'].tolist()})"
        raise ValueError(msg)
    return int(rows[0])


def stored_episode(
    run: EvalRun, update: int, seed: int, greedy: bool = False
) -> tuple[Replay, np.ndarray]:
    """The replayed episode of ``seed`` in an eval file, and its actions."""
    ev = load_eval(run.dir, update, greedy)
    i = _seed_row(ev, seed)
    row = ev["actions"][i]
    return replay_episode(run, seed, row), row[: ev["length"][i]]


def network_family(policy: Policy) -> Family:
    """The network family a policy re-runs through (`network_outputs`).

    Raises:
        TypeError: For a policy class of no family.
    """
    for cls, family in _FAMILIES:
        if isinstance(policy, cls):
            return family
    msg = f"no network family for policy {type(policy).__name__}"
    raise TypeError(msg)


def _q_outputs(
    policy: pqn_rnn.PQNRNNPolicy | SMDPPolicy, variables: Any, x: _Inputs
) -> tuple[jax.Array, jax.Array]:
    """Epsilon-greedy probabilities and max Q of the PQN-RNN or skill network."""
    net = policy.net
    if isinstance(policy, SMDPPolicy):  # as `smdp._skill_actions`
        frac = x.steps / step_limit(x.skill, policy.cfg.timeout_mult)
        extra = jnp.concatenate(
            [jax.nn.one_hot(x.skill, NUM_SKILLS), frac[:, None]], -1
        )
        obs = jnp.concatenate([x.obs, extra.astype(x.obs.dtype)], -1)
    else:
        obs = pqn_rnn.widen(x.obs, net.obs_dim)
    last = jnp.where(x.resets, pqn_rnn.NO_ACTION, jnp.roll(x.actions, 1))
    h0, seq = net.init_carry(1), (obs[:, None], x.resets[:, None], last[:, None])
    q = pqn_rnn.apply_network(net, variables, h0, *seq)[1][:, 0]
    n = q.shape[-1]
    probs = x.eps / n + (1 - x.eps) * jax.nn.one_hot(q.argmax(-1), n)
    if isinstance(policy, SMDPPolicy):  # a primitive option acts with a fixed action
        prim = jnp.asarray(PRIMITIVE)[x.skill][:, None]
        probs = jnp.where(prim, jax.nn.one_hot(x.actions, n), probs)
    return probs, q.max(-1)


def _outputs(policy: Policy, params: Any, x: _Inputs) -> tuple[jax.Array, jax.Array]:
    """Action probabilities and values of one episode's padded inputs."""
    family = network_family(policy)
    match family:
        case "ppo":
            logits, value = policy_value(cast(PPOPolicy, policy).net, params, x.obs)
            return jax.nn.softmax(logits), value
        case "ppo-rnn":
            net = cast(ppo_rnn.PPORNNPolicy, policy).net
            h0 = jnp.zeros((1, net.rnn_width))
            seq = (x.obs[:, None], x.resets[:, None])
            _, logits, value = ppo_rnn.apply_network(net, params, h0, *seq)
            return jax.nn.softmax(logits[:, 0]), value[:, 0]
        case "pqn-rnn" | "skills":
            return _q_outputs(cast(Any, policy), params, x)
        case _ as never:
            assert_never(never)


def network_outputs(
    run: EvalRun,
    update: int,
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    greedy: bool,
    skill: np.ndarray | None = None,
    skill_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Action probabilities ``(T, n_actions)`` and values ``(T,)`` at ``obs``, float32.

    The checkpoint's network re-runs over the episode's observations with its
    recurrent carry threaded from zeros (and reset after a ``dones`` step). PQN-RNN
    and the skill network get the previous entry of ``actions`` as last action; the
    skill network also the running ``skill`` and its ``skill_steps`` (the eval file's
    ``extras/skill`` and ``extras/steps_in_skill``). The PPO family returns the policy
    softmax and the critic value; value-based methods (`Q_FAMILIES`) max Q and the
    epsilon-greedy distribution at ``params["eps"]`` (one-hot when ``greedy``, and
    one-hot on the stored action at primitive-skill steps).

    The checkpoint is read once while calls stay on one ``update``.

    Raises:
        ValueError: If a skill method gets no ``skill`` or ``skill_steps``, or a
            per-step input is not as long as ``obs``.
    """
    family, n = network_family(run.policy), len(obs)
    if family == "skills" and (skill is None or skill_steps is None):
        msg = f"{run.method} needs the running skill and skill_steps of every step"
        raise ValueError(msg)
    per_step = {"actions": actions, "dones": dones, "skill": skill}
    per_step["skill_steps"] = skill_steps
    if bad := [
        f"{k} has {len(v)} steps"
        for k, v in per_step.items()
        if v is not None and len(v) != n
    ]:
        msg = f"{', '.join(bad)}, obs {n}"
        raise ValueError(msg)
    params = run._cached_params(update)
    q = family in Q_FAMILIES
    width = -(-max(n, 1) // CHUNK_STEPS) * CHUNK_STEPS  # one compile per bucket

    def pad(a: np.ndarray | None, dtype: Any) -> jax.Array:
        a = np.zeros(n, dtype) if a is None else np.asarray(a, dtype)[:n]
        return jnp.asarray(np.pad(a, [(0, width - n)] + [(0, 0)] * (a.ndim - 1)))

    resets = np.r_[True, np.asarray(dones, bool)[: n - 1]]
    eps = 0.0 if greedy or not q else params["eps"]
    with jax.default_device(device_roles().actor):  # where the eval rollouts ran
        x = _Inputs(
            pad(obs, np.asarray(obs).dtype),
            pad(actions, np.int32),
            pad(resets, np.bool_),
            pad(skill, np.int32),
            pad(skill_steps, np.int32),
            jnp.asarray(eps, jnp.float32),
        )
        probs, value = run._network(params["variables"] if q else params, x)
    return np.asarray(probs[:n], np.float32), np.asarray(value[:n], np.float32)


def episode_outputs(
    run: EvalRun, update: int, mode: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`network_outputs` of the stored episode of ``seed`` in ``eval/<mode>/<update>``.

    The episode is replayed from seed + actions; skill methods pass the file's
    ``extras/skill`` and ``extras/steps_in_skill``.

    Raises:
        ValueError: On a mode outside `MODES` or a seed the file does not hold.
    """
    if mode not in MODES:
        msg = f"mode must be one of {MODES}, got {mode!r}"
        raise ValueError(msg)
    greedy = mode == "greedy"
    ev = load_eval(run.dir, update, greedy)
    i = _seed_row(ev, seed)
    n = int(ev["length"][i])
    actions = ev["actions"][i, :n]
    ep = replay_episode(run, seed, actions)
    skill, steps = (
        ev[k][i, :n] if k in ev else None
        for k in ("extras/skill", "extras/steps_in_skill")
    )
    return network_outputs(run, update, ep.obs, actions, ep.done, greedy, skill, steps)


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
    p.add_argument("--skip-pruned", action="store_true", help="skip pruned checkpoints")
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split("-"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_runtime()
    run = EvalRun.load(a.run)
    updates = [int(u) for u in a.updates.split(",")] if a.updates else run.updates
    seeds = range(lo, hi + 1)
    evaluate(run, updates, a.mode, seeds=seeds, skip_pruned=a.skip_pruned)
    print(f"eval written to {run.dir / 'eval' / a.mode}")


if __name__ == "__main__":
    main()
