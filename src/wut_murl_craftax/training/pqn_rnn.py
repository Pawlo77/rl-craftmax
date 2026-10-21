# Portions adapted from purejaxql pqn_rnn_craftax.py (https://github.com/mttga/purejaxql,
# commit 47af6d7), Copyright 2024 Matteo Gallici. Licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
# agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and
# limitations under the License. Changes: the Q(lambda) targets are one reverse scan
# with the next return as the only carry; see the docstrings for the rest.
"""Recurrent PQN (LSTM) on Craftax(-Classic)-Symbolic: the PQN-RNN control and base.

Online Q(lambda) on the batch just collected, without replay or target network
(Gallici 2025; purejaxql ``pqn_rnn_craftax.py``; design: [[PQN-RNN design]]).
BatchRenorm input, Dense + LayerNorm + ReLU, last action, LSTM reset on done, Q head.
Env setup and device placement are ``training.ppo``'s; methods extend it only through
the `HookFields` of ``training.hooks``. `learn` is the update on any `Traj` with these
fields; `make_trainer` gives the registry's `Trainer`. Changes from the original: the
last-action input is zero after a reset, epsilon-greedy draws per batch, not per env
key, and the random warm-up rollout also runs the episode metrics.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding

from ..evaluation.episode_metrics import EpisodeAcc, episode_step, events_of, init_acc
from ..orchestration import shard_envs
from .batch_renorm import BatchRenorm
from .hooks import (
    HOOK_NAMES,
    MEMORY_LIMIT_KEY,
    HookFields,
    HookMethod,
    HookSpec,
    ResolvedHooks,
    Traj,
    check_method_keys,
    check_metric_names,
    resolve,
    with_method,
)
from .ppo import env_placement, make_train_env, place_on_actor
from .ppo_rnn import minibatches
from .records import LOSS_COLUMNS, achievement_names

if TYPE_CHECKING:  # the runner imports mlflow; tests of this module do not need it
    from ..orchestration.runner import ChunkMetrics

Carry = tuple[jax.Array, jax.Array]
"""LSTM ``(c, h)``, each ``(envs, hidden)``."""
NORM_TYPES = ("layer_norm", "batch_norm", "none")
"""``norm_type`` of the hidden layer: LayerNorm (repo), BatchRenorm (paper), none."""
INPUT_NORMS = ("batch_renorm", "layer_norm")
"""``input_norm`` under ``norm_input``: BatchRenorm (repo) or LayerNorm (no stats)."""
NO_ACTION = -1
"""Last action at an episode start: its one-hot input is all zeros."""
LR_END = 1e-20
"""Learning rate at the end of the linear decay, as the original."""
FILLED_LOSS_COLUMNS = ("value_loss", "grad_norm")
"""`LOSS_COLUMNS` an update fills: the TD loss and the gradient norm; others NaN."""
_LEARNER_HOOKS = {"observe", "extra_loss_fn"}
"""Hooks that read the method state on the learner, so it moves there each update;
without them ``metrics_fn`` reads it on the actor, where the rollout keeps it."""
_CLI_ONLY = {"seed"}
"""Fields the runner sets itself (the key), never in a config."""


@dataclass(frozen=True, slots=True)
class PQNRNNConfig(HookFields):
    """PQN-RNN settings: the repo YAML with the 1M rollout shape (256 x 16).

    Hooks and ``extra_steps`` come from `HookFields`.
    """

    env: str = "Craftax-Symbolic-v1"
    steps: int = 1_000_000
    envs: int = 256
    num_steps: int = 16
    """Steps per env and rollout; the last one only bootstraps."""
    minibatches: int = 4
    epochs: int = 4
    lr: float = 3e-4
    lr_linear_decay: bool = True
    """Lower the learning rate linearly per Q step to `LR_END` at the budget."""
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.5
    eps_start: float = 1.0
    eps_finish: float = 0.005
    eps_decay: float = 0.1
    """Share of the updates over which epsilon falls linearly to ``eps_finish``."""
    hidden: int = 512
    """Width of the Dense layer and of the LSTM."""
    norm_input: bool = True
    norm_type: Literal["layer_norm", "batch_norm", "none"] = "layer_norm"
    input_norm: Literal["batch_renorm", "layer_norm"] = "batch_renorm"
    add_last_action: bool = True
    rew_scale: float = 1.0
    """Factor on the extrinsic reward in the targets (``r_int`` is added unscaled)."""
    reset_ratio: int = 16
    """Envs per new world of an optimistic reset; the wrapper gets ``min(it, envs)``."""
    extra_obs_dim: int = 0
    """Network inputs after the env observation (e.g. a skill one-hot) that a caller
    appends; the trainer's own rollout and the policy feed zeros there."""
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject shapes and normalisations the network or minibatches cannot take."""
        ratio = self.reset_ratio
        checks = {
            f"{self.envs} envs do not split into {self.minibatches} minibatches": (
                self.envs % self.minibatches == 0
            ),
            f"norm_type must be one of {NORM_TYPES}: {self.norm_type!r}": (
                self.norm_type in NORM_TYPES
            ),
            f"input_norm must be one of {INPUT_NORMS}: {self.input_norm!r}": (
                self.input_norm in INPUT_NORMS
            ),
            f"num_steps must be >= 2 (the last only bootstraps): {self.num_steps}": (
                self.num_steps >= 2
            ),
            f"reset_ratio must be a positive int, got {ratio!r}": (
                isinstance(ratio, int) and not isinstance(ratio, bool) and ratio >= 1
            ),
            f"extra_obs_dim must be >= 0, got {self.extra_obs_dim}": (
                self.extra_obs_dim >= 0
            ),
        }
        if failed := [msg for msg, ok in checks.items() if not ok]:
            raise ValueError("; ".join(failed))

    @property
    def resets(self) -> str:
        """Always optimistic, as the repo: only that wrapper reports the step events."""
        return "optimistic"

    @property
    def batch(self) -> int:
        """Environment steps per update."""
        return self.envs * self.num_steps

    @property
    def n_updates(self) -> int:
        """Number of updates for the step budget."""
        return self.steps // self.batch

    @classmethod
    def from_config(cls, config: Mapping[str, Any], base: Self | None = None) -> Self:
        """Build from a run config: its keys, ``steps`` and the registry's ``method``.

        The keys override ``base`` (default: the field defaults); hooks are never keys,
        the runtime `MEMORY_LIMIT_KEY` is skipped.

        Raises:
            ValueError: On a key that is not a PQN-RNN setting.
        """
        keys = set(config) - {"method", MEMORY_LIMIT_KEY}
        names = {f.name for f in dataclasses.fields(cls)} - _CLI_ONLY - set(HOOK_NAMES)
        if unknown := sorted(keys - names):
            msg = f"unknown PQN-RNN config keys: {unknown}"
            raise ValueError(msg)
        kwargs = {k: config[k] for k in keys}
        return cls(**kwargs) if base is None else dataclasses.replace(base, **kwargs)


def q_lambda_targets(
    q_next_max: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float | jax.Array,
    lam: float | jax.Array,
) -> jax.Array:
    """Q(lambda) targets of steps 0..T-2 of a ``(T, envs)`` rollout.

    Every input is aligned on steps t = 0..T-2: ``q_next_max[t]`` is max_a Q(s_{t+1},
    a) (stop-gradient Q of the same forward pass), ``reward[t]`` and ``done[t]`` are
    step t's. ``G_t = r_t + gamma (1 - d_t) ((1 - lam) maxQ(s_{t+1}) + lam G_{t+1})``,
    seeded with ``G_{T-1} = maxQ(s_{T-1})``, so the last target bootstraps from the
    last state of the window and never from one past it (purejaxql issue #12).
    """

    def back(
        g_next: jax.Array, x: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        q, r, d = x
        g = r + gamma * (1 - d) * ((1 - lam) * q + lam * g_next)
        return g, g

    targets: Any
    _, targets = jax.lax.scan(
        back, q_next_max[-1], (q_next_max, reward, done), reverse=True
    )
    return cast(jax.Array, targets)


class ScannedLSTM(nn.Module):
    """LSTM over a ``(steps, envs)`` sequence; the carry is zeroed where ``resets``."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: Carry, x: tuple[jax.Array, jax.Array]
    ) -> tuple[Carry, jax.Array]:
        """One step: reset where a new episode starts, then the cell."""
        ins, resets = x
        carry = _reset(carry, resets)
        cell = nn.OptimizedLSTMCell(carry[1].shape[-1])
        return cast(tuple[Carry, jax.Array], cell(carry, ins))


class RNNQNetwork(nn.Module):
    """Input norm, Dense + norm + ReLU, last-action one-hot, LSTM, Dense to Q-values."""

    n_actions: int
    obs_dim: int
    """Input width: env observation plus ``extra_obs_dim``."""
    hidden: int = 512
    norm_input: bool = True
    input_norm: str = "batch_renorm"
    norm_type: str = "layer_norm"
    add_last_action: bool = True

    @nn.compact
    def __call__(
        self,
        carry: Carry,
        obs: jax.Array,
        resets: jax.Array,
        last_action: jax.Array,
        train: bool = False,
    ) -> tuple[Carry, jax.Array]:
        """``(last carry, Q-values)`` of a ``(steps, envs)`` sequence from ``carry``.

        ``resets[t]``: step t starts an episode (zero carry); ``last_action[t]``: the
        action of step t - 1, `NO_ACTION` for none. ``train`` normalises with batch
        statistics and updates the running ones (apply with ``mutable``).
        """
        # created whether used or not, as the original, so batch_stats always exist
        x = BatchRenorm(use_running_average=not train)(obs)
        if not self.norm_input:
            x = obs
        elif self.input_norm == "layer_norm":
            x = nn.LayerNorm()(obs)
        x = nn.Dense(self.hidden)(x)
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            x = BatchRenorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.add_last_action:
            x = jnp.concatenate([x, jax.nn.one_hot(last_action, self.n_actions)], -1)
        carry, y = ScannedLSTM()(carry, (x, resets))
        return carry, nn.Dense(self.n_actions)(y)

    def init_carry(self, n: int) -> Carry:
        """Zero LSTM carry for ``n`` envs."""
        return jnp.zeros((n, self.hidden)), jnp.zeros((n, self.hidden))

    def init_variables(self, key: jax.Array) -> Any:
        """Fresh params and BatchRenorm statistics (their int leaves as arrays)."""
        v = self.init(
            key,
            self.init_carry(1),
            jnp.zeros((1, 1, self.obs_dim)),
            jnp.zeros((1, 1), bool),
            jnp.zeros((1, 1), jnp.int32),
        )
        return jax.tree.map(jnp.asarray, v)


def build_network(cfg: PQNRNNConfig, n_actions: int, obs_dim: int) -> RNNQNetwork:
    """The network of ``cfg`` for an env observation of ``obs_dim`` features."""
    return RNNQNetwork(
        n_actions,
        obs_dim + cfg.extra_obs_dim,
        cfg.hidden,
        cfg.norm_input,
        cfg.input_norm,
        cfg.norm_type,
        cfg.add_last_action,
    )


def apply_network(
    net: RNNQNetwork,
    variables: Any,
    carry: Carry,
    obs: jax.Array,
    resets: jax.Array,
    last_action: jax.Array,
) -> tuple[Carry, jax.Array]:
    """``(last carry, Q-values)`` in inference mode (running statistics)."""
    out = net.apply(variables, carry, obs, resets, last_action)
    return cast(tuple[Carry, jax.Array], out)


def _reset(carry: Carry, resets: jax.Array) -> Carry:
    """Zero the carry of the envs where ``resets``."""
    c, h = (jnp.where(resets[:, None], 0.0, x) for x in carry)
    return c, h


def widen(obs: jax.Array, width: int) -> jax.Array:
    """Append zero features up to the network's input ``width`` (``extra_obs_dim``)."""
    pad = width - obs.shape[-1]
    if not pad:
        return obs
    return jnp.concatenate([obs, jnp.zeros((*obs.shape[:-1], pad), obs.dtype)], -1)


def eps_greedy(key: jax.Array, q: jax.Array, eps: jax.Array) -> jax.Array:
    """Argmax of ``q``, or a uniform action with probability ``eps``, per row."""
    k_a, k_e = jax.random.split(key)
    greedy = q.argmax(-1)
    rand = jax.random.randint(k_a, greedy.shape, 0, q.shape[-1])
    return jnp.where(jax.random.uniform(k_e, greedy.shape) < eps, rand, greedy)


def epsilon(cfg: PQNRNNConfig, update: jax.Array) -> jax.Array:
    """Epsilon of update ``update``: linear over ``eps_decay`` of the updates."""
    sched = optax.linear_schedule(
        cfg.eps_start, cfg.eps_finish, cfg.eps_decay * cfg.n_updates
    )
    return jnp.asarray(sched(update), jnp.float32)


def lr_schedule(cfg: PQNRNNConfig, extra_steps: int = 0) -> optax.Schedule:
    """Learning rate per optimizer step: linear from ``lr`` to `LR_END` (the original).

    The decay spans the budget's Q steps; the ``extra_steps`` after each update's Q
    steps run at the next update's rate, so they do not shift it.
    """
    q_steps = cfg.minibatches * cfg.epochs
    per_update = q_steps + extra_steps
    linear = optax.linear_schedule(cfg.lr, LR_END, cfg.n_updates * q_steps)

    def sched(count: jax.Array) -> jax.Array:
        done = count // per_update * q_steps + jnp.minimum(count % per_update, q_steps)
        return cast(jax.Array, linear(done))

    return sched


def optimizer(cfg: PQNRNNConfig, extra_steps: int = 0) -> optax.GradientTransformation:
    """RAdam with global-norm clipping and the optionally decaying learning rate."""
    lr = lr_schedule(cfg, extra_steps) if cfg.lr_linear_decay else cfg.lr
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.radam(lr))


class QTrainState(TrainState):
    """`TrainState` with the BatchRenorm statistics and the config `learn` runs."""

    batch_stats: Any
    cfg: PQNRNNConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def variables_of(ts: QTrainState) -> dict[str, Any]:
    """The network variables: params and BatchRenorm statistics."""
    return {"params": ts.params, "batch_stats": ts.batch_stats}


def _make_rollout(
    cfg: PQNRNNConfig, net: RNNQNetwork, env: Any, env_params: Any, hk: ResolvedHooks
) -> Callable[..., Any]:
    """Jitted epsilon-greedy rollout of ``num_steps`` x ``envs`` in inference mode."""
    rewarded = "intrinsic_reward_fn" in hk.active

    def step(
        with_traj: bool, eps: jax.Array, c: tuple[Any, ...], _: None
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        v, h, obs, last_done, last_act, st, ms, t, acc, key = c
        key, k_a, k_s = jax.random.split(key, 3)
        h = _reset(h, last_done)  # the carry this step acts from
        la = jnp.where(last_done, NO_ACTION, last_act)
        x = widen(obs, net.obs_dim)
        h2, q = apply_network(net, v, h, x[None], last_done[None], la[None])
        act = eps_greedy(k_a, q[0], eps)
        obs2, st2, r, d, info = env.step(k_s, st, act, env_params)
        rec = {"obs": x, "act": act, "r": r, "d": d, "q": q[0], "la": la}
        if rewarded:
            rec["r_int"], ms = hk.intrinsic_reward_fn(ms, st, st2, obs2, d, t)
        # the events come from the pre-reset state, so a done step keeps its unlocks
        acc2, total = episode_step(acc, r, d, events_of(info).unlocked)
        rec |= {"ach": total.ach, "ret": total.ret, "len": total.length}
        if with_traj:
            rec |= {"h": jnp.concatenate(h, -1).astype(jnp.float16)}
            rec |= {"ev": info["events"]}
        return (v, h2, obs2, d, act, st2, ms, t + cfg.envs, acc2, key), rec

    @functools.partial(jax.jit, static_argnames="with_traj")
    def rollout(
        variables: Any, carry: tuple[Any, ...], eps: jax.Array, with_traj: bool
    ) -> Any:
        h0 = _reset(carry[0], carry[2])
        body = functools.partial(step, with_traj, eps)
        carry, tr = jax.lax.scan(body, (variables, *carry), None, cfg.num_steps)
        r_int = tr["r_int"] if rewarded else jnp.zeros_like(tr["r"])
        extra = {"r_int": r_int, "q_vals": tr["q"], "last_actions": tr["la"]}
        hidden, events = (tr["h"], tr["ev"]) if with_traj else (None, None)
        traj = Traj(
            tr["obs"],
            tr["act"],
            tr["r"],
            tr["d"],
            cast(Any, hidden),
            cast(Any, events),
            extra,
        )
        ep = {k: tr[k] for k in ("d", "ach", "ret", "len")}
        return carry[1:], traj, h0, ep

    return rollout


def _learn_batch(
    cfg: PQNRNNConfig, traj: Traj, loss_mask: jax.Array | None, h0: Carry | None
) -> dict[str, Any]:
    """The `learn` inputs of a rollout: the time-major arrays and the start carry."""
    r = cfg.rew_scale * traj.rewards
    if "r_int" in traj.extra:
        r = r + traj.extra["r_int"]
    if h0 is None:  # the float16 acting carry of step 0, rounded
        c, h = jnp.split(traj.hidden[0].astype(jnp.float32), 2, -1)
        h0 = (c, h)
    d = traj.dones.astype(jnp.float32)
    # a skill method cuts targets at skill ends but resets its LSTM at episode ends
    ends = traj.extra.get("episode_dones", traj.dones).astype(bool)
    # the start carry is already reset, so step 0 never resets again
    first = jnp.zeros_like(ends[:1])
    return {
        "obs": traj.obs,
        "act": traj.actions,
        "r": r,
        "d": d,
        "reset": jnp.concatenate([first, ends[:-1]]),
        "la": traj.extra["last_actions"],
        "mask": jnp.ones_like(d) if loss_mask is None else loss_mask.astype(d.dtype),
        "h0": h0,
    }


def _make_learn(cfg: PQNRNNConfig, net: RNNQNetwork) -> Callable[..., Any]:
    """Jitted Q(lambda) update: epochs x minibatches of whole env sequences.

    Each minibatch reruns from its start carry in training mode (batch statistics,
    running ones updated); the loss is ``0.5 sum m (Q(s,a) - G)^2 / max(sum m, 1)``
    over steps 0..T-2.
    """

    def loss_fn(p: Any, bs: Any, mb: dict[str, Any]) -> tuple[jax.Array, Any]:
        (_, q), upd = net.apply(
            {"params": p, "batch_stats": bs},
            mb["h0"],
            mb["obs"],
            mb["reset"],
            mb["la"],
            train=True,
            mutable=["batch_stats"],
        )
        q_max = jax.lax.stop_gradient(q)[1:].max(-1)
        g = q_lambda_targets(q_max, mb["r"][:-1], mb["d"][:-1], cfg.gamma, cfg.lam)
        taken = jnp.take_along_axis(q, mb["act"][..., None], -1)[:-1, :, 0]
        m = mb["mask"][:-1]
        loss = 0.5 * jnp.sum(m * (taken - g) ** 2) / jnp.maximum(m.sum(), 1.0)
        return loss, upd["batch_stats"]

    grad = jax.value_and_grad(loss_fn, has_aux=True)

    def mb_step(ts: QTrainState, mb: dict[str, Any]) -> tuple[QTrainState, Any]:
        (loss, bs), g = grad(ts.params, ts.batch_stats, mb)
        ts = ts.apply_gradients(grads=g)  # type: ignore[no-untyped-call]  # flax API
        return ts.replace(batch_stats=bs), (loss, optax.tree.norm(g))

    def epoch(c: tuple[QTrainState, Any, jax.Array], _: None) -> Any:
        ts, b, key = c
        key, k = jax.random.split(key)
        ts, aux = jax.lax.scan(mb_step, ts, minibatches(b, k, cfg))
        return (ts, b, key), aux

    @jax.jit
    def learn(
        ts: QTrainState,
        traj: Traj,
        key: jax.Array,
        loss_mask: jax.Array | None,
        h0: Carry | None,
    ) -> tuple[QTrainState, dict[str, jax.Array]]:
        b = _learn_batch(cfg, traj, loss_mask, h0)
        (ts, _, _), (loss, gn) = jax.lax.scan(epoch, (ts, b, key), None, cfg.epochs)
        return ts, {"td_loss": loss.mean(), "grad_norm": gn.mean()}

    return learn


def _make_finish(hk: ResolvedHooks) -> Callable[..., Any]:
    """Jitted end of an update: ``extra_steps`` extra-loss steps, then ``metrics_fn``.

    Extra-loss metrics are averaged over the steps; ``metrics_fn`` runs here only if
    the state is on the learner.
    """
    on_learner = bool(_LEARNER_HOOKS & set(hk.active))
    extra_grad = jax.value_and_grad(hk.extra_loss_fn, has_aux=True)

    def extra_step(c: tuple[QTrainState, Any, jax.Array], _: None) -> Any:
        ts, ms, key = c
        key, k = jax.random.split(key)
        (_, (ms, m)), g = extra_grad(ts.params, ms, k)
        ts = ts.apply_gradients(grads=g)  # type: ignore[no-untyped-call]  # flax API
        return (ts, ms, key), m

    @jax.jit
    def finish(ts: QTrainState, ms: Any, key: jax.Array) -> Any:
        method: dict[str, jax.Array] = {}
        if "extra_loss_fn" in hk.active:
            (ts, ms, _), em = jax.lax.scan(
                extra_step, (ts, ms, key), None, hk.extra_steps
            )
            check_metric_names(em)
            method |= {k: v.mean() for k, v in em.items()}
        if on_learner:
            method |= hk.metrics_fn(ms)
        return ts, ms, method

    return finish


@dataclass(frozen=True, slots=True)
class _Setup:
    """Everything built once per config: network, env, optimizer, jitted steps."""

    net: RNNQNetwork
    env: Any
    env_params: Any
    tx: optax.GradientTransformation
    hooks: ResolvedHooks
    rollout: Callable[..., Any]
    learn: Callable[..., Any]
    observe: Callable[..., Any]
    finish: Callable[..., Any]
    metrics: Callable[..., Any]
    """Jitted ``metrics_fn`` on the actor, for a state that never leaves it."""
    learner: jax.Device
    actor: jax.Device
    mesh: Mesh | None
    """CPU ``envs`` mesh the env batch is split over, if there are several CPUs."""
    shared_at: jax.Device | NamedSharding
    """Placement of variables and keys the rollout reads: the actor, or replicated."""

    @property
    def with_traj(self) -> bool:
        """Whether the rollout records hidden and events (``observe`` reads them)."""
        return "observe" in self.hooks.active

    @property
    def learner_state(self) -> bool:
        """Whether the method state moves to the learner for the update."""
        return bool(_LEARNER_HOOKS & set(self.hooks.active))

    @property
    def actor_metrics(self) -> bool:
        """Whether ``metrics_fn`` runs on the actor (the state stays there)."""
        return "metrics_fn" in self.hooks.active and not self.learner_state

    @property
    def finishes(self) -> bool:
        """Whether an update ends with extra-loss steps or learner-side metrics."""
        return "extra_loss_fn" in self.hooks.active or (
            "metrics_fn" in self.hooks.active and self.learner_state
        )


@functools.cache
def _setup(cfg: PQNRNNConfig) -> _Setup:
    """Cached per config (hooks included) so every chunk reuses the compiled steps.

    Raises:
        ValueError: If the env batch does not split over the CPU devices, or a method
            sets ``after_extra_fn``.
    """
    hk = resolve(cfg)
    if "after_extra_fn" in hk.active:
        msg = "after_extra_fn is a PPO-RNN hook: PQN-RNN would never call it"
        raise ValueError(msg)
    roles, mesh, shared_at = env_placement(cfg.envs)
    wrap = hk.env_wrapper_fn if "env_wrapper_fn" in hk.active else None
    env, env_params = make_train_env(cfg, wrap)
    obs_dim = env.observation_space(env_params).shape[0]
    net = build_network(cfg, env.action_space(env_params).n, obs_dim)
    return _Setup(
        net,
        env,
        env_params,
        optimizer(cfg, hk.extra_steps),
        hk,
        _make_rollout(cfg, net, env, env_params, hk),
        _make_learn(cfg, net),
        jax.jit(hk.observe),
        _make_finish(hk),
        jax.jit(hk.metrics_fn),
        roles.learner,
        roles.actor,
        mesh,
        shared_at,
    )


def learn(
    train_state: QTrainState,
    traj: Traj,
    key: jax.Array,
    loss_mask: jax.Array | None = None,
    h0: Carry | None = None,
) -> tuple[QTrainState, dict[str, jax.Array]]:
    """One PQN-RNN update on a rollout, for `train_chunk` or another rollout.

    Reads ``traj.obs`` (network-wide), ``actions``, ``rewards``, ``dones`` (no target
    crosses one), ``extra["last_actions"]`` and, if present, ``extra["r_int"]`` and
    ``extra["episode_dones"]`` (LSTM resets; default ``dones``); ``loss_mask``
    ``(steps, envs)`` drops transitions from the loss. ``h0`` is the float32 start
    carry after step 0's reset (default: ``traj.hidden[0]``, rounded to float16).
    Returns the train state and the mean ``td_loss`` and ``grad_norm`` of its steps.

    Raises:
        ValueError: If the rollout, ``loss_mask`` or ``h0`` is not ``(num_steps,
            envs)`` / ``(envs, hidden)`` of the config, or ``h0`` is missing while
            ``traj.hidden`` is None.
    """
    s = _setup(train_state.cfg)
    _check_rollout(train_state.cfg, traj, loss_mask, h0)
    return cast(
        tuple[QTrainState, dict[str, jax.Array]],
        s.learn(train_state, traj, key, loss_mask, h0),
    )


def _check_rollout(
    cfg: PQNRNNConfig, traj: Traj, loss_mask: jax.Array | None, h0: Carry | None
) -> None:
    """Refuse shapes `learn` would silently mishandle (its env shuffle is per config).

    Raises:
        ValueError: On any mismatch, naming the shapes.
    """
    want = (cfg.num_steps, cfg.envs)
    arrays = {
        "traj.obs": traj.obs,
        "traj.actions": traj.actions,
        "traj.rewards": traj.rewards,
        "traj.dones": traj.dones,
        'traj.extra["last_actions"]': traj.extra["last_actions"],
    }
    if loss_mask is not None:
        arrays["loss_mask"] = loss_mask
    errors = [
        f"{name} has steps x envs {tuple(x.shape[:2])}, the config needs {want}"
        for name, x in arrays.items()
        if tuple(x.shape[:2]) != want
    ]
    if h0 is None and traj.hidden is None:
        errors.append("h0 is required when traj.hidden is None")
    carry = (cfg.envs, cfg.hidden)
    errors += [
        f"h0 leaves must be (envs, hidden) = {carry}, got {tuple(x.shape)}"
        for x in jax.tree.leaves(h0)
        if tuple(x.shape) != carry
    ]
    if errors:
        raise ValueError("; ".join(errors))


def _on_actor(tree: Any, s: _Setup, envs: int) -> Any:
    """`place_on_actor` with this setup's devices."""
    return place_on_actor(tree, envs, s.actor, s.mesh, s.shared_at)


class PQNRNNState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Full PQN-RNN training state: one pytree that `save_state` round-trips."""

    train_state: QTrainState
    env_state: Any
    obs: jax.Array
    hidden: Carry
    """LSTM carry of each env, float32, before the reset of the next step."""
    last_done: jax.Array
    """The previous step ended the env's episode."""
    last_action: jax.Array
    """Action of the previous step, int32 (`NO_ACTION` before the first)."""
    episode: EpisodeAcc
    """Return, length and achievements of each env's running episode."""
    env_steps: jax.Array
    """Env steps of the run so far, int32 (the hooks' ``t``)."""
    updates: jax.Array
    """Updates done, int32."""
    eps: jax.Array
    """Epsilon of the next rollout, float32 (`epsilon` of ``updates``)."""
    method_state: Any
    """The hooks' state (`HookFields.init_state_fn`); ``{}`` for the control."""
    key: jax.Array
    """Learner key (minibatch shuffles, extra steps)."""
    key_act: jax.Array
    """Actor key (actions, env steps)."""
    cfg: PQNRNNConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def init_state(
    cfg: PQNRNNConfig, key: jax.Array, memory_limit_bytes: int | None = None
) -> PQNRNNState:
    """Fresh network on the learner, envs on the actor, then the random warm-up.

    ``memory_limit_bytes`` goes to the method in `HookSpec`.
    """
    s = _setup(cfg)
    key, k_init, k_reset, k_method = jax.random.split(key, 4)
    with jax.default_device(s.learner):
        v = s.net.init_variables(k_init)
        ts = QTrainState.create(  # type: ignore[no-untyped-call]  # flax API
            apply_fn=s.net.apply,
            params=v["params"],
            tx=s.tx,
            batch_stats=v["batch_stats"],
            cfg=cfg,
        )
        # an array from the start, as after an update (see `ppo.init_state`)
        ts = ts.replace(step=jnp.zeros((), jnp.int32))
    n_ach = len(achievement_names(cfg.env))
    dims = (cfg.envs, s.net.obs_dim, s.net.n_actions, n_ach, cfg.steps)
    spec = HookSpec(*dims, memory_limit_bytes)
    with jax.default_device(s.actor):
        obs, st = s.env.reset(k_reset, s.env_params)
        per_env = (
            obs,
            st,
            s.net.init_carry(cfg.envs),
            jnp.zeros(cfg.envs, bool),
            jnp.full(cfg.envs, NO_ACTION, jnp.int32),
            init_acc(cfg.envs, n_ach),
        )
        method_state = s.hooks.init_state_fn(k_method, spec)
    if s.mesh is not None:
        per_env = shard_envs(per_env, s.mesh)
    obs, st, hidden, last_done, last_action, episode = per_env
    zero = jax.device_put(jnp.zeros((), jnp.int32), s.shared_at)
    state = PQNRNNState(
        train_state=cast(QTrainState, ts),
        env_state=st,
        obs=obs,
        hidden=hidden,
        last_done=last_done,
        last_action=last_action,
        episode=episode,
        env_steps=zero,
        updates=zero,
        eps=jax.device_put(jnp.ones((), jnp.float32), s.shared_at),
        method_state=_on_actor(method_state, s, cfg.envs),
        key=jax.device_put(key, s.learner),
        key_act=jax.device_put(jax.random.fold_in(key, 1), s.shared_at),
        cfg=cfg,
    )
    return _warm_up(state, s)


def _warm_up(state: PQNRNNState, s: _Setup) -> PQNRNNState:
    """The original's random rollout before update 0 (``MEMORY_WINDOW`` 0).

    ``num_steps`` steps at epsilon 1 with the initial network; its env, LSTM and
    episode states start the first rollout, its transitions, env steps and method
    state are dropped. Only `init_state` runs it, so a resumed run never does.
    """
    warm, _, _, _ = collect(state)
    eps = jax.device_put(epsilon(state.cfg, state.updates), s.shared_at)
    return warm.replace(
        env_steps=state.env_steps, method_state=state.method_state, eps=eps
    )


def collect(
    state: PQNRNNState, with_traj: bool = False
) -> tuple[PQNRNNState, Traj, Carry, dict[str, jax.Array]]:
    """One rollout on the actor at ``state.eps``: state, `Traj`, start carry, episodes.

    ``traj.hidden`` and ``traj.events`` are recorded only ``with_traj`` (else None).
    The start carry is float32, after step 0's reset. Episodes: d, ach, ret, len.
    """
    s = _setup(state.cfg)
    variables = jax.device_put(variables_of(state.train_state), s.shared_at)
    carry = (
        state.hidden,
        state.obs,
        state.last_done,
        state.last_action,
        state.env_state,
        _on_actor(state.method_state, s, state.cfg.envs),
        state.env_steps,
        state.episode,
        state.key_act,
    )
    carry, traj, h0, ep = s.rollout(variables, carry, state.eps, with_traj=with_traj)
    hidden, obs, last_done, last_action, st, ms, t, episode, key_act = carry
    new = state.replace(
        env_state=st,
        obs=obs,
        hidden=hidden,
        last_done=last_done,
        last_action=last_action,
        episode=episode,
        env_steps=t,
        method_state=ms,
        key_act=key_act,
    )
    return new, traj, h0, ep


def _update(
    state: PQNRNNState, s: _Setup, traj: Traj, h0: Carry
) -> tuple[PQNRNNState, jax.Array, dict[str, jax.Array]]:
    """Learner side of one update: ``observe``, `learn`, extra steps, method metrics.

    The losses come in ``records.LOSS_COLUMNS`` order: the TD loss as
    ``value_loss``, ``grad_norm`` (before clipping); the PPO-only columns are NaN,
    and `PQNRNNTrainer.loss_columns` keeps them out of logging and abort checks.
    """
    key, k_learn, k_extra = jax.random.split(state.key, 3)
    ms = state.method_state
    if s.learner_state:
        ms = jax.device_put(ms, s.learner)
    if "observe" in s.hooks.active:
        ms = s.observe(ms, traj)
    ts, m = s.learn(state.train_state, traj, k_learn, None, h0)
    method: dict[str, jax.Array] = {}
    if s.finishes:
        ts, ms, method = s.finish(ts, ms, k_extra)
    if s.actor_metrics:
        method |= s.metrics(ms)
    updates = state.updates + 1
    eps = jax.device_put(epsilon(state.cfg, updates), s.shared_at)
    state = state.replace(
        train_state=ts, method_state=ms, key=key, updates=updates, eps=eps
    )
    cols = dict(zip(FILLED_LOSS_COLUMNS, (m["td_loss"], m["grad_norm"]), strict=True))
    nan = jnp.full((), jnp.nan, jnp.float32)
    return state, jnp.stack([cols.get(c, nan) for c in LOSS_COLUMNS]), method


def train_chunk(
    state: PQNRNNState, n_updates: int
) -> tuple[PQNRNNState, "ChunkMetrics"]:
    """Run ``n_updates`` rollout-and-update steps; metrics stacked per update.

    Besides the shared metrics, every method metric name maps to its per-update
    values.
    """
    s = _setup(state.cfg)
    put = functools.partial(jax.device_put, device=s.learner)
    losses, eps, method = [], [], []
    for _ in range(n_updates):
        state, traj, h0, ep = collect(state, s.with_traj)
        state, loss, m = _update(state, s, put(traj), put(h0))
        losses.append(loss)
        eps.append(ep)
        method.append(m)
    metrics = {
        "losses": jnp.stack(losses),
        "done": jnp.stack([e["d"] for e in eps]).astype(bool),
        "ach": jnp.stack([e["ach"] for e in eps]),
        "ret": jnp.stack([e["ret"] for e in eps]),
        "length": jnp.stack([e["len"] for e in eps]),
    }
    metrics |= {k: jnp.stack([m[k] for m in method]) for k in method[0]}
    return state, cast("ChunkMetrics", metrics)


PolicyCarry = tuple[Carry, jax.Array]
"""LSTM carry and last action of each env."""


@dataclass(frozen=True, slots=True)
class PQNRNNPolicy:
    """Recurrent epsilon-greedy actor; params are `PQNRNNTrainer.params`."""

    net: RNNQNetwork

    def init_carry(self, n: int) -> PolicyCarry:
        """Zero LSTM carry and no last action (a zero one-hot) for ``n`` envs."""
        return self.net.init_carry(n), jnp.full(n, NO_ACTION, jnp.int32)

    def step(
        self,
        params: Mapping[str, Any],
        carry: PolicyCarry,
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, PolicyCarry]:
        """Argmax actions when ``greedy``, else epsilon-greedy at ``params["eps"]``.

        ``params["variables"]`` are the network's. Evaluation episodes end at their
        done, so no reset input is needed.
        """
        del reward, done  # the previous step's: not an input of this network
        lstm, last = carry
        resets = jnp.zeros((1, obs.shape[0]), bool)
        x = widen(obs, self.net.obs_dim)[None]
        lstm, q = apply_network(
            self.net, params["variables"], lstm, x, resets, last[None]
        )
        explore = eps_greedy(key, q[0], params["eps"])
        act = jnp.where(greedy, q[0].argmax(-1), explore)
        return act, (lstm, act)


@dataclass(frozen=True, slots=True)
class PQNRNNTrainer:
    """A PQN-RNN registry entry (`orchestration.runner.Trainer`).

    ``cfg`` is the factory config; the run config's base keys override its fields,
    then ``method`` builds the hooks from its own keys and that config (`with_method`).
    """

    cfg: PQNRNNConfig
    method: HookMethod[Any] | None = None
    loss_columns: ClassVar[tuple[str, ...]] = FILLED_LOSS_COLUMNS
    """The runner logs and abort-checks only these (the others are NaN)."""

    def __post_init__(self) -> None:
        """Refuse method keys that shadow PQNRNNConfig keys."""
        check_method_keys(self.method, PQNRNNConfig)

    def config(self, config: Mapping[str, Any]) -> PQNRNNConfig:
        """The factory config with the base keys, then the method's hooks, on top."""
        return with_method(self.cfg, self.method, config, PQNRNNConfig.from_config)

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Rollout batch: ``envs * num_steps``."""
        return self.config(config).batch

    def init(self, config: Mapping[str, Any], key: jax.Array) -> PQNRNNState:
        """Validated config to the initial state (after the warm-up rollout).

        Raises:
            ValueError: On an unknown key, or no ``steps`` (the epsilon and
                learning-rate schedules span the budget).
        """
        if "steps" not in config:
            msg = "PQN-RNN schedules span the budget: pass `steps` in the config"
            raise ValueError(msg)
        limit = config.get(MEMORY_LIMIT_KEY)
        return init_state(self.config(config), key, limit)

    def train_chunk(
        self, state: PQNRNNState, n_updates: int
    ) -> tuple[PQNRNNState, "ChunkMetrics"]:
        """See `train_chunk`."""
        return train_chunk(state, n_updates)

    def params(self, state: PQNRNNState) -> dict[str, Any]:
        """The ``ckpt/`` payload: network variables and the next rollout's epsilon."""
        return {"variables": variables_of(state.train_state), "eps": state.eps}

    def policy(self, state: PQNRNNState) -> PQNRNNPolicy:
        """The acting policy of this config's network (reads no state values)."""
        return PQNRNNPolicy(_setup(state.cfg).net)

    def env_wrapper(
        self, state: PQNRNNState
    ) -> Callable[[Any, Any], tuple[Any, Any]] | None:
        """The config's ``env_wrapper_fn``; evaluation wraps its env the same way."""
        return state.cfg.env_wrapper_fn

    def checkpoint_artifacts(
        self, state: PQNRNNState, run_dir: Path, update: int
    ) -> None:
        """The method's per-checkpoint files (`HookMethod.artifacts`), if it has any."""
        if self.method is not None and self.method.artifacts is not None:
            self.method.artifacts(run_dir, update, state.method_state)


def make_trainer(
    cfg: PQNRNNConfig, method: HookMethod[Any] | None = None
) -> PQNRNNTrainer:
    """The trainer of a method built on PQN-RNN: ``method`` builds its hooks per run.

    Raises:
        ValueError: If a method key shadows a `PQNRNNConfig` key.
    """
    return PQNRNNTrainer(cfg, method)
