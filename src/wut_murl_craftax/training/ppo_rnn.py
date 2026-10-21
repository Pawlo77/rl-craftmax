# Portions adapted from Craftax_Baselines ppo_rnn.py
# (https://github.com/MichaelTMatthews/Craftax_Baselines, commit 7ce36fa).
# Copyright (c) 2024 Michael Matthews. MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions: The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS
# IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""Recurrent PPO (GRU) on Craftax(-Classic)-Symbolic: the PPO-RNN control and base.

Adapted from Craftax_Baselines ``ppo_rnn.py``: Dense, GRU reset on done, two ReLU
heads; minibatches of whole env sequences rerun from the stored start hidden. Losses,
GAE, optimizer, env setup and device placement are ``training.ppo``'s. Methods extend
it only through the `HookFields` of ``training.hooks``; unset hooks are not traced, so
the control's graph stays as it is. `make_trainer` gives the registry's `Trainer`.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..evaluation.episode_metrics import EpisodeAcc, episode_step, events_of, init_acc
from ..orchestration import shard_envs
from .hooks import (
    HookFields,
    HookSpec,
    ResolvedHooks,
    Traj,
    check_metric_names,
    resolve,
)
from .ppo import (
    Batch,
    PPOConfig,
    env_placement,
    gae,
    make_train_env,
    optimizer,
    ppo_terms,
)
from .records import achievement_names

if TYPE_CHECKING:  # the runner imports mlflow; tests of this module do not need it
    from ..orchestration.runner import ChunkMetrics

ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
}
"""``activation`` values: ReLU as the original code, tanh as the paper's text."""
_LEARNER_HOOKS = {"observe", "extra_loss_fn", "metrics_fn"}
"""Hooks that read the method state on the learner, so it moves there each update."""


@dataclass(frozen=True, slots=True)
class PPORNNConfig(PPOConfig, HookFields):
    """`PPOConfig` plus the GRU; hooks and ``extra_steps`` come from `HookFields`."""

    rnn_width: int = 512
    activation: Literal["relu", "tanh"] = "relu"

    def __post_init__(self) -> None:
        """Also require optimistic resets, whole-env minibatches, a known activation."""
        PPOConfig.__post_init__(self)
        if self.resets != "optimistic":  # only that wrapper reports the step events
            msg = f"PPO-RNN needs optimistic resets, got {self.resets!r}"
            raise ValueError(msg)
        if self.envs % self.minibatches:
            msg = f"{self.envs} envs do not split into {self.minibatches} minibatches"
            raise ValueError(msg)
        if self.activation not in ACTIVATIONS:
            msg = f"activation must be one of {list(ACTIVATIONS)}: {self.activation!r}"
            raise ValueError(msg)


class ScannedGRU(nn.Module):
    """GRU over a ``(steps, envs)`` sequence; the carry is zeroed where ``resets``."""

    width: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, h: jax.Array, x: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """One step: reset where the previous step ended an episode, then the cell."""
        ins, resets = x
        h = jnp.where(resets[:, None], jnp.zeros_like(h), h)
        return cast(tuple[jax.Array, jax.Array], nn.GRUCell(self.width)(h, ins))


class ActorCriticRNN(nn.Module):
    """Dense embedding, GRU, then actor and critic heads of 2 hidden layers each."""

    n_actions: int
    obs_dim: int
    width: int
    rnn_width: int
    activation: str

    @nn.compact
    def __call__(
        self, h: jax.Array, obs: jax.Array, resets: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """``(last hidden, logits, values)`` of a ``(steps, envs)`` sequence from ``h``.

        ``resets[t]``: the episode ended at step t - 1, so step t starts from zeros.
        """
        act = ACTIVATIONS[self.activation]

        def dense(n: int, gain: float) -> nn.Dense:
            return nn.Dense(n, kernel_init=orthogonal(gain), bias_init=constant(0.0))

        x = act(dense(self.width, np.sqrt(2))(obs))
        h, y = ScannedGRU(self.rnn_width)(h, (x, resets))

        def head(out: int, gain: float) -> jax.Array:
            z = y
            for _ in range(2):
                z = act(dense(self.width, 2.0)(z))  # gain 2, as the original
            return cast(jax.Array, dense(out, gain)(z))

        return h, head(self.n_actions, 0.01), head(1, 1.0)[..., 0]

    def init_params(self, key: jax.Array) -> Any:
        """Fresh params (they do not depend on the batch size)."""
        h = jnp.zeros((1, self.rnn_width))
        return self.init(
            key, h, jnp.zeros((1, 1, self.obs_dim)), jnp.zeros((1, 1), bool)
        )


def build_network(cfg: PPORNNConfig, n_actions: int, obs_dim: int) -> ActorCriticRNN:
    """The network of ``cfg``; public so a method's loss can apply it."""
    return ActorCriticRNN(n_actions, obs_dim, cfg.width, cfg.rnn_width, cfg.activation)


def apply_network(
    net: ActorCriticRNN, params: Any, h: jax.Array, obs: jax.Array, resets: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(last hidden, logits, values)`` of ``net`` with ``params`` on a sequence."""
    return cast(
        tuple[jax.Array, jax.Array, jax.Array], net.apply(params, h, obs, resets)
    )


def minibatches(b: Batch, key: jax.Array, cfg: PPORNNConfig) -> Batch:
    """Shuffle the envs and split them into ``minibatches`` groups of whole sequences.

    A ``(steps, envs, ...)`` leaf becomes ``(minibatches, steps, envs / mb, ...)``;
    ``h0``, the ``(envs, hidden)`` start hidden, becomes ``(minibatches, envs / mb,
    hidden)``.
    """
    perm = jax.random.permutation(key, cfg.envs)

    def split(name: str, x: jax.Array) -> jax.Array:
        if name == "h0":
            return x[perm].reshape(cfg.minibatches, -1, *x.shape[1:])
        x = x[:, perm].reshape(x.shape[0], cfg.minibatches, -1, *x.shape[2:])
        return jnp.swapaxes(x, 0, 1)

    return {name: split(name, x) for name, x in b.items()}


def _make_rollout(
    cfg: PPORNNConfig, net: ActorCriticRNN, env: Any, env_params: Any, hk: ResolvedHooks
) -> Callable[..., Any]:
    """Jitted rollout of ``num_steps`` x ``envs`` with GAE on ``r_ext + r_int``."""
    rewarded = "intrinsic_reward_fn" in hk.active

    def step(
        with_traj: bool, c: tuple[Any, ...], _: None
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        params, h, obs, last_done, st, ms, t, acc, key = c
        key, k_a, k_s = jax.random.split(key, 3)
        h = jnp.where(last_done[:, None], 0.0, h)  # the hidden this step acts from
        h2, logits, v = apply_network(net, params, h, obs[None], last_done[None])
        logits, v = logits[0], v[0]
        act = jax.random.categorical(k_a, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(cfg.envs), act]
        obs2, st2, r, d, info = env.step(k_s, st, act, env_params)
        rec = {"obs": obs, "act": act, "logp": logp, "v": v, "r": r, "d": d}
        if rewarded:
            rec["r_int"], ms = hk.intrinsic_reward_fn(ms, st, st2, obs2, d, t)
        # the events come from the pre-reset state, so a done step keeps its unlocks
        acc2, total = episode_step(acc, r, d, events_of(info).unlocked)
        rec |= {"reset": last_done, "ach": total.ach, "ret": total.ret}
        rec["len"] = total.length
        if with_traj:
            rec |= {"h": h.astype(jnp.float16), "ev": info["events"]}
        return (params, h2, obs2, d, st2, ms, t + cfg.envs, acc2, key), rec

    @functools.partial(jax.jit, static_argnames="with_traj")
    def rollout(params: Any, carry: tuple[Any, ...], with_traj: bool) -> Any:
        h0 = carry[0]
        body = functools.partial(step, with_traj)
        carry, tr = jax.lax.scan(body, (params, *carry), None, cfg.num_steps)
        h, obs, last_done = carry[1:4]
        _, _, last_v = apply_network(net, params, h, obs[None], last_done[None])
        r = tr["r"] + tr["r_int"] if rewarded else tr["r"]
        adv = gae(r, tr["v"], tr["d"], last_v[0], cfg.gamma, cfg.gae_lambda)
        batch = {k: tr[k] for k in ("obs", "act", "logp", "v", "reset")}
        batch |= {"adv": adv, "tgt": adv + tr["v"], "h0": h0}
        traj = None
        if with_traj:
            r_int = tr["r_int"] if rewarded else jnp.zeros_like(r)
            extra = {"r_int": r_int, "values": tr["v"], "logp": tr["logp"]}
            traj = Traj(
                tr["obs"], tr["act"], tr["r"], tr["d"], tr["h"], tr["ev"], extra
            )
        ep = {k: tr[k] for k in ("d", "ach", "ret", "len")}
        return carry[1:], batch, traj, ep

    return rollout


def _seq_loss(
    cfg: PPORNNConfig, net: ActorCriticRNN, p: Any, mb: Batch
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """`ppo_terms` of a minibatch of sequences rerun from their start hidden."""
    _, logits, v = apply_network(net, p, mb["h0"], mb["obs"], mb["reset"])
    flat = {k: mb[k].reshape(-1) for k in ("act", "logp", "v", "adv", "tgt")}
    return ppo_terms(logits.reshape(-1, logits.shape[-1]), v.reshape(-1), flat, cfg)


def _make_learn(
    cfg: PPORNNConfig, net: ActorCriticRNN, hk: ResolvedHooks
) -> Callable[..., Any]:
    """Jitted update: ``observe``, PPO epochs, extra-loss steps, method metrics.

    Returns the losses in ``records.LOSS_COLUMNS`` order and the method's metrics
    (extra-loss metrics averaged over the extra steps, then ``metrics_fn``).
    """
    grad = jax.value_and_grad(lambda p, mb: _seq_loss(cfg, net, p, mb), has_aux=True)
    extra_grad = jax.value_and_grad(hk.extra_loss_fn, has_aux=True)

    def mb_step(ts: TrainState, mb: Batch) -> tuple[TrainState, Any]:
        (_, aux), g = grad(ts.params, mb)
        ts = ts.apply_gradients(grads=g)  # type: ignore[no-untyped-call]  # flax API
        return ts, (*aux, optax.tree.norm(g))

    def epoch(c: tuple[TrainState, Batch, jax.Array], _: None) -> Any:
        ts, b, key = c
        key, k = jax.random.split(key)
        ts, aux = jax.lax.scan(mb_step, ts, minibatches(b, k, cfg))
        return (ts, b, key), aux

    def extra_step(c: tuple[TrainState, Any, jax.Array], _: None) -> Any:
        ts, ms, key = c
        key, k = jax.random.split(key)
        (_, (ms, m)), g = extra_grad(ts.params, ms, k)
        ts = ts.apply_gradients(grads=g)  # type: ignore[no-untyped-call]  # flax API
        return (ts, ms, key), m

    @jax.jit
    def learn(
        ts: TrainState, b: Batch, traj: Traj | None, ms: Any, key: jax.Array
    ) -> Any:
        if "observe" in hk.active:
            ms = hk.observe(ms, cast(Traj, traj))  # built whenever observe is set
        (ts, _, key), aux = jax.lax.scan(epoch, (ts, b, key), None, cfg.update_epochs)
        ev = 1 - jnp.var(b["tgt"] - b["v"]) / (jnp.var(b["tgt"]) + 1e-8)
        m = [x.mean() for x in aux]
        method: dict[str, jax.Array] = {}
        if "extra_loss_fn" in hk.active:
            c, em = jax.lax.scan(extra_step, (ts, ms, key), None, hk.extra_steps)
            ts, ms, key = c
            check_metric_names(em)
            method |= {k: v.mean() for k, v in em.items()}
        method |= hk.metrics_fn(ms)
        return ts, ms, key, jnp.array([*m[:4], ev, *m[4:]]), method

    return learn


@dataclass(frozen=True, slots=True)
class _Setup:
    """Everything built once per config: network, env, optimizer, jitted steps."""

    net: ActorCriticRNN
    env: Any
    env_params: Any
    tx: optax.GradientTransformation
    hooks: ResolvedHooks
    rollout: Callable[..., Any]
    learn: Callable[..., Any]
    learner: jax.Device
    actor: jax.Device
    mesh: Mesh | None
    """CPU ``envs`` mesh the env batch is split over, if there are several CPUs."""
    shared_at: jax.Device | NamedSharding
    """Placement of params and keys the rollout reads: the actor, or replicated."""

    @property
    def with_traj(self) -> bool:
        """Whether the rollout builds a `Traj` (only ``observe`` reads it)."""
        return "observe" in self.hooks.active

    @property
    def learner_state(self) -> bool:
        """Whether the method state moves to the learner for the update."""
        return bool(_LEARNER_HOOKS & set(self.hooks.active))


@functools.cache
def _setup(cfg: PPORNNConfig) -> _Setup:
    """Cached per config (hooks included) so every chunk reuses the compiled steps.

    Raises:
        ValueError: If the env batch does not split over the CPU devices.
    """
    hk = resolve(cfg)
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
        _make_learn(cfg, net, hk),
        roles.learner,
        roles.actor,
        mesh,
        shared_at,
    )


def _on_actor(tree: Any, s: _Setup, envs: int) -> Any:
    """Place a pytree for the rollout: per-env leaves split over the mesh, if any."""
    if s.mesh is None:
        return jax.device_put(tree, s.actor)
    split = NamedSharding(s.mesh, PartitionSpec("envs"))  # type: ignore[no-untyped-call]

    def put(x: Any) -> Any:
        per_env = jnp.ndim(x) > 0 and jnp.shape(x)[0] == envs
        return jax.device_put(x, split if per_env else s.shared_at)

    return jax.tree.map(put, tree)


class PPORNNState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Full PPO-RNN training state: one pytree that `save_state` round-trips."""

    train_state: TrainState
    env_state: Any
    obs: jax.Array
    hidden: jax.Array
    """GRU carry of each env, float32, before the reset of the next step."""
    last_done: jax.Array
    """The previous step ended the env's episode."""
    episode: EpisodeAcc
    """Return, length and achievements of each env's running episode."""
    env_steps: jax.Array
    """Env steps of the run so far, int32 (the hooks' ``t``)."""
    method_state: Any
    """The hooks' state (`HookFields.init_state_fn`); ``{}`` for the control."""
    key: jax.Array
    """Learner key (minibatch shuffles, extra steps)."""
    key_act: jax.Array
    """Actor key (actions, env steps)."""
    cfg: PPORNNConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def init_state(cfg: PPORNNConfig, key: jax.Array) -> PPORNNState:
    """Fresh params on the learner; envs, GRU carry and method state on the actor."""
    s = _setup(cfg)
    key, k_init, k_reset, k_method = jax.random.split(key, 4)
    with jax.default_device(s.learner):
        ts = TrainState.create(  # type: ignore[no-untyped-call]  # flax API
            apply_fn=s.net.apply, params=s.net.init_params(k_init), tx=s.tx
        )
        # an array from the start, as after an update (see `ppo.init_state`)
        ts = ts.replace(step=jnp.zeros((), jnp.int32))
    n_ach = len(achievement_names(cfg.env))
    spec = HookSpec(cfg.envs, s.net.obs_dim, s.net.n_actions, n_ach, cfg.steps)
    with jax.default_device(s.actor):
        obs, st = s.env.reset(k_reset, s.env_params)
        per_env = (
            obs,
            st,
            jnp.zeros((cfg.envs, cfg.rnn_width)),
            jnp.zeros(cfg.envs, bool),
            init_acc(cfg.envs, n_ach),
        )
        method_state = s.hooks.init_state_fn(k_method, spec)
    if s.mesh is not None:
        per_env = shard_envs(per_env, s.mesh)
    obs, st, hidden, last_done, episode = per_env
    return PPORNNState(
        train_state=cast(TrainState, ts),
        env_state=st,
        obs=obs,
        hidden=hidden,
        last_done=last_done,
        episode=episode,
        env_steps=jax.device_put(jnp.zeros((), jnp.int32), s.shared_at),
        method_state=_on_actor(method_state, s, cfg.envs),
        key=jax.device_put(key, s.learner),
        key_act=jax.device_put(jax.random.fold_in(key, 1), s.shared_at),
        cfg=cfg,
    )


def collect(
    state: PPORNNState, with_traj: bool = False
) -> tuple[PPORNNState, Batch, Traj | None, Batch]:
    """One rollout on the actor: next state, PPO batch, `Traj` if asked, episodes.

    Batch: ``(steps, envs)`` obs, act, logp, v, adv, tgt, reset, and ``h0``, the
    float32 start hidden. Episodes: d, ach, ret, len, as ``training.ppo``.
    """
    s = _setup(state.cfg)
    params = jax.device_put(state.train_state.params, s.shared_at)
    carry = (
        state.hidden,
        state.obs,
        state.last_done,
        state.env_state,
        _on_actor(state.method_state, s, state.cfg.envs),
        state.env_steps,
        state.episode,
        state.key_act,
    )
    carry, batch, traj, ep = s.rollout(params, carry, with_traj=with_traj)
    hidden, obs, last_done, st, ms, t, episode, key_act = carry
    new = state.replace(
        env_state=st,
        obs=obs,
        hidden=hidden,
        last_done=last_done,
        episode=episode,
        env_steps=t,
        method_state=ms,
        key_act=key_act,
    )
    return new, batch, traj, ep


def train_chunk(
    state: PPORNNState, n_updates: int
) -> tuple[PPORNNState, "ChunkMetrics"]:
    """Run ``n_updates`` rollout-and-update steps; metrics stacked per update.

    Besides the PPO metrics, every method metric name maps to its per-update values.
    """
    s = _setup(state.cfg)
    losses, eps, method = [], [], []
    for _ in range(n_updates):
        state, batch, traj, ep = collect(state, s.with_traj)
        put = functools.partial(jax.device_put, device=s.learner)
        ms = put(state.method_state) if s.learner_state else None
        ts, ms, key, loss, m = s.learn(
            state.train_state, put(batch), put(traj), ms, state.key
        )
        if s.learner_state:
            state = state.replace(method_state=ms)
        state = state.replace(train_state=ts, key=key)
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


@dataclass(frozen=True, slots=True)
class PPORNNPolicy:
    """Recurrent actor: the carry is the GRU state, ``(n, rnn_width)``."""

    net: ActorCriticRNN

    def init_carry(self, n: int) -> jax.Array:
        """Zero GRU state for ``n`` envs."""
        return jnp.zeros((n, self.net.rnn_width))

    def step(
        self,
        params: Any,
        carry: jax.Array,
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sampled actions, or the most likely ones when ``greedy``; the next carry.

        Evaluation episodes end at their done, so no reset input is needed.
        """
        resets = jnp.zeros((1, obs.shape[0]), bool)
        carry, logits, _ = apply_network(self.net, params, carry, obs[None], resets)
        sampled = jax.random.categorical(key, logits[0])
        return jnp.where(greedy, logits[0].argmax(-1), sampled), carry


@dataclass(frozen=True, slots=True)
class PPORNNTrainer:
    """A PPO-RNN registry entry (`orchestration.runner.Trainer`).

    ``cfg`` is the method's factory config (its hooks); run-config keys override its
    fields.
    """

    cfg: PPORNNConfig

    def config(self, config: Mapping[str, Any]) -> PPORNNConfig:
        """The factory config with the run config's keys on top."""
        return PPORNNConfig.from_config(config, self.cfg)

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Rollout batch: ``envs * num_steps``."""
        return self.config(config).batch

    def init(self, config: Mapping[str, Any], key: jax.Array) -> PPORNNState:
        """Validated config to the initial state.

        Raises:
            ValueError: On an unknown key, or no ``steps`` while ``anneal_lr``.
        """
        cfg = self.config(config)
        if cfg.anneal_lr and "steps" not in config:
            msg = "PPO-RNN with anneal_lr needs the budget: pass `steps` in the config"
            raise ValueError(msg)
        return init_state(cfg, key)

    def train_chunk(
        self, state: PPORNNState, n_updates: int
    ) -> tuple[PPORNNState, "ChunkMetrics"]:
        """See `train_chunk`."""
        return train_chunk(state, n_updates)

    def params(self, state: PPORNNState) -> Any:
        """Network params (the ``ckpt/`` payload)."""
        return state.train_state.params

    def policy(self, state: PPORNNState) -> PPORNNPolicy:
        """The acting policy of this config's network."""
        return PPORNNPolicy(_setup(state.cfg).net)


def make_trainer(cfg: PPORNNConfig) -> PPORNNTrainer:
    """The trainer of a method built on PPO-RNN: ``cfg`` carries its hooks."""
    return PPORNNTrainer(cfg)
