# Portions adapted from Craftax_Baselines ppo.py
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
"""Baseline PPO on Craftax(-Classic)-Symbolic that records what the videos need.

Adapted from Craftax_Baselines ``ppo.py`` (Michael Matthews, MIT; itself after
purejaxrl by Chris Lu): same losses and defaults, no distrax, wandb or orbax.
Rollouts run on ``device_roles().actor`` (env batch split over the CPU devices when
there are several) and updates on ``.learner``. `PPOTrainer` is the runner's
registry entry; `train` writes the run layout of ``training.records`` standalone.

Usage: ``python -m wut_murl_craftax.training.ppo --steps 2e6 --out outputs/runs/c``
"""

import argparse
import dataclasses
import functools
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..orchestration import configure_runtime, device_roles, env_mesh, shard_envs
from .checkpoints import checkpoint_schedule, save_checkpoint
from .records import RunLog, achievement_names

if TYPE_CHECKING:  # the runner imports mlflow; the standalone CLI does not need it
    from ..orchestration.runner import ChunkMetrics

logger = logging.getLogger(__name__)

Batch = dict[str, jax.Array]
"""Flat PPO batch: obs, act, logp, v, adv, tgt."""
RESETS = ("optimistic", "auto")
"""Accepted ``resets`` values."""
_CLI_ONLY = {"seed", "n_ckpt"}
"""`PPOConfig` fields the runner sets itself (key, checkpoints), never in a config."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO hyperparameters; defaults follow Craftax_Baselines except the sizes."""

    env: str = "Craftax-Classic-Symbolic-v1"
    steps: int = 2_000_000
    envs: int = 256
    num_steps: int = 64
    width: int = 256
    lr: float = 2e-4
    update_epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip: float = 0.2
    ent: float = 0.01
    vf: float = 0.5
    max_grad_norm: float = 1.0
    anneal_lr: bool = True
    """Anneal the learning rate linearly to ``lr / n_updates`` at the last update."""
    resets: str = "optimistic"
    """Accepted and validated; envs auto-reset until PPO Task 2 vendors the
    optimistic-reset wrapper."""
    reset_ratio: int = 16
    n_ckpt: int = 12
    """Log-spaced checkpoints, plus update 0 and the last update."""
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject reset modes the trainer does not know."""
        if self.resets not in RESETS:
            msg = f"resets must be one of {RESETS}, got {self.resets!r}"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Build from a run config: its keys, ``steps`` and the registry's ``method``.

        Raises:
            ValueError: On a key that is not a PPO setting.
        """
        keys = set(config) - {"method"}
        names = {f.name for f in dataclasses.fields(cls)} - _CLI_ONLY
        if unknown := sorted(keys - names):
            msg = f"unknown PPO config keys: {unknown}"
            raise ValueError(msg)
        return cls(**{k: config[k] for k in keys})

    @property
    def batch(self) -> int:
        """Environment steps per update."""
        return self.envs * self.num_steps

    @property
    def n_updates(self) -> int:
        """Number of PPO updates for the step budget."""
        return self.steps // self.batch


class ActorCritic(nn.Module):
    """Separate 3-layer tanh MLPs for policy logits and value."""

    n_actions: int
    width: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(logits, value)`` for a batch of flat observations."""

        def mlp(h: jax.Array, out: int, gain: float) -> jax.Array:
            for _ in range(3):
                dense = nn.Dense(self.width, kernel_init=orthogonal(np.sqrt(2)))
                h = nn.tanh(dense(h))
            head = nn.Dense(out, kernel_init=orthogonal(gain), bias_init=constant(0.0))
            return head(h)

        return mlp(x, self.n_actions, 0.01), mlp(x, 1, 1.0)[..., 0]


def make_network(
    env_name: str, width: int, auto_reset: bool = True
) -> tuple[ActorCritic, Any, Any]:
    """Return ``(net, env, env_params)`` for an eval or training run."""
    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    params = env.default_params
    return ActorCritic(env.action_space(params).n, width), env, params


def policy_value(
    net: ActorCritic, params: Any, obs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Policy logits and value for a batch of observations."""
    return cast(tuple[jax.Array, jax.Array], net.apply(params, obs))


def _make_rollout(
    cfg: PPOConfig, net: ActorCritic, env: Any, env_params: Any, names: list[str]
) -> Callable[..., Any]:
    """Jitted rollout of ``num_steps`` x ``envs`` with GAE; returns the flat batch."""
    step_envs = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def step(c: tuple[Any, ...], _: None) -> tuple[tuple[Any, ...], dict[str, Any]]:
        params, st, obs, ep_ret, ep_len, key = c
        key, k_a, k_s = jax.random.split(key, 3)
        logits, v = policy_value(net, params, obs)
        act = jax.random.categorical(k_a, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(cfg.envs), act]
        keys = jax.random.split(k_s, cfg.envs)
        obs2, st2, r, d, info = step_envs(keys, st, act, env_params)
        # Craftax reports 100 * achieved, and only on the done step.
        ach = jnp.stack([info[f"Achievements/{n}"] for n in names], -1) > 0
        ep_ret2, ep_len2 = ep_ret + r, ep_len + 1
        rec = {"obs": obs, "act": act, "logp": logp, "v": v, "r": r, "d": d}
        rec |= {"ach": ach, "ret": ep_ret2, "len": ep_len2}
        carry = (params, st2, obs2, ep_ret2 * (1 - d), ep_len2 * (1 - d), key)
        return carry, rec

    def gae(c: tuple[jax.Array, jax.Array], t: dict[str, jax.Array]) -> Any:
        g, nv = c
        delta = t["r"] + cfg.gamma * nv * (1 - t["d"]) - t["v"]
        g = delta + cfg.gamma * cfg.gae_lambda * (1 - t["d"]) * g
        return (g, t["v"]), g

    @jax.jit
    def rollout(
        params: Any,
        st: Any,
        obs: jax.Array,
        ep_ret: jax.Array,
        ep_len: jax.Array,
        key: jax.Array,
    ) -> Any:
        carry = (params, st, obs, ep_ret, ep_len, key)
        carry, tr = jax.lax.scan(step, carry, None, cfg.num_steps)
        _, st, obs, ep_ret, ep_len, key = carry
        _, last_v = policy_value(net, params, obs)
        init = (jnp.zeros_like(last_v), last_v)
        _, adv = jax.lax.scan(gae, init, tr, reverse=True)
        flat = {
            k: tr[k].reshape(cfg.batch, *tr[k].shape[2:])
            for k in ("obs", "act", "logp", "v")
        }
        flat |= {
            "adv": adv.reshape(cfg.batch),
            "tgt": (adv + tr["v"]).reshape(cfg.batch),
        }
        ep = {k: tr[k] for k in ("d", "ach", "ret", "len")}
        return st, obs, ep_ret, ep_len, key, flat, ep

    return rollout


def _ppo_loss(
    cfg: PPOConfig, net: ActorCritic, p: Any, b: Batch
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Clipped PPO loss; aux = (policy loss, value loss, entropy, approx KL)."""
    logits, v = policy_value(net, p, b["obs"])
    lp_all = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lp_all, b["act"][:, None], 1)[:, 0]
    vc = b["v"] + (v - b["v"]).clip(-cfg.clip, cfg.clip)
    vloss = 0.5 * jnp.maximum((v - b["tgt"]) ** 2, (vc - b["tgt"]) ** 2).mean()
    ratio = jnp.exp(lp - b["logp"])
    g = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    clipped = ratio.clip(1 - cfg.clip, 1 + cfg.clip) * g
    ploss = -jnp.minimum(ratio * g, clipped).mean()
    ent = -(jnp.exp(lp_all) * lp_all).sum(-1).mean()
    kl = ((ratio - 1) - jnp.log(ratio)).mean()
    return ploss + cfg.vf * vloss - cfg.ent * ent, (ploss, vloss, ent, kl)


def _make_learn(cfg: PPOConfig, net: ActorCritic) -> Callable[..., Any]:
    """Jitted PPO update: epochs x shuffled minibatches; returns loss summary."""
    grad = jax.value_and_grad(lambda p, b: _ppo_loss(cfg, net, p, b), has_aux=True)

    def mb_step(ts: TrainState, b: Batch) -> tuple[TrainState, Any]:
        (_, aux), g = grad(ts.params, b)
        ts = ts.apply_gradients(grads=g)  # type: ignore[no-untyped-call]  # flax API
        return ts, aux

    def epoch(c: tuple[TrainState, Batch, jax.Array], _: None) -> Any:
        ts, flat, key = c
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, cfg.batch)
        shape = (cfg.minibatches, -1)
        mb = jax.tree.map(lambda x: x[perm].reshape(*shape, *x.shape[1:]), flat)
        ts, aux = jax.lax.scan(mb_step, ts, mb)
        return (ts, flat, key), aux

    @jax.jit
    def learn(ts: TrainState, flat: Batch, key: jax.Array) -> Any:
        (ts, _, key), aux = jax.lax.scan(
            epoch, (ts, flat, key), None, cfg.update_epochs
        )
        ev = 1 - jnp.var(flat["tgt"] - flat["v"]) / (jnp.var(flat["tgt"]) + 1e-8)
        return ts, key, jnp.array([x.mean() for x in aux] + [ev])

    return learn


def _optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    """Adam with global-norm clipping and an optionally annealed learning rate."""
    per_update = cfg.minibatches * cfg.update_epochs

    def sched(count: jax.Array) -> jax.Array:
        return cfg.lr * (1.0 - (count // per_update) / cfg.n_updates)

    lr = sched if cfg.anneal_lr else cfg.lr
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5)
    )


@dataclass(frozen=True, slots=True)
class _Setup:
    """Everything built once per config: network, env, optimizer, jitted steps."""

    net: ActorCritic
    env: Any
    env_params: Any
    tx: optax.GradientTransformation
    rollout: Callable[..., Any]
    learn: Callable[..., Any]
    learner: jax.Device
    actor: jax.Device
    mesh: Mesh | None
    """CPU ``envs`` mesh the env batch is split over, if there are several CPUs."""
    shared_at: jax.Device | NamedSharding
    """Placement of params and keys the rollout reads: the actor, or replicated."""


@functools.cache
def _setup(cfg: PPOConfig) -> _Setup:
    """Cached per config so every chunk and resume reuses the compiled steps.

    Raises:
        ValueError: If the env batch does not split over the CPU devices.
    """
    roles = device_roles()
    net, env, env_params = make_network(cfg.env, cfg.width)
    names = achievement_names(cfg.env)
    mesh: Mesh | None = None
    shared_at: jax.Device | NamedSharding = roles.actor
    n_cpu = len(jax.devices("cpu"))
    if roles.actor.platform == "cpu" and n_cpu > 1:
        if cfg.envs % n_cpu:
            msg = f"{cfg.envs} envs do not split over {n_cpu} CPU devices"
            raise ValueError(msg)
        mesh = env_mesh()
        shared_at = NamedSharding(mesh, PartitionSpec())  # type: ignore[no-untyped-call]
    return _Setup(
        net,
        env,
        env_params,
        _optimizer(cfg),
        _make_rollout(cfg, net, env, env_params, names),
        _make_learn(cfg, net),
        roles.learner,
        roles.actor,
        mesh,
        shared_at,
    )


class PPOState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Full PPO training state: one pytree that `save_state` round-trips."""

    train_state: TrainState
    env_state: Any
    obs: jax.Array
    ep_ret: jax.Array
    """Return of each env's running episode."""
    ep_len: jax.Array
    key: jax.Array
    """Learner key (minibatch shuffles)."""
    key_act: jax.Array
    """Actor key (actions, env steps)."""
    cfg: PPOConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def init_state(cfg: PPOConfig, key: jax.Array) -> PPOState:
    """Fresh params and optimizer on the learner, reset env batch on the actor."""
    s = _setup(cfg)
    key, k_init, k_reset = jax.random.split(key, 3)
    obs_dim = s.env.observation_space(s.env_params).shape
    with jax.default_device(s.learner):
        params = s.net.init(k_init, jnp.zeros((1, *obs_dim)))
        ts = TrainState.create(apply_fn=s.net.apply, params=params, tx=s.tx)  # type: ignore[no-untyped-call]  # flax API
        # an array from the start, as after an update: Python 0 would make the fresh
        # state (and the load_state template) differ from a trained one in leaf type
        ts = ts.replace(step=jnp.zeros((), jnp.int32))
    with jax.default_device(s.actor):
        reset_keys = jax.random.split(k_reset, cfg.envs)
        obs, st = jax.vmap(s.env.reset, in_axes=(0, None))(reset_keys, s.env_params)
        ep_ret, ep_len = jnp.zeros(cfg.envs), jnp.zeros(cfg.envs, jnp.int32)
    if s.mesh is not None:
        obs, st, ep_ret, ep_len = shard_envs((obs, st, ep_ret, ep_len), s.mesh)
    return PPOState(
        train_state=cast(TrainState, ts),
        env_state=st,
        obs=obs,
        ep_ret=ep_ret,
        ep_len=ep_len,
        key=jax.device_put(key, s.learner),
        key_act=jax.device_put(jax.random.fold_in(key, 1), s.shared_at),
        cfg=cfg,
    )


def train_chunk(state: PPOState, n_updates: int) -> tuple[PPOState, "ChunkMetrics"]:
    """Run ``n_updates`` rollout-and-update steps; metrics stacked per update."""
    s = _setup(state.cfg)
    ts, key = state.train_state, state.key
    carry = (state.env_state, state.obs, state.ep_ret, state.ep_len, state.key_act)
    losses, eps = [], []
    for _ in range(n_updates):
        params_act = jax.device_put(ts.params, s.shared_at)
        *carry, flat, ep = s.rollout(params_act, *carry)
        ts, key, loss = s.learn(ts, jax.device_put(flat, s.learner), key)
        losses.append(loss)
        eps.append(ep)
    st, obs, ep_ret, ep_len, key_act = carry
    new = state.replace(
        train_state=ts,
        env_state=st,
        obs=obs,
        ep_ret=ep_ret,
        ep_len=ep_len,
        key=key,
        key_act=key_act,
    )
    metrics: ChunkMetrics = {
        "losses": jnp.stack(losses),
        "done": jnp.stack([e["d"] for e in eps]).astype(bool),
        "ach": jnp.stack([e["ach"] for e in eps]),
        "ret": jnp.stack([e["ret"] for e in eps]),
        "length": jnp.stack([e["len"] for e in eps]),
    }
    return new, metrics


@dataclass(frozen=True, slots=True)
class PPOPolicy:
    """Feed-forward actor: the carry is an empty tuple."""

    net: ActorCritic

    def init_carry(self, n: int) -> tuple[()]:
        """No recurrent state, whatever the number of envs."""
        del n
        return ()

    def step(
        self,
        params: Any,
        carry: tuple[()],
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
    ) -> tuple[jax.Array, tuple[()]]:
        """Sampled actions, or the most likely ones when ``greedy``."""
        logits, _ = policy_value(self.net, params, obs)
        sampled = jax.random.categorical(key, logits)
        return jnp.where(greedy, logits.argmax(-1), sampled), carry


class PPOTrainer:
    """The ``ppo`` registry entry (`orchestration.runner.Trainer`)."""

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Rollout batch: ``envs * num_steps``."""
        return PPOConfig.from_config(config).batch

    def init(self, config: Mapping[str, Any], key: jax.Array) -> PPOState:
        """Validated config to the initial state.

        Raises:
            ValueError: On an unknown key, or no ``steps`` while ``anneal_lr``.
        """
        cfg = PPOConfig.from_config(config)
        if cfg.anneal_lr and "steps" not in config:
            msg = "PPO with anneal_lr needs the budget: pass `steps` in the config"
            raise ValueError(msg)
        if cfg.resets == "optimistic":
            logger.warning("resets=optimistic not vendored yet: envs auto-reset")
        return init_state(cfg, key)

    def train_chunk(
        self, state: PPOState, n_updates: int
    ) -> tuple[PPOState, "ChunkMetrics"]:
        """See `train_chunk`."""
        return train_chunk(state, n_updates)

    def params(self, state: PPOState) -> Any:
        """Actor-critic params (the ``ckpt/`` payload)."""
        return state.train_state.params

    def policy(self, state: PPOState) -> PPOPolicy:
        """The acting policy of this config's network."""
        return PPOPolicy(_setup(state.cfg).net)


def train(cfg: PPOConfig, out: Path) -> Path:
    """Train, writing ``config.json``, ``metrics.npz``, ``episodes.npz``, ``ckpt/``."""
    state = init_state(cfg, jax.random.key(cfg.seed))
    s, names = _setup(cfg), achievement_names(cfg.env)
    ckpts = checkpoint_schedule(cfg.n_updates, cfg.n_ckpt)
    out.mkdir(parents=True, exist_ok=True)
    meta = dataclasses.asdict(cfg) | {
        "n_updates": cfg.n_updates,
        "batch": cfg.batch,
        "achievements": names,
        "n_actions": s.net.n_actions,
        "ckpt_updates": ckpts,
        "devices": {"actor": str(s.actor), "learner": str(s.learner)},
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log, t0 = RunLog(out, cfg.env, names), time.time()
    save_checkpoint(out, 0, 0, 0.0, state.train_state.params)
    for u in range(1, cfg.n_updates + 1):
        state, metrics = train_chunk(state, 1)
        m = {k: np.asarray(v)[0] for k, v in metrics.items()}
        log.add_update(
            u,
            u * cfg.batch,
            time.time() - t0,
            m["losses"],
            m["done"],
            m["ach"],
            m["ret"],
            m["length"],
        )
        if u in ckpts:
            wall = time.time() - t0
            save_checkpoint(out, u, u * cfg.batch, wall, state.train_state.params)
            log.flush()  # a killed run keeps its curves up to the last checkpoint
            logger.info(
                "update %d/%d, %.0f steps/s", u, cfg.n_updates, u * cfg.batch / wall
            )
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: one ``--flag`` per ``PPOConfig`` field plus ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in dataclasses.fields(PPOConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if f.type in (bool, "bool"):
            action = argparse.BooleanOptionalAction
            parser.add_argument(flag, action=action, default=f.default)
            continue
        kind = float if f.type in (float, "float") or f.name == "steps" else f.type
        parser.add_argument(flag, type=kind, default=f.default)
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/ppo"))
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    args["steps"] = int(args["steps"])  # accept 2e6
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_runtime()
    train(PPOConfig(**args), out)
    print(f"run written to {out}")


if __name__ == "__main__":
    main()
