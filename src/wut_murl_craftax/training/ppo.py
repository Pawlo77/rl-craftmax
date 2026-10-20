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
Rollouts run on ``device_roles().actor`` and updates on ``.learner`` (CPU actor and
Metal learner on Apple silicon). Writes the run layout of ``training.records``.

Usage: ``python -m wut_murl_craftax.training.ppo --steps 2e6 --out outputs/runs/c``
"""

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from ..orchestration import configure_runtime, device_roles
from .checkpoints import checkpoint_schedule, save_checkpoint
from .records import RunLog, achievement_names

logger = logging.getLogger(__name__)

Batch = dict[str, jax.Array]
"""Flat PPO batch: obs, act, logp, v, adv, tgt."""


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
    n_ckpt: int = 12
    """Log-spaced checkpoints, plus update 0 and the last update."""
    seed: int = 0

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


def _init_state(
    cfg: PPOConfig, net: ActorCritic, obs_dim: tuple[int, ...], key: jax.Array
) -> TrainState:
    """Adam with global-norm clipping and a linearly annealed learning rate."""
    per_update = cfg.minibatches * cfg.update_epochs

    def sched(count: jax.Array) -> jax.Array:
        return cfg.lr * (1.0 - (count // per_update) / cfg.n_updates)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched, eps=1e-5)
    )
    params = net.init(key, jnp.zeros((1, *obs_dim)))
    state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)  # type: ignore[no-untyped-call]  # flax API
    return cast(TrainState, state)


def train(cfg: PPOConfig, out: Path) -> Path:
    """Train, writing ``config.json``, ``metrics.npz``, ``episodes.npz``, ``ckpt/``."""
    roles = device_roles()
    net, env, env_params = make_network(cfg.env, cfg.width)
    names = achievement_names(cfg.env)
    ckpts = checkpoint_schedule(cfg.n_updates, cfg.n_ckpt)
    out.mkdir(parents=True, exist_ok=True)
    meta = dataclasses.asdict(cfg) | {
        "n_updates": cfg.n_updates,
        "batch": cfg.batch,
        "achievements": names,
        "n_actions": net.n_actions,
        "ckpt_updates": ckpts,
        "devices": {"actor": str(roles.actor), "learner": str(roles.learner)},
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    key, k_init, k_reset = jax.random.split(jax.random.key(cfg.seed), 3)
    obs_dim = env.observation_space(env_params).shape
    with jax.default_device(roles.learner):
        ts = _init_state(cfg, net, obs_dim, k_init)
    with jax.default_device(roles.actor):
        reset_keys = jax.random.split(k_reset, cfg.envs)
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        ep_ret, ep_len = jnp.zeros(cfg.envs), jnp.zeros(cfg.envs, jnp.int32)
        key_act = jax.device_put(jax.random.fold_in(key, 1), roles.actor)
    key = jax.device_put(key, roles.learner)
    rollout, learn = (
        _make_rollout(cfg, net, env, env_params, names),
        _make_learn(cfg, net),
    )
    log, t0 = RunLog(out, cfg.env, names), time.time()
    save_checkpoint(out, 0, 0, 0.0, ts.params)
    for u in range(1, cfg.n_updates + 1):
        params_act = jax.device_put(ts.params, roles.actor)
        st, obs, ep_ret, ep_len, key_act, flat, ep = rollout(
            params_act, st, obs, ep_ret, ep_len, key_act
        )
        ts, key, losses = learn(ts, jax.device_put(flat, roles.learner), key)
        ep = jax.tree.map(np.asarray, ep)
        log.add_update(
            u,
            u * cfg.batch,
            time.time() - t0,
            np.asarray(losses),
            ep["d"],
            ep["ach"],
            ep["ret"],
            ep["len"],
        )
        if u in ckpts:
            save_checkpoint(out, u, u * cfg.batch, time.time() - t0, ts.params)
            log.flush()  # a killed run keeps its curves up to the last checkpoint
            sps = u * cfg.batch / (time.time() - t0)
            logger.info("update %d/%d, %.0f steps/s", u, cfg.n_updates, sps)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: one ``--flag`` per ``PPOConfig`` field plus ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in dataclasses.fields(PPOConfig):
        kind = float if f.type in (float, "float") or f.name == "steps" else f.type
        flag = f"--{f.name.replace('_', '-')}"
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
