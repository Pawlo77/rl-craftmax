"""Tests for the hooks inside the PQN-RNN trainer: zero-hook identity, hook effects."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from wut_murl_craftax.training.hooks import HookSpec, Traj
from wut_murl_craftax.training.pqn_rnn import (
    PQNRNNConfig,
    collect,
    init_state,
    learn,
    q_lambda_targets,
    train_chunk,
)

C = 0.25
"""Constant intrinsic reward of the shift test."""


def _zero_reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
    return jnp.zeros(d.shape, jnp.float32), s


def _const_reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
    return jnp.full(d.shape, C, jnp.float32), s


def _count_init(key: jax.Array, spec: HookSpec) -> dict[str, jax.Array]:
    return {"calls": jnp.zeros(spec.n_envs, jnp.int32)}


def _count(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: jax.Array) -> Any:
    return jnp.zeros(d.shape, jnp.float32), {"calls": s["calls"] + 1}


def _keep_traj(s: Any, traj: Traj) -> dict[str, Traj]:
    return {"traj": traj}


def _l2(params: Any, s: Any, key: jax.Array) -> Any:
    sq = sum(jnp.sum(x**2) for x in jax.tree.leaves(params))
    return 1e-3 * sq, (s, {})


def _norm(state: Any) -> float:
    return float(optax.tree.norm(state.train_state.params))


def test_zero_hooks_give_the_control_bit_for_bit(
    pqn_cfg: Callable[..., PQNRNNConfig], same: Callable[[Any, Any], bool]
) -> None:
    """r_int = 0, identity observe, extra_steps = 0: params, batch statistics and
    metrics of the control exactly."""
    zero = pqn_cfg(
        intrinsic_reward_fn=_zero_reward, observe=lambda s, traj: s, extra_loss_fn=_l2
    )
    runs = [train_chunk(init_state(c, jax.random.key(0)), 2) for c in (pqn_cfg(), zero)]
    (a, ma), (b, mb) = runs
    assert same(a.train_state, b.train_state)
    assert same(ma, mb)


def test_constant_intrinsic_reward_raises_every_target(
    pqn_cfg: Callable[..., PQNRNNConfig], same: Callable[[Any, Any], bool]
) -> None:
    """r_int = C enters the Q(lambda) reward: learn equals learn on r + C; by hand
    on 3 steps the targets rise by C times the lambda-weighted discount sum."""
    cfg = pqn_cfg(num_steps=3, intrinsic_reward_fn=_const_reward)
    state = init_state(cfg, jax.random.key(0))
    _, traj, h0, _ = collect(state)
    assert (np.asarray(traj.extra["r_int"]) == C).all()
    folded = traj.replace(
        rewards=traj.rewards + C,
        extra=traj.extra | {"r_int": jnp.zeros_like(traj.rewards)},
    )
    key = jax.random.key(1)
    a, _ = learn(state.train_state, traj, key, h0=h0)
    b, _ = learn(state.train_state, folded, key, h0=h0)
    assert same(a.params, b.params)
    q = jax.random.normal(jax.random.key(2), (2, cfg.envs))
    r, d = traj.rewards[:2], traj.dones[:2].astype(jnp.float32)
    gl = cfg.gamma * cfg.lam
    shift = q_lambda_targets(q, r + C, d, cfg.gamma, cfg.lam)
    shift -= q_lambda_targets(q, r, d, cfg.gamma, cfg.lam)
    want = np.stack([C + gl * (1 - d[0]) * C, np.full(cfg.envs, C)])
    np.testing.assert_allclose(shift, want, atol=1e-6)


def test_intrinsic_reward_skips_the_warm_up(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    """The warm-up rollout keeps the method state: envs x num_steps calls per update."""
    cfg = pqn_cfg(init_state_fn=_count_init, intrinsic_reward_fn=_count)
    state = init_state(cfg, jax.random.key(0))
    assert int(np.asarray(state.method_state["calls"]).sum()) == 0
    one, _ = train_chunk(state, 1)
    assert int(np.asarray(one.method_state["calls"]).sum()) == cfg.batch


def _next_lr_step(state: Any) -> jax.Array:
    """The optimizer's step for unit gradients at the next update's learning rate.

    Fresh RAdam moments, only the schedule's count moved on, so the step is the
    learning rate times a factor equal across configs of equal shapes.
    """
    ts = state.train_state

    def at_step(s: Any) -> Any:
        if isinstance(s, optax.ScaleByScheduleState):
            return s._replace(count=ts.step)
        return s

    fresh = jax.tree.map(
        at_step,
        ts.tx.init(ts.params),
        is_leaf=lambda s: isinstance(s, optax.ScaleByScheduleState),
    )
    ones = jax.tree.map(jnp.ones_like, ts.params)
    upd, _ = ts.tx.update(ones, fresh, ts.params)
    return jax.tree.leaves(upd)[0]


def test_extra_loss_steps_shrink_params_and_keep_lr_and_epsilon(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    """2 extra steps on 1e-3 sum(p^2) after 2 Q steps: smaller params, and the next
    update runs at the control's learning rate and epsilon."""
    base = {"epochs": 1}
    ctrl, hooked = pqn_cfg(**base), pqn_cfg(**base, extra_loss_fn=_l2, extra_steps=2)
    a, _ = train_chunk(init_state(ctrl, jax.random.key(0)), 1)
    b, _ = train_chunk(init_state(hooked, jax.random.key(0)), 1)
    assert int(b.train_state.step) == int(a.train_state.step) + 2
    assert _norm(b) < _norm(a)
    np.testing.assert_allclose(_next_lr_step(b), _next_lr_step(a), rtol=1e-6)
    assert float(b.eps) == float(a.eps)


def test_observe_sees_the_extrinsic_rollout(
    pqn_cfg: Callable[..., PQNRNNConfig], same: Callable[[Any, Any], bool]
) -> None:
    cfg = pqn_cfg(intrinsic_reward_fn=_const_reward, observe=_keep_traj)
    state = init_state(cfg, jax.random.key(0))
    _, want, _, _ = collect(state, with_traj=True)
    trained, _ = train_chunk(state, 1)
    got = trained.method_state["traj"]
    assert same(got, want)
    assert not (np.asarray(got.rewards) == C).all()  # extrinsic, without r_int
    assert got.hidden.dtype == jnp.float16
    assert got.hidden.shape == (cfg.num_steps, cfg.envs, 2 * cfg.hidden)
    assert got.events.unlocked.shape[:2] == (cfg.num_steps, cfg.envs)
    assert got.extra["q_vals"].shape[:2] == (cfg.num_steps, cfg.envs)
    assert set(got.extra) == {"r_int", "q_vals", "last_actions"}


def test_metrics_fn_names_reach_the_update_metrics(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    cfg = pqn_cfg(metrics_fn=lambda s: {"count/x": 1.0})
    _, metrics = train_chunk(init_state(cfg, jax.random.key(0)), 2)
    assert np.asarray(metrics["count/x"]).tolist() == [1.0, 1.0]  # type: ignore[typeddict-item]
    bad = init_state(pqn_cfg(metrics_fn=lambda s: {"x": 1.0}), jax.random.key(0))
    with pytest.raises(ValueError, match="group/name"):
        train_chunk(bad, 1)
