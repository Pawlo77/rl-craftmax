"""Tests for the hooks inside the PPO-RNN trainer: zero-hook identity, hook effects."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.hooks import HookSpec, Traj
from wut_murl_craftax.training.ppo import make_train_env
from wut_murl_craftax.training.ppo_rnn import (
    PPORNNConfig,
    collect,
    init_state,
    train_chunk,
)
from wut_murl_craftax.training.wrappers import GymnaxWrapper

C = 0.25
"""Constant intrinsic reward of the shift test."""


def _zero_reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
    return jnp.zeros(d.shape, jnp.float32), s


def _const_reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
    return jnp.full(d.shape, C, jnp.float32), s


def _count_init(key: jax.Array, spec: HookSpec) -> dict[str, jax.Array]:
    return {"calls": jnp.zeros(spec.n_envs, jnp.int32), "t": jnp.int32(-1)}


def _count(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: jax.Array) -> Any:
    return jnp.zeros(d.shape, jnp.float32), {"calls": s["calls"] + 1, "t": t}


def _keep_traj(s: Any, traj: Traj) -> dict[str, Traj]:
    return {"traj": traj}


def _l2(params: Any, s: Any, key: jax.Array) -> Any:
    sq = sum(jnp.sum(x**2) for x in jax.tree.leaves(params))
    return 1e-3 * sq, (s, {})


def _norm(state: Any) -> float:
    return float(optax.tree.norm(state.train_state.params))


def test_zero_hooks_give_the_control_bit_for_bit(
    rnn_cfg: Callable[..., PPORNNConfig], same: Callable[[Any, Any], bool]
) -> None:
    """r_int = 0, identity observe, extra_steps = 0: the control's numbers exactly."""
    zero = rnn_cfg(
        intrinsic_reward_fn=_zero_reward, observe=lambda s, traj: s, extra_loss_fn=_l2
    )
    runs = [train_chunk(init_state(c, jax.random.key(0)), 2) for c in (rnn_cfg(), zero)]
    (a, ma), (b, mb) = runs
    assert same(a.train_state, b.train_state)
    assert same(ma, mb)


def test_constant_intrinsic_reward_shifts_every_td_error(
    rnn_cfg: Callable[..., PPORNNConfig],
) -> None:
    """3 steps, lambda 0: advantages are one-step TD errors, each shifted by C."""
    base = {"num_steps": 3, "gae_lambda": 0.0}
    ctrl, hooked = rnn_cfg(**base), rnn_cfg(**base, intrinsic_reward_fn=_const_reward)
    key = jax.random.key(0)
    _, b0, _, _ = collect(init_state(ctrl, key))
    _, b1, traj, _ = collect(init_state(hooked, key), with_traj=True)
    np.testing.assert_allclose(b1["adv"] - b0["adv"], C, atol=1e-5)
    assert (np.asarray(traj.extra["r_int"]) == C).all()
    # by hand: r + C + gamma v' (1 - d) - v, with v' of the next step (or the bootstrap)
    v, r, d = (np.asarray(x) for x in (b1["v"], traj.rewards, traj.dones))
    td = r[:2] + C + 0.99 * v[1:] * (1 - d[:2]) - v[:2]
    np.testing.assert_allclose(np.asarray(b1["adv"])[:2], td, atol=1e-5)


def test_intrinsic_reward_counts_survive_a_resume(
    rnn_cfg: Callable[..., PPORNNConfig],
    same: Callable[[Any, Any], bool],
    tmp_path: Path,
) -> None:
    """envs x num_steps calls per update, the counter t, and save/load/resume."""
    cfg = rnn_cfg(init_state_fn=_count_init, intrinsic_reward_fn=_count)
    one, _ = train_chunk(init_state(cfg, jax.random.key(0)), 1)
    calls = np.asarray(one.method_state["calls"])
    assert calls.sum() == cfg.envs * cfg.num_steps
    assert int(one.method_state["t"]) == (cfg.num_steps - 1) * cfg.envs
    save_state(tmp_path, one)
    restored = load_state(tmp_path, init_state(cfg, jax.random.key(1)))
    assert same(restored.method_state, one.method_state)
    resumed, _ = train_chunk(restored, 1)
    whole, _ = train_chunk(init_state(cfg, jax.random.key(0)), 2)
    assert np.asarray(resumed.method_state["calls"]).sum() == 2 * calls.sum()
    assert same(resumed, whole)


def _next_lr_step(state: Any) -> jax.Array:
    """The optimizer's step for unit gradients at the next update's learning rate.

    Fresh Adam moments, only the schedule's count moved on, so the step is the
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


def test_extra_loss_steps_shrink_params_and_keep_the_lr_schedule(
    rnn_cfg: Callable[..., PPORNNConfig],
) -> None:
    """2 extra steps on 1e-3 sum(p^2) after 2 PPO steps: smaller params, and the
    next update still runs at the control's learning rate (an optimizer-count
    schedule would already be one update further)."""
    base = {"update_epochs": 1}
    ctrl, hooked = rnn_cfg(**base), rnn_cfg(**base, extra_loss_fn=_l2, extra_steps=2)
    a, _ = train_chunk(init_state(ctrl, jax.random.key(0)), 1)
    b, _ = train_chunk(init_state(hooked, jax.random.key(0)), 1)
    assert int(b.train_state.step) == int(a.train_state.step) + 2
    assert _norm(b) < _norm(a)
    np.testing.assert_allclose(_next_lr_step(b), _next_lr_step(a), rtol=1e-6)


def test_observe_sees_the_extrinsic_rollout(
    rnn_cfg: Callable[..., PPORNNConfig], same: Callable[[Any, Any], bool]
) -> None:
    cfg = rnn_cfg(intrinsic_reward_fn=_const_reward, observe=_keep_traj)
    state = init_state(cfg, jax.random.key(0))
    _, _, want, _ = collect(state, with_traj=True)
    trained, _ = train_chunk(state, 1)
    got = trained.method_state["traj"]
    assert same(got, want)
    assert not (np.asarray(got.rewards) == C).all()  # extrinsic, without r_int
    assert got.hidden.dtype == jnp.float16
    assert got.events.unlocked.shape[:2] == (cfg.num_steps, cfg.envs)


def test_metrics_fn_names_reach_the_update_metrics(
    rnn_cfg: Callable[..., PPORNNConfig],
) -> None:
    cfg = rnn_cfg(metrics_fn=lambda s: {"count/x": 1.0})
    _, metrics = train_chunk(init_state(cfg, jax.random.key(0)), 2)
    assert np.asarray(metrics["count/x"]).tolist() == [1.0, 1.0]  # type: ignore[typeddict-item]
    bad = init_state(rnn_cfg(metrics_fn=lambda s: {"x": 1.0}), jax.random.key(0))
    with pytest.raises(ValueError, match="group/name"):
        train_chunk(bad, 1)


class _ThreeFeatures(GymnaxWrapper):
    """Appends 3 constant features to every observation."""

    def _pad(self, obs: jax.Array) -> jax.Array:
        return jnp.concatenate([obs, jnp.ones((*obs.shape[:-1], 3))], -1)

    def reset(self, key: jax.Array, params: Any = None) -> Any:
        obs, state = self._env.reset(key, params)
        return self._pad(obs), state

    def step(self, key: jax.Array, state: Any, action: Any, params: Any = None) -> Any:
        obs, state, r, d, info = self._env.step(key, state, action, params)
        return self._pad(obs), state, r, d, info

    def observation_space(self, params: Any) -> Any:
        return SimpleNamespace(
            shape=(self._env.observation_space(params).shape[0] + 3,)
        )


def test_env_wrapper_widens_the_network_input(
    rnn_cfg: Callable[..., PPORNNConfig],
) -> None:
    plain = rnn_cfg()
    env, params = make_train_env(plain)
    obs_dim = env.observation_space(params).shape[0]
    cfg = rnn_cfg(env_wrapper_fn=lambda env, p: (_ThreeFeatures(env), p))
    state = init_state(cfg, jax.random.key(0))
    kernel = state.train_state.params["params"]["Dense_0"]["kernel"]
    assert kernel.shape == (obs_dim + 3, cfg.width)
    assert state.obs.shape == (cfg.envs, obs_dim + 3)
    trained, _ = train_chunk(state, 1)
    assert np.isfinite(_norm(trained))
