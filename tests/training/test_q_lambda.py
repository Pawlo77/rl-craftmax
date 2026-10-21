"""Tests for the Q(lambda) targets against an independent forward view."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.training.pqn_rnn import q_lambda_targets

T, B = 6, 3


def _rollout(seed: int, with_dones: bool = True) -> tuple[np.ndarray, ...]:
    """Max Q of states 0..T-1, rewards and dones of steps 0..T-1 (float64)."""
    rng = np.random.default_rng(seed)
    q_max = rng.normal(size=(T, B))
    r = rng.normal(size=(T, B))
    d = (rng.random((T, B)) < 0.25).astype(float) if with_dones else np.zeros((T, B))
    if with_dones:
        d[1, 0] = d[3, 2] = 1.0  # at least two episode ends inside the window
    return q_max, r, d


def _targets(q_max: np.ndarray, r: np.ndarray, d: np.ndarray, gamma: float, lam: float):
    """The function under test on steps 0..T-2, as the trainer calls it."""
    out = q_lambda_targets(
        jnp.asarray(q_max[1:]), jnp.asarray(r[:-1]), jnp.asarray(d[:-1]), gamma, lam
    )
    return np.asarray(out)


def _n_step(q_max, r, d, t, b, n, gamma):
    """G^(n) at step t, env b: n rewards, cut at an episode end, then max Q."""
    g = 0.0
    for k in range(n):
        g += gamma**k * r[t + k, b]
        if d[t + k, b]:
            return g
    return g + gamma**n * q_max[t + n, b]


def _forward_view(q_max, r, d, gamma, lam):
    """(1 - lam) sum_n lam^(n-1) G^(n) + lam^(N-1) G^(N), N = steps to T-1."""
    out = np.zeros((T - 1, B))
    for t in range(T - 1):
        n_max = T - 1 - t
        for b in range(B):
            g = sum(
                (1 - lam) * lam ** (n - 1) * _n_step(q_max, r, d, t, b, n, gamma)
                for n in range(1, n_max)
            )
            out[t, b] = g + lam ** (n_max - 1) * _n_step(
                q_max, r, d, t, b, n_max, gamma
            )
    return out


@pytest.mark.parametrize(("gamma", "lam"), [(0.99, 0.5), (0.9, 0.8), (0.95, 0.3)])
def test_targets_equal_the_forward_view_with_episode_ends(
    gamma: float, lam: float
) -> None:
    q_max, r, d = _rollout(0)
    want = _forward_view(q_max, r, d, gamma, lam)
    np.testing.assert_allclose(_targets(q_max, r, d, gamma, lam), want, atol=1e-5)


def test_lambda_zero_is_one_step_q_learning() -> None:
    q_max, r, d = _rollout(1)
    want = r[:-1] + 0.99 * (1 - d[:-1]) * q_max[1:]
    np.testing.assert_allclose(_targets(q_max, r, d, 0.99, 0.0), want, atol=1e-5)


def test_lambda_one_without_dones_is_the_discounted_sum_plus_the_last_max_q() -> None:
    q_max, r, d = _rollout(2, with_dones=False)
    gamma, got = 0.9, _targets(q_max, r, d, 0.9, 1.0)
    for t in range(T - 1):
        k = T - 1 - t
        want = sum(gamma**i * r[t + i] for i in range(k)) + gamma**k * q_max[T - 1]
        np.testing.assert_allclose(got[t], want, atol=1e-5)


def test_issue_12_the_last_target_bootstraps_from_the_last_state() -> None:
    """purejaxql issue #12: target T-2 is r + gamma max Q(s_{T-1}), not Q(s_T)."""
    q_max, r, d = _rollout(3, with_dones=False)
    got = _targets(q_max, r, d, 0.9, 0.8)
    np.testing.assert_allclose(got[T - 2], r[T - 2] + 0.9 * q_max[T - 1], atol=1e-5)
    assert got.shape == (T - 1, B)


def test_a_done_makes_the_target_the_reward() -> None:
    q_max, r, d = _rollout(4)
    got = _targets(q_max, r, d, 0.99, 0.5)
    ended = d[:-1].astype(bool)
    assert ended.any()
    np.testing.assert_allclose(got[ended], r[:-1][ended], atol=1e-6)


def test_targets_jit_on_float32() -> None:
    """Runs under jit on float32 inputs and keeps their shape and dtype."""
    q = jnp.ones((T - 1, B), jnp.float32)
    out = jax.jit(q_lambda_targets, static_argnums=(3, 4))(q, q, q * 0, 0.99, 0.5)
    assert out.shape == (T - 1, B)
    assert out.dtype == jnp.float32
