"""Tests for the BatchRenorm port: running averages, warm-up, clipping, momentum."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wut_murl_craftax.training.batch_renorm import BatchRenorm

EPS = 1e-3
"""BatchRenorm's default epsilon."""
F = 4


def _x(seed: int = 0) -> jax.Array:
    return 2.0 + 3.0 * jax.random.normal(jax.random.key(seed), (32, F))


def _variables(**stats: Any) -> Any:
    """Fresh variables, with the given batch statistics overwritten."""
    v = BatchRenorm(use_running_average=True).init(jax.random.key(0), _x())
    return {"params": v["params"], "batch_stats": v["batch_stats"] | stats}


def _train(v: Any, x: jax.Array) -> tuple[jax.Array, Any]:
    out, upd = BatchRenorm(use_running_average=False).apply(
        v, x, mutable=["batch_stats"]
    )
    return out, upd["batch_stats"]


def test_running_average_mode_uses_the_stored_statistics_and_keeps_them() -> None:
    mean, var = jnp.arange(F) * 0.5, jnp.arange(1, F + 1) * 2.0
    v = _variables(mean=mean, var=var, steps=7)
    out, upd = BatchRenorm(use_running_average=True).apply(
        v, _x(), mutable=["batch_stats"]
    )
    np.testing.assert_allclose(out, (_x() - mean) / jnp.sqrt(var + EPS), rtol=1e-5)
    for k, want in v["batch_stats"].items():
        assert (np.asarray(upd["batch_stats"][k]) == np.asarray(want)).all()


def test_training_before_warm_up_is_plain_batch_norm() -> None:
    """999 steps done, running statistics far off: r and d are not applied yet."""
    v = _variables(mean=jnp.full(F, 50.0), var=jnp.full(F, 1e-4), steps=999)
    out, _ = _train(v, _x())
    x = _x()
    want = (x - x.mean(0)) / jnp.sqrt(x.var(0) + EPS)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-5)


def test_after_warm_up_r_and_d_are_clipped() -> None:
    """Running statistics far off in both directions: r in [1/3, 3], d in [-5, 5]."""
    x = _x()
    mu, var = x.mean(0), x.var(0)
    ra_mean = jnp.array([50.0, -50.0, 0.0, 0.0])
    ra_var = jnp.array([1e-4, 1e-4, 1e4, var[3]])
    v = _variables(mean=ra_mean, var=ra_var, steps=1000)
    out, _ = _train(v, x)
    std, ra_std = jnp.sqrt(var + EPS), jnp.sqrt(ra_var + EPS)
    r = jnp.clip(std / ra_std, 1 / 3, 3)
    d = jnp.clip((mu - ra_mean) / ra_std, -5, 5)
    assert np.allclose(r[:3], [3, 3, 1 / 3])
    assert np.allclose(d[:2], [-5, 5])
    custom_var = var / r**2
    custom_mean = mu - d * jnp.sqrt(var) / r
    want = (x - custom_mean) / jnp.sqrt(custom_var + EPS)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)
    plain = (x - mu) / jnp.sqrt(var + EPS)
    assert not np.allclose(out[:, :3], plain[:, :3], atol=1e-2)


def test_one_training_step_moves_the_running_statistics_by_the_momentum() -> None:
    old_mean, old_var = jnp.full(F, 0.3), jnp.full(F, 2.0)
    v = _variables(mean=old_mean, var=old_var, steps=5)
    _, upd = _train(v, _x())
    x = _x()
    np.testing.assert_allclose(
        upd["mean"], 0.999 * old_mean + 0.001 * x.mean(0), rtol=1e-6
    )
    np.testing.assert_allclose(
        upd["var"], 0.999 * old_var + 0.001 * x.var(0), rtol=1e-5
    )
    assert int(upd["steps"]) == 6
    assert (int(upd["r_max"]), int(upd["d_max"])) == (3, 5)


def test_init_holds_the_original_defaults() -> None:
    stats = _variables()["batch_stats"]
    assert (np.asarray(stats["mean"]) == 0).all()
    assert (np.asarray(stats["var"]) == 1).all()
    assert (int(stats["r_max"]), int(stats["d_max"]), int(stats["steps"])) == (3, 5, 0)
