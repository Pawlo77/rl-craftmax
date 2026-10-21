"""Tests for the baseline PPO trainer: hand-computed losses, tiny Classic runs."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.training.checkpoints import load_index
from wut_murl_craftax.training.ppo import (
    PPOConfig,
    gae,
    init_state,
    lr_schedule,
    main,
    ppo_terms,
    train_chunk,
)
from wut_murl_craftax.training.records import (
    LOSS_COLUMNS,
    load_episodes,
    load_metrics,
)

TINY = PPOConfig(steps=2048, envs=16, num_steps=32, width=32, minibatches=2, n_ckpt=3)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ppo")
    argv = ["--steps", "2048", "--envs", "16", "--num-steps", "32", "--width", "32"]
    main([*argv, "--minibatches", "2", "--n-ckpt", "3", "--out", str(out)])
    return out


def test_tiny_run_writes_all_records(tiny_run: Path) -> None:
    cfg = json.loads((tiny_run / "config.json").read_text())
    assert cfg["n_updates"] == 4
    assert cfg["batch"] == TINY.envs * TINY.num_steps
    assert [e["update"] for e in load_index(tiny_run)] == cfg["ckpt_updates"]
    m = load_metrics(tiny_run)
    assert m.data.shape[0] == 4
    assert np.isfinite(m.col("policy_loss")).all()
    ep = load_episodes(tiny_run)
    assert len(ep.update) == m.col("n_episodes").sum()


def test_config_records_device_roles(tiny_run: Path) -> None:
    cfg = json.loads((tiny_run / "config.json").read_text())
    assert set(cfg["devices"]) == {"actor", "learner"}


def test_gae_on_a_hand_computed_rollout() -> None:
    """Env 0 ends an episode at step 2 (gamma 0.9, lambda 0.8: gamma lambda 0.72)."""
    r = jnp.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    v = jnp.array([[0.5, 0.0], [0.4, 0.0], [0.3, 0.0], [0.2, 0.0]])
    d = jnp.array([[0, 0], [0, 0], [1, 0], [0, 0]], bool)
    adv = gae(r, v, d, jnp.array([1.0, 0.0]), 0.9, 0.8)
    # env 0: A3 = 1 + 0.9 - 0.2; A2 = 2 - 0.3 (done); A1 = 0.27 - 0.4 + 0.72 A2;
    # A0 = 1 + 0.36 - 0.5 + 0.72 A1. env 1: A3 = 1, then x 0.72 per step back.
    expected = [[1.64768, 0.373248], [1.094, 0.5184], [1.7, 0.72], [1.7, 1.0]]
    np.testing.assert_allclose(adv, expected, rtol=1e-6)


def test_clipped_losses_on_a_hand_computed_batch() -> None:
    """Ratios 1.5 (clipped to 1.2) and 0.9 (inside 1 +- 0.2), advantages +1 and -1."""
    cfg = PPOConfig(clip=0.2, vf=0.5, ent=0.01)
    batch = {
        "act": jnp.array([0, 1]),
        "logp": jnp.log(jnp.array([0.5 / 1.5, 0.5 / 0.9])),
        "v": jnp.array([1.0, 0.0]),
        "adv": jnp.array([1.0, -1.0]),
        "tgt": jnp.array([2.0, 0.0]),
    }
    loss, aux = ppo_terms(jnp.zeros((2, 2)), jnp.array([1.5, 0.1]), batch, cfg)
    ploss, vloss, ent, kl, clip_frac = (float(x) for x in aux)
    # policy: -mean(min(1.5, 1.2) * 1, min(-0.9, -0.8)) = -(1.2 - 0.9) / 2
    assert ploss == pytest.approx(-0.15, rel=1e-6)
    # value: clipped prediction 1.2; 0.5 * mean(max(0.25, 0.64), 0.01)
    assert vloss == pytest.approx(0.1625, rel=1e-6)
    assert ent == pytest.approx(np.log(2), rel=1e-6)
    assert kl == pytest.approx((0.5 - np.log(1.5) - 0.1 - np.log(0.9)) / 2, rel=1e-5)
    assert clip_frac == 0.5
    assert float(loss) == pytest.approx(-0.15 + 0.5 * 0.1625 - 0.01 * np.log(2))


def test_learning_rate_anneals_to_lr_over_n_updates() -> None:
    cfg = PPOConfig(lr=3e-4, steps=1_000_000, envs=256, num_steps=16)
    sched, per_update = lr_schedule(cfg), cfg.minibatches * cfg.update_epochs
    assert cfg.n_updates == 244  # 1e6 // 4096: the last partial update is dropped
    assert float(sched(jnp.array(0))) == pytest.approx(3e-4, rel=1e-6)
    last = per_update * (cfg.n_updates - 1)
    for count in (last, last + per_update - 1):
        assert float(sched(jnp.array(count))) == pytest.approx(3e-4 / 244, rel=1e-5)


def test_same_seed_gives_identical_params_and_loss_summaries() -> None:
    cfg = dataclasses.replace(TINY, steps=4096, envs=32, num_steps=8)
    runs = [train_chunk(init_state(cfg, jax.random.key(0)), 2) for _ in range(2)]
    (a, ma), (b, mb) = runs
    pa, pb = (jax.tree.leaves(s.train_state.params) for s in (a, b))
    for x, y in zip(pa, pb, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    losses = np.asarray(ma["losses"])
    assert np.array_equal(losses, np.asarray(mb["losses"]))
    cols = dict(zip(LOSS_COLUMNS, losses.T, strict=True))
    assert ((cols["clip_frac"] >= 0) & (cols["clip_frac"] <= 1)).all()
    assert (cols["grad_norm"] > 0).all()


PPO_METRICS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "explained_variance",
    "grad_norm",
    "adv_mean",
    "adv_std",
    "lr",
)


def test_a_chunk_returns_the_ppo_metrics_per_update() -> None:
    cfg = dataclasses.replace(TINY, steps=4096, envs=32, num_steps=8, lr=1e-3)
    _, m = train_chunk(init_state(cfg, jax.random.key(0)), 3)
    losses = dict(zip(LOSS_COLUMNS, np.asarray(m["losses"]).T, strict=True))
    for name in PPO_METRICS:
        values = np.asarray(m[f"ppo/{name}"])  # type: ignore[literal-required]  # method metric
        assert values.shape == (3,), name
        assert np.isfinite(values).all(), name
    for name in set(LOSS_COLUMNS) - {"explained_var"}:
        np.testing.assert_array_equal(m[f"ppo/{name}"], losses[name])  # type: ignore[literal-required]
    np.testing.assert_array_equal(m["ppo/explained_variance"], losses["explained_var"])  # type: ignore[literal-required]
    assert (np.asarray(m["ppo/adv_std"]) > 0).all()  # type: ignore[literal-required]


def test_the_logged_learning_rate_is_the_annealed_one() -> None:
    cfg = dataclasses.replace(TINY, steps=4096, envs=32, num_steps=8, lr=1e-3)
    state = init_state(cfg, jax.random.key(1))
    _, m = train_chunk(state, 3)
    want = [1e-3 * (1 - u / cfg.n_updates) for u in range(3)]
    np.testing.assert_allclose(m["ppo/lr"], want, rtol=1e-6)  # type: ignore[literal-required]
    flat = dataclasses.replace(cfg, anneal_lr=False)
    _, m = train_chunk(init_state(flat, jax.random.key(1)), 2)
    np.testing.assert_allclose(m["ppo/lr"], [1e-3, 1e-3], rtol=1e-6)  # type: ignore[literal-required]
