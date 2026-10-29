"""Tests for the PPO-RNN control: GRU reset, sequence minibatches, recompute, policy.

Craftax-Classic unless ``CRAFTAX_TEST_ENV`` names another env (8 envs x 8 steps).
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _contract import assert_done_step_unlock_counted

from wut_murl_craftax.evaluation.episode_metrics import ONLINE_LAYOUT
from wut_murl_craftax.orchestration import device_roles
from wut_murl_craftax.orchestration.abort import unreported
from wut_murl_craftax.orchestration.queue import load_method, resolve_config
from wut_murl_craftax.training.hooks import HookFields
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import gae, make_train_env
from wut_murl_craftax.training.ppo_rnn import (
    ActorCriticRNN,
    PPORNNConfig,
    PPORNNState,
    build_network,
    collect,
    init_state,
    make_trainer,
    minibatches,
    train_chunk,
)
from wut_murl_craftax.training.records import LOSS_COLUMNS, achievement_names

REPO = Path(__file__).resolve().parents[2]
T, B, H = 8, 8, 512
PPO_NAMES = (
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
CONTROL_NAMES = (
    "rnn/hidden_norm",
    "rnn/hidden_saturation",
    "rnn/reset_share",
    "ppo/ratio_first_mb",
    "reward/extrinsic",
    "reward/intrinsic",
)
METRIC_NAMES = (*(f"ppo/{n}" for n in PPO_NAMES), *CONTROL_NAMES)


@pytest.fixture(scope="module")
def cfg(rnn_cfg: Callable[..., PPORNNConfig]) -> PPORNNConfig:
    return rnn_cfg()


@pytest.fixture(scope="module")
def net(cfg: PPORNNConfig) -> ActorCriticRNN:
    env, params = make_train_env(cfg)
    obs_dim = env.observation_space(params).shape[0]
    return build_network(cfg, env.action_space(params).n, obs_dim)


@pytest.fixture(scope="module")
def start(cfg: PPORNNConfig) -> PPORNNState:
    """A fresh state whose even envs start after a done, with a non-zero hidden.

    The resets and the stored start hidden are then both exercised by a rollout.
    """
    state = init_state(cfg, jax.random.key(0))
    hidden = jax.random.normal(jax.random.key(1), state.hidden.shape)
    return state.replace(hidden=hidden, last_done=jnp.arange(B) % 2 == 0)


@pytest.fixture(scope="module")
def rollout(start: PPORNNState) -> tuple[Any, ...]:
    return collect(start, with_traj=True)


def _logp(logits: jax.Array, act: jax.Array) -> jax.Array:
    return jnp.take_along_axis(jax.nn.log_softmax(logits), act[..., None], -1)[..., 0]


def test_gru_output_after_a_done_equals_a_fresh_zero_hidden(
    net: ActorCriticRNN,
) -> None:
    p = net.init_params(jax.random.key(0))
    obs = jax.random.uniform(jax.random.key(1), (3, B, net.obs_dim))
    resets = jnp.zeros((3, B), bool).at[2, :4].set(True)
    h0 = jax.random.normal(jax.random.key(2), (B, H))
    _, logits, v = net.apply(p, h0, obs, resets)
    _, fresh_logits, fresh_v = net.apply(
        p, jnp.zeros((B, H)), obs[2:], jnp.zeros((1, B), bool)
    )
    np.testing.assert_allclose(logits[2, :4], fresh_logits[0, :4], atol=1e-6)
    np.testing.assert_allclose(v[2, :4], fresh_v[0, :4], atol=1e-6)
    assert not np.allclose(logits[2, 4:], fresh_logits[0, 4:], atol=1e-6)


def test_recomputed_logp_and_values_match_the_rollout(
    start: PPORNNState,
    rollout: tuple[Any, ...],
    cfg: PPORNNConfig,
    net: ActorCriticRNN,
) -> None:
    """Epoch 1, before any gradient step: the sequence rerun from the stored start
    hidden gives the rollout's log-probs and values to 1e-5."""
    _, batch, _, _ = rollout
    batch = jax.device_put(batch, device_roles().learner)  # as `train_chunk` does
    _, k = jax.random.split(start.key)  # epoch 1's shuffle key
    mbs = minibatches(batch, k, cfg)
    params = start.train_state.params
    for i in range(cfg.minibatches):
        mb = {name: x[i] for name, x in mbs.items()}
        _, logits, v = net.apply(params, mb["h0"], mb["obs"], mb["reset"])
        np.testing.assert_allclose(_logp(logits, mb["act"]), mb["logp"], atol=1e-5)
        np.testing.assert_allclose(v, mb["v"], atol=1e-5)


def test_minibatches_hold_whole_env_sequences(cfg: PPORNNConfig) -> None:
    env_id = jnp.broadcast_to(jnp.arange(B), (T, B))
    step_id = jnp.broadcast_to(jnp.arange(T)[:, None], (T, B))
    batch = {"act": env_id, "obs": (100 * step_id + env_id)[..., None]}
    batch |= {"h0": jnp.arange(B)[:, None] * jnp.ones((1, 3))}
    mbs = minibatches(batch, jax.random.key(3), cfg)
    assert mbs["act"].shape == (cfg.minibatches, T, B // cfg.minibatches)
    seen: list[int] = []
    for i in range(cfg.minibatches):
        envs = np.asarray(mbs["act"][i, 0])
        assert (np.asarray(mbs["act"][i]) == envs).all()  # one env per column
        expected = 100 * np.arange(T)[:, None] + envs  # every step, in order
        assert (np.asarray(mbs["obs"][i, ..., 0]) == expected).all()
        assert (np.asarray(mbs["h0"][i, :, 0]) == envs).all()
        seen += envs.tolist()
    assert sorted(seen) == list(range(B))  # disjoint, and every env once


def test_advantages_are_the_shared_gae(
    start: PPORNNState,
    rollout: tuple[Any, ...],
    cfg: PPORNNConfig,
    net: ActorCriticRNN,
) -> None:
    nxt, batch, traj, _ = rollout
    p = start.train_state.params
    _, _, last_v = net.apply(p, nxt.hidden, nxt.obs[None], nxt.last_done[None])
    adv = gae(
        traj.rewards, batch["v"], traj.dones, last_v[0], cfg.gamma, cfg.gae_lambda
    )
    np.testing.assert_allclose(batch["adv"], adv, atol=1e-6)
    np.testing.assert_allclose(batch["tgt"], adv + batch["v"], atol=1e-6)
    assert (np.asarray(traj.extra["values"]) == np.asarray(batch["v"])).all()


def test_same_seed_gives_identical_params(
    cfg: PPORNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    runs = [train_chunk(init_state(cfg, jax.random.key(0)), 2) for _ in range(2)]
    (a, ma), (b, mb) = runs
    assert same(a.train_state.params, b.train_state.params)
    assert same(ma, mb)


def test_1b_config_is_table_7_with_the_relu_network() -> None:
    """The shared Craftax-1B column of the PPO config (Table 7) and the code's ReLU."""
    methods = REPO / "configs" / "methods"
    table_7 = resolve_config(load_method("ppo", methods), "ppo", "1b")
    cfg = PPORNNConfig.from_config(table_7 | {"method": "ppo-rnn"})
    assert (cfg.envs, cfg.num_steps, cfg.lr) == (1024, 64, 2e-4)
    assert (cfg.update_epochs, cfg.minibatches, cfg.gae_lambda) == (4, 8, 0.8)
    assert (cfg.width, cfg.rnn_width, cfg.activation) == (512, 512, "relu")
    assert cfg.env == "Craftax-Symbolic-v1"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"envs": 12, "minibatches": 8}, "minibatches"),
        ({"activation": "gelu"}, "gelu"),
        ({"resets": "auto"}, "optimistic"),
    ],
)
def test_bad_configs_fail_loudly(overrides: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        PPORNNConfig(**overrides)


def test_hooks_are_not_config_keys() -> None:
    with pytest.raises(ValueError, match="observe"):
        PPORNNConfig.from_config({"observe": None})


def test_extra_steps_comes_from_hook_fields_only() -> None:
    assert "extra_steps" in {f.name for f in dataclasses.fields(PPORNNConfig)}
    assert "extra_steps" in HookFields.__dataclass_fields__
    assert "extra_steps" not in PPORNNConfig.__annotations__
    assert issubclass(PPORNNConfig, HookFields)


def test_traj_holds_the_acting_hidden_and_the_wrapper_events(
    start: PPORNNState,
    rollout: tuple[Any, ...],
    cfg: PPORNNConfig,
    net: ActorCriticRNN,
) -> None:
    """``traj.hidden[t]``: the GRU state step t acted from (zeros after a done);
    ``traj.events``: the wrapper's ``info["events"]``, replayed with the same keys."""
    _, _, traj, _ = rollout
    assert traj.hidden.dtype == jnp.float16
    assert traj.hidden.shape == (T, B, H)
    env, env_params = make_train_env(cfg)
    p, h, done = start.train_state.params, start.hidden, start.last_done
    st, key = start.env_state, start.key_act
    for t in range(T):
        h = jnp.where(done[:, None], 0.0, h)
        want = np.asarray(h.astype(jnp.float16), np.float32)
        np.testing.assert_allclose(traj.hidden[t], want, rtol=1e-3, atol=1e-3)
        h, _, _ = net.apply(p, h, traj.obs[t][None], jnp.zeros((1, B), bool))
        key, _, k_s = jax.random.split(key, 3)
        _, st, _, done, info = env.step(k_s, st, traj.actions[t], env_params)
        assert (np.asarray(done) == np.asarray(traj.dones[t])).all()
        step = jax.tree.map(lambda x, t=t: x[t], traj.events)
        for got, want in zip(
            jax.tree.leaves(step), jax.tree.leaves(info["events"]), strict=True
        ):
            assert (np.asarray(got) == np.asarray(want)).all()


def test_policy_replays_the_first_episode_of_each_env(
    cfg: PPORNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    """From ``init_carry``, ``step`` with the rollout's action keys reproduces the
    rollout's actions until each env's first done."""
    state = init_state(cfg, jax.random.key(4))
    _, _, traj, _ = collect(state, with_traj=True)
    policy = make_trainer(cfg).policy(state)
    carry = policy.init_carry(B)
    assert same(carry, jnp.zeros((B, H)))
    step = jax.jit(policy.step, static_argnames="greedy")
    key, alive = state.key_act, np.ones(B, bool)
    for t in range(T):
        key, k_a, _ = jax.random.split(key, 3)
        act, carry = step(
            state.train_state.params, carry, traj.obs[t], k_a, greedy=False
        )
        assert (np.asarray(act)[alive] == np.asarray(traj.actions[t])[alive]).all()
        alive &= ~np.asarray(traj.dones[t])


def test_registry_entry_trains_with_the_ppo_metric_shapes(cfg: PPORNNConfig) -> None:
    trainer = METHODS["ppo-rnn"]
    config = {"method": "ppo-rnn", "env": cfg.env, "envs": B, "num_steps": T}
    config |= {"width": 64, "rnn_width": 64, "minibatches": 2, "steps": 4 * B * T}
    assert trainer.steps_per_update(config) == B * T
    state = trainer.init(config, jax.random.key(0))
    assert trainer.policy(state).init_carry(3).shape == (3, 64)
    _, m = trainer.train_chunk(state, 2)
    n_ach = len(achievement_names(cfg.env))
    assert {k: np.asarray(v).shape for k, v in m.items()} == {
        "losses": (2, len(LOSS_COLUMNS)),
        "done": (2, T, B),
        "ach": (2, T, B, n_ach),
        "ret": (2, T, B),
        "length": (2, T, B),
        "online": (2, ONLINE_LAYOUT["cause"].stop),
        "first_unlock": (2, n_ach),
        **dict.fromkeys(METRIC_NAMES, (2,)),
    }
    with pytest.raises(ValueError, match="steps"):
        trainer.init(
            {k: v for k, v in config.items() if k != "steps"}, jax.random.key(0)
        )


def test_done_step_unlocks_reach_the_episode_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared trainer contract: whole-episode flags with the done step's unlocks."""
    config = {"method": "ppo-rnn", "env": "Craftax-Classic-Symbolic-v1"}
    # a config no other test compiles, so the patched step events are traced
    config |= {"envs": 16, "num_steps": 32, "width": 19, "rnn_width": 19}
    config |= {"minibatches": 2, "update_epochs": 1, "steps": 100 * 16 * 32}
    assert_done_step_unlock_counted(METHODS["ppo-rnn"], config, monkeypatch)


@pytest.fixture(scope="module")
def three_updates(cfg: PPORNNConfig) -> Any:
    """The metrics of 3 updates from a fresh state."""
    return train_chunk(init_state(cfg, jax.random.key(0)), 3)[1]


def test_a_chunk_returns_the_ppo_and_recurrent_metrics_per_update(
    three_updates: Any,
) -> None:
    m = three_updates
    got = {k: np.asarray(m[k]) for k in METRIC_NAMES}  # type: ignore[literal-required]
    for name, values in got.items():
        assert values.shape == (3,), name
        assert np.isfinite(values).all(), name
    losses = dict(zip(LOSS_COLUMNS, np.asarray(m["losses"]).T, strict=True))
    for name in set(LOSS_COLUMNS) - {"explained_var"}:
        np.testing.assert_array_equal(got[f"ppo/{name}"], losses[name])
    np.testing.assert_array_equal(
        got["ppo/explained_variance"], losses["explained_var"]
    )
    assert (got["rnn/hidden_norm"] > 0).all()
    assert (
        (got["rnn/hidden_saturation"] >= 0) & (got["rnn/hidden_saturation"] <= 1)
    ).all()
    assert ((got["rnn/reset_share"] >= 0) & (got["rnn/reset_share"] <= 1)).all()
    assert (got["reward/intrinsic"] == 0).all()  # the control has no hooks
    assert (got["ppo/adv_std"] > 0).all()


def test_the_logged_learning_rate_is_the_annealed_one(
    cfg: PPORNNConfig, three_updates: Any, rnn_cfg: Callable[..., PPORNNConfig]
) -> None:
    want = [cfg.lr * (1 - u / cfg.n_updates) for u in range(3)]
    np.testing.assert_allclose(three_updates["ppo/lr"], want, rtol=1e-6)
    flat = rnn_cfg(anneal_lr=False)
    _, m = train_chunk(init_state(flat, jax.random.key(1)), 2)
    np.testing.assert_allclose(m["ppo/lr"], [flat.lr] * 2, rtol=1e-6)  # type: ignore[literal-required]


@pytest.mark.parametrize("budget", ["1m", "1b"])
def test_every_abort_rule_of_the_yaml_names_a_reported_metric(budget: str) -> None:
    """A rename of a reported metric must reach ppo-rnn.yaml (else the rule is dead)."""
    rules = load_method("ppo-rnn").abort_rules("ppo-rnn", budget) or {}
    assert (
        unreported(
            rules.get("flags", []),
            rules.get("finite", []),
            METRIC_NAMES,
            rules.get("limits", []),
        )
        == []
    )


def test_the_first_minibatch_ratio_is_one_before_any_gradient_step(
    three_updates: Any,
) -> None:
    np.testing.assert_allclose(three_updates["ppo/ratio_first_mb"], 1.0, atol=1e-3)


def test_the_rollout_stats_are_means_over_steps_and_envs(
    start: PPORNNState, cfg: PPORNNConfig
) -> None:
    """Hand check on a rollout from a known start: resets and rewards."""
    _, _, traj, _ = collect(start, with_traj=True)
    _, m = train_chunk(start, 1)
    resets = np.concatenate([start.last_done[None], np.asarray(traj.dones)[:-1]])
    np.testing.assert_allclose(m["rnn/reset_share"], [resets.mean()], rtol=1e-6)  # type: ignore[typeddict-item]
    np.testing.assert_allclose(  # type: ignore[call-overload]
        m["reward/extrinsic"], [np.asarray(traj.rewards).mean()], rtol=1e-6
    )


def test_a_hooked_reward_is_logged_apart_from_the_extrinsic_one(
    rnn_cfg: Callable[..., PPORNNConfig],
) -> None:
    def reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
        return jnp.full(d.shape, 0.25, jnp.float32), s

    plain, hooked = (
        train_chunk(init_state(rnn_cfg(**extra), jax.random.key(0)), 2)[1]
        for extra in ({}, {"intrinsic_reward_fn": reward})
    )
    assert np.asarray(hooked["reward/intrinsic"]).tolist() == [0.25, 0.25]  # type: ignore[typeddict-item]
    # the first rollout acts from the same params, so its extrinsic reward is the same
    assert hooked["reward/extrinsic"][0] == plain["reward/extrinsic"][0]  # type: ignore[typeddict-item]


@pytest.mark.parametrize("name", ["ppo/entropy", "rnn/hidden_norm", "reward/intrinsic"])
def test_a_method_cannot_reuse_a_control_metric_name(
    rnn_cfg: Callable[..., PPORNNConfig], name: str
) -> None:
    state = init_state(rnn_cfg(metrics_fn=lambda s: {name: 1.0}), jax.random.key(0))
    with pytest.raises(ValueError, match="control metric"):
        train_chunk(state, 1)
