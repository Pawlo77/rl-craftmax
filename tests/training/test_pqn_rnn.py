"""Tests for the PQN-RNN control: LSTM, last action, epsilon, learn, warm-up, policy.

Craftax-Classic unless ``CRAFTAX_TEST_ENV`` names another env (8 envs x 8 steps).
"""

import dataclasses
import subprocess
import sys
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
from wut_murl_craftax.training.checkpoints import (
    load_params,
    load_state,
    save_checkpoint,
    save_state,
)
from wut_murl_craftax.training.hooks import HOOK_NAMES, HookFields, Traj
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import make_train_env
from wut_murl_craftax.training.pqn_rnn import (
    NO_ACTION,
    PQNRNNConfig,
    PQNRNNState,
    RNNQNetwork,
    apply_network,
    build_network,
    collect,
    eps_greedy,
    epsilon,
    init_state,
    learn,
    make_trainer,
    q_lambda_targets,
    train_chunk,
    variables_of,
)
from wut_murl_craftax.training.records import LOSS_COLUMNS, achievement_names

T, B, H = 8, 8, 512
PQN_METRICS = (
    "pqn/td_loss",
    "pqn/q_taken_mean",
    "pqn/q_max_mean",
    "pqn/target_mean",
    "pqn/action_gap",
    "pqn/epsilon",
    "pqn/greedy_share",
    "pqn/lr",
    "pqn/grad_norm",
    "pqn/bn_warm",
    "pqn/bn_var_min",
    "pqn/bn_dead_share",
    "rnn/hidden_norm",
    "rnn/reset_share",
    "reward/extrinsic",
    "reward/intrinsic",
)
SYMBOLIC_OBS = 8268
"""Craftax-Symbolic observation width."""
REPO_1B = {
    "ENV_NAME": "Craftax-Symbolic-v1",
    "TOTAL_TIMESTEPS": 1e9,
    "NUM_ENVS": 1024,
    "NUM_STEPS": 128,
    "EPS_START": 1.0,
    "EPS_FINISH": 0.005,
    "EPS_DECAY": 0.1,
    "NUM_MINIBATCHES": 4,
    "NUM_EPOCHS": 4,
    "NORM_INPUT": True,
    "NORM_TYPE": "layer_norm",
    "HIDDEN_SIZE": 512,
    "ADD_LAST_ACTION": True,
    "LR": 0.0003,
    "MAX_GRAD_NORM": 0.5,
    "LR_LINEAR_DECAY": True,
    "REW_SCALE": 1.0,
    "GAMMA": 0.99,
    "LAMBDA": 0.5,
    "OPTIMISTIC_RESET_RATIO": 16,
}
"""purejaxql ``config/alg/pqn_rnn_craftax.yaml`` at 47af6d7, keys with a field."""
FIELDS = {
    "ENV_NAME": "env",
    "TOTAL_TIMESTEPS": "steps",
    "NUM_ENVS": "envs",
    "NUM_STEPS": "num_steps",
    "EPS_START": "eps_start",
    "EPS_FINISH": "eps_finish",
    "EPS_DECAY": "eps_decay",
    "NUM_MINIBATCHES": "minibatches",
    "NUM_EPOCHS": "epochs",
    "NORM_INPUT": "norm_input",
    "NORM_TYPE": "norm_type",
    "HIDDEN_SIZE": "hidden",
    "ADD_LAST_ACTION": "add_last_action",
    "LR": "lr",
    "MAX_GRAD_NORM": "max_grad_norm",
    "LR_LINEAR_DECAY": "lr_linear_decay",
    "REW_SCALE": "rew_scale",
    "GAMMA": "gamma",
    "LAMBDA": "lam",
    "OPTIMISTIC_RESET_RATIO": "reset_ratio",
}
"""YAML key -> `PQNRNNConfig` field."""


@pytest.fixture(scope="module")
def cfg(pqn_cfg: Callable[..., PQNRNNConfig]) -> PQNRNNConfig:
    return pqn_cfg()


@pytest.fixture(scope="module")
def net(cfg: PQNRNNConfig) -> RNNQNetwork:
    env, params = make_train_env(cfg)
    obs_dim = env.observation_space(params).shape[0]
    return build_network(cfg, env.action_space(params).n, obs_dim)


@pytest.fixture(scope="module")
def start(cfg: PQNRNNConfig, net: RNNQNetwork) -> PQNRNNState:
    """A state whose even envs start after a done, with a random carry and last action.

    The resets, the last-action input and the stored start carry are all exercised.
    """
    state = init_state(cfg, jax.random.key(0))
    k_c, k_h, k_a = jax.random.split(jax.random.key(1), 3)
    hidden = (jax.random.normal(k_c, (B, H)), jax.random.normal(k_h, (B, H)))
    last = jax.random.randint(k_a, (B,), 0, net.n_actions)
    return state.replace(
        hidden=hidden, last_done=jnp.arange(B) % 2 == 0, last_action=last
    )


@pytest.fixture(scope="module")
def rollout(start: PQNRNNState) -> tuple[Any, ...]:
    return collect(start, with_traj=True)


def _resets(dones: jax.Array) -> jax.Array:
    """Learner reset flags: none at step 0 (the start carry is post-reset)."""
    return jnp.concatenate([jnp.zeros_like(dones[:1], bool), dones[:-1].astype(bool)])


def test_lstm_output_after_a_done_equals_a_fresh_zero_carry(net: RNNQNetwork) -> None:
    v = net.init_variables(jax.random.key(0))
    obs = jax.random.uniform(jax.random.key(1), (3, B, net.obs_dim))
    last = jax.random.randint(jax.random.key(2), (3, B), 0, net.n_actions)
    resets = jnp.zeros((3, B), bool).at[2, :4].set(True)
    k_c, k_h = jax.random.split(jax.random.key(3))
    carry = (jax.random.normal(k_c, (B, H)), jax.random.normal(k_h, (B, H)))
    _, q = apply_network(net, v, carry, obs, resets, last)
    no_reset = jnp.zeros((1, B), bool)
    _, fresh = apply_network(net, v, net.init_carry(B), obs[2:], no_reset, last[2:])
    np.testing.assert_allclose(q[2, :4], fresh[0, :4], atol=1e-6)
    assert not np.allclose(q[2, 4:], fresh[0, 4:], atol=1e-6)


def _zero_action_rows(v: Any, row: int) -> Any:
    """Zero input row ``row`` of the LSTM's four input kernels."""

    def zero(path: Any, x: jax.Array) -> jax.Array:
        names = [getattr(p, "key", "") for p in path]
        return x.at[row].set(0.0) if names[-2] in {"ii", "if", "ig", "io"} else x

    return jax.tree_util.tree_map_with_path(zero, v)


def test_last_action_input_is_the_previous_action_and_zero_after_a_reset(
    start: PQNRNNState, rollout: tuple[Any, ...], net: RNNQNetwork
) -> None:
    _, traj, _, _ = rollout
    la, act = np.asarray(traj.extra["last_actions"]), np.asarray(traj.actions)
    done = np.asarray(traj.dones)
    first = np.where(start.last_done, NO_ACTION, start.last_action)
    assert (la[0] == first).all()
    assert (la[1:] == np.where(done[:-1], NO_ACTION, act[:-1])).all()
    assert (la == NO_ACTION).any()
    # the input is one_hot(a), all zeros for NO_ACTION: without the rows of action 3
    # the LSTM cannot tell 3 from NO_ACTION, with them it can
    v = net.init_variables(jax.random.key(0))
    obs = jax.random.uniform(jax.random.key(1), (1, B, net.obs_dim))
    no_reset, carry = jnp.zeros((1, B), bool), net.init_carry(B)

    def q(variables: Any, a: int) -> jax.Array:
        last = jnp.full((1, B), a, jnp.int32)
        return apply_network(net, variables, carry, obs, no_reset, last)[1]

    blind = _zero_action_rows(v, H + 3)
    np.testing.assert_allclose(q(blind, 3), q(blind, NO_ACTION), atol=1e-6)
    assert not np.allclose(q(v, 3), q(v, NO_ACTION), atol=1e-6)
    assert not np.allclose(q(blind, 4), q(blind, NO_ACTION), atol=1e-6)


def test_epsilon_falls_linearly_over_the_decay_share(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    cfg = pqn_cfg(steps=100 * B * T)  # 100 updates, decay over the first 10
    got = [float(epsilon(cfg, jnp.int32(u))) for u in (0, 5, 10, 11, 99)]
    np.testing.assert_allclose(got, [1.0, 1 - 0.995 / 2, 0.005, 0.005, 0.005], 1e-6)


def test_epsilon_greedy_is_argmax_at_0_and_uniform_at_1() -> None:
    q = jax.random.normal(jax.random.key(0), (20_000, 43))
    greedy = eps_greedy(jax.random.key(1), q, jnp.float32(0.0))
    assert (np.asarray(greedy) == np.asarray(q.argmax(-1))).all()
    rand = eps_greedy(jax.random.key(1), q, jnp.float32(1.0))
    assert set(np.unique(np.asarray(rand)).tolist()) == set(range(43))


def test_recomputed_q_values_match_the_rollout(
    start: PQNRNNState, rollout: tuple[Any, ...], net: RNNQNetwork
) -> None:
    """Inference mode from the stored float32 start carry, the learner's reset flags
    and the recorded last actions: the rollout's Q-values to 1e-5."""
    _, traj, h0, _ = rollout
    assert h0[0].dtype == h0[1].dtype == jnp.float32
    v = variables_of(start.train_state)
    last = traj.extra["last_actions"]
    _, q = apply_network(net, v, h0, traj.obs, _resets(traj.dones), last)
    np.testing.assert_allclose(q, traj.extra["q_vals"], atol=1e-5)


def test_same_seed_gives_identical_params(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    runs = [train_chunk(init_state(cfg, jax.random.key(0)), 2) for _ in range(2)]
    (a, ma), (b, mb) = runs
    assert same(a.train_state, b.train_state)
    assert same(ma, mb)


def test_1b_config_is_the_repo_yaml_field_by_field() -> None:
    """The 1M base with the repo's env, budget and rollout shape is the repo config;
    NUM_LAYERS 1, NUM_RNN_LAYERS 1 and MEMORY_WINDOW 0 are fixed in the code."""
    shape = {"steps": 10**9, "envs": 1024, "num_steps": 128}
    cfg = PQNRNNConfig.from_config({"method": "pqn-rnn", **shape})
    for key, value in REPO_1B.items():
        assert getattr(cfg, FIELDS[key]) == value, key
    assert cfg.resets == "optimistic"  # USE_OPTIMISTIC_RESETS: True
    assert cfg.input_norm == "batch_renorm"
    assert cfg.n_updates == 7629
    base = PQNRNNConfig()
    assert (base.envs, base.num_steps, base.steps) == (256, 16, 10**6)
    assert dataclasses.replace(base, **shape) == cfg
    names = {f.name for f in dataclasses.fields(PQNRNNConfig)}
    rest = names - set(FIELDS.values()) - set(HOOK_NAMES) - {"extra_steps"}
    assert rest == {"input_norm", "extra_obs_dim", "seed"}
    v = build_network(cfg, 43, SYMBOLIC_OBS).init_variables(jax.random.key(0))
    assert set(v["params"]) == {
        "BatchRenorm_0",
        "Dense_0",
        "LayerNorm_0",
        "ScannedLSTM_0",
        "Dense_1",
    }


def test_extra_steps_comes_from_hook_fields_only() -> None:
    assert "extra_steps" in {f.name for f in dataclasses.fields(PQNRNNConfig)}
    assert "extra_steps" in HookFields.__dataclass_fields__
    assert "extra_steps" not in PQNRNNConfig.__annotations__
    assert issubclass(PQNRNNConfig, HookFields)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"envs": 12, "minibatches": 8}, "minibatches"),
        ({"norm_type": "group_norm"}, "group_norm"),
        ({"input_norm": "batch_norm"}, "batch_norm"),
        ({"num_steps": 1}, "num_steps"),
        ({"reset_ratio": 0}, "reset_ratio"),
    ],
)
def test_bad_configs_fail_loudly(overrides: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        PQNRNNConfig(**overrides)


@pytest.mark.parametrize("key", ["observe", "seed", "typo_lr"])
def test_hooks_and_runner_fields_are_not_config_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        PQNRNNConfig.from_config({key: None})


def test_extra_inputs_widen_the_network_and_zero_features_equal_the_control(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    ctrl, wide = (
        build_network(pqn_cfg(extra_obs_dim=k), 43, SYMBOLIC_OBS) for k in (0, 5)
    )
    assert wide.obs_dim == SYMBOLIC_OBS + 5
    v = wide.init_variables(jax.random.key(0))
    assert v["params"]["Dense_0"]["kernel"].shape == (SYMBOLIC_OBS + 5, H)

    def narrow(x: jax.Array) -> jax.Array:
        return x[:SYMBOLIC_OBS] if jnp.ndim(x) and x.shape[0] == SYMBOLIC_OBS + 5 else x

    v_ctrl = jax.tree.map(narrow, v)
    obs = jax.random.normal(jax.random.key(1), (3, B, SYMBOLIC_OBS))
    wide_obs = jnp.concatenate([obs, jnp.zeros((3, B, 5))], -1)
    args = (ctrl.init_carry(B), jnp.zeros((3, B), bool), jnp.zeros((3, B), jnp.int32))
    for train in (False, True):
        (_, a), _ = ctrl.apply(
            v_ctrl, args[0], obs, *args[1:], train=train, mutable=["batch_stats"]
        )
        (_, b), _ = wide.apply(
            v, args[0], wide_obs, *args[1:], train=train, mutable=["batch_stats"]
        )
        np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.fixture(scope="module")
def one_step(pqn_cfg: Callable[..., PQNRNNConfig]) -> tuple[Any, ...]:
    """One minibatch, one epoch: ``td_loss`` is the loss at the initial variables."""
    cfg = pqn_cfg(minibatches=1, epochs=1)
    state = init_state(cfg, jax.random.key(0))
    _, traj, h0, _ = collect(state)
    return cfg, state.train_state, traj, h0


def test_loss_mask_selects_the_steps_that_enter_the_loss(
    one_step: tuple[Any, ...], same: Callable[[Any, Any], bool]
) -> None:
    cfg, ts, traj, h0 = one_step
    key = jax.random.key(5)
    plain = learn(ts, traj, key, h0=h0)
    assert same(learn(ts, traj, key, jnp.ones((T, B)), h0=h0), plain)
    assert not same(plain[0].params, ts.params)
    half = jnp.zeros((T, B)).at[::2].set(1.0)
    _, m = learn(ts, traj, key, half, h0=h0)
    (_, q), _ = ts.apply_fn(
        variables_of(ts),
        h0,
        traj.obs,
        _resets(traj.dones),
        traj.extra["last_actions"],
        train=True,
        mutable=["batch_stats"],
    )
    r = traj.rewards + traj.extra["r_int"]
    d = traj.dones.astype(jnp.float32)
    g = q_lambda_targets(q[1:].max(-1), r[:-1], d[:-1], cfg.gamma, cfg.lam)
    taken = jnp.take_along_axis(q, traj.actions[..., None], -1)[:-1, :, 0]
    keep = np.asarray(half[:-1], bool)
    want = 0.5 * np.mean(np.asarray(taken - g)[keep] ** 2)
    np.testing.assert_allclose(m["td_loss"], want, rtol=1e-5)
    frozen, _ = learn(ts, traj, key, jnp.zeros((T, B)), h0=h0)
    assert same(frozen.params, ts.params)


def test_learn_on_a_hand_built_traj_equals_one_train_chunk_update(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    """Only the fields `learn` reads: no hidden, events or Q-values."""
    state = init_state(cfg, jax.random.key(0))
    _, traj, h0, _ = collect(state)
    extra = {k: traj.extra[k] for k in ("r_int", "last_actions")}
    hand = Traj(traj.obs, traj.actions, traj.rewards, traj.dones, None, None, extra)
    hand, h0 = jax.device_put((hand, h0), device_roles().learner)
    _, k_learn, _ = jax.random.split(state.key, 3)
    ts, _ = learn(state.train_state, hand, k_learn, h0=h0)
    trained, _ = train_chunk(state, 1)
    assert same(ts, trained.train_state)


def _envs(tree: Any, n: int) -> Any:
    """Every env-axis leaf cut or tiled to ``n`` envs (time-major, axis 1)."""
    return jax.tree.map(lambda x: jnp.resize(x, (x.shape[0], n, *x.shape[2:])), tree)


@pytest.mark.parametrize("n", [2 * B, B // 2])
def test_learn_refuses_a_traj_of_another_env_count(
    one_step: tuple[Any, ...], n: int
) -> None:
    """Too many or too few envs: an error naming the shapes, never a silent drop
    or duplicate."""
    _, ts, traj, h0 = one_step
    key, other = jax.random.key(5), _envs(traj, n)
    with pytest.raises(ValueError, match=rf"\(8, {n}\).*\(8, 8\)"):
        learn(ts, other, key, h0=h0)
    other_h0 = jax.tree.map(lambda x: jnp.resize(x, (n, H)), h0)
    with pytest.raises(ValueError, match=rf"h0.*\({n}, {H}\)"):
        learn(ts, traj, key, h0=other_h0)
    with pytest.raises(ValueError, match=rf"loss_mask.*\(8, {n}\)"):
        learn(ts, traj, key, jnp.ones((T, n)), h0=h0)


def test_learn_without_h0_starts_from_the_float16_traj_hidden(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    """The fallback start carry is ``traj.hidden[0]``: the explicit float32 one
    rounded to float16; without a recorded hidden ``h0`` is required."""
    state = init_state(cfg, jax.random.key(0))
    state = state.replace(hidden=jax.tree.map(lambda x: x + 0.3, state.hidden))
    _, traj, h0, _ = collect(state, with_traj=True)
    key = jax.random.key(5)
    rounded = jax.tree.map(lambda x: x.astype(jnp.float16).astype(jnp.float32), h0)
    fallback, _ = learn(state.train_state, traj, key)
    assert same(fallback, learn(state.train_state, traj, key, h0=rounded)[0])
    exact, _ = learn(state.train_state, traj, key, h0=h0)
    for a, b in zip(jax.tree.leaves(fallback), jax.tree.leaves(exact), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-3)
    with pytest.raises(ValueError, match="h0"):
        learn(state.train_state, traj.replace(hidden=None), key)


def test_q_lambda_targets_run_without_a_trainer() -> None:
    script = (
        "import jax.numpy as jnp\n"
        "from wut_murl_craftax.training import pqn_rnn\n"
        "x = jnp.ones((3, 2))\n"
        "g = pqn_rnn.q_lambda_targets(x, x, x * 0, 0.9, 0.5)\n"
        "assert g.shape == (3, 2) and pqn_rnn._setup.cache_info().currsize == 0\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


def test_the_warm_up_rollout_runs_in_init_only(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    """num_steps random steps advance the episodes and the LSTM, not the env-step
    counter or the network; a resumed run equals the uninterrupted one."""
    state = init_state(cfg, jax.random.key(0))
    assert (int(state.env_steps), int(state.updates)) == (0, 0)
    assert float(state.eps) == 1.0
    assert int(np.asarray(state.episode.length).max()) == cfg.num_steps
    assert not np.allclose(state.hidden[1], 0.0)
    assert int(state.train_state.step) == 0
    stats = state.train_state.batch_stats["BatchRenorm_0"]
    assert int(stats["steps"]) == 0
    assert (np.asarray(stats["mean"]) == 0).all()
    one, _ = train_chunk(state, 1)
    assert int(one.env_steps) == cfg.batch
    save_state(tmp_path, one)
    restored = load_state(tmp_path, init_state(cfg, jax.random.key(1)))
    assert same(restored, one)
    resumed, _ = train_chunk(restored, 1)
    whole, _ = train_chunk(init_state(cfg, jax.random.key(0)), 2)
    assert same(resumed, whole)


def test_traj_holds_the_acting_carry_and_the_wrapper_events(
    start: PQNRNNState,
    rollout: tuple[Any, ...],
    cfg: PQNRNNConfig,
    net: RNNQNetwork,
) -> None:
    """``traj.hidden[t]``: the LSTM ``(c, h)`` step t acted from (zeros after a done);
    ``traj.events``: the wrapper's ``info["events"]``, replayed with the same keys."""
    _, traj, _, _ = rollout
    assert traj.hidden.dtype == jnp.float16
    assert traj.hidden.shape == (T, B, 2 * H)
    env, env_params = make_train_env(cfg)
    v, h, done = variables_of(start.train_state), start.hidden, start.last_done
    st, key = start.env_state, start.key_act
    no_reset = jnp.zeros((1, B), bool)
    for t in range(T):
        h = jax.tree.map(lambda x, done=done: jnp.where(done[:, None], 0.0, x), h)
        want = np.asarray(jnp.concatenate(h, -1).astype(jnp.float16), np.float32)
        np.testing.assert_allclose(traj.hidden[t], want, rtol=1e-3, atol=1e-3)
        last = traj.extra["last_actions"][t][None]
        h, _ = apply_network(net, v, h, traj.obs[t][None], no_reset, last)
        key, _, k_s = jax.random.split(key, 3)
        _, st, _, done, info = env.step(k_s, st, traj.actions[t], env_params)
        assert (np.asarray(done) == np.asarray(traj.dones[t])).all()
        step = jax.tree.map(lambda x, t=t: x[t], traj.events)
        for got, want in zip(
            jax.tree.leaves(step), jax.tree.leaves(info["events"]), strict=True
        ):
            assert (np.asarray(got) == np.asarray(want)).all()


def test_policy_replays_the_greedy_rollout_from_init_carry(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool]
) -> None:
    """From a fresh carry at epsilon 0 the rollout acts by argmax Q; ``step`` from
    ``init_carry`` with ``greedy`` takes the same actions until each env's done."""
    state = init_state(cfg, jax.random.key(4))
    fresh = state.replace(
        hidden=jax.tree.map(jnp.zeros_like, state.hidden),
        last_action=jnp.full_like(state.last_action, NO_ACTION),
        last_done=jnp.zeros_like(state.last_done),
        eps=jnp.zeros_like(state.eps),
    )
    _, traj, _, _ = collect(fresh)
    q_argmax = np.asarray(traj.extra["q_vals"]).argmax(-1)
    assert (np.asarray(traj.actions) == q_argmax).all()
    trainer = make_trainer(cfg)
    policy, params = trainer.policy(state), trainer.params(fresh)
    carry = policy.init_carry(B)
    zero = jnp.zeros((B, H))
    assert same(carry, ((zero, zero), jnp.full(B, NO_ACTION, jnp.int32)))
    step = jax.jit(policy.step, static_argnames="greedy")
    alive = np.ones(B, bool)
    for t in range(T):
        act, carry = step(params, carry, traj.obs[t], jax.random.key(t), greedy=True)
        assert (np.asarray(act)[alive] == q_argmax[t][alive]).all()
        alive &= ~np.asarray(traj.dones[t])


def test_checkpoints_hold_the_variables_and_act_at_their_epsilon(
    cfg: PQNRNNConfig, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    """``ckpt/`` holds params, BatchRenorm statistics and the update's epsilon; the
    policy of ``eval_shape(init)`` with loaded params acts as the live one."""
    trainer, config = make_trainer(cfg), {"method": "pqn-rnn", "steps": cfg.steps}
    state = trainer.init(config, jax.random.key(0))
    save_checkpoint(tmp_path, 0, 0, 0.0, trainer.params(state))
    trained, _ = trainer.train_chunk(state, cfg.n_updates)
    save_checkpoint(tmp_path, cfg.n_updates, cfg.steps, 1.0, trainer.params(trained))
    abstract = jax.eval_shape(lambda k: trainer.init(config, k), jax.random.key(0))
    policy, template = trainer.policy(abstract), trainer.params(abstract)
    first, last = (load_params(tmp_path, u, template) for u in (0, cfg.n_updates))
    assert float(first["eps"]) == 1.0
    np.testing.assert_allclose(float(last["eps"]), cfg.eps_finish, rtol=1e-6)
    stats = last["variables"]["batch_stats"]["BatchRenorm_0"]
    assert int(stats["steps"]) == cfg.n_updates * cfg.epochs * cfg.minibatches
    assert same(last, trainer.params(trained))
    live, obs = trainer.policy(trained), trained.obs
    for greedy in (True, False):
        want = live.step(
            trainer.params(trained), live.init_carry(B), obs, jax.random.key(1), greedy
        )
        got = policy.step(last, policy.init_carry(B), obs, jax.random.key(1), greedy)
        assert same(got, want)

    def acts(params: Any, greedy: bool) -> np.ndarray:
        keys = jax.random.split(jax.random.key(2), 200)
        step = jax.vmap(
            lambda k: policy.step(params, policy.init_carry(B), obs, k, greedy)[0]
        )
        return np.asarray(step(keys))

    off_greedy = [(acts(p, False) != acts(p, True)).mean() for p in (first, last)]
    assert off_greedy[0] > 0.85  # epsilon 1: uniform actions
    assert off_greedy[1] < 0.05  # epsilon 0.005


def test_registry_entry_trains_with_the_chunk_metric_shapes(cfg: PQNRNNConfig) -> None:
    trainer = METHODS["pqn-rnn"]
    config = {"method": "pqn-rnn", "env": cfg.env, "envs": B, "num_steps": T}
    config |= {"hidden": 64, "minibatches": 2, "steps": 4 * B * T}
    assert trainer.steps_per_update(config) == B * T
    state = trainer.init(config, jax.random.key(0))
    carry = trainer.policy(state).init_carry(3)
    assert [x.shape for x in jax.tree.leaves(carry)] == [(3, 64), (3, 64), (3,)]
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
        **dict.fromkeys(PQN_METRICS, (2,)),
    }
    losses = dict(zip(LOSS_COLUMNS, np.asarray(m["losses"]).T, strict=True))
    assert np.isfinite(losses.pop("value_loss")).all()
    assert np.isfinite(losses.pop("grad_norm")).all()
    assert all(np.isnan(v).all() for v in losses.values())  # no policy terms in PQN
    with pytest.raises(ValueError, match="steps"):
        trainer.init(
            {k: v for k, v in config.items() if k != "steps"}, jax.random.key(0)
        )


def test_done_step_unlocks_reach_the_episode_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared trainer contract: whole-episode flags with the done step's unlocks."""
    config = {"method": "pqn-rnn", "env": "Craftax-Classic-Symbolic-v1"}
    # a config no other test compiles, so the patched step events are traced; epsilon
    # decays over the whole budget, so the agent keeps moving and dies
    config |= {"envs": 16, "num_steps": 32, "hidden": 19, "minibatches": 2}
    config |= {"epochs": 1, "eps_decay": 1.0, "steps": 100 * 16 * 32}
    assert_done_step_unlock_counted(METHODS["pqn-rnn"], config, monkeypatch)


# --- metrics logged per update --------------------------------------------------


def _metrics(m: Any) -> dict[str, np.ndarray]:
    return {k: np.asarray(m[k]) for k in PQN_METRICS}


def test_a_chunk_returns_the_pqn_metrics_per_update(cfg: PQNRNNConfig) -> None:
    _, m = train_chunk(init_state(cfg, jax.random.key(0)), 3)
    got = _metrics(m)
    for name, values in got.items():
        assert values.shape == (3,), name
        assert np.isfinite(values).all(), name
    losses = dict(zip(LOSS_COLUMNS, np.asarray(m["losses"]).T, strict=True))
    np.testing.assert_array_equal(got["pqn/td_loss"], losses["value_loss"])
    np.testing.assert_array_equal(got["pqn/grad_norm"], losses["grad_norm"])
    assert (got["pqn/action_gap"] >= 0).all()
    assert (got["pqn/q_max_mean"] >= got["pqn/q_taken_mean"] - 1e-6).all()
    assert ((got["pqn/greedy_share"] >= 0) & (got["pqn/greedy_share"] <= 1)).all()
    assert (got["rnn/hidden_norm"] > 0).all()
    assert (got["reward/intrinsic"] == 0).all()  # the control has no hooks
    assert ((got["rnn/reset_share"] >= 0) & (got["rnn/reset_share"] <= 1)).all()


def test_epsilon_and_the_learning_rate_are_those_of_each_update(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    cfg = pqn_cfg(steps=10 * B * T)  # 10 updates, epsilon falls over the first one
    _, m = train_chunk(init_state(cfg, jax.random.key(0)), 3)
    want = [float(epsilon(cfg, jnp.int32(u))) for u in range(3)]
    np.testing.assert_allclose(m["pqn/epsilon"], want, rtol=1e-6)  # type: ignore[typeddict-item]
    q_steps = cfg.minibatches * cfg.epochs  # the schedule decays per Q step
    lr = [cfg.lr * (1 - u * q_steps / (cfg.n_updates * q_steps)) for u in range(3)]
    np.testing.assert_allclose(m["pqn/lr"], lr, rtol=1e-5)  # type: ignore[typeddict-item]
    flat = pqn_cfg(steps=10 * B * T, lr_linear_decay=False)
    _, m = train_chunk(init_state(flat, jax.random.key(0)), 2)
    np.testing.assert_allclose(m["pqn/lr"], [cfg.lr] * 2, rtol=1e-6)  # type: ignore[typeddict-item]


def test_a_greedy_behaviour_takes_the_greedy_action_every_step(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    cfg = pqn_cfg(eps_start=0.0, eps_finish=0.0)
    _, m = train_chunk(init_state(cfg, jax.random.key(0)), 2)
    assert np.asarray(m["pqn/greedy_share"]).tolist() == [1.0, 1.0]  # type: ignore[typeddict-item]
    assert np.asarray(m["pqn/epsilon"]).tolist() == [0.0, 0.0]  # type: ignore[typeddict-item]
    random = dataclasses.replace(cfg, eps_start=1.0)
    _, m = train_chunk(init_state(random, jax.random.key(0)), 1)
    assert float(m["pqn/greedy_share"][0]) < 0.5  # type: ignore[typeddict-item]


def test_the_q_statistics_are_means_over_the_steps_that_enter_the_loss(
    one_step: tuple[Any, ...],
) -> None:
    cfg, ts, traj, h0 = one_step
    _, m = learn(ts, traj, jax.random.key(5), h0=h0)
    (_, q), _ = ts.apply_fn(
        variables_of(ts),
        h0,
        traj.obs,
        _resets(traj.dones),
        traj.extra["last_actions"],
        train=True,
        mutable=["batch_stats"],
    )
    r = traj.rewards + traj.extra["r_int"]
    d = traj.dones.astype(jnp.float32)
    g = q_lambda_targets(q[1:].max(-1), r[:-1], d[:-1], cfg.gamma, cfg.lam)
    taken = jnp.take_along_axis(q, traj.actions[..., None], -1)[:-1, :, 0]
    top = jnp.sort(q[:-1], -1)
    want = {
        "q_taken_mean": taken.mean(),
        "q_max_mean": top[..., -1].mean(),
        "target_mean": g.mean(),
        "action_gap": (top[..., -1] - top[..., -2]).mean(),
    }
    for name, value in want.items():
        np.testing.assert_allclose(m[name], value, rtol=1e-5, atol=1e-7, err_msg=name)


def test_the_batch_renorm_metrics_follow_the_input_statistics(
    cfg: PQNRNNConfig,
) -> None:
    from wut_murl_craftax.training.batch_renorm import WARM_UP

    state = init_state(cfg, jax.random.key(0))
    _, m = train_chunk(state, 1)
    assert float(m["pqn/bn_warm"][0]) == 0.0  # type: ignore[typeddict-item]
    assert 0 < float(m["pqn/bn_var_min"][0]) <= 1.0  # type: ignore[typeddict-item]
    assert float(m["pqn/bn_dead_share"][0]) == 0.0  # type: ignore[typeddict-item]
    stats = state.train_state.batch_stats
    steps = stats["BatchRenorm_0"]["steps"]
    near = {
        **stats,
        "BatchRenorm_0": {**stats["BatchRenorm_0"], "steps": steps + WARM_UP - 1},
    }
    warm = state.replace(train_state=state.train_state.replace(batch_stats=near))
    _, m = train_chunk(warm, 1)  # the update's own Q steps take it past the warm-up
    assert float(m["pqn/bn_warm"][0]) == 1.0  # type: ignore[typeddict-item]
    var = stats["BatchRenorm_0"]["var"]
    small = {
        **stats,
        "BatchRenorm_0": {**stats["BatchRenorm_0"], "var": jnp.full_like(var, 1e-9)},
    }
    low = state.replace(train_state=state.train_state.replace(batch_stats=small))
    _, m = train_chunk(low, 1)
    assert float(m["pqn/bn_var_min"][0]) < 1e-6  # type: ignore[typeddict-item]
    # one update's 16 Q steps add (1 - 0.999) x the batch variance: only the inputs
    # that never vary in it stay dead
    share = float(m["pqn/bn_dead_share"][0])  # type: ignore[typeddict-item]
    assert 0.0 < share < 1.0


def test_the_rollout_stats_are_means_over_steps_and_envs(
    start: PQNRNNState,
) -> None:
    _, traj, _, _ = collect(start, with_traj=True)
    _, m = train_chunk(start, 1)
    dones = np.asarray(traj.dones)
    resets = np.concatenate([np.asarray(start.last_done)[None], dones[:-1]])
    np.testing.assert_allclose(m["rnn/reset_share"], [resets.mean()], rtol=1e-6)  # type: ignore[typeddict-item]
    np.testing.assert_allclose(  # type: ignore[call-overload]
        m["reward/extrinsic"], [np.asarray(traj.rewards).mean()], rtol=1e-6
    )


def test_a_hooked_reward_is_logged_apart_from_the_extrinsic_one(
    pqn_cfg: Callable[..., PQNRNNConfig],
) -> None:
    def reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
        return jnp.full(d.shape, 0.25, jnp.float32), s

    plain, hooked = (
        train_chunk(init_state(pqn_cfg(**extra), jax.random.key(0)), 1)[1]
        for extra in ({}, {"intrinsic_reward_fn": reward})
    )
    assert float(hooked["reward/intrinsic"][0]) == 0.25  # type: ignore[typeddict-item]
    assert float(hooked["reward/extrinsic"][0]) == float(plain["reward/extrinsic"][0])  # type: ignore[typeddict-item]


@pytest.mark.parametrize("name", ["pqn/td_loss", "rnn/hidden_norm", "reward/extrinsic"])
def test_a_method_cannot_reuse_a_control_metric_name(
    pqn_cfg: Callable[..., PQNRNNConfig], name: str
) -> None:
    state = init_state(pqn_cfg(metrics_fn=lambda s: {name: 1.0}), jax.random.key(0))
    with pytest.raises(ValueError, match="control metric"):
        train_chunk(state, 1)
