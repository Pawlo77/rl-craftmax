"""Tests for SIL sampling and loss: slot and priority draws, loss, stored replays."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import make_train_env
from wut_murl_craftax.training.ppo_rnn import (
    PPORNNConfig,
    build_network,
    collect,
    init_state,
)
from wut_murl_craftax.training.records import achievement_names
from wut_murl_craftax.training.sil import (
    SILConfig,
    SILState,
    after_extra,
    buffer_bytes,
    extra_loss,
    init,
    metrics,
    observe,
    replay,
    sample,
    sample_rows,
    sil_loss,
    sil_terms,
)

A, D, DRAWS = 22, 1345, 10_000


def _filled(cfg: SILConfig, windows: dict[int, list[float]]) -> SILState:
    """Buffer holding a window of priority p for each p in ``windows[achievement]``."""
    st = init(cfg, 2, A, D)
    buf = {k: np.asarray(st.buffer[k]).copy() for k in ("stamp", "prio", "ach")}
    n = 0
    for a, prios in windows.items():
        for j, p in enumerate(prios):
            row = a * st.slot_size + j if cfg.layout == "slots" else n
            buf["stamp"][row], buf["prio"][row], buf["ach"][row] = n, p, a
            n += 1
    return st.replace(buffer=st.buffer | {k: jnp.asarray(v) for k, v in buf.items()})


def _share(x: np.ndarray, values: list[int]) -> list[float]:
    return [float(np.mean(x == v)) for v in values]


@pytest.mark.parametrize(
    ("layout", "want"), [("slots", [0.5, 0.5]), ("single", [0.25, 0.75])]
)
def test_slot_frequencies(layout: str, want: list[float]) -> None:
    st = _filled(SILConfig(slots_windows=3, layout=layout), {0: [1.0], 1: [1.0] * 3})
    rows = sample_rows(st, jax.random.key(0), DRAWS)
    ach = np.asarray(st.buffer["ach"])[np.asarray(rows)]
    np.testing.assert_allclose(_share(ach, [0, 1]), want, atol=0.02)


def test_window_frequencies_follow_priority_to_alpha() -> None:
    prios = [1.0, 2.0, 4.0]
    st = _filled(SILConfig(slots_windows=3), {1: prios})
    rows = np.asarray(sample_rows(st, jax.random.key(1), DRAWS))
    want = np.array(prios) ** 0.6
    np.testing.assert_allclose(_share(rows, [3, 4, 5]), want / want.sum(), atol=0.02)


def test_two_step_loss_matches_the_hand_value() -> None:
    cfg = SILConfig(lambda_sil=0.5, beta_v=0.01)
    logp, v = jnp.array([-1.0, -2.0, -3.0]), jnp.array([0.5, 1.0, 0.0])
    ret, mask = jnp.array([1.5, 0.5, 9.0]), jnp.array([True, True, False])
    loss, aux = sil_terms(logp, v, ret, mask, cfg)
    # (R - V)+ = [1, 0] on the two valid steps; the masked third step is left out
    policy, value = (1.0 * 1.0 + 0.0) / 2, (0.5 * 1.0**2 + 0.0) / 2
    assert float(loss) == pytest.approx(0.5 * (policy + 0.01 * value))
    assert (float(aux["policy"]), float(aux["value"])) == pytest.approx((policy, value))
    assert float(aux["active"]) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def replayed(rnn_cfg: Callable[..., PPORNNConfig]) -> dict[str, Any]:
    """One window stored by `observe` from real PPO-RNN rollouts, unchanged params.

    Unlocks are cleared except one, on env 0 at the last step, so the buffer holds
    exactly the last 128 steps of env 0 (staged rollouts plus the current one).
    """
    cfg = rnn_cfg()
    ppo = init_state(cfg, jax.random.key(0))
    env, env_params = make_train_env(cfg)
    obs_dim = env.observation_space(env_params).shape[0]
    n_ach = len(achievement_names(cfg.env))
    sil_cfg = SILConfig()
    names = tuple(achievement_names(cfg.env))
    kw = {"gamma": cfg.gamma, "rnn_width": cfg.rnn_width, "names": names}
    st = init(sil_cfg, cfg.envs, n_ach, obs_dim, **kw)
    n = sil_cfg.window // cfg.num_steps + 1
    trajs = []
    for k in range(n):
        ppo, _, traj, _ = collect(ppo, with_traj=True)
        unl = jnp.zeros_like(traj.events.unlocked).at[-1, 0, 0].set(k == n - 1)
        traj = traj.replace(events=traj.events.replace(unlocked=unl))
        st = jax.jit(observe)(st, traj)
        trajs.append(traj)
    cat = jax.tree.map(lambda *x: jnp.concatenate(x)[-sil_cfg.window :, 0], *trajs)
    net = build_network(cfg, env.action_space(env_params).n, obs_dim)
    dims = {"n_ach": n_ach, "obs_dim": obs_dim}
    return {
        "net": net,
        "params": ppo.train_state.params,
        "state": st,
        "traj": cat,
    } | dims


def test_replay_from_the_stored_state_gives_the_rollout_logits(
    replayed: dict[str, Any],
) -> None:
    st, tr = replayed["state"], replayed["traj"]
    assert int((st.buffer["stamp"] >= 0).sum()) == 1
    w = sample(st, jax.random.key(2))
    np.testing.assert_array_equal(np.asarray(w["obs"][:, 0]), np.asarray(tr.obs))
    np.testing.assert_array_equal(np.asarray(w["act"][:, 0]), np.asarray(tr.actions))
    logits, v = replay(replayed["net"], replayed["params"], w)
    logp = jax.nn.log_softmax(logits[:, 0])[jnp.arange(len(tr.actions)), tr.actions]
    # the stored start state is float16; the rollout acted from float32
    np.testing.assert_allclose(logp, tr.extra["logp"], atol=2e-3)
    np.testing.assert_allclose(v[:, 0], tr.extra["values"], atol=2e-3)


def _grad(net: Any, params: Any, st: SILState, key: jax.Array) -> Any:
    """The learner's call: ``value_and_grad`` of ``extra_loss`` bound to ``net``."""
    fn = jax.value_and_grad(functools.partial(extra_loss, net), has_aux=True)
    return jax.jit(fn)(params, st, key)


def _is_zero(tree: Any) -> bool:
    return all(bool((x == 0).all()) for x in jax.tree.leaves(tree))


def test_loss_and_gradient_vanish_where_the_value_beats_the_return(
    replayed: dict[str, Any],
) -> None:
    net, params, st = replayed["net"], replayed["params"], replayed["state"]
    w = sample(st, jax.random.key(3))
    w = w | {"ret": jnp.full_like(w["ret"], -1e3)}
    loss, g = jax.value_and_grad(lambda p: sil_loss(net, st.cfg, p, w)[0])(params)
    assert float(loss) == 0.0
    assert _is_zero(g)


def test_sampled_priorities_are_rewritten(replayed: dict[str, Any]) -> None:
    net, params, st = replayed["net"], replayed["params"], replayed["state"]
    (row,) = np.flatnonzero(np.asarray(st.buffer["stamp"]) >= 0)
    # a return of 1 at every step beats the fresh critic, so SIL has work to do
    st = st.replace(buffer=st.buffer | {"ret": st.buffer["ret"].at[row].set(1.0)})
    (_, (st2, metrics)), g = _grad(net, params, st, jax.random.key(4))
    w = sample(st, jax.random.key(4))
    _, v = replay(net, params, w)
    adv = np.maximum(np.asarray(w["ret"][:, 0] - v[:, 0]), 0.0)
    want = adv[np.asarray(w["mask"][:, 0])].mean() + 0.01
    prio = np.asarray(st2.buffer["prio"])
    assert prio[row] == pytest.approx(want, rel=1e-4)
    assert prio[row] != pytest.approx(float(st.buffer["prio"][row]))
    np.testing.assert_array_equal(np.delete(prio, row), 0.0)
    assert want > 0.1
    assert float(st2.max_prio) == pytest.approx(max(1.0, want))
    assert all(name.startswith("sil/") for name in metrics)
    assert not _is_zero(g)


def test_empty_buffer_gives_zero_loss_and_finite_metrics(
    replayed: dict[str, Any],
) -> None:
    net, params, st = replayed["net"], replayed["params"], replayed["state"]
    envs = st.staging["done"].shape[0]
    width = st.staging["hidden"].shape[-1]
    empty = init(st.cfg, envs, replayed["n_ach"], replayed["obs_dim"], rnn_width=width)
    (loss, (st2, metrics)), g = _grad(net, params, empty, jax.random.key(5))
    assert float(loss) == 0.0
    assert _is_zero(g)
    assert metrics
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    np.testing.assert_array_equal(st2.buffer["prio"], empty.buffer["prio"])
    assert float(st2.max_prio) == float(empty.max_prio)


# --- metrics -------------------------------------------------------------------------


def test_sample_metrics_match_the_hand_values(replayed: dict[str, Any]) -> None:
    """One stored window (inserted at update 3, now 7): every draw is it."""
    net, params, st = replayed["net"], replayed["params"], replayed["state"]
    (row,) = np.flatnonzero(np.asarray(st.buffer["stamp"]) >= 0)
    born = st.buffer["born"].at[row].set(3)
    st = st.replace(buffer=st.buffer | {"born": born}, update=jnp.int32(7))
    (_, (_, m)), _ = _grad(net, params, st, jax.random.key(6))
    assert float(m["sil/sample_age_mean"]) == float(m["sil/sample_age_max"]) == 4.0
    w = sample(st, jax.random.key(6))
    _, v = replay(net, params, w)
    mask = np.asarray(w["mask"])
    bias = (np.asarray(v) - np.asarray(w["ret"]))[mask].mean()
    assert float(m["sil/value_bias"]) == pytest.approx(bias, rel=1e-4, abs=1e-6)
    first = st.names[int(st.buffer["ach"][row])]
    shares = {k: float(x) for k, x in m.items() if k.startswith("sil/sample_share/")}
    assert len(shares) == replayed["n_ach"]
    assert shares.pop(f"sil/sample_share/{first}") == 1.0
    assert set(shares.values()) == {0.0}


def test_buffer_metrics_count_windows_per_achievement() -> None:
    cfg = SILConfig(slots_windows=3)
    st = _filled(cfg, {0: [1.0], 1: [1.0] * 3, 5: [2.0, 1.0]})
    m = {k: float(v) for k, v in metrics(st).items()}
    fill = {k.removeprefix("sil/slot_fill/"): v for k, v in m.items() if "fill" in k}
    assert fill == {str(a): {0: 1, 1: 3, 5: 2}.get(a, 0) for a in range(A)}
    assert m["sil/nonempty_slots"] == 3
    size = buffer_bytes(cfg, 2, A, D)
    assert (m["sil/buffer_bytes"], m["sil/staging_bytes"]) == (
        size.buffer,
        size.staging,
    )


SMALL = {
    "env": "Craftax-Classic-Symbolic-v1",
    "envs": 16,
    "num_steps": 32,
    "minibatches": 2,
    "update_epochs": 1,
    "width": 16,
    "rnn_width": 16,
    "steps": 4 * 512,
    "lambda_sil": 0.3,
    "slots_windows": 2,
    "sil_steps": 2,
    "window": 128,
    "beta_v": 0.01,
    "batch": 32,
    "alpha": 0.6,
    "insert_cap": 64,
    "layout": "slots",
}
"""A SIL run config on Classic, 512 env steps per update: random play unlocks."""
NAMES = [
    "inserted",
    "overflow",
    "nonempty_slots",
    "buffer_bytes",
    "staging_bytes",
    "active",
    "adv_pos_mean",
    "policy_loss",
    "value_loss",
    "value_bias",
    "sample_age_mean",
    "sample_age_max",
    "grad_norm",
    "kl_after",
    "dominates",
]
"""The ``sil/`` metrics besides the per-achievement ones."""


def test_updates_log_every_metric_and_slots_never_overfill() -> None:
    trainer = METHODS["sil"]
    state, m = trainer.train_chunk(trainer.init(SMALL, jax.random.key(0)), 3)
    names = achievement_names(SMALL["env"])
    method = {k for k in m if k not in {"losses", "done", "ach", "ret", "length"}}
    assert all(k.startswith("sil/") for k in method)
    assert method == {f"sil/{n}" for n in NAMES} | {
        f"sil/{group}/{a}" for group in ("slot_fill", "sample_share") for a in names
    }
    assert all(np.isfinite(np.asarray(m[k], float)).all() for k in method)
    fill = np.stack([np.asarray(m[f"sil/slot_fill/{a}"]) for a in names])
    assert fill.max() <= 2
    assert fill[:, -1].sum() == int((state.method_state.buffer["stamp"] >= 0).sum()) > 0
    assert (np.asarray(m["sil/nonempty_slots"]) == (fill > 0).sum(0)).all()


def test_the_dominance_guard_turns_nan_after_20_updates_over_the_limit() -> None:
    st = _filled(SILConfig(), {0: [1.0]})
    over, under = {"sil/kl_after": jnp.float32(0.6)}, {"sil/kl_after": jnp.float32(0.4)}
    for _ in range(19):
        st = after_extra(st, over)
    assert float(metrics(st)["sil/dominates"]) == 0.0
    assert np.isnan(float(metrics(after_extra(st, over))["sil/dominates"]))
    st = after_extra(after_extra(st, under), over)  # a dip starts the count again
    assert int(st.kl_streak) == 1
    assert float(metrics(st)["sil/dominates"]) == 0.0
