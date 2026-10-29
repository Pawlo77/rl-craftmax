"""Tests for the top-level SMDP table on a hand-computed 3-skill toy.

Toy: keys k0 = 3 and k1 = 5 of an 8-row table, gamma 0.9, alpha 0.5, the table zero
except ``Q[k1] = [2, 0, 1]``.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.training.ppo import make_train_env
from wut_murl_craftax.training.pqn_rnn import (
    PQNRNNConfig,
    QTrainState,
    _check_rollout,
    build_network,
    learn,
    optimizer,
)
from wut_murl_craftax.training.progress_key import hash_fields
from wut_murl_craftax.training.skills import (
    NUM_SKILLS,
    REASONS,
    SKILL_IDS,
    available,
    decode_obs,
    key_fields,
    low_stats,
    skill_step,
    step_limit,
)
from wut_murl_craftax.training.smdp import (
    EXTRA_DIM,
    MCRecords,
    RolloutCarry,
    SMDPConfig,
    SMDPPolicy,
    accumulate,
    init_rollout,
    init_table,
    mc_init,
    mc_push,
    mc_update,
    rollout_step,
    select_skill,
    smdp_update,
    to_traj,
    top_key,
)

K0, K1, GAMMA, ALPHA = 3, 5, 0.9, 0.5
ALL = jnp.ones(3, bool)


def toy() -> jax.Array:
    return init_table(3, 3, 0.0).at[K1].set(jnp.array([2.0, 0.0, 1.0]))


def env_a() -> tuple[jax.Array, jax.Array]:
    """``o0`` from k0 for 3 steps with native rewards 0, 0, 1."""
    ret, disc = jnp.float32(0.0), jnp.float32(1.0)
    for r in (0.0, 0.0, 1.0):
        ret, disc = accumulate(ret, disc, jnp.float32(r), GAMMA)
    return ret, disc


def update(
    q: jax.Array,
    ret: list[float],
    disc: list[float],
    done: list[bool],
    ended: list[bool],
    avail: jax.Array = ALL,
    k: list[int] | None = None,
    o: list[int] | None = None,
) -> jax.Array:
    n = len(ret)
    return smdp_update(
        q,
        jnp.array(k or [K0] * n),
        jnp.array(o or [0] * n),
        jnp.array(ret, jnp.float32),
        jnp.array(disc, jnp.float32),
        jnp.full(n, K1),
        jnp.tile(avail, (n, 1)),
        jnp.array(done),
        jnp.array(ended),
        ALPHA,
    )


def test_init_table() -> None:
    q = init_table(4, 25, 10.0)
    assert q.shape == (16, 25)
    assert q.dtype == jnp.float32
    assert float(q.min()) == float(q.max()) == 10.0
    with pytest.raises(ValueError, match="envs"):
        init_table(3, 25, 10.0, envs=8)  # rows would be read as the env axis


def test_accumulator() -> None:
    ret, disc = env_a()
    assert float(ret) == pytest.approx(0.81)
    assert float(disc) == pytest.approx(0.729)


def test_update_bootstraps_from_the_best_available_skill() -> None:
    ret, disc = env_a()
    q = update(toy(), [float(ret)], [float(disc)], [False], [True])
    assert float(q[K0, 0]) == pytest.approx(0.5 * 2.268)
    masked = jnp.array([False, True, True])
    q = update(toy(), [0.81], [0.729], [False], [True], avail=masked)
    assert float(q[K0, 0]) == pytest.approx(0.7695)


def test_death_does_not_bootstrap() -> None:
    q = update(toy(), [0.81], [0.729], [True], [True])
    assert float(q[K0, 0]) == pytest.approx(0.405)


def test_same_pair_in_one_step_takes_the_mean_td() -> None:
    q = update(toy(), [0.81, 0.0], [0.729, 0.729], [False, True], [True, True])
    assert float(q[K0, 0]) == pytest.approx(0.567)  # a plain scatter-add: 1.134


def test_running_skills_leave_the_table_unchanged() -> None:
    q0 = toy()
    q = update(q0, [0.81, 5.0], [0.729, 0.5], [False, False], [False, False])
    np.testing.assert_array_equal(q, q0)
    # an unfinished env on the same pair neither updates nor dilutes the mean
    q = update(q0, [0.81, 5.0], [0.729, 0.5], [False, False], [True, False])
    assert float(q[K0, 0]) == pytest.approx(1.134)
    mixed = update(q0, [0.81, 9.0], [0.729, 1.0], [True, False], [True, False],
                   k=[K0, K1], o=[0, 2])  # fmt: skip
    np.testing.assert_array_equal(np.asarray(mixed)[K1], np.asarray(q0)[K1])


def push(rec: MCRecords, ended: list[bool], k: list[int], o: list[int],
         r: list[float], disc: list[float]) -> MCRecords:  # fmt: skip
    return mc_push(rec, *(jnp.array(x) for x in (ended, k, o, r, disc)))


def test_monte_carlo_returns_without_bootstrap() -> None:
    rec = mc_init(envs=1, length=4)
    rec = push(rec, [True], [K0], [0], [0.81], [0.729])
    rec = push(rec, [True], [K1], [1], [1.0], [0.9])
    q, rec = mc_update(toy(), rec, jnp.array([True]), ALPHA)
    assert float(q[K0, 0]) == pytest.approx(0.7695)  # target 0.81 + 0.729 x 1
    assert float(q[K1, 1]) == pytest.approx(0.5)  # target 1
    assert int(rec.n[0]) == 0


def test_monte_carlo_push_and_update_run_under_jit() -> None:
    push_j, update_j = jax.jit(mc_push), jax.jit(mc_update, static_argnames="alpha")
    rec = mc_init(envs=1, length=4)
    for k, o, r, disc in ((K0, 0, 0.81, 0.729), (K1, 1, 1.0, 0.9)):
        rec = push_j(rec, *(jnp.array([x]) for x in (True, k, o, r, disc)))
    q, rec = update_j(toy(), rec, jnp.array([True]), alpha=ALPHA)
    assert float(q[K0, 0]) == pytest.approx(0.7695)  # as without jit
    assert float(q[K1, 1]) == pytest.approx(0.5)
    assert int(rec.n[0]) == 0


def test_monte_carlo_waits_for_the_episode_end() -> None:
    rec = mc_init(envs=2, length=4)
    rec = push(rec, [True, False], [K0, K0], [0, 0], [1.0, 1.0], [0.5, 0.5])
    q, rec2 = mc_update(toy(), rec, jnp.array([False, False]), ALPHA)
    np.testing.assert_array_equal(q, toy())
    np.testing.assert_array_equal(rec2.n, [1, 0])


def test_monte_carlo_keeps_the_last_records_when_full() -> None:
    """An episode longer than the buffer: the oldest records are dropped, the kept
    ones still get their exact returns."""
    rec = mc_init(envs=1, length=2)
    for k, o, r in ((K0, 0, 7.0), (K0, 1, 0.81), (K1, 1, 1.0)):
        rec = push(rec, [True], [k], [o], [r], [0.729])
    q, _ = mc_update(toy(), rec, jnp.array([True]), ALPHA)
    assert float(q[K0, 0]) == 0.0
    assert float(q[K0, 1]) == pytest.approx(0.7695)
    assert float(q[K1, 1]) == pytest.approx(0.5)


def test_select_skill_is_masked_epsilon_greedy() -> None:
    q = toy()
    avail = jnp.array([False, True, True])
    assert int(select_skill(q, jnp.int32(K1), avail, 0.0, jax.random.key(0))) == 2
    assert int(select_skill(q, jnp.int32(K1), ALL, 0.0, jax.random.key(0))) == 0
    keys = jax.random.split(jax.random.key(1), 10_000)
    draws = jax.vmap(lambda key: select_skill(q, jnp.int32(K1), avail, 1.0, key))(keys)
    assert set(np.unique(np.asarray(draws)).tolist()) == {1, 2}


def test_select_skill_breaks_ties_at_random() -> None:
    """With the optimistic start every entry is equal; greedy picks any of them."""
    q = init_table(3, 3, 10.0)
    keys = jax.random.split(jax.random.key(2), 1000)
    draws = jax.vmap(lambda key: select_skill(q, jnp.int32(0), ALL, 0.0, key))(keys)
    assert set(np.unique(np.asarray(draws)).tolist()) == {0, 1, 2}


def test_top_key_hashes_the_fields_and_low_stat_flags() -> None:
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    s = env.reset(jax.random.key(0), env.default_params)[1]
    v, low = (
        decode_obs(env.get_obs(s)),
        decode_obs(env.get_obs(s.replace(player_food=2))),
    )
    flags = low_stats(v).astype(jnp.int32)
    want = hash_fields(jnp.concatenate([key_fields(v, "log2"), flags]), 20)
    assert int(top_key(v, "log2", 20)) == int(want)
    assert int(top_key(low, "log2", 20)) != int(top_key(v, "log2", 20))
    no_flags = top_key(v, "log2", 20, low_flags=False)
    assert int(no_flags) == int(hash_fields(key_fields(v, "log2"), 20))
    assert int(top_key(low, "log2", 20, low_flags=False)) == int(no_flags)


def test_jit_and_vmap() -> None:
    q = jax.jit(smdp_update)(
        toy(), jnp.array([K0, K0]), jnp.array([0, 0]), jnp.array([0.81, 0.0]),
        jnp.array([0.729, 0.729]), jnp.array([K1, K1]), jnp.ones((2, 3), bool),
        jnp.array([False, True]), jnp.array([True, True]), ALPHA,
    )  # fmt: skip
    assert float(q[K0, 0]) == pytest.approx(0.567)
    keys = jax.random.split(jax.random.key(3), 4)
    ks = jnp.array([K1, K1, 0, 0])
    picks = jax.jit(jax.vmap(select_skill, in_axes=(None, 0, None, None, 0)))(
        toy(), ks, ALL, 0.0, keys
    )
    assert picks.tolist()[:2] == [0, 0]


# --- rollout step on the training wrapper chain ---

ENVS = 2
FIGHT, DESCEND = SKILL_IDS["fight"], SKILL_IDS["descend"]
NET_CFG = PQNRNNConfig(
    env="Craftax-Symbolic-v1", envs=ENVS, num_steps=4, minibatches=1, hidden=16,
    extra_obs_dim=EXTRA_DIM,
)  # fmt: skip
TOY = SMDPConfig(gamma_top=GAMMA, alpha=ALPHA, log2_rows=10)


def script_hook(state: Any, st: Any, st2: Any, obs2: Any, done: Any, t: Any) -> Any:
    """r+ = 1 on the third step (t counts env steps), and a call counter as state."""
    return jnp.where(t == 2 * ENVS, 1.0, 0.0) * jnp.ones(ENVS), state + 1


def zero_hook(state: Any, st: Any, st2: Any, obs2: Any, done: Any, t: Any) -> Any:
    return jnp.zeros(ENVS), state


class Rig:
    """The PQN-RNN training env chain, a 16-wide skill network and jitted steps."""

    def __init__(self) -> None:
        self.env, self.params = make_train_env(NET_CFG)
        self.net = build_network(NET_CFG, 43, 8268)
        self.variables = self.net.init_variables(jax.random.key(0))
        self._steps: dict[Any, Any] = {}

    def carry(self, cfg: SMDPConfig = TOY, seed: int = 0, **kw: Any) -> RolloutCarry:
        key = jax.random.key(seed)
        return init_rollout(
            cfg, self.env, self.net, self.variables, key, env_params=self.params, **kw
        )

    def run(self, carry: RolloutCarry, n: int, hook: Any = None) -> tuple[Any, Any]:
        """``n`` jitted steps; outputs stacked on a leading time axis."""
        if hook not in self._steps:
            self._steps[hook] = jax.jit(
                lambda c: rollout_step(
                    c, self.env, self.net, hook, env_params=self.params
                )
            )
        outs = []
        for _ in range(n):
            carry, out = self._steps[hook](carry)
            outs.append(out)
        return carry, jax.tree.map(lambda *x: jnp.stack(x), *outs)


@pytest.fixture(scope="module")
def rig() -> Rig:
    return Rig()


def force(c: RolloutCarry, skill: list[int], steps: list[int] | None = None) -> Any:
    """Set the running skills (and their step counts) of every env."""
    sk = c.per_env.skill.replace(
        skill=jnp.array(skill, jnp.int32),
        steps=jnp.array(steps or [0] * ENVS, jnp.int32),
    )
    return c.replace(per_env=c.per_env.replace(skill=sk))


def views(obs: jax.Array) -> Any:
    return jax.vmap(decode_obs)(obs[..., :8268])


def test_config_refuses_tiers_and_a_table_as_wide_as_the_envs(rig: Rig) -> None:
    with pytest.raises(ValueError, match="granularity"):
        SMDPConfig(granularity="tiers")
    with pytest.raises(ValueError, match="envs"):
        rig.carry(SMDPConfig(log2_rows=1))


def test_skill_state_accumulates_the_return_and_discount(rig: Rig) -> None:
    c, out = rig.run(force(rig.carry(method_state=jnp.int32(0)), [DESCEND] * 2), 3,
                     script_hook)  # fmt: skip
    assert not out["done"].any()
    r = np.asarray(out["native_reward"]) + np.array([[0.0], [0.0], [1.0]])
    want = r[0] + GAMMA * r[1] + GAMMA**2 * r[2]
    np.testing.assert_allclose(c.per_env.skill.ret, want, rtol=1e-6)
    np.testing.assert_allclose(c.per_env.skill.disc, [0.729, 0.729], rtol=1e-6)
    np.testing.assert_array_equal(c.per_env.skill.steps, [3, 3])
    assert not out["reward"].any()  # r+ never reaches the skill pseudo-reward
    assert int(c.shared.method_state) == 3


def test_a_skill_end_rechooses_from_the_new_observation(rig: Rig) -> None:
    c = rig.carry(eps_top=0.0)
    c = c.replace(shared=c.shared.replace(q=c.shared.q.at[:, FIGHT].set(100.0)))
    wood = SKILL_IDS["collect_wood"]
    limit = int(step_limit(wood, TOY.timeout_mult))
    c = force(c, [wood] * 2, [limit - 1] * 2)
    k0 = np.asarray(c.per_env.skill.k_start)
    c2, out = rig.run(c, 1)
    assert [REASONS[int(x)] for x in out["reason"][0]] == ["timeout"] * 2
    s = c2.per_env.skill
    np.testing.assert_array_equal(s.skill, [FIGHT, FIGHT])
    np.testing.assert_array_equal(s.steps, [0, 0])
    np.testing.assert_array_equal(s.ret, [0.0, 0.0])
    np.testing.assert_array_equal(s.disc, [1.0, 1.0])
    keys = jax.vmap(lambda v: top_key(v, "recipe", 10))(views(c2.per_env.obs))
    np.testing.assert_array_equal(s.k_start, keys)
    r = np.asarray(out["native_reward"][0])
    for i in range(ENVS):  # envs on one (key, skill) pair share the mean step
        want = 10 + ALPHA * np.mean((r + GAMMA * 100 - 10)[k0 == k0[i]])
        assert float(c2.shared.q[k0[i], wood]) == pytest.approx(want, rel=1e-6)


def test_an_episode_end_resets_and_does_not_bootstrap(rig: Rig) -> None:
    c = rig.carry()
    st = c.per_env.env_state
    inner = st.env_state.replace(
        timestep=st.env_state.timestep.at[0].set(rig.params.max_timesteps - 1)
    )
    c = force(c.replace(per_env=c.per_env.replace(
        env_state=st.replace(env_state=inner))), [DESCEND] * 2)  # fmt: skip
    c = c.replace(shared=c.shared.replace(q=c.shared.q.at[:, FIGHT].set(100.0)))
    k0 = int(c.per_env.skill.k_start[0])
    c2, out = rig.run(c, 1)
    assert bool(out["episode_done"][0, 0])
    assert not bool(out["episode_done"][0, 1])
    assert REASONS[int(out["reason"][0, 0])] == "episode_end"
    assert not bool(out["success"][0, 0])
    assert float(out["reward"][0, 0]) == 0.0
    r0 = float(out["native_reward"][0, 0])
    assert float(c2.shared.q[k0, DESCEND]) == pytest.approx(10 + ALPHA * (r0 - 10))
    s = c2.per_env.skill
    assert (int(s.steps[0]), float(s.ret[0]), float(s.disc[0])) == (0, 0.0, 1.0)
    first = jax.tree.map(lambda x: x[0], views(c2.per_env.obs))
    assert int(s.k_start[0]) == int(top_key(first, "recipe", 10))
    assert int(s.steps[1]) == 1
    assert bool(c2.per_env.last_done[0])


def test_a_primitive_option_takes_one_unlearned_step(rig: Rig) -> None:
    c = force(rig.carry(), [SKILL_IDS["read_book"], DESCEND])
    _, out = rig.run(c, 1)
    assert int(out["action"][0, 0]) == 35
    np.testing.assert_array_equal(out["done"][0], [True, False])
    np.testing.assert_array_equal(out["loss_mask"][0], [False, True])


def test_extra_input_is_the_skill_and_its_step_fraction(rig: Rig) -> None:
    c = force(rig.carry(), [DESCEND, FIGHT], [250, 30])
    _, out = rig.run(c, 1)
    extra = np.asarray(out["obs"][0, :, 8268:])
    assert extra.shape == (ENVS, EXTRA_DIM)
    np.testing.assert_array_equal(extra[:, :NUM_SKILLS].argmax(1), [DESCEND, FIGHT])
    assert extra[:, :NUM_SKILLS].sum() == ENVS
    np.testing.assert_allclose(extra[:, -1], [250 / 1000, 30 / 300])


def test_ends_come_from_the_skills_functions(rig: Rig) -> None:
    c, out = rig.run(rig.carry(eps=1.0, eps_top=1.0), 8)
    obs = jnp.concatenate([out["obs"][:, :, :8268], c.per_env.obs[None]])
    for t in range(8):
        a, b = views(obs[t]), views(obs[t + 1])
        got = jax.vmap(
            lambda k, va, act, r, vb, d, n: skill_step(k, va, act, r, vb, d, n, 1.0)
        )(out["skill"][t], a, out["action"][t], out["native_reward"][t], b,
          out["episode_done"][t], out["steps_in_skill"][t])  # fmt: skip
        np.testing.assert_array_equal(got[0], out["success"][t])
        np.testing.assert_array_equal(got[1], out["done"][t])
        np.testing.assert_array_equal(got[2], out["reason"][t])
        # a skill is chosen from the available ones where the last one ended (t = 0
        # runs the carry's starting skill); in between only its end is tested
        chosen = jax.vmap(available)(a)[jnp.arange(ENVS), out["skill"][t]]
        decided = out["done"][t - 1] if t else jnp.zeros(ENVS, bool)
        assert bool((chosen | ~decided).all())
    ends = np.asarray(c.shared.ends)
    assert ends.shape == (NUM_SKILLS, len(REASONS))
    assert ends.sum() == int(out["done"].sum())
    agree = np.asarray(c.shared.agreement)
    assert (agree[:, 1] <= agree[:, 0]).all()
    assert agree[:, 0].sum() <= ends.sum()


def test_same_key_same_skills(rig: Rig) -> None:
    runs = [rig.run(rig.carry(seed=3, eps=1.0, eps_top=1.0), 6) for _ in range(2)]
    (c1, o1), (c2, o2) = runs
    np.testing.assert_array_equal(o1["skill"], o2["skill"])
    np.testing.assert_array_equal(o1["action"], o2["action"])
    np.testing.assert_array_equal(c1.shared.q, c2.shared.q)


def test_no_hook_is_bit_identical_to_a_zero_hook(
    rig: Rig, same: Callable[[Any, Any], bool]
) -> None:
    c = rig.carry(seed=4, eps=1.0, eps_top=1.0, method_state=jnp.int32(0))
    assert same(rig.run(c, 5), rig.run(c, 5, zero_hook))


def test_monte_carlo_mode_waits_for_the_episode(rig: Rig) -> None:
    cfg = SMDPConfig(gamma_top=GAMMA, alpha=ALPHA, log2_rows=10, monte_carlo=True,
                     mc_length=8)  # fmt: skip
    wood = SKILL_IDS["collect_wood"]
    c = force(rig.carry(cfg), [wood] * 2, [int(step_limit(wood, 1.0)) - 1] * 2)
    c2, out = rig.run(c, 1)
    assert out["done"].all()
    np.testing.assert_array_equal(c2.shared.q, c.shared.q)
    np.testing.assert_array_equal(c2.per_env.mc.n, [1, 1])


def test_the_transition_fits_pqn_rnn_learn(rig: Rig) -> None:
    _, out = rig.run(rig.carry(eps=1.0, eps_top=1.0), NET_CFG.num_steps)
    traj, mask = to_traj(out)
    _check_rollout(NET_CFG, traj, mask, rig.net.init_carry(ENVS))
    assert traj.obs.shape == (4, ENVS, 8268 + EXTRA_DIM)
    assert traj.rewards.dtype == jnp.float32
    np.testing.assert_array_equal(traj.dones, out["done"])
    np.testing.assert_array_equal(traj.extra["episode_dones"], out["episode_done"])
    v = rig.variables
    ts = QTrainState.create(  # type: ignore[no-untyped-call]
        apply_fn=rig.net.apply, params=v["params"], tx=optimizer(NET_CFG),
        batch_stats=v["batch_stats"], cfg=NET_CFG,
    )  # fmt: skip
    ts2, m = learn(ts, traj, jax.random.key(5), mask, rig.net.init_carry(ENVS))
    assert np.isfinite(float(m["td_loss"]))
    assert int(ts2.step) == NET_CFG.epochs * NET_CFG.minibatches


def test_an_episode_end_is_not_judged_by_the_events(rig: Rig) -> None:
    """The done step forces success off, so `test_agreement` skips it (``fight`` is
    judged at every other end); the step's events stay out of the outputs."""
    c = rig.carry()
    st = c.per_env.env_state
    inner = st.env_state.replace(
        timestep=st.env_state.timestep.at[0].set(rig.params.max_timesteps - 1)
    )
    c = force(c.replace(per_env=c.per_env.replace(
        env_state=st.replace(env_state=inner))), [FIGHT] * 2)  # fmt: skip
    c2, out = rig.run(c, 1)
    assert bool(out["episode_done"][0, 0])
    assert REASONS[int(out["reason"][0, 1])] == "running"
    np.testing.assert_array_equal(c2.shared.agreement[FIGHT], [0, 0])
    assert int(c2.shared.ends[FIGHT].sum()) == 1
    assert {"unlocked", "events"}.isdisjoint(out)


def test_a_network_without_the_skill_inputs_is_refused(rig: Rig) -> None:
    narrow = build_network(NET_CFG, 43, 8268 - EXTRA_DIM)
    with pytest.raises(ValueError, match="extra_obs_dim=26"):
        init_rollout(
            TOY, rig.env, narrow, None, jax.random.key(0), env_params=rig.params
        )


@pytest.mark.parametrize("flags", [True, False])
def test_the_low_stat_interruption_follows_the_flags(rig: Rig, flags: bool) -> None:
    """Food falls to 3 on the next step: a running skill ends there with the flags,
    and runs on in the no-flags ablation, in the rollout and in the policy."""
    cfg = SMDPConfig(gamma_top=GAMMA, alpha=ALPHA, log2_rows=10, low_flags=flags)
    c = rig.carry(cfg)
    st = c.per_env.env_state
    hungry = st.env_state.replace(
        player_food=jnp.full(ENVS, 4, jnp.int32), player_hunger=jnp.full(ENVS, 25.0)
    )
    c = force(c.replace(per_env=c.per_env.replace(
        env_state=st.replace(env_state=hungry))), [DESCEND] * 2)  # fmt: skip
    obs0 = c.per_env.obs
    c2, out = rig.run(c, 1)
    reason = "low_stat" if flags else "running"
    assert [REASONS[int(x)] for x in out["reason"][0]] == [reason] * ENVS
    policy = SMDPPolicy(rig.net, cfg)
    pc = policy.init_carry(ENVS)
    act = out["action"][0]
    pc = (pc[0], act, jnp.full(ENVS, DESCEND), jnp.zeros(ENVS, jnp.int32), obs0,
          jnp.ones(ENVS, bool))  # fmt: skip
    rows = {"rows": np.zeros(0, np.int32), "q_rows": np.zeros((0, NUM_SKILLS))}
    params = rows | {"variables": rig.variables, "eps": 0.0, "eps_top": 0.0}
    r, d = out["native_reward"][0], out["episode_done"][0]
    _, pc2 = policy.step(params, pc, c2.per_env.obs, jax.random.key(0), True, r, d)
    np.testing.assert_array_equal(pc2[3], 0 if flags else 1)  # steps in the skill
