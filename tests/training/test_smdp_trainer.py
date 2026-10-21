"""Tests for the SMDP skills registry entry: config file, trainer, policy, runner.

Tiny Craftax-Classic runs (8 envs x 8 steps, a 16-wide skill network, a 2^12-row
table) on the PQN-RNN control's tuned block of a temporary method folder.
"""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from _contract import assert_done_step_unlock_counted

from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import smdp
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.pqn_rnn import (
    PQNRNNConfig,
    _learn_batch,
    q_lambda_targets,
)
from wut_murl_craftax.training.progress_key import CLASSIC_FIELDS
from wut_murl_craftax.training.records import LOSS_COLUMNS
from wut_murl_craftax.training.skills import (
    CLASSIC_OBS,
    LOW_STAT,
    NUM_SKILLS,
    REASONS,
    SKILL_IDS,
    SKILLS,
    SUCCESS,
    TIMEOUT,
)
from wut_murl_craftax.visualisation import eval_rollouts

REPO = Path(__file__).resolve().parents[2]
CLASSIC = "Craftax-Classic-Symbolic-v1"
ENVS, NUM_STEPS, N_ACH = 8, 8, 22
TINY = {
    "env": CLASSIC,
    "envs": ENVS,
    "num_steps": NUM_STEPS,
    "minibatches": 2,
    "hidden": 16,
}
"""Tuned block of the temporary PQN-RNN control copy."""
STEPS = 2 * ENVS * NUM_STEPS
"""Budget of every tiny run: 2 updates, as the runner's 0.128k phase."""
NAME = "smdp-skills"
TOP_KNOBS = {
    "alpha",
    "top_horizon",
    "eps_top_decay",
    "eps_top_end",
    "timeout_mult",
    "granularity",
    "log2_rows",
    "q_init",
    "low_flags",
    "monte_carlo",
    "mc_length",
}
"""The method's own run-config keys (`smdp.TOP_KEYS`)."""


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of configs/methods with the PQN-RNN control's ``tuned`` filled in."""
    out = tmp_path_factory.mktemp("methods")
    for path in (REPO / "configs" / "methods").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if path.stem == "pqn-rnn":
            tuned = yaml.safe_dump({"tuned": TINY}, sort_keys=False)
            text = text.replace("\ntuned: null\n", "\n" + tuned)
        (out / path.name).write_text(text, encoding="utf-8")
    return out


def _config(methods_dir: Path, ref: str = "base", **over: Any) -> dict[str, Any]:
    """A resolved run config: tiny skill network, 2^12 table rows, 2-update budget."""
    method = load_method(NAME, methods_dir)
    cfg = resolve_config(method, NAME, "1m", ref=ref, methods_dir=methods_dir)
    return cfg | {"log2_rows": 12, "steps": STEPS} | over


# --- config file -------------------------------------------------------------------


def test_the_method_file_has_the_base_and_both_grids() -> None:
    method = load_method(NAME)
    assert method.registry_names == [NAME]
    assert method.base["base_from"] == "pqn-rnn.tuned"
    assert method.tuned is None
    # every method knob is set in the base, so no dataclass default decides a run
    assert method.base.keys() - {"base_from", "extra_obs_dim"} == TOP_KNOBS
    assert method.base["extra_obs_dim"] == smdp.EXTRA_DIM == NUM_SKILLS + 1
    knobs = ("alpha", "top_horizon", "eps_top_decay", "timeout_mult", "granularity")
    grid = [tuple(e[k] for k in knobs) for e in method.coarse]
    want = itertools.product(
        (0.05, 0.2), (200, 1000), (0.1, 0.3), (0.5, 1.0, 2.0), ("recipe", "log2")
    )
    assert len(method.coarse) == len(set(grid)) == 48
    assert set(grid) == set(want)
    assert all(e.keys() == {"name", *knobs} for e in method.coarse)
    fine = method.fine
    assert fine is not None
    assert fine["rule"] == {"select": 2, "combine": "cross"}
    numeric = {"alpha", "top_horizon", "eps_top_decay", "timeout_mult"}
    assert {k["key"] for k in fine["knobs"]} == numeric
    assert all(k["mul"] == [0.5, 1.0, 2.0] for k in fine["knobs"])


def test_the_horizon_knob_sets_the_top_level_gamma(methods_dir: Path) -> None:
    """Stage 2 scales gamma through its horizon: 1000 x {0.5, 1, 2} is 0.998, 0.999,
    0.9995."""
    trainer = METHODS[NAME]
    gammas = [
        trainer.config(_config(methods_dir, top_horizon=h)).top.gamma_top
        for h in (500, 1000, 2000)
    ]
    np.testing.assert_allclose(gammas, [0.998, 0.999, 0.9995])
    cfg = trainer.config(_config(methods_dir, ref="a0.2-h1000-e0.1-t1-log2"))
    assert (cfg.top.alpha, cfg.top.granularity, cfg.top.timeout_mult) == (
        0.2,
        "log2",
        1.0,
    )
    assert cfg.net.extra_obs_dim == smdp.EXTRA_DIM
    assert cfg.net.hidden == 16
    with pytest.raises(ValueError, match="extra_obs_dim"):
        trainer.config(_config(methods_dir, extra_obs_dim=0))
    with pytest.raises(ValueError, match="gamma_top"):
        trainer.config(_config(methods_dir, gamma_top=0.99))


# --- Q(lambda) over skill ends ------------------------------------------------------


def test_q_lambda_targets_stop_at_a_skill_end() -> None:
    """One env, 4 steps, a skill end at step 2 (gamma 0.99, lambda 0.5).

    ``G_2 = r_2 = 1`` (no bootstrap from the next skill's state),
    ``G_1 = 0.99 (0.5 x 2 + 0.5 x 1) = 1.485``,
    ``G_0 = 0.99 (0.5 x 1 + 0.5 x 1.485) = 1.230075``.
    """
    q_next = jnp.array([[1.0], [2.0], [3.0]])
    r = jnp.array([[0.0], [0.0], [1.0]])
    d = jnp.array([[0.0], [0.0], [1.0]])
    g = q_lambda_targets(q_next, r, d, 0.99, 0.5)
    np.testing.assert_allclose(g[:, 0], [1.230075, 1.485, 1.0], rtol=1e-6)


def test_the_learner_cuts_targets_at_skill_ends_and_resets_at_episode_ends() -> None:
    """R42: ``dones`` are skill ends, the LSTM resets only after episode ends."""
    cfg = PQNRNNConfig(**TINY)
    skill_end = jnp.zeros((NUM_STEPS, ENVS), bool).at[2].set(True)
    episode_end = jnp.zeros((NUM_STEPS, ENVS), bool).at[5, 0].set(True)
    out = {
        "obs": jnp.zeros((NUM_STEPS, ENVS, 4)),
        "action": jnp.zeros((NUM_STEPS, ENVS), jnp.int32),
        "reward": jnp.zeros((NUM_STEPS, ENVS)),
        "done": skill_end | episode_end,
        "loss_mask": jnp.ones((NUM_STEPS, ENVS), bool),
        "last_action": jnp.zeros((NUM_STEPS, ENVS), jnp.int32),
        "episode_done": episode_end,
    }
    traj, mask = smdp.to_traj(out)
    h0 = (jnp.zeros((ENVS, 16)), jnp.zeros((ENVS, 16)))
    b = _learn_batch(cfg, traj, mask, h0)
    np.testing.assert_array_equal(b["d"], (skill_end | episode_end).astype(float))
    want = jnp.zeros((NUM_STEPS, ENVS), bool).at[6, 0].set(True)
    np.testing.assert_array_equal(b["reset"], want)


# --- the trainer interface ------------------------------------------------------------


@pytest.fixture(scope="module")
def entry(methods_dir: Path) -> Any:
    """(trainer, config, start state, 2-update result) of the entry."""
    trainer = METHODS[NAME]
    cfg = _config(methods_dir)
    start = trainer.init(cfg, jax.random.key(0))
    return trainer, cfg, start, trainer.train_chunk(start, 2)


def test_the_entry_exposes_the_trainer_interface(entry: Any) -> None:
    trainer, cfg, _, _ = entry
    names = ("steps_per_update", "init", "train_chunk", "params", "policy")
    assert all(callable(getattr(trainer, n)) for n in (*names, "env_wrapper"))
    assert trainer.steps_per_update(cfg) == ENVS * NUM_STEPS
    assert trainer.loss_columns == ("value_loss", "grad_norm")


def test_two_updates_report_the_chunk_metrics(entry: Any) -> None:
    _, _, _, (_, m) = entry
    shapes = {k: np.asarray(v).shape for k, v in m.items() if not k.startswith("smdp/")}
    assert shapes == {
        "losses": (2, len(LOSS_COLUMNS)),
        "done": (2, NUM_STEPS, ENVS),
        "ach": (2, NUM_STEPS, ENVS, N_ACH),
        "ret": (2, NUM_STEPS, ENVS),
        "length": (2, NUM_STEPS, ENVS),
    }
    losses = np.asarray(m["losses"])
    for c in ("value_loss", "grad_norm"):
        assert np.isfinite(losses[:, LOSS_COLUMNS.index(c)]).all()
    assert np.asarray(m["done"]).dtype == np.asarray(m["ach"]).dtype == bool


def test_training_moves_the_network_and_the_table(
    entry: Any, same: Callable[[Any, Any], bool]
) -> None:
    trainer, _, start, (state, _) = entry
    p0, p1 = trainer.params(start), trainer.params(state)
    assert not same(p0["variables"]["params"], p1["variables"]["params"])
    assert int((np.asarray(p1["q"]) != 10.0).sum()) > 0
    assert int(state.carry.shared.t) == STEPS
    assert int(state.updates) == 2


def test_chunks_of_one_equal_one_chunk_of_two(
    entry: Any, same: Callable[[Any, Any], bool]
) -> None:
    """Also: a chunk leaves its input state usable."""
    trainer, _, start, (whole, m) = entry
    one, m1 = trainer.train_chunk(start, 1)
    two, m2 = trainer.train_chunk(one, 1)
    assert same(two, whole)
    joined = {k: np.concatenate([np.asarray(m1[k]), np.asarray(m2[k])]) for k in m}
    assert same(joined, {k: np.asarray(v) for k, v in m.items()})


def test_the_table_and_skill_state_are_saved_and_a_resume_continues(
    entry: Any, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    trainer, cfg, start, (whole, _) = entry
    one, _ = trainer.train_chunk(start, 1)
    save_state(tmp_path, one)
    restored = load_state(tmp_path, trainer.init(cfg, jax.random.key(1)))
    assert same(restored, one)
    assert same(restored.carry.shared.q, one.carry.shared.q)
    assert same(restored.carry.per_env.skill, one.carry.per_env.skill)
    resumed, _ = trainer.train_chunk(restored, 1)
    assert same(resumed, whole)


def test_a_done_step_unlock_reaches_the_chunk_metrics(
    methods_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(methods_dir, envs=16, num_steps=32, hidden=17)  # uncompiled
    assert_done_step_unlock_counted(METHODS[NAME], cfg, monkeypatch)


# --- method metrics ------------------------------------------------------------------

PER_SKILL = (
    "chosen_share",
    "success_rate",
    "mean_len",
    "end/success",
    "end/failure",
    "end/timeout",
    "end/episode_end",
    "end/interrupt",
    "test_agreement",
)
TOP_METRICS = {
    "smdp/eps",
    "smdp/td_abs_mean",
    "smdp/decisions_per_episode",
    "smdp/keys_visited",
    "smdp/occupancy",
    "smdp/collision_est",
    "smdp/q_start",
    "smdp/q_max_updated",
    "smdp/low/loss",
    "smdp/low/q_mean",
    "smdp/low/eps",
    "smdp/diamond_episodes",
    "smdp/diamond_first_step",
    "smdp/diamond_tenth_step",
    "smdp/nan",
    "smdp/table_full",
    "smdp/divergence",
}
RATES = ("chosen_share", "success_rate", "test_agreement")


def test_one_update_logs_every_method_metric(entry: Any) -> None:
    _, _, _, (state, m) = entry
    names = {k for k in m if k not in ("losses", "done", "ach", "ret", "length")}
    skills = {f"smdp/skill/{s.name}/{x}" for s in SKILLS for x in PER_SKILL}
    assert names == TOP_METRICS | skills
    for k in names:
        v = np.asarray(m[k], float)
        assert v.shape == (2,), k
        assert np.isfinite(v).all(), k
    for s, x in itertools.product(
        SKILLS, (*RATES, *(p for p in PER_SKILL if "/" in p))
    ):
        v = np.asarray(m[f"smdp/skill/{s.name}/{x}"])
        assert ((v >= 0) & (v <= 1)).all(), (s.name, x)
    shares = sum(np.asarray(m[f"smdp/skill/{s.name}/chosen_share"]) for s in SKILLS)
    ends = int(state.carry.shared.ends.sum())  # every end is followed by a decision
    assert ends > 0
    np.testing.assert_allclose(shares[shares > 0], 1.0, rtol=1e-6)
    # both epsilons decay over 10% of the 2 updates: at their ends from update 1
    np.testing.assert_allclose(m["smdp/eps"], [1.0, 0.05])
    np.testing.assert_allclose(m["smdp/low/eps"], [1.0, 0.005])
    rows = 2**12
    visited = int((np.asarray(state.carry.shared.seen)[:, 0] >= 0).sum())
    assert m["smdp/keys_visited"][-1] == visited > 0
    assert m["smdp/occupancy"][-1] == pytest.approx(visited / rows)
    np.testing.assert_array_equal(m["smdp/diamond_first_step"], -1)
    for guard in ("smdp/nan", "smdp/table_full", "smdp/divergence"):
        np.testing.assert_array_equal(m[guard], 0.0)


def _record(**over: Any) -> dict[str, np.ndarray]:
    """One update's stacked record with a hand-made skill 0 (`smdp.method_metrics`)."""
    ends = np.zeros((1, NUM_SKILLS, len(REASONS)), np.int32)
    ends[0, 0, [SUCCESS, TIMEOUT, LOW_STAT]] = [2, 1, 1]
    agreement = np.zeros((1, NUM_SKILLS, 2), np.int32)
    agreement[0, 0] = [4, 3]
    chosen = np.zeros((1, NUM_SKILLS), np.int32)
    chosen[0, :2] = [3, 1]
    length = np.zeros((1, NUM_SKILLS), np.int32)
    length[0, 0] = 40
    r = {
        "ends": ends, "agreement": agreement, "chosen": chosen, "length": length,
        "td_abs": np.array([6.0]), "td_n": np.array([3]), "q_upd": np.array([12.5]),
        "q_bad": np.array([False]), "q_mean": np.array([0.5]),
        "episodes": np.array([2]), "visited": np.array([100]),
        "q_start": np.array([10.0]), "td_loss": np.array([0.25]),
        "grad_norm": np.array([1.0]), "finite": np.array([True]),
        "found": np.array([[3, 50, -1]]), "eps_top": np.array([0.5]),
        "eps": np.array([0.2]),
    }  # fmt: skip
    return r | over


def test_method_metrics_are_shares_of_the_update_counts() -> None:
    m = smdp.method_metrics(_record(), rows=1000)
    skill = f"smdp/skill/{SKILLS[0].name}"
    assert m[f"{skill}/chosen_share"][0] == 0.75
    assert m[f"{skill}/success_rate"][0] == m[f"{skill}/end/success"][0] == 0.5
    assert m[f"{skill}/end/timeout"][0] == m[f"{skill}/end/interrupt"][0] == 0.25
    assert m[f"{skill}/mean_len"][0] == 10.0
    assert m[f"{skill}/test_agreement"][0] == 0.75
    other = f"smdp/skill/{SKILLS[1].name}"  # chosen but not ended: no rates yet
    assert m[f"{other}/success_rate"][0] == m[f"{other}/mean_len"][0] == 0.0
    assert m[f"{other}/test_agreement"][0] == 1.0
    assert m["smdp/td_abs_mean"][0] == 2.0
    assert m["smdp/decisions_per_episode"][0] == 2.0
    assert m["smdp/occupancy"][0] == 0.1
    assert m["smdp/collision_est"][0] == 0.05
    assert m["smdp/q_max_updated"][0] == 12.5
    assert (m["smdp/diamond_episodes"][0], m["smdp/diamond_first_step"][0]) == (3, 50)
    assert m["smdp/diamond_tenth_step"][0] == -1
    assert m["smdp/low/q_mean"][0] == 0.5
    none = smdp.method_metrics(_record(q_upd=np.array([-np.inf])), rows=1000)
    assert none["smdp/q_max_updated"][0] == 0.0


def test_the_diamond_counter_keeps_the_first_and_tenth_episode() -> None:
    at = jnp.array([8, 16, 24, 32])
    found = smdp._diamonds(jnp.array([0, -1, -1]), jnp.array([0, 1, 0, 9]), at)
    np.testing.assert_array_equal(found, [10, 16, 32])
    again = smdp._diamonds(found, jnp.array([1, 0, 0, 0]), at + 32)
    np.testing.assert_array_equal(again, [11, 16, 32])
    none = smdp._diamonds(jnp.array([0, -1, -1]), jnp.zeros(4, int), at)
    np.testing.assert_array_equal(none, [0, -1, -1])


def test_the_greedy_table_lists_every_decided_key(entry: Any, tmp_path: Path) -> None:
    trainer, _, _, (state, _) = entry
    params = trainer.params(state)
    path = smdp.write_greedy_table(tmp_path, 2, params)
    assert path == tmp_path / "eval" / smdp.GREEDY_TABLE_FILE
    smdp.write_greedy_table(tmp_path, 4, params)
    smdp.write_greedy_table(tmp_path, 2, params)  # rewritten, not duplicated
    table = pq.read_table(path).to_pydict()
    seen = np.asarray(params["seen"])
    keys = np.flatnonzero(seen[:, 0] >= 0)
    assert sorted(set(table["update"])) == [2, 4]
    assert len(table["key"]) == 2 * len(keys)
    first = [i for i, u in enumerate(table["update"]) if u == 2]
    assert [table["key"][i] for i in first] == keys.tolist()
    names = [*CLASSIC_FIELDS, "low_food", "low_drink", "low_energy", "low_health"]
    q = np.asarray(params["q"])
    for i, key in zip(first, keys, strict=True):
        assert [table[n][i] for n in names] == seen[key].tolist()
        assert table["greedy_skill"][i] == SKILLS[int(q[key].argmax())].name
        assert table["value"][i] == pytest.approx(float(q[key].max()))


# --- the two-level policy -----------------------------------------------------------


def _eval_obs(state: Any) -> jax.Array:
    """The training observations with the reward column of the eval env."""
    return state.carry.per_env.obs


def test_the_policy_carries_the_lstm_and_skill_state(entry: Any) -> None:
    trainer, _, _, (state, _) = entry
    policy, params = trainer.policy(state), trainer.params(state)
    assert {"variables", "eps", "eps_top", "q", "seen"} <= params.keys()
    carry = policy.init_carry(ENVS)
    lstm, last, skill, steps, prev, started = carry
    assert lstm[0].shape == (ENVS, 16)
    np.testing.assert_array_equal(last, -1)
    assert skill.shape == steps.shape == started.shape == (ENVS,)
    assert prev.shape == (ENVS, CLASSIC_OBS)
    step = jax.jit(policy.step, static_argnames="greedy")
    obs = _eval_obs(state)
    assert obs.shape == (ENVS, CLASSIC_OBS + 1)
    act, carry2 = step(params, carry, obs, jax.random.key(0), greedy=False)
    assert act.shape == (ENVS,)
    assert ((act >= 0) & (act < 17)).all()
    extras = jax.vmap(policy.extras)(carry2)
    assert extras["skill"].dtype == jnp.uint8
    assert extras["steps_in_skill"].dtype == jnp.uint16
    np.testing.assert_array_equal(extras["steps_in_skill"], 0)  # a first decision
    assert bool(carry2[5].all())
    # one more step: the skills keep running and count their steps
    _, carry3 = step(params, carry2, obs, jax.random.key(1), greedy=True)
    running = np.asarray(carry3[2]) == np.asarray(carry2[2])
    np.testing.assert_array_equal(np.asarray(carry3[3])[running], 1)
    assert trainer.env_wrapper(state) is smdp.reward_in_obs


def test_the_policy_picks_skills_from_the_checkpoint_table(entry: Any) -> None:
    """Greedy: the best available skill of the row; eps_top 1: any available one."""
    trainer, _, _, (state, _) = entry
    policy, params = trainer.policy(state), trainer.params(state)
    fight = SKILL_IDS["fight"]
    q = jnp.full_like(params["q"], -1.0).at[:, fight].set(5.0)
    step = jax.jit(policy.step, static_argnames="greedy")
    obs, carry = _eval_obs(state), policy.init_carry(ENVS)
    _, c = step(params | {"q": q}, carry, obs, jax.random.key(0), greedy=True)
    np.testing.assert_array_equal(c[2], fight)
    explore = params | {"q": q, "eps_top": 1.0}
    picks = [
        step(explore, carry, obs, jax.random.key(k), greedy=False)[1][2]
        for k in range(4)
    ]
    assert len(np.unique(np.concatenate(picks))) > 1


# --- through the runner ---------------------------------------------------------------


def _phase(tmp_path: Path, methods_dir: Path, **over: Any) -> list[Any]:
    """The specs of a 2-update Classic phase training the base on a small table."""
    phase = tmp_path / "smdp.yaml"
    item = {"ref": "base", "set": {"log2_rows": 12, **over}}
    entry = {"config": NAME, "configs": [item], "seeds": 1}
    raw = {"budget": "0.128k", "eval": "final", "env": CLASSIC, "runs": {NAME: entry}}
    phase.write_text(yaml.safe_dump(raw))
    return expand_phase(phase, methods_dir=methods_dir)


def test_the_runner_trains_and_evaluation_replays_the_skills(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    """FINISHED with smdp/* metrics; both eval modes of the final checkpoint store one
    skill id and skill-step count per step."""
    (spec,) = _phase(tmp_path, methods_dir)
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FINISHED"
    assert run.data.metrics.keys() >= TOP_METRICS
    out = artifact_path(run.info.artifact_uri)
    opened = eval_rollouts.EvalRun.load(out, METHODS)
    assert opened.method == NAME
    last = opened.updates[-1]
    for mode in eval_rollouts.MODES:
        eval_rollouts.evaluate(opened, [last], mode, seeds=(10000,), chunk=256)
        ev = eval_rollouts.load_eval(out, last, mode == "greedy")
        live = ev["actions"] >= 0
        assert ev["extras/skill"].dtype == np.uint8
        assert ev["extras/steps_in_skill"].dtype == np.uint16
        assert ev["extras/skill"].shape == ev["actions"].shape
        assert (ev["extras/skill"][live] < NUM_SKILLS).all()
        assert (ev["extras/steps_in_skill"][:, 0] == 0).all()


_ON_TWO_CPUS = """
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime(host_devices=2)
from wut_murl_craftax.training.interface import METHODS

cfg = {"env": "Craftax-Classic-Symbolic-v1", "envs": 8, "num_steps": 8,
       "minibatches": 2, "hidden": 16, "log2_rows": 12, "steps": 128}
trainer = METHODS["smdp-skills"]
state = trainer.init(cfg, jax.random.key(0))

def one_device(s):
    return {d for x in jax.tree.leaves(s) for d in x.sharding.device_set}

assert len(jax.devices("cpu")) == 2
assert one_device(state.carry) == {jax.devices("cpu")[0]}
state, m = trainer.train_chunk(state, 2)
assert one_device(state.carry) == {jax.devices("cpu")[0]}
print("ok")
"""


def test_the_rollout_stays_on_one_actor_device(
    start_on_cpus: Callable[[str, int, Path], Any], tmp_path: Path
) -> None:
    """Two host CPUs: envs and table on the actor alone (`smdp.SHARD_ROLLOUT`)."""
    proc = start_on_cpus(_ON_TWO_CPUS, 2, tmp_path)
    out, err = proc.communicate(timeout=900)
    assert proc.returncode == 0, err[-3000:]
    assert out.strip().splitlines()[-1] == "ok"
