"""Tests for the Dyna-Q registry entry: config file, trainer, metrics, runner, aborts.

Tiny Craftax-Classic runs (8 envs x 8 steps, a 16-wide skill network, a 2^12-row
table, 2^10 model pairs, step limits x 0.02) on the tuned blocks of a temporary method
folder: the PQN-RNN control's tiny settings, an `smdp-skills` block equal to its base.
"""

import itertools
import time
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
from _methods import copy_methods

from wut_murl_craftax.orchestration.abort import (
    TAG_ABORT_GUARD,
    TAG_ABORT_METRIC,
    TAG_ABORT_UPDATE,
    flag_hits,
)
from wut_murl_craftax.orchestration.queue import (
    _knob_values,
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.orchestration.runner import ChunkMetrics
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import dyna_q, smdp
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.progress_key import CLASSIC_FIELDS
from wut_murl_craftax.training.skills import SKILLS
from wut_murl_craftax.visualisation import eval_rollouts

REPO = Path(__file__).resolve().parents[2]
CLASSIC = "Craftax-Classic-Symbolic-v1"
ENVS, NUM_STEPS = 8, 8
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
NAME = "dyna-q"
SMALL = {"log2_rows": 12, "timeout_mult": 0.02}
"""Table rows of the tiny runs; short step limits, so skills end in every rollout."""
MODEL = {"model_pairs": 1024}
"""Model pairs of the tiny runs."""
DYNA_KNOBS = {
    "plan_variant",
    "plan_n",
    "plan_kappa",
    "model_slots",
    "model_pairs",
    "plan_batch",
    "ps_theta",
    "vi_tol",
    "vi_max_sweeps",
}
"""The method's own run-config keys (`dyna_q.DYNA_KEYS`)."""
TIMED = {"dyna/plan_seconds", "dyna/plan_wall_share"}
"""Wall-clock metrics: not reproducible across runs."""
DYNA_METRICS = TIMED | {
    "dyna/pairs_visited",
    "dyna/pairs_full_share",
    "dyna/model_accuracy",
    "dyna/backups",
    "dyna/plan_td_abs",
    "dyna/bonus_mean",
    "dyna/stale_share",
    "dyna/ps_above_theta",
    "dyna/vi_sweeps",
    "dyna/vi_residual",
}
"""Every ``dyna/...`` metric but the guards."""
SMDP_GUARDS = ["smdp/nan", "smdp/table_full", "smdp/divergence"]
DYNA_GUARDS = ["dyna/nan", "dyna/model_full", "dyna/divergence"]
SHARES = (
    "dyna/pairs_full_share",
    "dyna/model_accuracy",
    "dyna/stale_share",
    "dyna/plan_wall_share",
)


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of configs/methods with the PQN-RNN and smdp-skills ``tuned`` set."""
    tuned = {"pqn-rnn": TINY, "smdp-skills": {"alpha": 0.2}}
    return copy_methods(tmp_path_factory.mktemp("methods"), tuned)


def _config(
    methods_dir: Path, name: str = NAME, ref: str = "base", **over: Any
) -> dict[str, Any]:
    """A resolved run config: tiny network, small table and model, 2-update budget."""
    method = load_method(name, methods_dir)
    cfg = resolve_config(method, name, "1m", ref=ref, methods_dir=methods_dir)
    small = SMALL | MODEL if name == NAME else SMALL
    return cfg | small | {"steps": STEPS} | over


def _strip(metrics: Any) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in metrics.items() if k not in TIMED}


# --- config file -------------------------------------------------------------------


def test_the_method_file_has_the_base_and_both_grids() -> None:
    method = load_method(NAME)
    assert method.registry_names == [NAME]
    assert method.base["base_from"] == "smdp-skills.tuned"
    assert method.tuned is None
    # every planning knob is set in the base, so no dataclass default decides a run
    assert method.base.keys() - {"base_from"} == DYNA_KNOBS
    assert method.base["model_pairs"] == 2**16
    assert method.base["model_slots"] == 8
    assert method.base["plan_batch"] == 1024
    assert (method.base["ps_theta"], method.base["vi_tol"]) == (1e-4, 1e-3)
    assert method.base["vi_max_sweeps"] == 50
    knobs = ("plan_variant", "plan_n", "plan_kappa")
    grid = [tuple(e.get(k) for k in knobs) for e in method.coarse]
    kappas = (0.0, 0.003, 0.03)
    want = [
        *itertools.product(("uniform", "prioritized"), (1, 5, 20), kappas),
        *(("value_iteration", None, k) for k in kappas),
    ]
    assert len(method.coarse) == len(set(grid)) == 21
    assert set(grid) == set(want)
    fine = method.fine
    assert fine is not None
    assert fine["rule"] == {"select": 2, "combine": "cross"}
    by_key = {k["key"]: k for k in fine["knobs"]}
    assert by_key.keys() == {"plan_kappa", "model_slots"}
    assert by_key["plan_kappa"]["mul"] == [0.5, 1.0, 2.0]
    assert by_key["plan_kappa"]["zero"] == [0.0, 0.0015, 0.003]
    assert by_key["model_slots"]["mul"] == [0.5, 1.0, 2.0]
    assert by_key["model_slots"]["integer"] is True


def test_the_config_splits_the_planning_keys_off(methods_dir: Path) -> None:
    trainer = METHODS[NAME]
    cfg = trainer.config(_config(methods_dir, ref="prioritized-n20-k0.03"))
    assert (cfg.dyna.plan_variant, cfg.dyna.plan_n, cfg.dyna.plan_kappa) == (
        "prioritized",
        20,
        0.03,
    )
    assert (cfg.dyna.model_slots, cfg.dyna.model_pairs) == (8, 1024)
    smdp_cfg = METHODS["smdp-skills"].config(_config(methods_dir, "smdp-skills"))
    assert cfg.smdp == smdp_cfg
    assert dyna_q.DYNA_KEYS == DYNA_KNOBS
    with pytest.raises(ValueError, match="vi_tol"):
        trainer.config({k: v for k, v in _config(methods_dir).items() if k != "vi_tol"})
    with pytest.raises(ValueError, match="plan_variant"):
        trainer.config(_config(methods_dir, plan_variant="sweeps"))


def test_the_model_refuses_pairs_equal_to_the_envs(methods_dir: Path) -> None:
    with pytest.raises(ValueError, match="envs"):
        METHODS[NAME].init(_config(methods_dir, model_pairs=ENVS), jax.random.key(0))


# --- the trainer -------------------------------------------------------------------


@pytest.fixture(scope="module")
def entry(methods_dir: Path) -> Any:
    """(trainer, config, start state, 2-update result) of the entry (uniform, n 5)."""
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
    assert trainer.env_wrapper(entry[2]) is None


def test_with_no_planning_it_is_the_smdp_trainer(
    methods_dir: Path, same: Callable[[Any, Any], bool]
) -> None:
    """n = 0: the same table, network, envs, counters and smdp/* metrics."""
    seed = jax.random.key(0)
    base = METHODS["smdp-skills"]
    s_state, s_m = base.train_chunk(
        base.init(_config(methods_dir, "smdp-skills"), seed), 2
    )
    trainer = METHODS[NAME]
    d_state, d_m = trainer.train_chunk(
        trainer.init(_config(methods_dir, plan_n=0), seed), 2
    )
    assert same(d_state.smdp, s_state)
    assert same(d_state.smdp.carry.shared.q, s_state.carry.shared.q)
    assert same(trainer.params(d_state), base.params(s_state))
    assert {k for k in d_m if not k.startswith("dyna/")} == set(s_m)
    assert same(_strip(s_m), {k: np.asarray(d_m[k]) for k in s_m})
    np.testing.assert_array_equal(d_m["dyna/backups"], 0)
    assert int(d_state.model.n_pairs) > 0  # the model still learns


def test_planning_changes_the_table_but_uses_no_env_steps(
    entry: Any, methods_dir: Path, same: Callable[[Any, Any], bool]
) -> None:
    trainer, _, _, (state, m) = entry
    idle, _ = trainer.train_chunk(
        trainer.init(_config(methods_dir, plan_n=0), jax.random.key(0)), 2
    )
    assert int(state.smdp.carry.shared.t) == int(idle.smdp.carry.shared.t) == STEPS
    backups = np.asarray(m["dyna/backups"])  # n = 5 per real skill end
    assert (backups > 0).all()
    np.testing.assert_array_equal(backups % 5, 0)
    assert not same(state.smdp.carry.shared.q, idle.smdp.carry.shared.q)
    assert int(state.smdp.updates) == 2


def test_chunks_of_one_equal_one_chunk_of_two(
    entry: Any, same: Callable[[Any, Any], bool]
) -> None:
    """Also: a chunk leaves its input state (model included) usable."""
    trainer, _, start, (whole, m) = entry
    one, m1 = trainer.train_chunk(start, 1)
    two, m2 = trainer.train_chunk(one, 1)
    assert same(two, whole)
    joined = {k: np.concatenate([m1[k], m2[k]]) for k in _strip(m)}
    assert same(joined, _strip(m))


def test_the_model_is_saved_and_a_resume_continues(
    entry: Any, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    trainer, cfg, start, (whole, _) = entry
    one, _ = trainer.train_chunk(start, 1)
    assert int(one.model.n_pairs) > 0
    save_state(tmp_path, one)
    restored = load_state(tmp_path, trainer.init(cfg, jax.random.key(1)))
    assert same(restored, one)
    assert same(restored.model, one.model)
    resumed, _ = trainer.train_chunk(restored, 1)
    assert same(resumed, whole)


def test_the_policy_and_checkpoint_are_those_of_smdp_skills(entry: Any) -> None:
    """The two-level policy of the table; the model stays out of ``ckpt/``."""
    trainer, _, _, (state, _) = entry
    policy = trainer.policy(state)
    assert isinstance(policy, smdp.SMDPPolicy)
    assert policy.cfg == state.smdp.cfg.top
    params = trainer.params(state)
    assert params.keys() == {"variables", "eps", "eps_top", "rows", "q_rows", "fields"}
    want = METHODS["smdp-skills"].params(state.smdp)
    for k in ("rows", "q_rows", "fields"):
        np.testing.assert_array_equal(params[k], want[k])


def test_updates_after_the_first_reuse_its_executables(
    methods_dir: Path, lowerings: Callable[[], int]
) -> None:
    """Rollout, planner and learner compile once (R56)."""
    trainer = METHODS[NAME]
    cfg = _config(methods_dir, hidden=21)  # a config no other test compiles
    state, _ = trainer.train_chunk(trainer.init(cfg, jax.random.key(0)), 1)
    before = lowerings()
    for _ in range(3):
        state, _ = trainer.train_chunk(state, 1)
    assert lowerings() == before


def test_a_done_step_unlock_reaches_the_chunk_metrics(
    methods_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(methods_dir, envs=16, num_steps=32, hidden=17)  # uncompiled
    assert_done_step_unlock_counted(METHODS[NAME], cfg, monkeypatch)


# --- method metrics ------------------------------------------------------------------


def test_one_update_logs_every_method_metric(entry: Any) -> None:
    _, _, _, (state, m) = entry
    new = {
        k for k in m if not k.startswith("smdp/")
    } - ChunkMetrics.__annotations__.keys()
    assert new >= DYNA_METRICS
    assert all(k.startswith("dyna/") for k in new)
    for k in new:
        v = np.asarray(m[k], float)
        assert v.shape == (2,), k
        assert np.isfinite(v).all(), k
    for k in SHARES:
        v = np.asarray(m[k])
        assert ((v >= 0) & (v <= 1)).all(), k
    pairs = np.asarray(m["dyna/pairs_visited"])
    model = jax.device_get(state.model)
    n = int(model.n_pairs)
    assert (np.diff(pairs) >= 0).all()
    assert pairs[-1] == n > 0
    # planning backs up only pairs with real outcomes: the table's tried pairs
    np.testing.assert_array_equal(m["smdp/pairs_tried"], pairs)
    count, slots = model.count[:n], model.m_R.shape[1]
    assert m["dyna/pairs_full_share"][-1] == pytest.approx((count >= slots).mean())
    stale = 1000.0 - model.last_tried[:n] > 100  # time at the end: the whole budget
    assert m["dyna/stale_share"][-1] == pytest.approx(stale.mean())
    assert (np.asarray(m["dyna/plan_td_abs"]) > 0).all()
    np.testing.assert_array_equal(m["dyna/bonus_mean"], 0.0)  # kappa 0
    for k in ("dyna/vi_sweeps", "dyna/vi_residual", "dyna/ps_above_theta"):
        np.testing.assert_array_equal(m[k], 0)  # uniform
    assert new == DYNA_METRICS | set(DYNA_GUARDS)
    for guard in (*SMDP_GUARDS, *DYNA_GUARDS):
        np.testing.assert_array_equal(m[guard], 0.0)


def _record(**over: Any) -> dict[str, np.ndarray]:
    """One update's stacked Dyna record (`dyna_q.method_metrics`)."""
    r = {
        "backups": [10], "sweeps": [3], "td_abs": [3.0], "td_n": [6], "bonus": [1.2],
        "q_upd": [12.0], "q_bad": [False], "above_theta": [7], "residual": [2e-4],
        "n_pairs": [4.0], "full_pairs": [1], "stale": [2], "full": [False],
        "hits": [3], "ends": [4], "plan_seconds": [0.5], "update_seconds": [2.0],
    }  # fmt: skip
    return {k: np.asarray(v) for k, v in (r | over).items()}


def test_method_metrics_are_shares_of_the_update_counts() -> None:
    m = dyna_q.method_metrics(_record(), max_q=300.0)
    assert m["dyna/pairs_visited"][0] == 4
    assert m["dyna/pairs_full_share"][0] == 0.25
    assert m["dyna/model_accuracy"][0] == 0.75
    assert m["dyna/backups"][0] == 10
    assert m["dyna/plan_td_abs"][0] == 0.5
    assert m["dyna/bonus_mean"][0] == pytest.approx(0.2)
    assert m["dyna/stale_share"][0] == 0.5
    assert m["dyna/ps_above_theta"][0] == 7
    assert (m["dyna/vi_sweeps"][0], m["dyna/vi_residual"][0]) == (3, 2e-4)
    assert m["dyna/plan_seconds"][0] == 0.5
    assert m["dyna/plan_wall_share"][0] == 0.25
    fast = dyna_q.method_metrics(
        _record(plan_seconds=[0.01], update_seconds=[0.02]), max_q=300.0
    )
    assert fast["dyna/plan_wall_share"][0] == 0.5  # seconds, not counts: no floor 1
    empty = dyna_q.method_metrics(
        _record(ends=[0], hits=[0], td_n=[0], n_pairs=[0.0], full_pairs=[0], stale=[0]),
        max_q=300.0,
    )
    for k in ("model_accuracy", "plan_td_abs", "pairs_full_share", "stale_share"):
        assert empty[f"dyna/{k}"][0] == 0.0, k


def test_the_model_graph_lists_every_visited_pair(entry: Any, tmp_path: Path) -> None:
    _, _, _, (state, _) = entry
    path = dyna_q.write_model_graph(tmp_path, 2, state)
    assert path == tmp_path / "eval" / dyna_q.MODEL_GRAPH_FILE
    dyna_q.write_model_graph(tmp_path, 4, state)
    dyna_q.write_model_graph(tmp_path, 2, state)  # rewritten, not duplicated
    table = pq.read_table(path).to_pydict()
    model = jax.device_get(state.model)
    n = int(model.n_pairs)
    assert sorted(set(table["update"])) == [2, 4]
    rows = [i for i, u in enumerate(table["update"]) if u == 2]
    assert len(rows) == n > 0
    seen = np.asarray(state.smdp.carry.shared.seen)
    gamma = state.smdp.cfg.top.gamma_top
    names = [*CLASSIC_FIELDS, *smdp.FLAG_NAMES]
    for p, i in enumerate(rows):
        k, o, c = int(model.pair_k[p]), int(model.pair_o[p]), int(model.count[p])
        filled = min(c, model.m_R.shape[1])
        nxt = model.m_next[p, :filled].tolist()
        assert (table["pair"][i], table["key"][i], table["count"][i]) == (p, k, c)
        assert table["skill"][i] == SKILLS[o].name
        assert [table[f][i] for f in names] == seen[k].tolist()
        assert nxt.count(table["next_key"][i]) == max(nxt.count(x) for x in nxt)
        nk = table["next_key"][i]
        assert [table[f"next_{f}"][i] for f in names] == seen[nk].tolist()
        r, disc = model.m_R[p, :filled], model.m_disc[p, :filled]
        assert table["mean_return"][i] == pytest.approx(float(r.mean()), rel=1e-5)
        tau = np.log(disc.astype(float)) / np.log(gamma)
        assert table["mean_tau"][i] == pytest.approx(float(tau.mean()), rel=1e-3)


# --- through the runner ---------------------------------------------------------------


def _phase(
    tmp_path: Path, methods_dir: Path, budget: str = "0.128k", **over: Any
) -> list[Any]:
    """The specs of a 2-update Classic phase training the base on small tables."""
    phase = tmp_path / "dyna.yaml"
    item = {"ref": "base", "set": SMALL | MODEL | over}
    entry = {"config": NAME, "configs": [item], "seeds": 1}
    raw = {"budget": budget, "eval": "final", "env": CLASSIC, "runs": {NAME: entry}}
    phase.write_text(yaml.safe_dump(raw))
    return expand_phase(phase, methods_dir=methods_dir)


def test_the_runner_trains_and_evaluates_a_checkpoint(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    """FINISHED with dyna/* metrics; both eval modes of a checkpoint play the skills."""
    (spec,) = _phase(tmp_path, methods_dir)
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FINISHED"
    assert any(k.startswith("dyna/") for k in run.data.metrics)
    assert "smdp/diamond_episodes" in run.data.metrics
    out = artifact_path(run.info.artifact_uri)
    opened = eval_rollouts.EvalRun.load(out, METHODS)
    assert opened.method == NAME
    last = opened.updates[-1]
    for mode in eval_rollouts.MODES:
        eval_rollouts.evaluate(opened, [last], mode, seeds=(10000,), chunk=256)
        ev = eval_rollouts.load_eval(out, last, mode == "greedy")
        live = ev["actions"] >= 0
        assert live.any()
        assert (ev["extras/skill"][live] < smdp.NUM_SKILLS).all()


# --- abort rules ----------------------------------------------------------------------


def test_the_abort_list_holds_only_the_guards_and_flags_slow_planning() -> None:
    """The guards are 0 until broken (`test_one_update_logs_every_method_metric`)."""
    assert load_method(NAME).abort == {
        "finite": SMDP_GUARDS + DYNA_GUARDS,
        "permanent": SMDP_GUARDS[1:] + DYNA_GUARDS[1:],
        "flags": [
            {
                "name": "plan_slow",
                "metric": "dyna/plan_wall_share",
                "above": 0.3,
                "updates": 3,
            }
        ],
    }


@pytest.mark.parametrize(
    ("shares", "hit"),
    [
        ([0.9, 0.1, 0.1, 0.1], None),  # the planner's first-call compile
        ([0.9, 0.9, 0.1, 0.1], None),  # and a second-update recompile
        ([0.1, 0.9, 0.9, 0.9], 4),
    ],
)
def test_slow_planning_needs_three_updates_in_a_row(
    shares: list[float], hit: int | None
) -> None:
    rules = load_method(NAME).abort["flags"]  # type: ignore[index]
    hits = flag_hits(rules, {"dyna/plan_wall_share": shares}, 100)
    assert hits.get("plan_slow") == hit


@pytest.mark.parametrize(
    ("over", "guard"),
    [
        ({"q_bad": [True]}, "dyna/nan"),
        ({"full": [True]}, "dyna/model_full"),
        ({"q_upd": [300.5]}, "dyna/divergence"),
    ],
)
def test_each_guard_turns_nan_on_its_rule(over: dict[str, Any], guard: str) -> None:
    m = dyna_q.method_metrics(_record(**over), max_q=300.0)
    assert {g for g in DYNA_GUARDS if np.isnan(m[g][0])} == {guard}
    fine = dyna_q.method_metrics(_record(q_upd=[300.0]), max_q=300.0)
    assert not any(np.isnan(fine[g][0]) for g in DYNA_GUARDS)


def test_the_divergence_bound_adds_the_largest_dyna_q_plus_return() -> None:
    """kappa sqrt(1000) per decision at most, a decision at least one step."""
    assert dyna_q.max_q(0.0, 0.999) == smdp.MAX_Q == 300.0
    bound = dyna_q.max_q(0.03, 0.999)
    assert bound == pytest.approx(300.0 + 0.03 * np.sqrt(1000.0) / 0.001)
    m = dyna_q.method_metrics(_record(q_upd=[0.9 * bound]), max_q=bound)
    assert m["dyna/divergence"][0] == 0.0
    m = dyna_q.method_metrics(_record(q_upd=[1.1 * bound]), max_q=bound)
    assert np.isnan(m["dyna/divergence"][0])
    real = smdp.method_metrics(_smdp_record(q_upd=400.0), rows=1000, max_q=bound)
    assert real["smdp/divergence"][0] == 0.0
    real = smdp.method_metrics(_smdp_record(q_upd=1e6), rows=1000, max_q=bound)
    assert np.isnan(real["smdp/divergence"][0])
    plain = smdp.method_metrics(_smdp_record(q_upd=400.0), rows=1000)
    assert np.isnan(plain["smdp/divergence"][0])  # smdp-skills keeps 300


def _smdp_record(q_upd: float) -> dict[str, np.ndarray]:
    """A minimal `smdp.method_metrics` record with one updated value ``q_upd``."""
    z, n = np.zeros, smdp.NUM_SKILLS
    r = {
        "ends": z((1, n, 7), int), "agreement": z((1, n, 2), int),
        "chosen": z((1, n), int), "length": z((1, n), int),
        "td_abs": [0.0], "td_n": [0], "q_upd": [q_upd], "q_bad": [False],
        "q_mean": [0.0], "episodes": [0], "visited": [1], "tried": [1],
        "q_start": [10.0],
        "td_loss": [0.0], "grad_norm": [0.0], "finite": [True], "found": [[0, -1, -1]],
        "eps_top": [0.5], "eps": [0.5],
    }  # fmt: skip
    return {k: np.asarray(v) for k, v in r.items()}


def test_a_stale_self_loop_settles_past_300_but_below_the_bound() -> None:
    """level_up-like pair (tau 1, back to its own key, R 0), 1000 units untried,
    kappa 0.03, horizon 1000: value iteration heads for 0.03 sqrt(1000) / 0.001 =
    949, a healthy Dyna-Q+ value that the plain 300 guard would abort."""
    m = dyna_q.init_model(4, 2, 8, 4)
    rec = dyna_q.Outcome(
        jnp.array([1]), jnp.array([0]), jnp.array([0.0]), jnp.array([0.999]),
        jnp.array([1]), jnp.array([False]), jnp.ones((1, 2), bool),
    )  # fmt: skip
    m = dyna_q.write(m, rec, jnp.array([True]), jnp.float32(0.0))
    q = smdp.init_table(4, 2, 0.0)
    q, stats = dyna_q.plan(
        q, m, 0, "value_iteration", 0.03, 1000.0, jax.random.key(0), alpha=1.0,
        batch=1024, theta=1e-4, tol=1e-3, max_sweeps=5000,
    )  # fmt: skip
    fixed = 0.03 * np.sqrt(1000.0) / (1 - 0.999)
    assert 300.0 < float(stats["q_upd"]) <= fixed
    record = _record(q_upd=[float(stats["q_upd"])])
    bound = dyna_q.max_q(0.03, 0.999)
    assert dyna_q.method_metrics(record, max_q=bound)["dyna/divergence"][0] == 0.0
    assert np.isnan(dyna_q.method_metrics(record, max_q=300.0)["dyna/divergence"][0])


def _configured_kappas_and_horizons() -> tuple[set[float], set[float]]:
    """Every kappa dyna-q.yaml can run (coarse, fine rule around each coarse value,
    the kappa ablation, the 1B finals' tuned value) and every smdp-skills horizon."""
    method, top = load_method(NAME), load_method("smdp-skills")
    coarse = {float(e["plan_kappa"]) for e in method.coarse}
    knobs = {k["key"]: k for k in (method.fine or {})["knobs"]}
    fine = {v for best in coarse for v in _knob_values(knobs["plan_kappa"], best)}
    ablation = {0.0} | coarse  # kappa 0, or the best nonzero stage-1 kappa
    kappas = coarse | fine | ablation  # the 1B finals take the tuned (fine) value
    horizons = {float(top.base["top_horizon"])} | {
        float(e["top_horizon"]) for e in top.coarse
    }
    h_knob = {k["key"]: k for k in (top.fine or {})["knobs"]}["top_horizon"]
    horizons |= {v for h in set(horizons) for v in _knob_values(h_knob, h)}
    return kappas, horizons


def test_every_configured_kappa_keeps_healthy_runs_below_the_bound() -> None:
    """The largest healthy planned value: real returns up to 300 plus a tau = 1
    self-loop collecting the full bonus every step."""
    kappas, horizons = _configured_kappas_and_horizons()
    assert max(kappas) >= 0.06
    assert 0.0 in kappas
    for kappa, h in itertools.product(kappas, horizons):
        gamma = 1 - 1 / h
        healthy = smdp.MAX_Q + kappa * np.sqrt(dyna_q.TIME_UNITS) / (1 - gamma)
        assert healthy <= dyna_q.max_q(kappa, gamma) * (1 + 1e-12), (kappa, h)
        loop = kappa * np.sqrt(dyna_q.TIME_UNITS) / (1 - gamma)
        if loop > smdp.MAX_Q:  # the cases the plain guard would abort
            assert loop < dyna_q.max_q(kappa, gamma), (kappa, h)


def _aborted(run: Any) -> tuple[str, str, str, str | None]:
    """Status, metric, update and the permanent guard (None: retried once)."""
    tags = run.data.tags
    guard = tags.get(TAG_ABORT_GUARD)
    return run.info.status, tags[TAG_ABORT_METRIC], tags[TAG_ABORT_UPDATE], guard


def test_a_nan_after_planning_aborts_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = dyna_q._records
    monkeypatch.setattr(
        dyna_q, "_records", lambda *a: real(*a)._replace(ret=jnp.float32(jnp.nan))
    )
    dyna_q._planner.cache_clear()  # trace the patch; drop the patched planner after
    try:
        (spec,) = _phase(tmp_path, methods_dir)
        run = run_store.run([spec])[spec.name]
    finally:
        dyna_q._planner.cache_clear()
    assert _aborted(run) == ("FAILED", "dyna/nan", "1", None)


def test_a_full_model_aborts_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    (spec,) = _phase(tmp_path, methods_dir, model_pairs=2)
    run = run_store.run([spec])[spec.name]
    assert _aborted(run) == ("FAILED", "dyna/model_full", "1", "dyna/model_full")


def test_a_planned_value_above_the_return_bound_aborts_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored returns of 1e6 in planning only: planned entries pass the bound, real
    updates do not."""
    real = dyna_q._records
    monkeypatch.setattr(
        dyna_q, "_records", lambda *a: real(*a)._replace(ret=jnp.float32(1e6))
    )
    dyna_q._planner.cache_clear()  # trace the patch; drop the patched planner after
    try:
        (spec,) = _phase(tmp_path, methods_dir, plan_kappa=0.03)
        run = run_store.run([spec])[spec.name]
    finally:
        dyna_q._planner.cache_clear()
    assert _aborted(run) == ("FAILED", "dyna/divergence", "1", "dyna/divergence")


def test_table_values_between_300_and_the_bound_do_not_abort(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    """kappa 0.03 at horizon 1000: entries of 400 (optimistic start) are healthy
    for Dyna-Q+ (bound 1249); `smdp-skills` alone would abort on smdp/divergence."""
    (spec,) = _phase(tmp_path, methods_dir, plan_kappa=0.03, q_init=400.0)
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["smdp/divergence"] == 0.0
    assert run.data.metrics["dyna/divergence"] == 0.0


def test_slow_planning_is_flagged_without_stopping_the_run(
    entry: Any,
    methods_dir: Path,
    run_store: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planning step of over a second in 5 updates: the last 3 are slow even
    if compiles dilute the share of the first 2."""
    real = dyna_q._run_plan

    def slow(planner: Any, *args: Any) -> Any:
        def sleepy(*a: Any) -> Any:
            time.sleep(1.0)
            return planner(*a)

        return real(sleepy, *args)

    monkeypatch.setattr(dyna_q, "_run_plan", slow)
    (spec,) = _phase(tmp_path, methods_dir, budget="0.32k")
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FINISHED"
    assert "flag/plan_slow" in run.data.tags
    assert run.data.metrics["dyna/plan_wall_share"] > 0.3
