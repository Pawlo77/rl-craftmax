"""Tests for the Dyna-Q registry entry: config file, trainer, metrics, runner, aborts.

Tiny Craftax-Classic runs (8 envs x 8 steps, a 16-wide skill network, a 2^12-row
table, 2^10 model pairs, step limits x 0.02) on the tuned blocks of a temporary method
folder: the PQN-RNN control's tiny settings, an `smdp-skills` block equal to its base.
"""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
import yaml
from _contract import assert_done_step_unlock_counted

from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import dyna_q, smdp
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.interface import METHODS
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


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of configs/methods with the PQN-RNN and smdp-skills ``tuned`` set."""
    out = tmp_path_factory.mktemp("methods")
    tuned = {"pqn-rnn": TINY, "smdp-skills": {"alpha": 0.2}}
    for path in (REPO / "configs" / "methods").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if path.stem in tuned:
            block = yaml.safe_dump({"tuned": tuned[path.stem]}, sort_keys=False)
            text = text.replace("\ntuned: null\n", "\n" + block)
        (out / path.name).write_text(text, encoding="utf-8")
    return out


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


def test_a_done_step_unlock_reaches_the_chunk_metrics(
    methods_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(methods_dir, envs=16, num_steps=32, hidden=17)  # uncompiled
    assert_done_step_unlock_counted(METHODS[NAME], cfg, monkeypatch)


# --- through the runner ---------------------------------------------------------------


def _phase(tmp_path: Path, methods_dir: Path, **over: Any) -> list[Any]:
    """The specs of a 2-update Classic phase training the base on small tables."""
    phase = tmp_path / "dyna.yaml"
    item = {"ref": "base", "set": SMALL | MODEL | over}
    entry = {"config": NAME, "configs": [item], "seeds": 1}
    raw = {"budget": "0.128k", "eval": "final", "env": CLASSIC, "runs": {NAME: entry}}
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
