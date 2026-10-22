"""Tests for the two count-bonus registry entries: config, hooks, targets, runs."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
import yaml
from _contract import assert_done_step_unlock_counted
from _methods import copy_methods

from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.orchestration.runner import EXTRAS_FILE
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import count_bonus, ppo_rnn, pqn_rnn
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.count_bonus import CountBonusConfig
from wut_murl_craftax.training.hooks import HOOK_NAMES, HookSpec, resolve
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.records import LOSS_COLUMNS
from wut_murl_craftax.visualisation import eval_rollouts

REPO = Path(__file__).resolve().parents[2]
CLASSIC = "Craftax-Classic-Symbolic-v1"
VARIANTS = ("ppo-rnn", "pqn-rnn")
KNOBS = {"beta", "granularity", "decay", "log2_k", "key", "max_occupancy"}
"""The method's run-config keys (`CountBonusConfig` fields)."""
TINY = {
    "ppo-rnn": {
        "env": CLASSIC,
        "envs": 8,
        "num_steps": 8,
        "minibatches": 2,
        "update_epochs": 1,
        "width": 16,
        "rnn_width": 16,
    },
    "pqn-rnn": {
        "env": CLASSIC,
        "envs": 8,
        "num_steps": 8,
        "minibatches": 2,
        "epochs": 1,
        "hidden": 16,
    },
}
"""Tuned blocks of the temporary control copies: tiny Classic runs, 64 steps each."""
CONFIG_A = "b0.1-recipe-constant"
"""The coarse entry of smoke config A (the design's default knobs)."""


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of configs/methods with the two controls' ``tuned`` filled in."""
    return copy_methods(tmp_path_factory.mktemp("methods"), TINY)


def _config(methods_dir: Path, variant: str, ref: str = CONFIG_A, **over: Any) -> Any:
    """A resolved count-bonus run config with a 4-update budget and overrides."""
    method = load_method("count-bonus", methods_dir)
    name = f"count-bonus-{variant}"
    cfg = resolve_config(method, name, "1m", ref=ref, methods_dir=methods_dir)
    return cfg | {"steps": 4 * 64} | over


def _control(methods_dir: Path, variant: str) -> dict[str, Any]:
    """The control's tuned run config with the same budget."""
    method = load_method(variant, methods_dir)
    cfg = resolve_config(method, variant, "1m", ref="tuned", methods_dir=methods_dir)
    return cfg | {"steps": 4 * 64}


# --- config file -----------------------------------------------------------------


def test_the_method_file_has_both_variants_and_the_coarse_grid() -> None:
    method = load_method("count-bonus")
    assert method.registry_names == ["count-bonus-ppo-rnn", "count-bonus-pqn-rnn"]
    assert method.base["ppo-rnn"]["base_from"] == "ppo-rnn.tuned"
    assert method.base["pqn-rnn"]["base_from"] == "pqn-rnn.tuned"
    assert method.tuned is None
    grid = {(e["beta"], e["granularity"], e["decay"]) for e in method.coarse}
    assert len(method.coarse) == len(grid) == 24
    assert grid == {
        (b, g, d)
        for b in (0.01, 0.03, 0.1, 0.3)
        for g in ("tiers", "recipe", "log2")
        for d in ("constant", "linear")
    }
    # every knob is set in the base, so no dataclass default decides a run
    for block in method.base.values():
        assert block.keys() >= KNOBS
        assert (block["log2_k"], block["key"], block["max_occupancy"]) == (
            20,
            "progress",
            0.25,
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_resolves_on_its_own_control(
    methods_dir: Path, variant: str
) -> None:
    cfg = _config(methods_dir, variant, "b0.03-log2-linear")
    control = _control(methods_dir, variant)
    assert cfg == control | {
        "method": f"count-bonus-{variant}",
        "beta": 0.03,
        "granularity": "log2",
        "decay": "linear",
        "log2_k": 20,
        "key": "progress",
        "max_occupancy": 0.25,
    }
    built = METHODS[f"count-bonus-{variant}"].config(cfg)
    assert built.envs == 8
    assert built.intrinsic_reward_fn is count_bonus.intrinsic_reward


def test_a_missing_knob_fails_instead_of_taking_a_default(methods_dir: Path) -> None:
    cfg = _config(methods_dir, "ppo-rnn")
    del cfg["beta"]
    with pytest.raises(TypeError, match="beta"):
        METHODS["count-bonus-ppo-rnn"].config(cfg)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_hooks_are_reward_state_and_metrics_only(
    methods_dir: Path, variant: str
) -> None:
    cfg = METHODS[f"count-bonus-{variant}"].config(_config(methods_dir, variant))
    set_hooks = {"init_state_fn", "intrinsic_reward_fn", "metrics_fn"}
    assert resolve(cfg).active == tuple(n for n in HOOK_NAMES if n in set_hooks)
    assert all(getattr(cfg, n) is None for n in set(HOOK_NAMES) - set_hooks)
    assert cfg.extra_steps == 0
    assert cfg.metrics_fn is count_bonus.metrics


# --- against the control ----------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_beta_zero_trains_the_control_bit_for_bit(
    methods_dir: Path, variant: str, same: Callable[[Any, Any], bool]
) -> None:
    control, method = METHODS[variant], METHODS[f"count-bonus-{variant}"]
    runs = [
        trainer.train_chunk(trainer.init(cfg, jax.random.key(0)), 2)[0]
        for trainer, cfg in (
            (control, _control(methods_dir, variant)),
            (method, _config(methods_dir, variant, beta=0.0)),
        )
    ]
    assert same(runs[0].train_state, runs[1].train_state)
    assert int(runs[1].method_state.table.sum()) > 0  # it counted all the same


ONE_ENV = {"envs": 1, "minibatches": 1, "num_steps": 3, "beta": 0.3}
"""A 3-step, 1-env rollout of the hand-computed targets."""


def test_ppo_gae_targets_are_over_reward_plus_bonus(methods_dir: Path) -> None:
    """GAE (gamma 0.99, lambda 0.8) by hand over r + r_int equals the batch target."""
    trainer = METHODS["count-bonus-ppo-rnn"]
    cfg = trainer.config(
        _config(methods_dir, "ppo-rnn", gae_lambda=0.8, gamma=0.99, **ONE_ENV)
    )
    state = ppo_rnn.init_state(cfg, jax.random.key(0))
    new, batch, traj, _ = ppo_rnn.collect(state, with_traj=True)
    r_int, d = np.asarray(traj.extra["r_int"])[:, 0], np.asarray(traj.dones)[:, 0]
    assert r_int[0] == pytest.approx(0.3) or d[0]  # the first visit pays beta
    assert (r_int > 0).sum() == (~d).sum()  # every live step pays
    net = ppo_rnn.build_network(cfg, 17, traj.obs.shape[-1])
    params = state.train_state.params
    _, _, last_v = ppo_rnn.apply_network(
        net, params, new.hidden, new.obs[None], new.last_done[None]
    )
    r = np.asarray(traj.rewards)[:, 0] + r_int
    v = np.append(np.asarray(batch["v"])[:, 0], np.asarray(last_v)[0, 0])
    adv, run = np.zeros(3), 0.0
    for t in reversed(range(3)):
        delta = r[t] + 0.99 * v[t + 1] * (1 - d[t]) - v[t]
        run = delta + 0.99 * 0.8 * (1 - d[t]) * run
        adv[t] = run
    np.testing.assert_allclose(np.asarray(batch["tgt"])[:, 0], adv + v[:3], atol=1e-5)


def test_pqn_q_lambda_targets_are_over_reward_plus_bonus(
    methods_dir: Path, same: Callable[[Any, Any], bool]
) -> None:
    """learn on r and r_int equals learn on r + r_int; Q(lambda 0.5) by hand."""
    trainer = METHODS["count-bonus-pqn-rnn"]
    cfg = trainer.config(_config(methods_dir, "pqn-rnn", lam=0.5, **ONE_ENV))
    state = pqn_rnn.init_state(cfg, jax.random.key(0))
    _, traj, h0, _ = pqn_rnn.collect(state)
    r_int = traj.extra["r_int"]
    assert float(r_int.sum()) > 0
    folded = traj.replace(
        rewards=traj.rewards + r_int, extra=traj.extra | {"r_int": 0 * r_int}
    )
    key = jax.random.key(1)
    a, _ = pqn_rnn.learn(state.train_state, traj, key, h0=h0)
    b, _ = pqn_rnn.learn(state.train_state, folded, key, h0=h0)
    assert same(a.params, b.params)
    q = np.asarray(jax.random.normal(jax.random.key(2), (2, 1)))[:, 0]
    r = np.asarray(traj.rewards + r_int)[:2, 0]
    d = np.asarray(traj.dones)[:2, 0].astype(float)
    g1 = r[1] + 0.99 * (1 - d[1]) * q[1]
    g0 = r[0] + 0.99 * (1 - d[0]) * (0.5 * q[0] + 0.5 * g1)
    got = pqn_rnn.q_lambda_targets(
        q[:, None], r[:, None], d[:, None], cfg.gamma, cfg.lam
    )
    np.testing.assert_allclose(np.asarray(got)[:, 0], [g0, g1], atol=1e-5)


# --- the trainer interface ----------------------------------------------------------


@pytest.fixture(scope="module", params=VARIANTS)
def entry(request: pytest.FixtureRequest, methods_dir: Path) -> Any:
    """(trainer, config, start state, 2-update result) of one variant."""
    trainer = METHODS[f"count-bonus-{request.param}"]
    cfg = _config(methods_dir, request.param)
    start = trainer.init(cfg, jax.random.key(0))
    return trainer, cfg, start, trainer.train_chunk(start, 2)


def test_both_entries_expose_the_trainer_interface(entry: Any) -> None:
    trainer, cfg, _, _ = entry
    names = ("steps_per_update", "init", "train_chunk", "params", "policy")
    assert all(callable(getattr(trainer, n)) for n in (*names, "env_wrapper"))
    assert trainer.steps_per_update(cfg) == 64


def test_two_updates_report_chunk_and_count_metrics(entry: Any) -> None:
    _, _, _, (_, m) = entry
    shapes = {k: np.asarray(v).shape for k, v in m.items()}
    count = {k for k in m if k.startswith("count/")}
    assert {"count/distinct_keys", "count/occupancy"} <= count
    assert {"rnn/hidden_norm", "reward/extrinsic", "reward/intrinsic"} <= set(m)
    assert shapes == {
        "losses": (2, len(LOSS_COLUMNS)),
        "done": (2, 8, 8),
        "ach": (2, 8, 8, 22),
        "ret": (2, 8, 8),
        "length": (2, 8, 8),
        **{k: (2,) for k in m if "/" in k},  # the control's own metrics come along
    }
    assert all(np.isfinite(np.asarray(m[k], float)).all() for k in count)


def test_chunks_of_one_equal_one_chunk_of_two(
    entry: Any, same: Callable[[Any, Any], bool]
) -> None:
    trainer, _, start, (whole, m) = entry
    one, m1 = trainer.train_chunk(start, 1)
    two, m2 = trainer.train_chunk(one, 1)
    assert same(two, whole)
    joined = {k: np.concatenate([np.asarray(m1[k]), np.asarray(m2[k])]) for k in m}
    assert same(joined, {k: np.asarray(v) for k, v in m.items()})


def test_the_count_table_is_saved_and_a_resume_continues_it(
    entry: Any, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    trainer, cfg, start, (whole, m) = entry
    one, m1 = trainer.train_chunk(start, 1)
    save_state(tmp_path, one)
    restored = load_state(tmp_path, trainer.init(cfg, jax.random.key(1)))
    assert same(restored.method_state, one.method_state)
    resumed, _ = trainer.train_chunk(restored, 1)
    assert same(resumed.method_state, whole.method_state)
    live = 2 * 64 - int(np.asarray(m["done"]).sum())  # done steps count nothing
    assert int(resumed.method_state.table.sum()) == live
    assert int(one.method_state.table.sum()) == 64 - int(np.asarray(m1["done"]).sum())


def test_the_policy_acts_on_observations(entry: Any) -> None:
    trainer, _, _, (state, _) = entry
    policy, params = trainer.policy(state), trainer.params(state)
    act, _ = policy.step(
        params, policy.init_carry(8), state.obs, jax.random.key(0), greedy=True
    )
    assert act.shape == (8,)
    assert trainer.env_wrapper(state) is None


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_done_step_unlock_reaches_the_chunk_metrics(
    methods_dir: Path, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    width = {"width": 17} if variant == "ppo-rnn" else {"hidden": 17}  # uncompiled
    cfg = _config(methods_dir, variant, envs=16, num_steps=32, **width)
    trainer = METHODS[f"count-bonus-{variant}"]
    assert_done_step_unlock_counted(trainer, cfg, monkeypatch)


_ON_TWO_CPUS = """
import sys
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime(host_devices=2)
from wut_murl_craftax.training.interface import METHODS

base = {"env": "Craftax-Classic-Symbolic-v1", "envs": 8, "num_steps": 8,
        "minibatches": 2, "steps": 128, "beta": 0.1, "granularity": "recipe",
        "decay": "constant", "log2_k": 20, "key": "progress",
        "max_occupancy": 0.25}
sizes = {"ppo-rnn": {"width": 16, "rnn_width": 16}, "pqn-rnn": {"hidden": 16}}
for variant, size in sizes.items():
    trainer = METHODS["count-bonus-" + variant]
    state, m = trainer.train_chunk(trainer.init(base | size, jax.random.key(0)), 2)
    table = state.method_state.table
    assert len(table.sharding.device_set) == 2, (variant, table.sharding)
    assert int(table.sum()) == 128 - int(m["done"].sum()), variant
print("ok")
"""


def test_the_table_stays_on_the_actor_devices(
    start_on_cpus: Callable[[str, int, Path], Any], tmp_path: Path
) -> None:
    """Two host CPUs: the table is replicated over the actor mesh after updates (the
    metrics read it there, not on the learner device)."""
    proc = start_on_cpus(_ON_TWO_CPUS, 2, tmp_path)
    out, err = proc.communicate(timeout=900)
    assert proc.returncode == 0, err[-3000:]
    assert out.strip().splitlines()[-1] == "ok"


# --- through the runner ------------------------------------------------------------


def test_the_runner_trains_and_evaluation_opens_both_variants(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    """A 2-update Classic run per variant: FINISHED, count/* logged, and the run
    directory reopens with the same configured trainer."""
    phase = tmp_path / "count.yaml"
    entry = {"config": "count-bonus", "configs": [CONFIG_A], "seeds": 1}
    phase.write_text(
        yaml.safe_dump({
            "budget": "0.128k",
            "eval": "final",
            "env": CLASSIC,
            "runs": {f"count-bonus-{v}": entry for v in VARIANTS},
        })
    )
    specs = expand_phase(phase, methods_dir=methods_dir)
    assert len(specs) == 2
    runs = run_store.run(specs)
    for spec in specs:
        run = runs[spec.name]
        assert run.info.status == "FINISHED"
        assert {"count/distinct_keys", "count/occupancy"} <= run.data.metrics.keys()
        out = artifact_path(run.info.artifact_uri)
        with np.load(out / EXTRAS_FILE) as z:
            distinct = z["count/distinct_keys"]
        assert len(distinct) == 2
        assert (np.diff(distinct) >= 0).all()
        assert distinct[0] > 0
        saved = json.loads((out / "config.json").read_text())
        trainer = METHODS[spec.registry_name]
        assert trainer.config(saved) == trainer.config(spec.config)
        opened = eval_rollouts.EvalRun.load(out, METHODS)
        assert opened.method == spec.registry_name
        knobs = {k: saved[k] for k in KNOBS}
        assert count_bonus.CountBonusConfig(**knobs) == CountBonusConfig(
            beta=0.1,
            granularity="recipe",
            decay="constant",
            log2_k=20,
            key="progress",
            max_occupancy=0.25,
        )
        params = opened.params(opened.updates[-1])
        obs, _ = opened.env.reset(jax.random.key(0), opened.env_params)
        act, _ = opened.policy.step(
            params, opened.policy.init_carry(1), obs[None], jax.random.key(1), True
        )
        assert act.shape == (1,)


def test_the_simhash_ablation_overrides_build_a_trainer(methods_dir: Path) -> None:
    """The final-1m ablation set count-bonus.yaml documents, on the PPO variant."""
    text = (REPO / "configs" / "methods" / "count-bonus.yaml").read_text()
    line = "set: {key: simhash, log2_k: 24, max_occupancy: 0.1}"
    assert line in text
    over = yaml.safe_load(line)["set"]
    cfg = METHODS["count-bonus-ppo-rnn"].config(_config(methods_dir, "ppo-rnn", **over))
    state = jax.eval_shape(lambda: cfg.init_state_fn(jax.random.key(0), _SPEC))
    assert state.table.shape == (2**24,)
    assert state.seen.shape[0] == 0


_SPEC = HookSpec(
    n_envs=8, obs_dim=8268, n_actions=43, n_achievements=67, total_steps=10**6
)
