"""Tests for the SIL registry entry: config file, hooks, gate, placement, runs."""

import dataclasses
import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from _contract import assert_done_step_unlock_counted
from _methods import copy_methods

from wut_murl_craftax.evaluation.episode_metrics import ONLINE_LAYOUT
from wut_murl_craftax.orchestration.abort import (
    TAG_ABORT_MEMORY,
    TAG_ABORT_METRIC,
    TAG_ABORT_UPDATE,
)
from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.orchestration.runner import (
    EXTRAS_FILE,
    TAG_FAIL_COUNT,
    TAG_FAIL_ERROR,
)
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import ppo_rnn, sil
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.hooks import (
    HOOK_NAMES,
    MEMORY_LIMIT_KEY,
    HookSpec,
    MemoryLimitError,
    resolve,
)
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import PPOConfig
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig
from wut_murl_craftax.training.records import LOSS_COLUMNS
from wut_murl_craftax.visualisation import eval_rollouts

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "tests" / "orchestration" / "fixtures" / "profiles"
CLASSIC = "Craftax-Classic-Symbolic-v1"
TINY = {
    "env": CLASSIC,
    "envs": 8,
    "num_steps": 8,
    "minibatches": 2,
    "update_epochs": 1,
    "width": 16,
    "rnn_width": 16,
}
"""Tuned block of the temporary PPO-RNN control copy: tiny Classic runs, 64 steps."""
KNOBS = {f.name for f in dataclasses.fields(sil.SILConfig)}
"""The method's run-config keys."""
CONFIG = "l0.3-s2-k2"
"""The coarse entry the tests train."""
LEARNER_HOOKS = (
    "init_state_fn",
    "observe",
    "extra_loss_fn",
    "after_extra_fn",
    "metrics_fn",
)


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of configs/methods with the PPO-RNN control's ``tuned`` filled in."""
    return copy_methods(tmp_path_factory.mktemp("methods"), {"ppo-rnn": TINY})


def _config(methods_dir: Path, ref: str = CONFIG, **over: Any) -> dict[str, Any]:
    """A resolved SIL run config with a 4-update budget and overrides."""
    method = load_method("sil", methods_dir)
    cfg = resolve_config(method, "sil", "1m", ref=ref, methods_dir=methods_dir)
    return cfg | {"steps": 4 * 64} | over


def _control(methods_dir: Path, **over: Any) -> dict[str, Any]:
    """The PPO-RNN control's tuned run config with the same budget."""
    method = load_method("ppo-rnn", methods_dir)
    cfg = resolve_config(method, "ppo-rnn", "1m", ref="tuned", methods_dir=methods_dir)
    return cfg | {"steps": 4 * 64} | over


def _adam(ts: Any) -> list[Any]:
    """The Adam moments and count of a train state."""
    states = jax.tree.leaves(
        ts.opt_state, is_leaf=lambda s: isinstance(s, optax.ScaleByAdamState)
    )
    return [s for s in states if isinstance(s, optax.ScaleByAdamState)]


def _schedule_count(ts: Any) -> int:
    (s,) = [
        s
        for s in jax.tree.leaves(
            ts.opt_state, is_leaf=lambda s: isinstance(s, optax.ScaleByScheduleState)
        )
        if isinstance(s, optax.ScaleByScheduleState)
    ]
    return int(s.count)


# --- config file -------------------------------------------------------------------


def test_the_method_file_has_the_base_and_both_grids() -> None:
    method = load_method("sil")
    assert method.registry_names == ["sil"]
    assert method.base["base_from"] == "ppo-rnn.tuned"
    assert method.tuned is None or set(method.tuned) <= KNOBS  # null until marked
    # every knob is set in the base, so no dataclass default decides a run
    assert method.base.keys() - {"base_from"} == KNOBS
    grid = {
        (e["lambda_sil"], e["slots_windows"], e["sil_steps"]) for e in method.coarse
    }
    assert len(method.coarse) == len(grid) == 18
    assert grid == {
        (lam, s, k) for lam in (0.1, 0.3, 1.0) for s in (2, 8) for k in (1, 2, 4)
    }
    fine = method.fine
    assert fine is not None
    assert fine["rule"] == {"select": 2, "combine": "cross"}
    assert {k["key"] for k in fine["knobs"]} == {
        "lambda_sil",
        "slots_windows",
        "sil_steps",
    }
    for knob in fine["knobs"]:
        assert knob["mul"] == [0.5, 1.0, 2.0]
        assert knob.get("integer", False) == (knob["key"] != "lambda_sil")


def test_gamma_and_the_gru_width_come_from_the_control(methods_dir: Path) -> None:
    """No duplicated YAML keys: the method has neither key, its state copies both."""
    assert {"gamma", "rnn_width"}.isdisjoint(KNOBS)
    cfg = METHODS["sil"].config(_config(methods_dir, gamma=0.9))
    spec = HookSpec(8, 1345, 17, 22, 256)
    state = jax.eval_shape(lambda: cfg.init_state_fn(jax.random.key(0), spec))
    assert state.gamma == 0.9
    assert state.staging["hidden"].shape == (8, 128, 16)
    assert state.cfg.sil_steps == cfg.extra_steps == 2


def test_the_hooks_are_the_learner_hooks_and_metrics(methods_dir: Path) -> None:
    cfg = METHODS["sil"].config(_config(methods_dir))
    assert resolve(cfg).active == LEARNER_HOOKS
    assert cfg.intrinsic_reward_fn is None
    assert cfg.env_wrapper_fn is None
    assert cfg.observe is sil.observe
    assert cfg.metrics_fn is sil.metrics
    assert cfg.after_extra_fn is sil.after_extra
    loss = cfg.extra_loss_fn
    assert isinstance(loss, functools.partial)
    assert loss.func is sil.extra_loss
    assert loss.args == (ppo_rnn.build_network(cfg, 17, 1345),)
    assert set(HOOK_NAMES) - set(LEARNER_HOOKS) == {
        "intrinsic_reward_fn",
        "env_wrapper_fn",
    }


# --- against the control ------------------------------------------------------------


BIG = {"envs": 16, "num_steps": 32}
"""A longer rollout: random Classic play unlocks something within 2 updates."""


def test_no_sil_step_trains_the_control_bit_for_bit(
    methods_dir: Path, same: Callable[[Any, Any], bool]
) -> None:
    """K = 0: the control's params after 2 updates, while observe fills the buffer."""
    ctrl = METHODS["ppo-rnn"]
    a, _ = ctrl.train_chunk(
        ctrl.init(_control(methods_dir, **BIG), jax.random.key(0)), 2
    )
    trainer = METHODS["sil"]
    start = trainer.init(_config(methods_dir, sil_steps=0, **BIG), jax.random.key(0))
    b, _ = trainer.train_chunk(start, 2)
    assert same(a.train_state, b.train_state)
    assert int((b.method_state.buffer["stamp"] >= 0).sum()) > 0
    assert not np.asarray(b.method_state.staging["done"]).all()


@pytest.fixture(scope="module")
def no_observe(methods_dir: Path) -> Any:
    """The SIL trainer config with ``observe`` off: the buffer is only what a test
    writes into it."""
    cfg = METHODS["sil"].config(_config(methods_dir))
    return dataclasses.replace(cfg, observe=None)


def _with_window(state: Any, ret: float) -> Any:
    """``state`` with one stored window of return ``ret`` at every step in slot 0."""
    ms = state.method_state
    buf = ms.buffer | {
        "stamp": ms.buffer["stamp"].at[0].set(0),
        "mask": ms.buffer["mask"].at[0].set(True),
        "ret": ms.buffer["ret"].at[0].set(ret),
        "ach": ms.buffer["ach"].at[0].set(0),
    }
    return state.replace(method_state=ms.replace(buffer=buf))


@pytest.mark.parametrize("ret", [None, -100.0])
def test_a_sil_step_with_no_active_step_leaves_adam_untouched(
    methods_dir: Path,
    no_observe: Any,
    same: Callable[[Any, Any], bool],
    ret: float | None,
) -> None:
    """Empty buffer, or windows where V beats R: params and Adam stay the control's
    after 1 update; only the learning-rate clock moves on by K."""
    ctrl = METHODS["ppo-rnn"]
    a, _ = ctrl.train_chunk(ctrl.init(_control(methods_dir), jax.random.key(0)), 1)
    start = ppo_rnn.init_state(no_observe, jax.random.key(0))
    if ret is not None:
        start = _with_window(start, ret)
    b, m = ppo_rnn.train_chunk(start, 1)
    assert same(a.train_state.params, b.train_state.params)
    assert same(_adam(a.train_state), _adam(b.train_state))
    assert _schedule_count(b.train_state) == _schedule_count(a.train_state) + 2
    assert int(b.train_state.step) == int(a.train_state.step) + 2
    assert float(m["sil/active"][0]) == 0.0  # type: ignore[typeddict-item]
    # skipped steps: no gradient and an unchanged policy
    assert float(m["sil/grad_norm"][0]) == float(m["sil/kl_after"][0]) == 0.0  # type: ignore[typeddict-item]


def test_kl_after_skips_its_forward_passes_when_every_step_is_skipped(
    methods_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``kl_after`` is 0 without the two batch passes; an applied step runs them."""
    calls: list[int] = []
    real = ppo_rnn._policy_kl

    def counted(*args: Any) -> jax.Array:
        jax.debug.callback(lambda: calls.append(1))
        return real(*args)

    monkeypatch.setattr(ppo_rnn, "_policy_kl", counted)
    cfg = METHODS["sil"].config(_config(methods_dir, width=21))  # traces the patch
    cfg = dataclasses.replace(cfg, observe=None)  # the buffer is what the test writes
    start = ppo_rnn.init_state(cfg, jax.random.key(0))
    _, m = ppo_rnn.train_chunk(start, 1)
    jax.effects_barrier()
    assert (calls, float(m["sil/kl_after"][0])) == ([], 0.0)  # type: ignore[typeddict-item]
    _, m = ppo_rnn.train_chunk(_with_window(start, 100.0), 1)
    jax.effects_barrier()
    assert calls == [1]
    assert float(m["sil/kl_after"][0]) > 0  # type: ignore[typeddict-item]


def test_a_sil_step_with_active_steps_moves_the_params(
    methods_dir: Path, no_observe: Any, same: Callable[[Any, Any], bool]
) -> None:
    ctrl = METHODS["ppo-rnn"]
    a, _ = ctrl.train_chunk(ctrl.init(_control(methods_dir), jax.random.key(0)), 1)
    start = _with_window(ppo_rnn.init_state(no_observe, jax.random.key(0)), 100.0)
    _, batch, _, _ = ppo_rnn.collect(start)
    b, m = ppo_rnn.train_chunk(start, 1)
    assert not same(a.train_state.params, b.train_state.params)
    # the control's params are SIL's before its steps: same seed, same PPO epochs
    net = ppo_rnn._setup(no_observe).net

    def logp(params: Any) -> jax.Array:
        h0, obs, reset = batch["h0"], batch["obs"], batch["reset"]
        return jax.nn.log_softmax(ppo_rnn.apply_network(net, params, h0, obs, reset)[1])

    before, after = logp(a.train_state.params), logp(b.train_state.params)
    kl = float(jnp.mean(jnp.sum(jnp.exp(before) * (before - after), -1)))
    assert kl > 0
    assert float(m["sil/kl_after"][0]) == pytest.approx(kl, rel=1e-4)  # type: ignore[typeddict-item]
    assert float(m["sil/grad_norm"][0]) > 0  # type: ignore[typeddict-item]
    assert int(_adam(b.train_state)[0].count) == int(_adam(a.train_state)[0].count) + 2
    assert float(m["sil/active"][0]) == 1.0  # type: ignore[typeddict-item]


# --- the trainer interface ------------------------------------------------------------


@pytest.fixture(scope="module")
def entry(methods_dir: Path) -> Any:
    """(trainer, config, start state, 2-update result) of the SIL entry."""
    trainer = METHODS["sil"]
    cfg = _config(methods_dir)
    start = trainer.init(cfg, jax.random.key(0))
    return trainer, cfg, start, trainer.train_chunk(start, 2)


def test_the_entry_exposes_the_trainer_interface(entry: Any) -> None:
    trainer, cfg, _, _ = entry
    names = ("steps_per_update", "init", "train_chunk", "params", "policy")
    assert all(callable(getattr(trainer, n)) for n in (*names, "env_wrapper"))
    assert trainer.steps_per_update(cfg) == 64


def test_two_updates_report_chunk_and_sil_metrics(entry: Any) -> None:
    _, _, _, (_, m) = entry
    shapes = {k: np.asarray(v).shape for k, v in m.items()}
    names = {k for k in m if k.startswith("sil/")}
    assert {"sil/active", "sil/policy_loss", "sil/inserted", "sil/overflow"} <= names
    control = {
        "ppo/entropy",
        "ppo/ratio_first_mb",
        "rnn/hidden_norm",
        "reward/intrinsic",
    }
    assert control <= set(m)  # the PPO-RNN control's own metrics come with the method's
    assert shapes == {
        "losses": (2, len(LOSS_COLUMNS)),
        "done": (2, 8, 8),
        "ach": (2, 8, 8, 22),
        "ret": (2, 8, 8),
        "length": (2, 8, 8),
        "online": (2, ONLINE_LAYOUT["cause"].stop),
        "first_unlock": (2, 22),
        **{k: (2,) for k in m if "/" in k},
    }
    assert all(np.isfinite(np.asarray(m[k], float)).all() for k in names)


def test_chunks_of_one_equal_one_chunk_of_two(
    entry: Any, same: Callable[[Any, Any], bool]
) -> None:
    """Also: a chunk leaves its input state usable (the learner donates a copy)."""
    trainer, _, start, (whole, m) = entry
    one, m1 = trainer.train_chunk(start, 1)
    two, m2 = trainer.train_chunk(one, 1)
    assert same(two, whole)
    joined = {k: np.concatenate([np.asarray(m1[k]), np.asarray(m2[k])]) for k in m}
    assert same(joined, {k: np.asarray(v) for k, v in m.items()})


def test_the_buffer_is_saved_and_a_resume_continues_it(
    entry: Any, same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    trainer, cfg, start, (whole, _) = entry
    one, _ = trainer.train_chunk(start, 1)
    save_state(tmp_path, one)
    restored = load_state(tmp_path, trainer.init(cfg, jax.random.key(1)))
    assert same(restored.method_state, one.method_state)
    assert not np.asarray(restored.method_state.staging["done"]).all()
    resumed, _ = trainer.train_chunk(restored, 1)
    assert same(resumed.method_state, whole.method_state)
    assert same(resumed.train_state, whole.train_state)


def test_the_policy_acts_on_observations(entry: Any) -> None:
    trainer, _, _, (state, _) = entry
    policy, params = trainer.policy(state), trainer.params(state)
    act, _ = policy.step(
        params, policy.init_carry(8), state.obs, jax.random.key(0), greedy=True
    )
    assert act.shape == (8,)
    assert trainer.env_wrapper(state) is None


def test_a_done_step_unlock_reaches_the_chunk_metrics(
    methods_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(methods_dir, envs=16, num_steps=32, width=17)  # uncompiled
    assert_done_step_unlock_counted(METHODS["sil"], cfg, monkeypatch)


_ON_TWO_CPUS = """
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime(host_devices=2)
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo import PPOConfig
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig

cfg = {"env": "Craftax-Classic-Symbolic-v1", "envs": 8, "num_steps": 8,
       "minibatches": 2, "steps": 128, "width": 16, "rnn_width": 16,
       "lambda_sil": 0.3, "slots_windows": 2, "sil_steps": 1, "window": 128,
       "beta_v": 0.01, "batch": 32, "alpha": 0.6, "insert_cap": 64,
       "layout": "slots"}
trainer = METHODS["sil"]
state = trainer.init(cfg, jax.random.key(0))

def one_device(s):
    return all(len(x.sharding.device_set) == 1 for x in jax.tree.leaves(s))

assert len(jax.devices("cpu")) == 2
assert one_device(state.method_state)
assert len(state.obs.sharding.device_set) == 2
state, m = trainer.train_chunk(state, 2)
assert one_device(state.method_state)
print("ok")
"""


def test_the_buffer_stays_on_the_learner(
    start_on_cpus: Callable[[str, int, Path], Any], tmp_path: Path
) -> None:
    """Two host CPUs: the env batch is split over them, the SIL state is on the
    learner alone (not replicated to the actor mesh) before and after updates."""
    proc = start_on_cpus(_ON_TWO_CPUS, 2, tmp_path)
    out, err = proc.communicate(timeout=900)
    assert proc.returncode == 0, err[-3000:]
    assert out.strip().splitlines()[-1] == "ok"


# --- through the runner ---------------------------------------------------------------


def _phase(tmp_path: Path, methods_dir: Path, config: Any = CONFIG) -> list[Any]:
    """The specs of a 2-update Classic phase training one SIL config."""
    phase = tmp_path / "sil.yaml"
    entry = {"config": "sil", "configs": [config], "seeds": 1}
    phase.write_text(
        yaml.safe_dump({
            "budget": "0.128k",
            "eval": "final",
            "env": CLASSIC,
            "runs": {"sil": entry},
        })
    )
    return expand_phase(phase, methods_dir=methods_dir)


def test_the_runner_trains_and_evaluation_opens_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path
) -> None:
    """A 2-update Classic run: FINISHED, sil/* logged, and the run directory reopens
    with the same configured trainer."""
    (spec,) = _phase(tmp_path, methods_dir)
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FINISHED"
    assert {"sil/active", "sil/policy_loss"} <= run.data.metrics.keys()
    out = artifact_path(run.info.artifact_uri)
    with np.load(out / EXTRAS_FILE) as z:
        assert len(z["sil/active"]) == 2
    saved = json.loads((out / "config.json").read_text())
    assert "memory_limit_bytes" not in saved
    trainer = METHODS["sil"]
    assert trainer.config(saved) == trainer.config(spec.config)
    opened = eval_rollouts.EvalRun.load(out, METHODS)
    assert opened.method == "sil"
    params = opened.params(opened.updates[-1])
    obs, _ = opened.env.reset(jax.random.key(0), opened.env_params)
    act, _ = opened.policy.step(
        params, opened.policy.init_carry(1), obs[None], jax.random.key(1), True
    )
    assert act.shape == (1,)


# --- abort rules ----------------------------------------------------------------------


def test_the_abort_list_holds_only_always_finite_metrics() -> None:
    """The loss parts are 0, not NaN, on an empty buffer (`test_sil_loss`), and NaN
    once the loss or the params are; the PPO-RNN control's rules come first."""
    control = load_method("ppo-rnn").abort or {}
    own = ["sil/policy_loss", "sil/value_loss", "sil/dominates"]
    assert load_method("sil").abort == {
        "finite": control["finite"] + own,
        "permanent": ["sil/dominates"],  # the KL streak recurs on a retry
        "limits": control["limits"],  # the SIL steps follow the PPO epochs: ratio 1
        "flags": control["flags"],  # kl_high watches the off-policy SIL steps
    }


def test_trainer_configs_take_the_memory_limit_as_a_runtime_key(
    methods_dir: Path,
) -> None:
    """The runner adds it to every run config like ``steps``; no config keeps it."""
    over = {MEMORY_LIMIT_KEY: 10**9}
    assert PPOConfig.from_config(over) == PPOConfig()
    assert PQNRNNConfig.from_config(over) == PQNRNNConfig()
    configs = {"sil": _config(methods_dir), "ppo-rnn": _control(methods_dir)}
    for name, cfg in configs.items():
        assert METHODS[name].config(cfg | over) == METHODS[name].config(cfg)


def test_a_buffer_over_the_machine_limit_fails_at_init(methods_dir: Path) -> None:
    """Craftax-Symbolic, S = 2 on 8 envs (155 MB) under a 100 MB profile limit."""
    cfg = _config(methods_dir, env="Craftax-Symbolic-v1", width=19, rnn_width=16)
    need = sil.buffer_bytes(sil.SILConfig(slots_windows=2), 8, 67, rnn_width=16)
    assert need.total > 100 * 10**6
    trainer = METHODS["sil"]
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        trainer.init(cfg | {MEMORY_LIMIT_KEY: 100 * 10**6}, jax.random.key(0))
    state = jax.eval_shape(  # without the key (evaluation), nothing is checked
        lambda: trainer.init(cfg, jax.random.key(0))
    )
    assert state.method_state.buffer["map"].shape[0] == 134


def test_a_run_over_the_profile_limit_is_a_permanent_abort(
    methods_dir: Path,
    run_store: Any,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FAILED with ``abort/memory``; later queue calls skip it instead of retrying."""
    (spec,) = _phase(tmp_path, methods_dir)
    profile = yaml.safe_load((PROFILES / "test.yaml").read_text())
    need = sil.buffer_bytes(sil.SILConfig(), 8, 22, 1345, rnn_width=16).total
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(profile | {"memory_limit_bytes": 3 * need - 1}))
    run = run_store.run([spec], path)[spec.name]  # an abort: the queue goes on
    assert run.info.status == "FAILED"
    assert run.data.tags[TAG_ABORT_MEMORY] == "true"
    assert "memory_limit_bytes" in run.data.tags[TAG_FAIL_ERROR]
    assert not (artifact_path(run.info.artifact_uri) / EXTRAS_FILE).exists()
    with caplog.at_level("WARNING"):
        again = run_store.run([spec], path)[spec.name]
    assert f"{spec.name} is over the machine's memory limit" in caplog.text
    assert again.data.tags[TAG_FAIL_COUNT] == "1"
    assert again.info.end_time == run.info.end_time  # not reopened


def test_check_memory_raises_the_memory_limit_error() -> None:
    spec = HookSpec(256, 8268, 43, 67, 10**6)
    with pytest.raises(MemoryLimitError):
        sil.check_memory(sil.SILConfig(), spec, 10**6)


def test_a_nan_sil_loss_aborts_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = sil.sil_terms

    def nan_terms(*args: Any) -> Any:
        loss, aux = real(*args)
        return loss * jnp.nan, aux | {"policy": aux["policy"] * jnp.nan}

    monkeypatch.setattr(sil, "sil_terms", nan_terms)
    # a width no other test compiles, so the patched loss is traced
    (spec,) = _phase(tmp_path, methods_dir, {"ref": CONFIG, "set": {"width": 21}})
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FAILED"
    assert run.data.tags[TAG_ABORT_METRIC] == "sil/policy_loss"
    assert run.data.tags[TAG_ABORT_UPDATE] == "1"


def test_kl_after_over_the_limit_for_long_enough_aborts_the_run(
    methods_dir: Path, run_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the limit at -1 and 2 updates in a row, the 2nd update aborts."""
    monkeypatch.setattr(sil, "KL_LIMIT", -1.0)
    monkeypatch.setattr(sil, "KL_UPDATES", 2)
    # a width no other test compiles, so the patched constants are traced
    (spec,) = _phase(tmp_path, methods_dir, {"ref": CONFIG, "set": {"width": 23}})
    run = run_store.run([spec])[spec.name]
    assert run.info.status == "FAILED"
    assert run.data.tags[TAG_ABORT_METRIC] == "sil/dominates"
    assert run.data.tags[TAG_ABORT_UPDATE] == "2"
