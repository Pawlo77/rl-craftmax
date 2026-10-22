"""Tests for `combine` and the `combo` entry: composition, config, Classic runs."""

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from _methods import copy_methods

from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.training import count_bonus, sil
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.dyna_q import DynaTrainer
from wut_murl_craftax.training.hooks import (
    HookFields,
    HookMethod,
    HookSpec,
    SplitState,
    resolve,
)
from wut_murl_craftax.training.interface import METHODS, combine, combo_trainer
from wut_murl_craftax.training.ppo_rnn import PPORNNTrainer

REPO = Path(__file__).resolve().parents[2]
SPEC = HookSpec(4, 1345, 17, 22, 256)
KEY = jax.random.key(0)
DONE = jnp.zeros(4, bool)


def _method(
    prefix: str,
    *,
    loss: bool = False,
    wrap: bool = False,
    group: str = "",
    reward: bool = True,
    size: int | None = None,
) -> HookMethod[Any]:
    """A fixture hook method with keys ``<prefix>_r`` and ``<prefix>_k``.

    Its state (int32, shape ``(size,)`` or scalar) counts calls: +1 per reward step,
    +10 per observe, +100 per extra step, +1000 per after-extra; reward and extra loss
    are ``<prefix>_r``; its metric is ``<group or prefix>/n``; ``artifacts`` records the
    state it gets, ``init_state_fn`` the spec.
    """
    fields = [(f"{prefix}_r", float, 1.0), (f"{prefix}_k", int, 1)]
    config = dataclasses.make_dataclass(f"{prefix}Config", fields, frozen=True)
    name = f"{group or prefix}/n"

    def hooks(cfg: Any, base: Any) -> HookFields:
        r, k = getattr(cfg, f"{prefix}_r"), getattr(cfg, f"{prefix}_k")

        def bonus(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
            return jnp.full(d.shape, r, jnp.float32), s + 1

        def init(key: jax.Array, spec: HookSpec) -> jax.Array:
            _SPECS[prefix] = spec
            return jnp.zeros(() if size is None else (size,), jnp.int32)

        def extra(params: Any, s: Any, key: jax.Array) -> Any:
            return jnp.float32(r), (s + 100, {f"{prefix}/loss": jnp.float32(r)})

        return HookFields(
            init_state_fn=init,
            intrinsic_reward_fn=bonus if reward else None,
            observe=lambda s, traj: s + 10,
            extra_loss_fn=extra if loss else None,
            extra_steps=k if loss else 0,
            after_extra_fn=(lambda s, m: s + 1000) if loss else None,
            env_wrapper_fn=(lambda env, p: (env, p)) if wrap else None,
            metrics_fn=lambda s: {name: s},
        )

    seen: list[Any] = []
    method = HookMethod(config, hooks, lambda run_dir, update, s: seen.append(s))
    _SEEN[prefix] = seen
    return method


_SEEN: dict[str, list[Any]] = {}
"""States each fixture method's ``artifacts`` received, by prefix."""
_SPECS: dict[str, HookSpec] = {}
"""The spec each fixture method's ``init_state_fn`` last received, by prefix."""


@dataclasses.dataclass(frozen=True)
class _SplitBase:
    """A base config whose trainer places a `SplitState` by parts (as PPO-RNN)."""

    splits_method_state: ClassVar[bool] = True


def _hooks(
    a: HookMethod[Any], b: HookMethod[Any], base: Any = None, **knobs: Any
) -> HookFields:
    pair = combine(a, b)
    return pair.hooks(pair.config(**knobs), base)


# --- composition -----------------------------------------------------------------


def test_the_pair_takes_both_methods_keys() -> None:
    pair = combine(_method("a"), _method("b", loss=True))
    assert pair.keys == {"a_r", "a_k", "b_r", "b_k"}
    assert pair.config(a_r=2.0).a_r == 2.0


def test_rewards_add_and_observe_runs_both_on_their_own_state() -> None:
    h = _hooks(_method("a"), _method("b"), a_r=1.0, b_r=0.5)
    state = h.init_state_fn(KEY, SPEC)
    r, state = h.intrinsic_reward_fn(state, None, None, None, DONE, 0)
    assert np.asarray(r).tolist() == [1.5] * 4
    assert [int(s) for s in h.observe(state, None)] == [11, 11]


def test_the_extra_loss_and_after_extra_act_on_their_methods_part() -> None:
    h = _hooks(_method("a"), _method("b", loss=True), b_r=2.0, b_k=3)
    assert h.extra_steps == 3
    state = h.init_state_fn(KEY, SPEC)
    loss, (state, metrics) = h.extra_loss_fn(None, state, KEY)
    assert float(loss) == 2.0
    assert metrics == {"b/loss": 2.0}
    assert [int(s) for s in h.after_extra_fn(state, metrics)] == [0, 1100]


def test_metrics_merge_with_each_methods_prefix() -> None:
    h = _hooks(_method("a"), _method("b"))
    state = h.observe(h.init_state_fn(KEY, SPEC), None)
    assert {k: int(v) for k, v in h.metrics_fn(state).items()} == {"a/n": 10, "b/n": 10}


def test_a_method_without_extra_steps_adds_no_extra_loss() -> None:
    h = _hooks(_method("a"), _method("b", loss=True), b_k=0)
    assert (h.extra_loss_fn, h.after_extra_fn, h.extra_steps) == (None, None, 0)
    active = ("init_state_fn", "intrinsic_reward_fn", "observe", "metrics_fn")
    assert resolve(h).active == active


@pytest.mark.parametrize(
    ("a", "b", "field"),
    [
        (_method("a", loss=True), _method("b", loss=True), "extra_steps"),
        (_method("a", wrap=True), _method("b", wrap=True), "env_wrapper_fn"),
    ],
)
def test_a_field_set_by_both_fails_loudly(
    a: HookMethod[Any], b: HookMethod[Any], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _hooks(a, b)


def test_a_metric_name_logged_by_both_fails_loudly() -> None:
    h = _hooks(_method("a"), _method("c", group="a"))
    with pytest.raises(ValueError, match="a/n"):
        h.metrics_fn(h.init_state_fn(KEY, SPEC))


def test_a_key_of_both_methods_fails_loudly() -> None:
    with pytest.raises(ValueError, match="a_k"):
        combine(_method("a"), _method("a"))


def test_combine_refuses_a_skill_method() -> None:
    with pytest.raises(TypeError, match="hook methods only"):
        combine(_method("a"), DynaTrainer())  # type: ignore[arg-type]


def test_artifacts_get_their_methods_part_of_the_state(tmp_path: Path) -> None:
    pair = combine(_method("a"), _method("b"))
    assert pair.artifacts is not None
    pair.artifacts(tmp_path, 3, ("sa", "sb"))
    assert (_SEEN["a"], _SEEN["b"]) == (["sa"], ["sb"])


def test_a_learner_only_method_keeps_its_part_off_the_actor() -> None:
    """On a splitting base, the rollout reads only the rewarding method's part."""
    a, b = _method("a"), _method("b", loss=True, reward=False)
    h = _hooks(a, b, _SplitBase())
    state = h.init_state_fn(KEY, SPEC)
    assert isinstance(state, SplitState)
    r, actor = h.intrinsic_reward_fn(state.actor, None, None, None, DONE, 0)
    assert np.asarray(r).tolist() == [1.0] * 4
    state = h.observe(state.replace(actor=actor), None)
    _, (state, _) = h.extra_loss_fn(None, state, KEY)
    assert (
        [int(x) for x in state.actor.values()],
        [int(x) for x in state.learner.values()],
    ) == (
        [11],
        [110],
    )
    assert {k: int(v) for k, v in h.metrics_fn(state).items()} == {
        "a/n": 11,
        "b/n": 110,
    }


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (_method("a"), _method("b")),  # both read by the rollout
        (_method("a", reward=False), _method("b", reward=False)),  # both resident
    ],
)
def test_a_pair_on_one_side_keeps_one_state(
    a: HookMethod[Any], b: HookMethod[Any]
) -> None:
    assert isinstance(_hooks(a, b, _SplitBase()).init_state_fn(KEY, SPEC), tuple)


def test_each_methods_memory_limit_leaves_room_for_the_other() -> None:
    h = _hooks(_method("a", size=10), _method("b", size=30))
    h.init_state_fn(KEY, dataclasses.replace(SPEC, memory_limit_bytes=1000))
    assert _SPECS["a"].memory_limit_bytes == 1000 - 30 * 4
    assert _SPECS["b"].memory_limit_bytes == 1000 - 10 * 4
    h.init_state_fn(KEY, SPEC)
    assert _SPECS["a"].memory_limit_bytes is None


# --- combo.yaml and the registry entry ----------------------------------------------


def test_the_combo_entry_combines_the_pair_named_in_combo_yaml() -> None:
    combo = load_method("combo")
    assert combo.combine == {
        "base": "ppo-rnn",
        "methods": ["count-bonus-ppo-rnn", "sil"],
    }
    assert combo.base["base_from"] == "ppo-rnn.tuned"
    knobs = count_bonus.METHOD.keys | sil.METHOD.keys
    assert combo.base.keys() - {"base_from"} == knobs  # every knob set, no defaults
    assert [e["name"] for e in combo.coarse] == ["combined"]
    assert (combo.fine, combo.tuned) == (None, None)
    trainer = METHODS["combo"]
    assert isinstance(trainer, PPORNNTrainer)
    assert trainer.method is not None
    assert trainer.method.keys == knobs


def _write_combo(out: Path, base: str, methods: list[str]) -> Path:
    combo = yaml.safe_load((REPO / "configs/methods/combo.yaml").read_text())
    combo["combine"] = {"base": base, "methods": methods}
    combo["base"] = {"base_from": f"{base}.tuned"}
    (out / "combo.yaml").write_text(yaml.safe_dump(combo))
    return out


@pytest.mark.parametrize(
    ("base", "methods", "error", "message"),
    [
        ("pqn-rnn", ["smdp-skills", "count-bonus-pqn-rnn"], TypeError, "hook methods"),
        ("ppo-rnn", ["sil", "dyna-q"], TypeError, "hook methods only"),
        ("ppo-rnn", ["count-bonus-pqn-rnn", "sil"], ValueError, "not built on ppo-rnn"),
    ],
)
def test_combo_yaml_refuses_skill_methods_and_other_bases(
    tmp_path: Path, base: str, methods: list[str], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        combo_trainer(METHODS, _write_combo(tmp_path, base, methods))


@pytest.mark.parametrize(("phase", "seeds", "envs"), [("1m", 10, 256), ("1b", 3, 1024)])
def test_the_combo_phases_run_the_combined_config(
    phase: str, seeds: int, envs: int
) -> None:
    specs = expand_phase(REPO / f"configs/phases/combo-{phase}.yaml")
    assert [(s.registry_name, s.seed) for s in specs] == [
        ("combo", s) for s in range(seeds)
    ]
    assert {(s.budget, s.env, s.eval_mode) for s in specs} == {
        (phase, "Craftax-Symbolic-v1", "log_spaced")
    }
    assert {s.config["envs"] for s in specs} == {envs}


# --- on Craftax-Classic against the PPO-RNN control ------------------------------


TINY = {
    "env": "Craftax-Classic-Symbolic-v1",
    "envs": 16,
    "num_steps": 32,
    "minibatches": 2,
    "update_epochs": 1,
    "width": 16,
    "rnn_width": 16,
}
"""Tuned block of the temporary PPO-RNN control copy; random Classic play unlocks
something within 2 updates of 16 x 32 steps, so the SIL buffer fills."""
STEPS = {"steps": 4 * 16 * 32}


@pytest.fixture(scope="module")
def methods_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return copy_methods(tmp_path_factory.mktemp("methods"), {"ppo-rnn": TINY})


def _config(methods_dir: Path, registry: str, ref: str, **over: Any) -> dict[str, Any]:
    method = load_method(registry, methods_dir)
    cfg = resolve_config(method, registry, "1m", ref=ref, methods_dir=methods_dir)
    return cfg | STEPS | over


@pytest.fixture(scope="module")
def on(methods_dir: Path) -> tuple[Any, Any, Any]:
    """Config, state and metrics of 2 combo updates with both components on."""
    cfg = _config(methods_dir, "combo", "combined")
    state, metrics = METHODS["combo"].train_chunk(METHODS["combo"].init(cfg, KEY), 2)
    return cfg, state, metrics


def test_both_components_off_train_the_control_bit_for_bit(
    methods_dir: Path, same: Callable[[Any, Any], bool]
) -> None:
    ctrl = METHODS["ppo-rnn"]
    a, _ = ctrl.train_chunk(ctrl.init(_config(methods_dir, "ppo-rnn", "tuned"), KEY), 2)
    off = _config(methods_dir, "combo", "combined", beta=0.0, sil_steps=0)
    b, _ = METHODS["combo"].train_chunk(METHODS["combo"].init(off, KEY), 2)
    assert same(a.train_state, b.train_state)


def test_both_components_on_are_active(on: tuple[Any, Any, Any]) -> None:
    _, state, m = on
    assert float(np.asarray(m["reward/intrinsic"]).sum()) > 0  # the count bonus pays
    assert int(np.asarray(m["count/distinct_keys"])[-1]) > 0
    assert int(np.asarray(m["sil/inserted"]).sum()) > 0  # observe fills the buffer
    assert {"sil/policy_loss", "sil/grad_norm", "sil/kl_after"} <= set(m)
    counts, buffer = state.method_state.actor[0], state.method_state.learner[1]
    assert int(counts.table.sum()) > 0
    assert int((buffer.buffer["stamp"] >= 0).sum()) > 0


def test_the_combo_entry_passes_the_interface_test(
    on: tuple[Any, Any, Any], same: Callable[[Any, Any], bool], tmp_path: Path
) -> None:
    """The trainer interface of infrastructure Task 3, on a 2-update combo state."""
    trainer = METHODS["combo"]
    cfg, state, _ = on
    names = ("steps_per_update", "init", "train_chunk", "params", "policy")
    assert all(callable(getattr(trainer, n)) for n in (*names, "env_wrapper"))
    assert trainer.steps_per_update(cfg) == 16 * 32
    save_state(tmp_path, state)
    assert same(load_state(tmp_path, trainer.init(cfg, jax.random.key(1))), state)
    policy = trainer.policy(state)
    act, _ = policy.step(
        trainer.params(state), policy.init_carry(16), state.obs, KEY, greedy=True
    )
    assert act.shape == (16,)
    assert trainer.env_wrapper(state) is None
    trainer.checkpoint_artifacts(state, tmp_path, 2)  # the count part's top keys
    assert (tmp_path / "eval" / count_bonus.TOP_KEYS_FILE).is_file()


_BROKEN_COMBO = """
import sys
from pathlib import Path
from wut_murl_craftax.orchestration import queue
queue.METHODS_DIR = Path(sys.argv[1])  # holds only a combo.yaml naming a skill method
from wut_murl_craftax.training.interface import METHODS

assert list(METHODS)[-1] == "combo" and "combo" in METHODS and len(METHODS) == 9
cfg = METHODS["ppo-rnn"].config({"env": "Craftax-Classic-Symbolic-v1", "envs": 8})
assert cfg.envs == 8
try:
    METHODS["combo"]
except TypeError as e:
    assert "hook methods only" in str(e), e
else:
    raise AssertionError("a skill-method pair built a combo entry")
print("ok")
"""


def test_a_broken_combo_yaml_breaks_only_the_combo_entry(
    start_on_cpus: Callable[[str, int, Path], Any], tmp_path: Path
) -> None:
    """The registry imports and builds the other entries; only ``combo`` fails."""
    _write_combo(tmp_path, "pqn-rnn", ["smdp-skills", "count-bonus-pqn-rnn"])
    proc = start_on_cpus(_BROKEN_COMBO, 1, tmp_path)
    out, err = proc.communicate(timeout=900)
    assert proc.returncode == 0, err[-3000:]
    assert out.strip().splitlines()[-1] == "ok"


_ON_TWO_CPUS = """
import json
import sys
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime(host_devices=2)
from wut_murl_craftax.training import ppo_rnn
from wut_murl_craftax.training.hooks import SplitState
from wut_murl_craftax.training.interface import METHODS

cfg = json.loads(sys.argv[2])
trainer = METHODS["combo"]
state = trainer.init(cfg, jax.random.key(0))

def devices(tree):
    return {len(x.sharding.device_set) for x in jax.tree.leaves(tree)}

for _ in range(2):
    ms = state.method_state
    assert isinstance(ms, SplitState), type(ms)
    assert devices(ms.actor) == {2}, devices(ms.actor)  # the count table, replicated
    assert devices(ms.learner) == {1}, devices(ms.learner)  # the SIL buffer, resident
    state = ppo_rnn.collect(state, with_traj=True)[0]  # the rollout keeps the split
# the update reads both parts on the learner; the SIL part never leaves it
state, _ = trainer.train_chunk(state, 2)
assert devices(state.method_state.learner) == {1}
print("ok")
"""


def test_only_the_count_part_goes_to_the_actor_devices(
    methods_dir: Path, tmp_path: Path
) -> None:
    """Two host CPUs: for the rollout only the count table is replicated over the
    actor mesh; the SIL buffer stays on the learner throughout."""
    cfg = _config(methods_dir, "combo", "combined", envs=8, num_steps=8)
    script = [sys.executable, "-c", _ON_TWO_CPUS, str(tmp_path), json.dumps(cfg)]
    flags = "--xla_force_host_platform_device_count=2"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags}
    proc = subprocess.run(script, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"
