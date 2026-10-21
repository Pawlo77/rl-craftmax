"""Tests for the trainer registry and its PPO entry on tiny Craftax-Classic runs."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _contract import assert_done_step_unlock_counted

from wut_murl_craftax.orchestration.queue import load_method, resolve_config
from wut_murl_craftax.training import ppo_rnn, pqn_rnn
from wut_murl_craftax.training.checkpoints import load_state, save_state
from wut_murl_craftax.training.hooks import HookFields, HookMethod
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.records import LOSS_COLUMNS

REPO = Path(__file__).resolve().parents[2]
ENVS, NUM_STEPS, N_ACH = 8, 8, 22
TINY = {
    "env": "Craftax-Classic-Symbolic-v1",
    "envs": ENVS,
    "num_steps": NUM_STEPS,
    "width": 16,
    "minibatches": 2,
    "update_epochs": 1,
}
"""Overrides of the shipped ppo.yaml base that keep a test update fast."""

_SHARDED = """
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime(host_devices=2)
from wut_murl_craftax.training.interface import METHODS

cfg = {"env": "Craftax-Classic-Symbolic-v1", "envs": 8, "num_steps": 8, "width": 16,
       "minibatches": 2, "update_epochs": 1, "steps": 128}
trainer = METHODS["ppo"]
state = trainer.init(cfg, jax.random.key(0))

def sharded(s):
    leaves = jax.tree.leaves((s.env_state, s.obs, s.episode))
    return all(
        len(x.sharding.device_set) == 2 and x.sharding.spec[0] == "envs"
        for x in leaves
    )

assert len(jax.devices("cpu")) == 2
assert sharded(state)
state, _ = trainer.train_chunk(state, 1)
assert sharded(state)
try:
    trainer.init(cfg | {"envs": 7}, jax.random.key(0))
except ValueError as e:
    assert "7 envs do not split over 2 CPU devices" in str(e), e
else:
    raise AssertionError("7 envs on 2 devices did not fail")
print("ok")
"""


def _config(**overrides: Any) -> dict[str, Any]:
    """The shipped PPO config (every key of ppo.yaml) with tiny sizes and a budget."""
    base = resolve_config(load_method("ppo", REPO / "configs" / "methods"), "ppo", "1m")
    return base | TINY | {"steps": 4 * ENVS * NUM_STEPS} | overrides


def _same(a: Any, b: Any) -> bool:
    """Every leaf equal bit for bit; typed keys compared by their key data."""

    def raw(x: Any) -> bytes:
        if isinstance(x, jax.Array) and jax.dtypes.issubdtype(
            x.dtype, jax.dtypes.prng_key
        ):
            x = jax.random.key_data(x)
        return np.asarray(x).tobytes()

    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        raw(x) == raw(y) for x, y in zip(la, lb, strict=False)
    )


@pytest.fixture(scope="module")
def ppo() -> Any:
    return METHODS["ppo"]


@pytest.fixture(scope="module")
def start(ppo: Any) -> Any:
    return ppo.init(_config(), jax.random.key(0))


@pytest.fixture(scope="module")
def two_updates(ppo: Any, start: Any) -> tuple[Any, Any]:
    return ppo.train_chunk(start, 2)


def test_every_method_exposes_the_trainer_interface() -> None:
    assert next(iter(METHODS)) == "ppo"
    names = ("steps_per_update", "init", "train_chunk", "params", "policy")
    names += ("env_wrapper",)
    for trainer in METHODS.values():
        assert all(callable(getattr(trainer, n, None)) for n in names)


def test_steps_per_update_is_the_rollout_batch(ppo: Any) -> None:
    assert ppo.steps_per_update(_config()) == ENVS * NUM_STEPS


def test_two_updates_report_chunk_metrics(two_updates: tuple[Any, Any]) -> None:
    _, m = two_updates
    shapes = {k: np.asarray(v).shape for k, v in m.items()}
    assert shapes == {
        "losses": (2, len(LOSS_COLUMNS)),
        "done": (2, NUM_STEPS, ENVS),
        "ach": (2, NUM_STEPS, ENVS, N_ACH),
        "ret": (2, NUM_STEPS, ENVS),
        "length": (2, NUM_STEPS, ENVS),
        **{k: (2,) for k in m if k.startswith("ppo/")},
    }
    assert len([k for k in m if k.startswith("ppo/")]) == 10
    assert np.isfinite(np.asarray(m["losses"])).all()
    assert np.asarray(m["done"]).dtype == np.asarray(m["ach"]).dtype == bool


def test_training_changes_the_params(
    ppo: Any, start: Any, two_updates: tuple[Any, Any]
) -> None:
    assert not _same(ppo.params(start), ppo.params(two_updates[0]))


def test_chunking_does_not_change_the_result(
    ppo: Any, start: Any, two_updates: tuple[Any, Any]
) -> None:
    """Two chunks of 1 update equal one chunk of 2: params, optimizer, keys, envs."""
    one, m1 = ppo.train_chunk(start, 1)
    two, m2 = ppo.train_chunk(one, 1)
    whole, m = two_updates
    assert _same(two, whole)
    joined = {k: np.concatenate([np.asarray(m1[k]), np.asarray(m2[k])]) for k in m}
    assert _same(joined, {k: np.asarray(v) for k, v in m.items()})


def test_the_state_round_trips_through_save_state(
    ppo: Any, two_updates: tuple[Any, Any], tmp_path: Path
) -> None:
    """A restored state equals the saved one and trains on identically."""
    state = two_updates[0]
    save_state(tmp_path, state)
    restored = load_state(tmp_path, ppo.init(_config(), jax.random.key(1)))
    assert _same(restored, state)
    assert _same(ppo.train_chunk(restored, 1), ppo.train_chunk(state, 1))


def test_a_done_step_unlock_reaches_the_chunk_metrics(
    ppo: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared trainer contract: whole-episode flags with the done step's unlocks."""
    config = _config(envs=16, num_steps=32, width=17)  # a config no other test compiles
    assert_done_step_unlock_counted(ppo, config, monkeypatch)


def test_the_policy_acts_on_observations(
    ppo: Any, two_updates: tuple[Any, Any]
) -> None:
    state = two_updates[0]
    policy, params = ppo.policy(state), ppo.params(state)
    carry = policy.init_carry(ENVS)
    step = jax.jit(policy.step, static_argnames="greedy")
    act, carry2 = step(params, carry, state.obs, jax.random.key(0), greedy=False)
    assert act.shape == (ENVS,)
    assert ((act >= 0) & (act < 17)).all()
    assert _same(carry2, carry)
    greedy = [
        step(params, carry, state.obs, jax.random.key(k), greedy=True)[0]
        for k in (0, 1)
    ]
    assert _same(greedy[0], greedy[1])


@dataclass(frozen=True, slots=True)
class _ShadowsLr:
    lr: float = 1.0


BASES = {
    "ppo-rnn": (ppo_rnn.make_trainer, ppo_rnn.PPORNNConfig),
    "pqn-rnn": (pqn_rnn.make_trainer, pqn_rnn.PQNRNNConfig),
}
"""The base trainers a hook method builds on: factory and config class."""


@pytest.mark.parametrize("base", list(BASES))
def test_a_hook_method_takes_its_knobs_from_the_run_config(
    base: str, knob_method: HookMethod
) -> None:
    """Grid entries of one method train different hooks; base keys reach the base."""
    make, config_cls = BASES[base]
    trainer = make(config_cls(), method=knob_method)
    for value in (0.0, 0.25):
        cfg = trainer.config({"method": "knob", "knob_reward": value, "lr": 1e-3})
        assert cfg.intrinsic_reward_fn is not None
        r_int, _ = cfg.intrinsic_reward_fn({}, None, None, None, jnp.zeros(2, bool), 0)
        assert r_int.tolist() == [value, value]
        assert cfg.lr == 1e-3
    with pytest.raises(ValueError, match="knob_rewrd"):
        trainer.config({"method": "knob", "knob_rewrd": 0.25})
    with pytest.raises(ValueError, match="'lr'"):
        make(config_cls(), method=HookMethod(_ShadowsLr, lambda c: HookFields()))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"seed": 3}, "seed"),
        ({"n_ckpt": 3}, "n_ckpt"),
        ({"typo_lr": 1.0}, "typo_lr"),
        ({"resets": "never"}, "never"),
    ],
)
def test_bad_configs_fail_loudly(
    ppo: Any, overrides: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ppo.init(_config(**overrides), jax.random.key(0))


def test_annealing_needs_the_budget(ppo: Any) -> None:
    cfg = _config()
    del cfg["steps"]
    with pytest.raises(ValueError, match="steps"):
        ppo.init(cfg, jax.random.key(0))
    ppo.init(cfg | {"anneal_lr": False}, jax.random.key(0))


def test_env_states_are_sharded_over_the_host_devices() -> None:
    """host_devices=2 splits the env batch over the envs mesh; 7 envs cannot split."""
    result = subprocess.run(
        [sys.executable, "-c", _SHARDED],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1] == "ok"
