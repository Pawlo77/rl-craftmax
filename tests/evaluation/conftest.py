"""Shared fixtures of the evaluation tests: real start states, a tiny skills run."""

import json
import os
from pathlib import Path
from typing import Any

import jax
import pytest
from craftax.craftax_env import make_craftax_env_from_name


def _start(name: str) -> Any:
    env = make_craftax_env_from_name(name, auto_reset=False)
    return jax.jit(env.reset)(jax.random.PRNGKey(0), env.default_params)[1]


@pytest.fixture(autouse=True)
def _restore_xla_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the host-device split a CLI's ``configure_runtime`` writes into XLA_FLAGS.

    Without it the flag outlives the test and reaches later subprocess tests that copy
    ``os.environ`` (8 CPU devices for a 2-env fixture).
    """
    monkeypatch.setenv("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))


@pytest.fixture(scope="session")
def craftax_start() -> Any:
    """Craftax state after reset with key 0 (player at (24, 24) on the overworld)."""
    return _start("Craftax-Symbolic-v1")


@pytest.fixture(scope="session")
def classic_start() -> Any:
    """Craftax-Classic state after reset with key 0."""
    return _start("Craftax-Classic-Symbolic-v1")


SKILL_CONFIG = {
    "method": "smdp-skills",
    "env": "Craftax-Classic-Symbolic-v1",
    "envs": 16,
    "num_steps": 8,
    "minibatches": 2,
    "hidden": 16,
    "epochs": 1,
    "eps_decay": 2.0,
    "log2_rows": 12,
    "extra_obs_dim": 26,
    "timeout_mult": 0.5,
}
"""A tiny Classic skills run as the runner writes its config (`_method_runs`)."""
REPLAY_SEEDS = (10000, 10001)
"""The first production eval seeds."""


@pytest.fixture(scope="session")
def skill_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-update smdp-skills run (checkpoints 0 and 2), both eval modes written."""
    from wut_murl_craftax.training.checkpoints import save_checkpoint
    from wut_murl_craftax.training.interface import METHODS
    from wut_murl_craftax.visualisation.eval_rollouts import MODES, EvalRun, evaluate

    root = tmp_path_factory.mktemp("skill_run")
    (root / "config.json").write_text(json.dumps(SKILL_CONFIG))
    trainer = METHODS["smdp-skills"]
    spu = trainer.steps_per_update(SKILL_CONFIG)
    state = trainer.init(SKILL_CONFIG | {"steps": 2 * spu}, jax.random.key(0))
    save_checkpoint(root, 0, 0, 0.0, trainer.params(state))
    state, _ = trainer.train_chunk(state, 2)
    save_checkpoint(root, 2, 2 * spu, 1.0, trainer.params(state))
    run = EvalRun.load(root)
    for mode in MODES:
        evaluate(run, run.updates, mode, seeds=REPLAY_SEEDS)
    return root
