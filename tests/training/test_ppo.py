"""Tests for the baseline PPO trainer on a tiny Craftax-Classic run."""

import json
from pathlib import Path

import numpy as np
import pytest

from wut_murl_craftax.training.checkpoints import load_index
from wut_murl_craftax.training.ppo import PPOConfig, main
from wut_murl_craftax.training.records import load_episodes, load_metrics

TINY = PPOConfig(steps=2048, envs=16, num_steps=32, width=32, minibatches=2, n_ckpt=3)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ppo")
    argv = ["--steps", "2048", "--envs", "16", "--num-steps", "32", "--width", "32"]
    main([*argv, "--minibatches", "2", "--n-ckpt", "3", "--out", str(out)])
    return out


def test_tiny_run_writes_all_records(tiny_run: Path) -> None:
    cfg = json.loads((tiny_run / "config.json").read_text())
    assert cfg["n_updates"] == 4
    assert cfg["batch"] == TINY.envs * TINY.num_steps
    assert [e["update"] for e in load_index(tiny_run)] == cfg["ckpt_updates"]
    m = load_metrics(tiny_run)
    assert m.data.shape[0] == 4
    assert np.isfinite(m.col("policy_loss")).all()
    ep = load_episodes(tiny_run)
    assert len(ep.update) == m.col("n_episodes").sum()


def test_config_records_device_roles(tiny_run: Path) -> None:
    cfg = json.loads((tiny_run / "config.json").read_text())
    assert set(cfg["devices"]) == {"actor", "learner"}
