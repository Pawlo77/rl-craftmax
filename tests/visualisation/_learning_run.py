"""One tiny PPO run and its eval rollouts, built once per test session and shared."""

import atexit
import functools
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from wut_murl_craftax.training.ppo import PPOConfig, train
from wut_murl_craftax.visualisation.eval_rollouts import MODES, EvalRun, evaluate

TINY = PPOConfig(steps=2048, envs=16, num_steps=32, width=32, minibatches=2, n_ckpt=3)
EVAL_SEEDS = (10000, 10001)
"""The first production eval seeds: tests play fewer seeds, never other ones."""
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def runner_config(cfg: PPOConfig) -> dict[str, Any]:
    """``config.json`` as the runner writes it: the registry name and config keys."""
    runtime = {"steps", "seed", "n_ckpt"}  # set by the runner, never in a config
    return {"method": "ppo"} | {
        k: v for k, v in asdict(cfg).items() if k not in runtime
    }


@functools.cache
def tiny_run() -> Path:
    """Train 4 updates on Craftax-Classic (checkpoints 0, 1, 2, 4) and evaluate them."""
    root = Path(tempfile.mkdtemp(prefix="learning_run_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    run = train(TINY, root / "run")
    (run / "config.json").write_text(json.dumps(runner_config(TINY)))
    opened = EvalRun.load(run)
    for mode in MODES:
        evaluate(opened, opened.updates, mode, seeds=EVAL_SEEDS)
    return run
