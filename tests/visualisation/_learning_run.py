"""One tiny PPO run and its eval rollouts, built once per test session and shared."""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import pytest

from wut_murl_craftax.training.ppo import PPOConfig, train
from wut_murl_craftax.visualisation.eval_rollouts import evaluate

TINY = PPOConfig(steps=2048, envs=16, num_steps=32, width=32, minibatches=2, n_ckpt=3)
EVAL_SEEDS = (0, 1)
EVAL_T = 48
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@functools.cache
def tiny_run() -> Path:
    """Train 4 updates on Craftax-Classic (checkpoints 0, 1, 2, 4) and evaluate them."""
    root = Path(tempfile.mkdtemp(prefix="learning_run_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    run = train(TINY, root / "run")
    evaluate(run, seeds=EVAL_SEEDS, steps=EVAL_T)
    return run
