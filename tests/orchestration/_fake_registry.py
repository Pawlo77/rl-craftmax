"""Fake trainers a runner child process imports by reference: ``REGISTRY``."""

import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import mlflow
from _fake_trainer import FakeTrainer

from wut_murl_craftax.orchestration.queue import RunSpec

REGISTRY = "_fake_registry:METHODS"
"""The ``registry`` argument of ``runner.main`` for these trainers."""
NO_EVAL = "_fake_registry:no_eval"
"""The ``evaluate`` argument: skip the evaluation."""


class PidTrainer(FakeTrainer):
    """Tags its run with the PID of the process that trains it."""

    def init(self, config: Any, key: jax.Array) -> dict[str, Any]:
        mlflow.set_tag("pid", str(os.getpid()))
        return super().init(config, key)


class CrashTrainer(FakeTrainer):
    """Dies as the OOM killer would kill it: SIGKILL in the first update."""

    def train_chunk(self, state: Any, n_updates: int) -> Any:
        os.kill(os.getpid(), signal.SIGKILL)


def no_eval(run_dir: Path, spec: RunSpec, should_stop: Callable[[], bool]) -> None:
    """An `Evaluate` that writes nothing."""


METHODS = {
    "toy": PidTrainer(),
    "crash": CrashTrainer(),
    "stop": FakeTrainer(signal_at=(2, signal.SIGINT), to_parent=True),
}
"""Registry name -> fake trainer; specs pick one by ``registry_name``."""
