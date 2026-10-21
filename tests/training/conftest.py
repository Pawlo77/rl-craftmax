"""Shared fixtures of the training tests: recurrent configs, equality, CPU runs."""

import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from wut_murl_craftax.training.hooks import HookFields, HookMethod
from wut_murl_craftax.training.ppo_rnn import PPORNNConfig
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig

TEST_ENV = os.environ.get("CRAFTAX_TEST_ENV", "Craftax-Classic-Symbolic-v1")
"""Env of the recurrent tests; Craftax-Symbolic reruns them at the real scale."""
PROFILE = Path(__file__).parents[1] / "orchestration" / "fixtures" / "profiles"
"""Folder of the runner tests' machine profile ``test.yaml``."""


@dataclass(frozen=True, slots=True)
class KnobConfig:
    """A hook method's own config: one YAML knob."""

    knob_reward: float = 0.0
    """Constant intrinsic reward of every step."""


def _knob_hooks(cfg: KnobConfig, base: Any) -> HookFields:
    r_int = cfg.knob_reward * base.lr  # reads the resolved base config

    def reward(s: Any, e: Any, e2: Any, o: Any, d: jax.Array, t: Any) -> Any:
        return jnp.full(d.shape, r_int, jnp.float32), s

    return HookFields(intrinsic_reward_fn=reward)


KNOB = HookMethod(KnobConfig, _knob_hooks)
"""A hook method: intrinsic reward ``knob_reward`` times the base config's ``lr``."""


@pytest.fixture(scope="session")
def knob_method() -> HookMethod[KnobConfig]:
    """`KNOB`: the intrinsic reward is ``knob_reward * lr`` of the run config."""
    return KNOB


@pytest.fixture(scope="session")
def rnn_cfg() -> Callable[..., PPORNNConfig]:
    """Tiny PPO-RNN config (8 envs x 8 steps, published widths, 4 updates of budget)."""

    def make(**overrides: Any) -> PPORNNConfig:
        base = {"env": TEST_ENV, "envs": 8, "num_steps": 8, "width": 512}
        base |= {"minibatches": 2, "steps": 4 * 8 * 8}
        return PPORNNConfig(**(base | overrides))

    return make


@pytest.fixture(scope="session")
def pqn_cfg() -> Callable[..., PQNRNNConfig]:
    """Tiny PQN-RNN config (8 envs x 8 steps, the repo widths, 4 updates of budget)."""

    def make(**overrides: Any) -> PQNRNNConfig:
        base = {"env": TEST_ENV, "envs": 8, "num_steps": 8, "minibatches": 2}
        return PQNRNNConfig(**(base | {"steps": 4 * 8 * 8} | overrides))

    return make


def _same(a: Any, b: Any) -> bool:

    def raw(x: Any) -> bytes:
        if isinstance(x, jax.Array) and jax.dtypes.issubdtype(
            x.dtype, jax.dtypes.prng_key
        ):
            x = jax.random.key_data(x)
        return np.asarray(x).tobytes()

    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        raw(x) == raw(y) for x, y in zip(la, lb, strict=True)
    )


@pytest.fixture(scope="session")
def same() -> Callable[[Any, Any], bool]:
    """Every leaf equal bit for bit; typed keys compared by their key data."""
    return _same


def _start_on_cpus(script: str, devices: int, out: Path) -> subprocess.Popen[str]:
    flags = f"--xla_force_host_platform_device_count={devices}" if devices > 1 else ""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(out)],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture(scope="session")
def start_on_cpus() -> Callable[[str, int, Path], subprocess.Popen[str]]:
    """Start ``python -c script out`` on ``devices`` host CPUs (JAX fixes the count
    when it starts, hence a subprocess)."""
    return _start_on_cpus


class RunStore:
    """A throwaway MLflow store whose specs train in this process, unevaluated."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # the runner imports mlflow (seconds): only the tests that train runs need it
        self.runner = importlib.import_module("wut_murl_craftax.orchestration.runner")
        monkeypatch.setattr(self.runner, "configure_runtime", lambda **kw: None)
        # the platform choice would change JAX_PLATFORMS of the test process
        monkeypatch.setattr(self.runner, "select_platforms", lambda mode: None)
        self.uri = f"sqlite:///{tmp / 'mlruns.db'}"
        self.config = tmp / "experiment.yaml"
        mlflow_cfg = {"tracking_uri": self.uri, "experiment_name": "x"}
        text = yaml.safe_dump({"experiment": {"name": "craftax"}, "mlflow": mlflow_cfg})
        self.config.write_text(text, encoding="utf-8")

    def run(self, specs: list[Any]) -> dict[str, Any]:
        """Train ``specs`` with the registry's trainers; the store's runs by name."""
        methods = importlib.import_module("wut_murl_craftax.training.interface").METHODS
        self.runner.main(
            specs,
            PROFILE / "test.yaml",
            experiment_config=self.config,
            methods={s.registry_name: methods[s.registry_name] for s in specs},
            evaluate=lambda run_dir, spec, should_stop: None,
        )
        client = self.runner.mlflow.MlflowClient(tracking_uri=self.uri)
        ids = [e.experiment_id for e in client.search_experiments()]
        return {r.info.run_name: r for r in client.search_runs(ids)}


@pytest.fixture
def run_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunStore:
    """`RunStore` in the test's temporary directory."""
    return RunStore(tmp_path, monkeypatch)
