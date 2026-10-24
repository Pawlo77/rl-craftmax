"""Toy registry runs on Craftax-Classic shared by the interpretation tests."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.visualisation.eval_rollouts import EvalRun

CLASSIC = "Craftax-Classic-Symbolic-v1"
CRAFTAX = "Craftax-Symbolic-v1"


class ToyTrainer:
    """Registry entry of a toy policy; no params are read."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def init(self, config: Mapping[str, Any], key: jax.Array) -> dict[str, jax.Array]:
        del config, key
        return {"w": jnp.zeros((), jnp.int32)}

    def params(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return state

    def policy(self, state: dict[str, jax.Array]) -> Any:
        return self._policy

    def env_wrapper(self, state: dict[str, jax.Array]) -> None:
        return None


def toy_run(root: Path, policy: Any, env: str = CLASSIC) -> EvalRun:
    """A run of method ``toy`` acting through ``policy``, checkpoint 0."""
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"method": "toy", "env": env}))
    save_checkpoint(root, 0, 0, 0.0, {"w": np.int32(0)})
    return EvalRun.load(root, {"toy": ToyTrainer(policy)})
