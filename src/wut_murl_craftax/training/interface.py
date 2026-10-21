"""Trainer registry: every method's `Trainer`, keyed by registry name, and `Policy`.

The runner imports this module after ``configure_runtime``, so it and the trainers
may import Craftax (which starts the JAX backends) at module top.
"""

from collections.abc import Mapping
from typing import Any, Protocol

import jax

from ..orchestration.runner import Trainer
from .ppo import PPOTrainer
from .ppo_rnn import PPORNNConfig, make_trainer


class Policy(Protocol):
    """How a trained agent acts; evaluation drives every method through it."""

    def init_carry(self, n: int) -> Any:
        """Recurrent state for ``n`` envs; an empty pytree for feed-forward nets."""
        ...

    def step(
        self, params: Any, carry: Any, obs: jax.Array, key: jax.Array, greedy: bool
    ) -> tuple[jax.Array, Any]:
        """Actions for a batch of observations and the next carry.

        ``greedy`` takes the most likely action instead of sampling; jit-safe as a
        static argument or a traced bool.
        """
        ...


METHODS: Mapping[str, Trainer] = {
    "ppo": PPOTrainer(),
    "ppo-rnn": make_trainer(PPORNNConfig()),
}
"""Registry name (`RunSpec.registry_name`, ``config["method"]``) -> trainer; each
method plan adds its entry."""
