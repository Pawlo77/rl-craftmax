"""Count-based progress bonus: a global count table and ``beta_t / sqrt(N)`` per step.

The hooks of `training/hooks.py`: `init` builds the state (``init_state_fn``),
`intrinsic_reward` counts the key of each next state and pays the bonus
(``intrinsic_reward_fn``), `metrics` reports the table (``metrics_fn``). The key is
`progress_key` of the inner next state, or a SimHash of the next observation (the
ablation). Done steps are neither counted nor paid: their next state already starts
a new episode. `METHOD` builds these hooks per run config for the registry entries
``count-bonus-ppo-rnn`` and ``count-bonus-pqn-rnn``. Design: [[Count-Based Progress
Bonus design]].
"""

import functools
from dataclasses import dataclass
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
from flax import struct

from .hooks import HookFields, HookMethod, HookSpec
from .progress_key import Granularity, fields, hash_fields, unwrap

type Decay = Literal["constant", "linear"]
"""β schedule: ``beta`` throughout, or ``beta * (1 - t / T)``, 0 at the budget T."""
type KeyKind = Literal["progress", "simhash"]
"""What is counted: the progress key, or a SimHash of the observation (ablation)."""
SIMHASH_BITS = 64
"""Sign bits of the SimHash projection, folded by `hash_fields`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CountBonusConfig:
    """Method knobs of the count bonus; no defaults, so every run config sets each."""

    beta: float
    granularity: Granularity
    decay: Decay
    log2_k: int
    """Table size ``2^log2_k`` slots, also the hash width."""
    key: KeyKind

    def __post_init__(self) -> None:
        """Refuse a negative or NaN β, a table outside 1-31 bits, unknown knob values.

        Raises:
            ValueError: On any of these.
        """
        if not self.beta >= 0 or not 0 < self.log2_k < 32:
            msg = f"need beta >= 0 and 0 < log2_k < 32, got {self.beta}, {self.log2_k}"
            raise ValueError(msg)
        knobs = (("granularity", Granularity), ("decay", Decay), ("key", KeyKind))
        for name, kind in knobs:
            allowed = get_args(kind.__value__)
            if getattr(self, name) not in allowed:
                msg = f"{name} must be one of {allowed}, got {getattr(self, name)!r}"
                raise ValueError(msg)


class CountState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Method state: one pytree in the train state, round-tripped by ``save_state``.

    No leaf has the env count as leading dim, so the trainers replicate every leaf
    over the env mesh (`ppo.place_on_actor`).
    """

    table: jax.Array
    """Visits per key slot since the run began, int32 ``[2^log2_k]``, never reset."""
    t: jax.Array
    """Env-step counter of the last call (the hooks' ``t``), int32; gives β_t."""
    proj: jax.Array
    """Fixed Gaussian SimHash projection, float32 ``[obs_dim, 64]``."""
    cfg: CountBonusConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    total_steps: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Run budget T of the linear decay (`HookSpec.total_steps`)."""


def init(cfg: CountBonusConfig, key: jax.Array, spec: HookSpec) -> CountState:
    """Zero table of ``2^log2_k`` slots, the SimHash projection and T from ``spec``."""
    return CountState(
        table=jnp.zeros(2**cfg.log2_k, jnp.int32),
        t=jnp.zeros((), jnp.int32),
        proj=jax.random.normal(key, (spec.obs_dim, SIMHASH_BITS), jnp.float32),
        cfg=cfg,
        total_steps=spec.total_steps,
    )


def simhash_key(obs: jax.Array, proj: jax.Array, bits: int) -> jax.Array:
    """Key of each observation on the last axis: signs of ``obs @ proj``, hashed."""
    # full precision, so a TF32 GPU matmul does not flip signs near zero
    z = jnp.matmul(obs, proj, precision=jax.lax.Precision.HIGHEST)
    return hash_fields((z > 0).astype(jnp.int32), bits)


def _beta_t(state: CountState, t: jax.Array) -> jax.Array:
    left = jnp.maximum(1 - jnp.asarray(t, jnp.float32) / state.total_steps, 0.0)
    scale = left if state.cfg.decay == "linear" else jnp.ones((), jnp.float32)
    return state.cfg.beta * scale


def intrinsic_reward(
    state: CountState,
    env_state: Any,
    next_env_state: Any,
    obs_next: jax.Array,
    done: jax.Array,
    t: jax.Array,
) -> tuple[jax.Array, CountState]:
    """Count each env's next key unless done, then pay ``beta_t / sqrt(N)`` (0 if done).

    ``env_state`` is unused, so no state is compared across a done; counts are added
    before they are read, so ``N >= 1`` and one key seen by k envs in a step counts k.
    """
    del env_state
    cfg = state.cfg
    if cfg.key == "simhash":
        keys = simhash_key(obs_next, state.proj, cfg.log2_k)
    else:
        per_env = jax.vmap(functools.partial(fields, granularity=cfg.granularity))
        keys = hash_fields(per_env(unwrap(next_env_state)), cfg.log2_k)
    live = jnp.logical_not(done)
    table = state.table.at[keys].add(live.astype(jnp.int32))
    n = jnp.maximum(table[keys], 1).astype(jnp.float32)  # done slots may still be 0
    r_int = jnp.where(live, _beta_t(state, t) / jnp.sqrt(n), 0.0)
    t = jnp.asarray(t, jnp.int32)
    return r_int.astype(jnp.float32), state.replace(table=table, t=t)


def metrics(state: CountState) -> dict[str, jax.Array]:
    """Occupied slots (``count/distinct_keys``) and their share of the table."""
    distinct = jnp.count_nonzero(state.table)
    return {
        "count/distinct_keys": distinct,
        "count/occupancy": distinct / state.table.shape[0],
    }


def hooks(cfg: CountBonusConfig, base: Any) -> HookFields:
    """The count-bonus hooks of one run config, on either recurrent base."""
    del base  # the bonus reads no base setting
    return HookFields(
        init_state_fn=functools.partial(init, cfg),
        intrinsic_reward_fn=intrinsic_reward,
        metrics_fn=metrics,
    )


METHOD = HookMethod(CountBonusConfig, hooks)
"""The method both registry entries build their trainers with."""
