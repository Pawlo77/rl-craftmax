"""Count-based progress bonus: a global count table and ``beta_t / sqrt(N)`` per step.

The hooks of `training/hooks.py`: `init` builds the state (``init_state_fn``),
`intrinsic_reward` counts the key of each next state and pays the bonus
(``intrinsic_reward_fn``), `metrics` reports ``count/...`` per update (``metrics_fn``).
The key is `progress_key` of the inner next state, or a SimHash of the next observation
(the ablation). Done steps are neither counted nor paid: their next state already
starts a new episode. `METHOD` builds the hooks of both registry entries per run
config; `top_keys` decodes the most and least visited slots on the host.
Design: [[Count-Based Progress Bonus design]].
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..evaluation.milestones import Game, game_of
from .hooks import HookFields, HookMethod, HookSpec
from .progress_key import (
    CLASSIC_FIELDS,
    FIELDS,
    Granularity,
    fields,
    hash_fields,
    unwrap,
)

type Decay = Literal["constant", "linear"]
"""β schedule: ``beta`` throughout, or ``beta * (1 - t / T)``, 0 at the budget T."""
type KeyKind = Literal["progress", "simhash"]
"""What is counted: the progress key, or a SimHash of the observation (ablation)."""
SIMHASH_BITS = 64
"""Sign bits of the SimHash projection, folded by `hash_fields`."""
N_FLOORS = 9
"""Craftax floors (``StaticEnvParams.num_levels`` of craftax 1.6.1); Classic has one."""
TOP_KEYS_FILE = "count_top_keys.json"
"""File in the run directory's ``eval/``: `top_keys` by checkpoint update."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CountBonusConfig:
    """Method knobs of the count bonus; no defaults, so every run config sets each."""

    beta: float
    granularity: Granularity
    decay: Decay
    log2_k: int
    """Table size ``2^log2_k`` slots, also the hash width."""
    key: KeyKind
    max_occupancy: float
    """Occupied share of the table above which the run aborts (``count/table_full``);
    about twice the share of keys that then collide, ``n / 2K``."""

    def __post_init__(self) -> None:
        """Refuse bad β, table bits, threshold or knob values.

        β must be >= 0 (not NaN), ``log2_k`` in 1-31, ``max_occupancy`` in (0, 1].

        Raises:
            ValueError: On any of these.
        """
        if not self.beta >= 0 or not 0 < self.log2_k < 32:
            msg = f"need beta >= 0 and 0 < log2_k < 32, got {self.beta}, {self.log2_k}"
            raise ValueError(msg)
        if not 0 < self.max_occupancy <= 1:
            msg = f"max_occupancy must be in (0, 1], got {self.max_occupancy}"
            raise ValueError(msg)
        knobs = (("granularity", Granularity), ("decay", Decay), ("key", KeyKind))
        for name, kind in knobs:
            allowed = get_args(kind.__value__)
            if getattr(self, name) not in allowed:
                msg = f"{name} must be one of {allowed}, got {getattr(self, name)!r}"
                raise ValueError(msg)


class Window(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Sums over the env steps of one update, ``t // batch``; the next restarts them."""

    index: jax.Array
    """Update the sums belong to, int32; -1 before the first call."""
    start_distinct: jax.Array
    """Occupied slots when the update began, int32."""
    r_sum: jax.Array
    """Bonus paid, done steps included (they pay 0)."""
    steps: jax.Array
    """Env steps, int32."""
    floor_sum: jax.Array
    """Bonus of the live steps by floor of the next state, ``[floors]``."""
    floor_n: jax.Array
    """Live steps by floor, int32 ``[floors]``."""
    fin_int: jax.Array
    """Bonus earned by the episodes that ended."""
    fin_ext: jax.Array
    """Extrinsic return of those episodes up to their done step (that step's own
    reward is not visible: its next state is already the reset one)."""


def _window(floors: int, index: Any = -1, start_distinct: Any = 0) -> Window:
    zero = jnp.zeros((), jnp.float32)
    return Window(
        index=jnp.asarray(index, jnp.int32),
        start_distinct=jnp.asarray(start_distinct, jnp.int32),
        r_sum=zero,
        steps=jnp.zeros((), jnp.int32),
        floor_sum=jnp.zeros(floors, jnp.float32),
        floor_n=jnp.zeros(floors, jnp.int32),
        fin_int=zero,
        fin_ext=zero,
    )


class CountState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Method state: one pytree in the train state, round-tripped by ``save_state``.

    Only ``ep_int`` and ``ep_ext`` have the env count as leading dim, so the trainers
    split them over the env mesh and replicate the rest (`ppo.place_on_actor`).
    """

    table: jax.Array
    """Visits per key slot since the run began, int32 ``[2^log2_k]``, never reset."""
    seen: jax.Array
    """Progress fields of the first state counted in each slot, int8
    ``[2^log2_k, fields]``: decodes the keys in `top_keys`. Empty (no rows) for
    SimHash keys, which the progress fields do not decode."""
    t: jax.Array
    """Env-step counter of the last call (the hooks' ``t``), int32; gives β_t."""
    proj: jax.Array
    """Fixed Gaussian SimHash projection, float32 ``[obs_dim, 64]``."""
    window: Window
    """Sums of the current update, read by `metrics`."""
    share: jax.Array
    """``count/r_int_share`` of the last update in which an episode ended."""
    ep_int: jax.Array
    """Bonus of each env's running episode, float32 ``[envs]``."""
    ep_ext: jax.Array
    """Extrinsic return of each env's running episode so far, float32 ``[envs]``."""
    max_floor: jax.Array
    """Deepest floor of any counted state, int32."""
    cfg: CountBonusConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    total_steps: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Run budget T of the linear decay (`HookSpec.total_steps`)."""
    batch: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Env steps of one update: the window of the metrics."""
    game: Game = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Field layout of the progress key: Classic's or Craftax's."""


def init(
    cfg: CountBonusConfig, key: jax.Array, spec: HookSpec, *, batch: int, game: Game
) -> CountState:
    """Zero counts and sums, the SimHash projection and T from ``spec``.

    ``batch`` is the env steps of one update, ``game`` the progress-key layout.
    """
    width = len(CLASSIC_FIELDS if game == "classic" else FIELDS)
    k = 2**cfg.log2_k
    return CountState(
        table=jnp.zeros(k, jnp.int32),
        seen=jnp.zeros((k if cfg.key == "progress" else 0, width), jnp.int8),
        t=jnp.zeros((), jnp.int32),
        proj=jax.random.normal(key, (spec.obs_dim, SIMHASH_BITS), jnp.float32),
        window=_window(1 if game == "classic" else N_FLOORS),
        share=jnp.zeros((), jnp.float32),
        ep_int=jnp.zeros(spec.n_envs, jnp.float32),
        ep_ext=jnp.zeros(spec.n_envs, jnp.float32),
        max_floor=jnp.zeros((), jnp.int32),
        cfg=cfg,
        total_steps=spec.total_steps,
        batch=batch,
        game=game,
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


def _share(w: Window, last: jax.Array) -> jax.Array:
    """Bonus share of the episodes that ended in ``w``; ``last`` if none did.

    Bonus over bonus plus extrinsic return, clipped at 0 so the share is in [0, 1].
    """
    total = w.fin_int + jnp.maximum(w.fin_ext, 0.0)
    return jnp.where(total > 0, w.fin_int / jnp.where(total > 0, total, 1.0), last)


def _roll(state: CountState, t: jax.Array) -> CountState:
    """Start new sums when ``t`` is in a later update than the window's."""
    index = t // state.batch
    w = state.window
    new = index != w.index

    def fresh() -> Window:  # counts the table once per update, not per step
        return _window(w.floor_n.shape[0], index, jnp.count_nonzero(state.table))

    return state.replace(
        window=jax.lax.cond(new, fresh, lambda: w),
        share=jnp.where(new, _share(w, state.share), state.share),
    )


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
    The episode sums read the log wrapper's running return of ``next_env_state``.
    """
    del env_state
    cfg, t = state.cfg, jnp.asarray(t, jnp.int32)
    inner = unwrap(next_env_state)
    f = jax.vmap(functools.partial(fields, granularity=cfg.granularity))(inner)
    if cfg.key == "simhash":
        keys = simhash_key(obs_next, state.proj, cfg.log2_k)
    else:
        keys = hash_fields(f, cfg.log2_k)
    state = _roll(state, t)
    live = jnp.logical_not(done)
    old = state.table[keys]
    table = state.table.at[keys].add(live.astype(jnp.int32))
    n = jnp.maximum(table[keys], 1).astype(jnp.float32)  # done slots may still be 0
    r_int = jnp.where(live, _beta_t(state, t) / jnp.sqrt(n), 0.0).astype(jnp.float32)
    seen = state.seen
    if cfg.key == "progress":
        first = (live & (old == 0))[:, None]
        seen = seen.at[keys].set(jnp.where(first, f.astype(jnp.int8), seen[keys]))
    floor = inner.player_level if state.game == "craftax" else jnp.zeros_like(keys)
    ep_int = state.ep_int + r_int
    w = state.window
    w = w.replace(
        r_sum=w.r_sum + r_int.sum(),
        steps=w.steps + done.shape[0],
        floor_sum=w.floor_sum.at[floor].add(jnp.where(live, r_int, 0.0)),
        floor_n=w.floor_n.at[floor].add(live.astype(jnp.int32)),
        fin_int=w.fin_int + jnp.where(done, ep_int, 0.0).sum(),
        fin_ext=w.fin_ext + jnp.where(done, state.ep_ext, 0.0).sum(),
    )
    ret = jnp.asarray(next_env_state.episode_returns, jnp.float32)
    return r_int, state.replace(
        table=table,
        seen=seen,
        t=t,
        window=w,
        ep_int=jnp.where(done, 0.0, ep_int),
        ep_ext=jnp.where(done, 0.0, ret),
        max_floor=jnp.maximum(state.max_floor, jnp.where(live, floor, 0).max()),
    )


def metrics(state: CountState) -> dict[str, jax.Array]:
    """The ``count/...`` metrics: the table, and the bonus of the current update.

    The guards ``count/table_full`` (occupancy above ``max_occupancy``) and
    ``count/negative_counts`` (a count below 0: counts only grow) are 0, or NaN once
    broken, so the runner's ``abort.finite`` check fails the run on them.
    """
    k = state.table.shape[0]
    distinct = jnp.count_nonzero(state.table).astype(jnp.int32)
    w = state.window
    by_floor = w.floor_sum / jnp.maximum(w.floor_n, 1)  # 0 on floors not visited

    def guard(broken: jax.Array) -> jax.Array:
        return jnp.where(broken, jnp.nan, 0.0).astype(jnp.float32)

    return {
        "count/table_full": guard(distinct > state.cfg.max_occupancy * k),
        "count/negative_counts": guard(state.table.min() < 0),
        "count/distinct_keys": distinct,
        "count/new_keys": distinct - w.start_distinct,
        "count/occupancy": distinct / k,
        "count/collision_est": distinct / (2 * k),  # share of keys sharing a slot
        "count/beta_t": _beta_t(state, state.t),
        "count/r_int_mean": w.r_sum / jnp.maximum(w.steps, 1),
        "count/r_int_share": _share(w, state.share),
        "count/max_key_floor": state.max_floor,
    } | {f"count/r_int_by_floor/{i}": v for i, v in enumerate(by_floor)}


def top_keys(state: CountState, n: int = 50) -> dict[str, Any]:
    """The ``n`` most and ``n`` least visited occupied slots with their fields.

    On the host (NumPy) for the per-checkpoint file; equal counts go by slot. SimHash
    slots have no decoded fields (None).
    """
    table, seen = np.asarray(state.table), np.asarray(state.seen)
    names = CLASSIC_FIELDS if state.game == "classic" else FIELDS
    slots = np.flatnonzero(table)
    least = slots[np.argsort(table[slots], kind="stable")][:n]
    most = slots[np.lexsort((slots, -table[slots]))][:n]

    def entry(slot: int) -> dict[str, Any]:
        decoded = (
            dict(zip(names, seen[slot].tolist(), strict=True)) if len(seen) else None
        )
        return {"slot": int(slot), "count": int(table[slot]), "fields": decoded}

    return {
        "distinct_keys": len(slots),
        "most": [entry(s) for s in most],
        "least": [entry(s) for s in least],
    }


def write_top_keys(run_dir: Path, update: int, state: CountState) -> Path:
    """Add `top_keys` of ``update`` to the run's ``eval/`` file, atomically."""
    path = Path(run_dir) / "eval" / TOP_KEYS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved[str(update)] = top_keys(state)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=1))
    tmp.replace(path)
    return path


def hooks(cfg: CountBonusConfig, base: Any) -> HookFields:
    """The count-bonus hooks of a run config; one ``base`` update is a window."""
    init_fn = functools.partial(init, cfg, batch=base.batch, game=game_of(base.env))
    return HookFields(
        init_state_fn=init_fn,
        intrinsic_reward_fn=intrinsic_reward,
        metrics_fn=metrics,
    )


METHOD = HookMethod(CountBonusConfig, hooks)
"""The method both registry entries build their trainers with."""
