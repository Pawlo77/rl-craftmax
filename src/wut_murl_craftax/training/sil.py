"""Self-Imitation Learning: a per-env staging ring and a buffer of unlock windows.

`init` builds the state (``init_state_fn``). `observe` (the ``observe`` hook) copies
the window of the last ``window`` steps ending at each new unlock
(``traj.events.unlocked``: privileged, it only picks the data and its slot) into that
achievement's FIFO slot, with returns cut at the unlock, then writes the rollout into
the staging ring. Observations are kept as a ``uint8`` map and a ``float32`` tail
(`compress_obs`). `extra_loss` (``extra_loss_fn``, bound to the network) samples
windows (`sample`), replays them from their stored GRU state and imitates the actions
whose return beat the critic. `buffer_bytes` and `check_memory` size the state unbuilt;
`METHOD` gives the ``sil`` entry's hooks. Design: [[Self-Imitation Learning design]].
"""

import functools
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..evaluation.milestones import Game, constants, game_of
from .hooks import HookFields, HookMethod, HookSpec, Traj
from .ppo_rnn import ActorCriticRNN, PPORNNConfig, apply_network, build_network
from .records import achievement_names, achievement_weights

type Layout = Literal["slots", "single"]
"""One FIFO slot per achievement, or one shared FIFO of the same capacity (ablation)."""
MAP_DIMS: dict[int, int] = {8268: 8217, 1345: 1323}
"""Length of the binary map view (one-hot blocks, items, mobs; light) at the start of
each observation, by observation length: Craftax 9 x 11 x 83, Classic 7 x 9 x 21."""
CRAFTAX_OBS_DIM = 8268
"""Craftax-Symbolic observation length, the default of `buffer_bytes`."""
OBS_DIMS: dict[Game, int] = {"craftax": CRAFTAX_OBS_DIM, "classic": 1345}
"""Symbolic observation length by game, for the network SIL replays."""
CONTROL_GAMMA, CONTROL_RNN_WIDTH = 0.99, 512
"""gamma and GRU width of the PPO-RNN control: `init` defaults; `hooks` passes the
run's."""
TIE_SEED = 0
"""Seed of the random order among unlocks of one tier when a rollout has more unlocks
than ``insert_cap`` (folded with the next stamp, so each rollout draws anew)."""
PRIORITY_EPS = 0.01
"""Added to a window's mean ``(R - V)_+``, so a caught-up window can still be drawn."""
STEP_KEYS = ("map", "tail", "act", "rew", "done")
"""Per-step arrays of the staging ring and the buffer (plus ``hidden``, ``ret``,
``mask``)."""


@dataclass(frozen=True, slots=True)
class SILConfig:
    """Method knobs of SIL, its YAML keys; gamma and the GRU width are the control's."""

    lambda_sil: float = 1.0
    """Weight of the SIL loss against the PPO loss."""
    slots_windows: int = 2
    """Windows per slot ``S`` (the single layout holds ``S`` per achievement)."""
    sil_steps: int = 1
    """SIL optimizer steps per PPO update ``K`` (``extra_steps``)."""
    window: int = 128
    beta_v: float = 0.01
    batch: int = 32
    """Windows per SIL step."""
    alpha: float = 0.6
    """Priority exponent of the window sampling."""
    insert_cap: int = 64
    """Most windows copied per rollout; static, so shapes stay fixed."""
    layout: Layout = "slots"

    def __post_init__(self) -> None:
        """Refuse sizes below 1, negative weights and other layouts.

        Raises:
            ValueError: Naming the field.
        """
        sizes = ("slots_windows", "window", "batch", "insert_cap")
        bad = [n for n in sizes if getattr(self, n) < 1]
        bad += [n for n in ("lambda_sil", "beta_v", "alpha", "sil_steps")
                if getattr(self, n) < 0]  # fmt: skip
        bad += [] if self.layout in get_args(Layout.__value__) else ["layout"]
        if bad:
            msg = f"bad SIL settings {bad}: {[getattr(self, n) for n in bad]}"
            raise ValueError(msg)


class SILBytes(NamedTuple):
    """Bytes of the SIL state: the staging ring and the rest (buffer, counters)."""

    staging: int
    buffer: int

    @property
    def total(self) -> int:
        """Staging plus buffer."""
        return self.staging + self.buffer


class SILState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Method state: one pytree in the train state, round-tripped by ``save_state``.

    Only the staging leaves have the env count as leading dim, so only they are split
    over the env mesh (`ppo.place_on_actor`); buffer leaves lead with the row count.
    """

    staging: dict[str, jax.Array]
    """Last ``window`` steps per env, ``(envs, window, ...)``: `STEP_KEYS` and the
    float16 ``hidden`` each step acted from; ``done`` starts true (before the run)."""
    ptr: jax.Array
    """Staging column of the oldest step, int32; the ring is written there next."""
    buffer: dict[str, jax.Array]
    """Windows, ``(rows, window, ...)``: `STEP_KEYS`, ``ret`` and ``mask``; per row
    ``h0`` (float16 start state), ``prio``, ``stamp`` (insertion order, -1 empty),
    ``born`` (`update` at insertion) and ``ach`` (the unlocked achievement). Slot ``s``
    holds rows ``s*S .. s*S+S-1``."""
    next_stamp: jax.Array
    """Stamp of the next inserted window, int32."""
    max_prio: jax.Array
    """Largest priority so far, float32; new windows get it."""
    inserted: jax.Array
    overflow: jax.Array
    """Buffer rows written, and unlocks over ``insert_cap`` dropped, by the last
    rollout (a slot that gets more than ``S`` keeps its newest ``S``)."""
    update: jax.Array
    """Rollouts observed so far, int32: the age clock of the windows."""
    cfg: SILConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    slot_size: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Rows per slot: ``S`` (slots) or ``S * n_ach`` (single)."""
    gamma: float = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Discount of the stored returns, the control's."""
    tiers: tuple[int, ...] = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Tier rank of each achievement (0 lowest); the insertion cap keeps high ones."""
    names: tuple[str, ...] = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API
    """Achievement names in slot order, for the per-achievement metrics."""


def _map_dim(obs_dim: int) -> int:
    if obs_dim not in MAP_DIMS:
        msg = f"obs length {obs_dim} is not a Craftax one ({sorted(MAP_DIMS)})"
        raise ValueError(msg)
    return MAP_DIMS[obs_dim]


def compress_obs(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(uint8 map, float32 tail)`` of observations on the last axis, lossless.

    Raises:
        ValueError: On an observation length outside `MAP_DIMS`.
    """
    m = _map_dim(obs.shape[-1])
    return obs[..., :m].astype(jnp.uint8), obs[..., m:].astype(jnp.float32)


def decompress_obs(map_view: jax.Array, tail: jax.Array) -> jax.Array:
    """The float32 observations `compress_obs` split."""
    return jnp.concatenate([map_view.astype(jnp.float32), tail], axis=-1)


def _steps(lead: tuple[int, ...], obs_dim: int) -> dict[str, jax.Array]:
    m = _map_dim(obs_dim)
    return {
        "map": jnp.zeros((*lead, m), jnp.uint8),
        "tail": jnp.zeros((*lead, obs_dim - m), jnp.float32),
        "act": jnp.zeros(lead, jnp.uint8),
        "rew": jnp.zeros(lead, jnp.float32),
        "done": jnp.zeros(lead, bool),
    }


def init(
    cfg: SILConfig,
    n_envs: int,
    n_ach: int,
    obs_dim: int,
    *,
    gamma: float = CONTROL_GAMMA,
    rnn_width: int = CONTROL_RNN_WIDTH,
    tiers: tuple[float, ...] | None = None,
    names: tuple[str, ...] | None = None,
) -> SILState:
    """Empty buffer of ``n_ach * S`` windows and an empty staging ring per env.

    ``tiers``: a value per achievement, higher kept first by the insertion cap (the
    reward weights); None ranks all alike. ``names`` label the per-achievement
    metrics; None numbers them.

    Raises:
        ValueError: On an observation length outside `MAP_DIMS`, gamma outside (0, 1],
            a GRU width below 1 or ``tiers`` or ``names`` not of length ``n_ach``.
    """
    tiers = (0.0,) * n_ach if tiers is None else tiers
    names = tuple(map(str, range(n_ach))) if names is None else names
    if not 0 < gamma <= 1 or rnn_width < 1 or {len(tiers), len(names)} != {n_ach}:
        msg = f"need 0 < gamma <= 1, rnn_width >= 1, {n_ach} tiers and names: "
        msg += f"{gamma}, {rnn_width}, {len(tiers)} tiers, {len(names)} names"
        raise ValueError(msg)
    slot_size = cfg.slots_windows * (1 if cfg.layout == "slots" else n_ach)
    rows = n_ach * cfg.slots_windows
    rows += rows == n_envs  # one unused row, so the buffer is never taken as per-env
    lw, h = cfg.window, rnn_width
    staging = _steps((n_envs, lw), obs_dim)
    staging["done"] = jnp.ones((n_envs, lw), bool)  # no step before the run counts
    staging["hidden"] = jnp.zeros((n_envs, lw, h), jnp.float16)
    buffer = _steps((rows, lw), obs_dim) | {
        "ret": jnp.zeros((rows, lw), jnp.float32),
        "mask": jnp.zeros((rows, lw), bool),
        "h0": jnp.zeros((rows, h), jnp.float16),
        "prio": jnp.zeros(rows, jnp.float32),
        "stamp": jnp.full(rows, -1, jnp.int32),
        "born": jnp.zeros(rows, jnp.int32),
        "ach": jnp.full(rows, -1, jnp.int32),
    }
    zero = jnp.zeros((), jnp.int32)
    return SILState(
        staging=staging,
        ptr=zero,
        buffer=buffer,
        next_stamp=zero,
        max_prio=jnp.ones((), jnp.float32),
        inserted=zero,
        overflow=zero,
        update=zero,
        cfg=cfg,
        slot_size=slot_size,
        gamma=gamma,
        tiers=tuple(int(r) for r in np.unique(tiers, return_inverse=True)[1]),
        names=names,
    )


def _nbytes(tree: object) -> int:
    """Bytes of a pytree's array leaves, from shapes alone (works under a trace)."""
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(tree))


def buffer_bytes(
    cfg: SILConfig,
    n_envs: int,
    n_ach: int,
    obs_dim: int = CRAFTAX_OBS_DIM,
    rnn_width: int = CONTROL_RNN_WIDTH,
) -> SILBytes:
    """Bytes `init` allocates, from its shapes alone (nothing is built)."""
    build = functools.partial(init, cfg, n_envs, n_ach, obs_dim, rnn_width=rnn_width)
    st = jax.eval_shape(build)

    return SILBytes(_nbytes(st.staging), _nbytes(st) - _nbytes(st.staging))


def check_memory(
    cfg: SILConfig,
    spec: HookSpec,
    memory_limit_bytes: int,
    rnn_width: int = CONTROL_RNN_WIDTH,
) -> SILBytes:
    """`buffer_bytes` of the run in ``spec``, refused above the machine's limit.

    A lower bound of the footprint: the state as allocated, without the copies an
    update may make of it.

    Raises:
        ValueError: If staging plus buffer exceed ``memory_limit_bytes``.
    """
    dims = (spec.n_envs, spec.n_achievements, spec.obs_dim, rnn_width)
    need = buffer_bytes(cfg, *dims)
    if need.total > memory_limit_bytes:
        msg = (
            f"SIL needs {need.total:,} B (staging {need.staging:,}, buffer "
            f"{need.buffer:,}), over memory_limit_bytes {memory_limit_bytes:,}"
        )
        raise ValueError(msg)
    return need


def _windows(
    staging: dict[str, jax.Array],
    ptr: jax.Array,
    roll: dict[str, jax.Array],
    t: jax.Array,
    env: jax.Array,
) -> dict[str, jax.Array]:
    """Window of each ``(t, env)`` ending at rollout step ``t``, ``(n, window, ...)``.

    Position ``c`` of staging followed by the rollout is staging column
    ``(ptr + c) % window`` while ``c < window``, else rollout step ``c - window``.
    """
    lw = staging["done"].shape[1]
    c = t[:, None] + 1 + jnp.arange(lw)
    old, e = c < lw, env[:, None]

    def pick(s: jax.Array, r: jax.Array) -> jax.Array:
        a, b = s[e, (ptr + c) % lw], r[jnp.maximum(c - lw, 0), e]
        return jnp.where(old.reshape(*old.shape, *(1,) * (a.ndim - 2)), a, b)

    return {k: pick(staging[k], roll[k]) for k in staging}


def _returns(rew: jax.Array, gamma: float) -> jax.Array:
    """Discounted returns along axis 1, cut after the last step (no bootstrap)."""

    def back(g: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
        g = r + gamma * g
        return g, g

    _, ret = jax.lax.scan(back, jnp.zeros(rew.shape[0]), rew.T, reverse=True)
    return ret.T


def _episode_mask(done: jax.Array) -> jax.Array:
    """Steps with no done between them and the window's last step (its episode)."""
    later = jnp.cumsum(done[:, :-1][:, ::-1], axis=1)[:, ::-1]
    return jnp.concatenate([later == 0, jnp.ones((done.shape[0], 1), bool)], axis=1)


def _target_rows(state: SILState, slot: jax.Array, live: jax.Array) -> jax.Array:
    """Buffer row of each candidate; out of range (dropped) when not written.

    A slot that gets ``k`` candidates keeps the newest ``min(k, S)`` and writes them
    over its empty, then oldest, rows, so the slot stays a FIFO.
    """
    size = state.slot_size
    same = (slot[:, None] == slot[None, :]) & live[:, None] & live[None, :]
    rank = jnp.tril(same, -1).sum(1)
    skip = jnp.maximum(same.sum(1) - size, 0)
    stamps = state.buffer["stamp"]
    n_rows = stamps.shape[0]
    order = jnp.argsort(stamps[: n_rows - n_rows % size].reshape(-1, size), axis=1)
    row = slot * size + order[slot, jnp.clip(rank - skip, 0, size - 1)]
    return jnp.where(live & (rank >= skip), row, n_rows)


def _write_staging(
    staging: dict[str, jax.Array], ptr: jax.Array, roll: dict[str, jax.Array]
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Staging after the rollout: its last ``window`` steps over the oldest columns."""
    lw = staging["done"].shape[1]
    n = min(roll["done"].shape[0], lw)
    cols = (ptr + jnp.arange(n)) % lw
    new = {k: s.at[:, cols].set(jnp.swapaxes(roll[k][-n:], 0, 1))
           for k, s in staging.items()}  # fmt: skip
    return new, (ptr + n) % lw


def _capped(state: SILState, unl: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Flat indices of the unlocks kept under ``insert_cap``, oldest first, and live.

    The highest tiers go first; within a tier the order is random (`TIE_SEED`), so
    rare late unlocks are not crowded out by common early ones.
    """
    cap = state.cfg.insert_cap
    key = jax.random.fold_in(jax.random.key(TIE_SEED), state.next_stamp)
    # 2 x rank + [0, 1): float32 rounding never lifts a tier into the next one
    rank = 2.0 * jnp.asarray(state.tiers, jnp.float32)
    score = jnp.where(unl, rank + jax.random.uniform(key, unl.shape), -1.0)
    k = min(cap, unl.size)
    top, idx = jax.lax.top_k(score.ravel(), k)
    idx = jnp.pad(
        jnp.where(top >= 0, idx, unl.size), (0, cap - k), constant_values=unl.size
    )
    idx = jnp.sort(idx)  # time order: the slot FIFO keeps the newest
    return jnp.minimum(idx, unl.size - 1), idx < unl.size


def observe(state: SILState, traj: Traj) -> SILState:
    """Copy the window of each new unlock into its slot, then stage the rollout.

    Raises:
        ValueError: If ``traj.hidden`` is not the staged GRU width, or
            ``traj.events.unlocked`` has another achievement count than the state.
    """
    cfg = state.cfg
    width, n_ach = state.staging["hidden"].shape[-1], len(state.tiers)
    if (traj.hidden.shape[-1], traj.events.unlocked.shape[-1]) != (width, n_ach):
        msg = (
            f"traj has hidden width {traj.hidden.shape[-1]} and "
            f"{traj.events.unlocked.shape[-1]} achievements, the state {width}, {n_ach}"
        )
        raise ValueError(msg)
    mp, tail = compress_obs(traj.obs)
    roll = {"map": mp, "tail": tail, "act": traj.actions.astype(jnp.uint8),
            "rew": traj.rewards.astype(jnp.float32), "done": traj.dones.astype(bool),
            "hidden": traj.hidden}  # fmt: skip
    unl = traj.events.unlocked
    total = jnp.sum(unl, dtype=jnp.int32)
    flat, live = _capped(state, unl)
    t, env, ach = jnp.unravel_index(flat, unl.shape)
    win = _windows(state.staging, state.ptr, roll, t, env)
    n = jnp.minimum(total, cfg.insert_cap)
    new = {k: win[k] for k in STEP_KEYS} | {
        "ret": _returns(win["rew"], state.gamma),
        "mask": _episode_mask(win["done"]),
        "h0": win["hidden"][:, 0],
        "prio": jnp.full(cfg.insert_cap, state.max_prio),
        "stamp": state.next_stamp + jnp.arange(cfg.insert_cap, dtype=jnp.int32),
        "born": jnp.full(cfg.insert_cap, state.update + 1),
        "ach": ach.astype(jnp.int32),
    }
    slot = ach if cfg.layout == "slots" else jnp.zeros_like(ach)
    rows = _target_rows(state, slot, live)
    buffer = {k: x.at[rows].set(new[k], mode="drop") for k, x in state.buffer.items()}
    staging, ptr = _write_staging(state.staging, state.ptr, roll)
    return state.replace(
        staging=staging,
        ptr=ptr,
        buffer=buffer,
        next_stamp=state.next_stamp + n,
        inserted=jnp.sum(rows < state.buffer["stamp"].shape[0], dtype=jnp.int32),
        overflow=total - n,
        update=state.update + 1,
    )


def sample_rows(state: SILState, key: jax.Array, n: int) -> jax.Array:
    """Rows of ``n`` draws: a non-empty slot uniformly, then a window by ``prio^alpha``.

    An empty buffer draws row 0, whose mask is all false.
    """
    size = state.slot_size
    n_rows = state.buffer["stamp"].shape[0]
    real = n_rows - n_rows % size  # the unused row of `init`, if any, is no slot
    full = (state.buffer["stamp"][:real] >= 0).reshape(-1, size)
    prio = state.buffer["prio"][:real].reshape(-1, size)
    k_slot, k_win = jax.random.split(key)
    slot_logits = jnp.where(full.any(1), 0.0, -jnp.inf)
    slot = jax.random.categorical(k_slot, slot_logits, shape=(n,))
    logits = jnp.where(
        full, state.cfg.alpha * jnp.log(jnp.where(full, prio, 1.0)), -jnp.inf
    )
    return slot * size + jax.random.categorical(k_win, logits[slot])


def sample(state: SILState, key: jax.Array) -> dict[str, jax.Array]:
    """``batch`` windows, time-major ``(window, batch, ...)`` as the network reads them.

    Also ``h0`` (float32 start states), ``rows`` and ``valid`` (a stored window).
    """
    rows = sample_rows(state, key, state.cfg.batch)
    buf = {
        k: state.buffer[k][rows] for k in ("map", "tail", "act", "done", "ret", "mask")
    }
    obs = decompress_obs(buf.pop("map"), buf.pop("tail"))
    out = {k: jnp.swapaxes(x, 0, 1) for k, x in buf.items()}
    return out | {
        "obs": jnp.swapaxes(obs, 0, 1),
        "act": out["act"].astype(jnp.int32),
        "h0": state.buffer["h0"][rows].astype(jnp.float32),
        "rows": rows,
        "valid": state.buffer["stamp"][rows] >= 0,
    }


def replay(
    net: ActorCriticRNN, params: Any, windows: dict[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """``(logits, values)`` of the windows, the GRU run from their stored start state.

    Resets follow the stored dones; the first step needs none, ``h0`` already has it.
    """
    done = windows["done"]
    resets = jnp.concatenate([jnp.zeros_like(done[:1]), done[:-1]])
    _, logits, v = apply_network(net, params, windows["h0"], windows["obs"], resets)
    return logits, v


def sil_terms(
    logp: jax.Array, v: jax.Array, ret: jax.Array, mask: jax.Array, cfg: SILConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """SIL loss over the masked steps and its parts.

    ``lambda_sil * mean(-logp * sg((R - V)+) + beta_v / 2 * (R - V)+^2)``; aux holds
    ``adv`` (``(R - V)+`` per step, 0 where masked), ``bias`` (``V - R`` per step),
    ``policy``, ``value`` (the two means, unweighted) and ``active`` (share of steps
    with ``R > V``).
    """
    w = mask.astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)  # an empty buffer gives 0, not NaN
    adv = jnp.maximum(ret - v, 0.0)
    policy = -jnp.sum(logp * jax.lax.stop_gradient(adv) * w) / n
    value = jnp.sum(0.5 * adv**2 * w) / n
    aux = {
        "adv": jax.lax.stop_gradient(adv) * w,
        "bias": jax.lax.stop_gradient(v - ret),
        "policy": policy,
        "value": value,
        "active": jnp.sum((adv > 0) * w) / n,
    }
    return cfg.lambda_sil * (policy + cfg.beta_v * value), aux


def sil_loss(
    net: ActorCriticRNN, cfg: SILConfig, params: Any, windows: dict[str, jax.Array]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """`sil_terms` of sampled windows replayed with ``params``."""
    logits, v = replay(net, params, windows)
    logp = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp, windows["act"][..., None], axis=-1)[..., 0]
    return sil_terms(logp, v, windows["ret"], windows["mask"], cfg)


def extra_loss(
    net: ActorCriticRNN, params: Any, state: SILState, key: jax.Array
) -> tuple[jax.Array, tuple[SILState, dict[str, jax.Array]]]:
    """The ``extra_loss_fn`` hook once bound to ``net`` (``functools.partial``).

    One SIL step: sample, loss, and the sampled windows' priorities rewritten as their
    mean ``(R - V)_+ + 0.01`` under ``params``.
    """
    w = sample(state, key)
    loss, aux = sil_loss(net, state.cfg, params, w)
    steps = jnp.maximum(w["mask"].sum(0), 1)
    prio = aux["adv"].sum(0) / steps + PRIORITY_EPS
    n_rows = state.buffer["prio"].shape[0]
    rows = jnp.where(w["valid"], w["rows"], n_rows)  # empty rows keep priority 0
    buffer = state.buffer | {
        "prio": state.buffer["prio"].at[rows].set(prio, mode="drop")
    }
    top = jnp.max(jnp.where(w["valid"], prio, 0.0))
    new = state.replace(buffer=buffer, max_prio=jnp.maximum(state.max_prio, top))
    return loss, (new, _sample_metrics(state, w, aux))


def _sample_metrics(
    state: SILState, w: dict[str, jax.Array], aux: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    """``sil/`` metrics of one SIL step's sample; all 0 when nothing is stored."""
    n = jnp.maximum(w["mask"].sum(), 1)
    valid = w["valid"]
    n_win = jnp.maximum(valid.sum(), 1)
    age = jnp.where(valid, state.update - state.buffer["born"][w["rows"]], 0)
    ach = jnp.where(valid, state.buffer["ach"][w["rows"]], -1)
    share = (ach[:, None] == jnp.arange(len(state.names))).sum(0) / n_win
    return {
        "sil/policy_loss": aux["policy"],
        "sil/value_loss": aux["value"],
        "sil/active": aux["active"],
        "sil/adv_pos_mean": aux["adv"].sum() / n,
        "sil/value_bias": jnp.sum(aux["bias"] * w["mask"]) / n,
        "sil/sample_age_mean": age.sum() / n_win,
        "sil/sample_age_max": age.max(),
    } | {f"sil/sample_share/{a}": x for a, x in zip(state.names, share, strict=True)}


def metrics(state: SILState) -> dict[str, jax.Array]:
    """The ``sil/...`` metrics of the buffer after the last rollout (``metrics_fn``).

    ``slot_fill/<achievement>`` counts the stored windows of each achievement (at most
    ``S`` in the slots layout); the byte sizes are float32, constant over a run.
    """
    stored = state.buffer["stamp"] >= 0
    fill = (
        jnp
        .zeros(len(state.names), jnp.int32)
        .at[state.buffer["ach"]]
        .add(stored.astype(jnp.int32), mode="drop")
    )
    staging, total = _nbytes(state.staging), _nbytes(state)
    return {
        "sil/inserted": state.inserted,
        "sil/overflow": state.overflow,
        "sil/nonempty_slots": jnp.count_nonzero(fill).astype(jnp.int32),
        "sil/buffer_bytes": jnp.float32(total - staging),
        "sil/staging_bytes": jnp.float32(staging),
    } | {f"sil/slot_fill/{a}": x for a, x in zip(state.names, fill, strict=True)}


def _init_hook(
    cfg: SILConfig, base: PPORNNConfig, _key: jax.Array, spec: HookSpec
) -> SILState:
    """``init_state_fn``: `init` sized by ``spec``, with the control's gamma and GRU.

    Raises:
        ValueError: If the state exceeds ``spec.memory_limit_bytes`` (`check_memory`).
    """
    if spec.memory_limit_bytes is not None:  # before anything is allocated
        check_memory(cfg, spec, spec.memory_limit_bytes, base.rnn_width)
    tiers = tuple(float(w) for w in achievement_weights(base.env))
    names = tuple(achievement_names(base.env))
    return init(
        cfg,
        spec.n_envs,
        spec.n_achievements,
        spec.obs_dim,
        gamma=base.gamma,
        rnn_width=base.rnn_width,
        tiers=tiers,
        names=names,
    )


def hooks(cfg: SILConfig, base: PPORNNConfig) -> HookFields:
    """The SIL hooks of a run config on the resolved PPO-RNN ``base`` config.

    The extra loss replays windows with the network ``base`` trains; gamma and the GRU
    width come from ``base`` too, so the method has no keys for them.
    """
    game = game_of(base.env)
    net = build_network(base, len(constants(game).Action), OBS_DIMS[game])
    return HookFields(
        init_state_fn=functools.partial(_init_hook, cfg, base),
        observe=observe,
        extra_loss_fn=functools.partial(extra_loss, net),
        extra_steps=cfg.sil_steps,
        metrics_fn=metrics,
    )


METHOD = HookMethod(SILConfig, hooks)
"""The method the ``sil`` registry entry builds its trainer with."""
