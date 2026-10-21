"""SMDP skills method: top-level table ``Q[2^bits, 25]``, rollout, trainer, policy.

`top_key` hashes `key_vector` (`skills.key_fields` plus 4 low-stat flags) to a row;
`select_skill` is masked ε-greedy over `skills.available`; `accumulate` keeps a
running skill's discounted return and ``gamma^tau``; `smdp_update` is the semi-Markov
backup, one mean TD step per (key, skill) pair ended in the same env step; `mc_init`,
`mc_push`, `mc_update` are the Monte Carlo ablation; `rollout_step` is one env step of
every env. `SMDPTrainer` (registry ``smdp-skills``) trains the skill network with
`pqn_rnn.learn` and acts through `SMDPPolicy`, whose skill tests read the reward that
evaluation hands `Policy.step`. Imported after ``configure_runtime`` (via ``pqn_rnn``).
Design: [[SMDP Q-Learning over Hand-Defined Skills design]].
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from flax import struct
from jax.sharding import Mesh, NamedSharding

from ..evaluation.episode_metrics import EpisodeAcc, episode_step, init_acc
from ..orchestration import shard_envs
from .hooks import Traj
from .ppo import env_placement, make_train_env
from .pqn_rnn import (
    FILLED_LOSS_COLUMNS,
    NO_ACTION,
    Carry,
    PQNRNNConfig,
    QTrainState,
    RNNQNetwork,
    apply_network,
    build_network,
    eps_greedy,
    epsilon,
    learn,
    optimizer,
    variables_of,
)
from .progress_key import CLASSIC_FIELDS, FIELDS, Granularity, hash_fields
from .records import LOSS_COLUMNS, achievement_names
from .skills import (
    EPISODE_END,
    FAILURE,
    FLOOR_CHANGE,
    LOW_STAT,
    NUM_SKILLS,
    PRIMITIVE,
    REASONS,
    SKILLS,
    SUCCESS,
    TIMEOUT,
    ObsView,
    available,
    decode_obs,
    key_fields,
    low_stats,
    primitive_action,
    skill_step,
    step_limit,
    test_agreement,
)

if TYPE_CHECKING:  # the runner imports mlflow; tests of this module do not need it
    from ..orchestration.runner import ChunkMetrics

EXTRA_DIM = NUM_SKILLS + 1
"""Skill-network inputs after the observation: skill one-hot, step-limit fraction."""


def init_table(
    log2_rows: int, n_skills: int, q_init: float, *, envs: int | None = None
) -> jax.Array:
    """Table of ``2^log2_rows`` keys x ``n_skills``, every entry ``q_init``, float32.

    Raises:
        ValueError: If the row count equals ``envs``: the trainers would split the
            table over the env mesh as a per-env leaf (`ppo.place_on_actor`).
    """
    rows = 2**log2_rows
    if rows == envs:
        msg = f"table rows {rows} equal envs {envs}; pick another log2_rows"
        raise ValueError(msg)
    return jnp.full((rows, n_skills), q_init, jnp.float32)


def top_key(
    view: ObsView, granularity: Granularity, bits: int, *, low_flags: bool = True
) -> jax.Array:
    """Table row of one observation: key fields plus low-stat flags, hashed, int32.

    ``low_flags=False`` is the no-low-stat-flags ablation.
    """
    return hash_fields(key_vector(view, granularity, low_flags=low_flags), bits)


def key_vector(
    view: ObsView, granularity: Granularity, *, low_flags: bool = True
) -> jax.Array:
    """The fields `top_key` hashes: `skills.key_fields`, then the 4 low-stat flags."""
    f = key_fields(view, granularity)
    if low_flags:
        f = jnp.concatenate([f, low_stats(view).astype(jnp.int32)])
    return f


def select_skill(
    q: jax.Array, k: jax.Array, avail: jax.Array, eps: jax.Array | float, key: jax.Array
) -> jax.Array:
    """ε-greedy skill of one env over the available skills of row ``k``, int32.

    Greedy ties are broken at random (the optimistic start makes every entry equal).
    """
    k_eps, k_tie = jax.random.split(key)
    noise = jax.random.uniform(k_tie, avail.shape)
    row = jnp.where(avail, q[k], -jnp.inf)
    greedy = jnp.argmax(jnp.where(row == row.max(), noise, -1.0))
    explore = jnp.argmax(jnp.where(avail, noise, -1.0))
    pick = jnp.where(jax.random.uniform(k_eps) < eps, explore, greedy)
    return pick.astype(jnp.int32)


def accumulate(
    ret: jax.Array, disc: jax.Array, r: jax.Array, gamma: float
) -> tuple[jax.Array, jax.Array]:
    """One step of a running skill: ``R += disc x r`` and ``disc *= gamma``.

    Start at ``R = 0``, ``disc = 1``; after tau steps ``disc = gamma^tau``.
    """
    return ret + disc * r, disc * gamma


def _mean_step(
    q: jax.Array,
    k: jax.Array,
    o: jax.Array,
    td: jax.Array,
    mask: jax.Array,
    alpha: float,
) -> jax.Array:
    """Add ``alpha`` x the mean of ``td`` over the masked entries of each (k, o).

    Sums and counts are scatter-added over the distinct pairs in the batch (sized
    like the batch, not the table), so several ends of one pair make one step.
    """
    flat = (k * q.shape[1] + o).ravel()
    uniq, inv = jnp.unique(flat, return_inverse=True, size=flat.size)
    inv = inv.ravel()
    mask = mask.ravel()
    sums = jnp.zeros(flat.size, jnp.float32).at[inv].add(jnp.where(mask, td.ravel(), 0))
    counts = jnp.zeros(flat.size, jnp.float32).at[inv].add(mask)
    step = jnp.where(counts > 0, sums / jnp.maximum(counts, 1), 0.0)
    return q.reshape(-1).at[uniq].add(alpha * step).reshape(q.shape)


def smdp_target(
    q: jax.Array,
    ret: jax.Array,
    disc: jax.Array,
    k_next: jax.Array,
    avail_next: jax.Array,
    done: jax.Array,
) -> jax.Array:
    """``R + gamma^tau (1 - done) max_{o' available} Q[k', o']`` per row."""
    best = jnp.max(jnp.where(avail_next, q[k_next], -jnp.inf), axis=-1)
    return ret + jnp.where(done, 0.0, disc * best)


def smdp_update(
    q: jax.Array,
    k: jax.Array,
    o: jax.Array,
    ret: jax.Array,
    disc: jax.Array,
    k_next: jax.Array,
    avail_next: jax.Array,
    done: jax.Array,
    ended: jax.Array,
    alpha: float,
) -> jax.Array:
    """Semi-Markov Q-learning step for the envs whose skill ``ended`` (per-env args).

    Target ``R + gamma^tau (1 - done) max_{o' available} Q[k', o']``; envs ending
    the same (k, o) in this step share one step of ``alpha`` x their mean TD error.
    """
    return _smdp_step(q, k, o, ret, disc, k_next, avail_next, done, ended, alpha)[0]


def _smdp_step(
    q: jax.Array,
    k: jax.Array,
    o: jax.Array,
    ret: jax.Array,
    disc: jax.Array,
    k_next: jax.Array,
    avail_next: jax.Array,
    done: jax.Array,
    ended: jax.Array,
    alpha: float,
) -> tuple[jax.Array, jax.Array]:
    """`smdp_update` and the TD error of every env."""
    td = smdp_target(q, ret, disc, k_next, avail_next, done) - q[k, o]
    return _mean_step(q, k, o, td, ended, alpha), td


class MCRecords(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Per-env ring of this episode's skill ends for the Monte Carlo ablation.

    Every leaf has the env count as leading dim. An episode with more than
    ``length`` skill ends keeps its last ``length``; their returns stay exact.
    """

    key: jax.Array
    skill: jax.Array
    """Table row and skill of each record, int32 ``[envs, length]``."""
    ret: jax.Array
    disc: jax.Array
    """The skill's discounted return and ``gamma^tau``, float32 ``[envs, length]``."""
    n: jax.Array
    """Skill ends this episode, int32 ``[envs]`` (may exceed ``length``)."""


def mc_init(envs: int, length: int) -> MCRecords:
    """Empty records of ``envs`` envs, ``length`` slots each."""
    zi, zf = jnp.zeros((envs, length), jnp.int32), jnp.zeros((envs, length))
    return MCRecords(zi, zi, zf, zf, jnp.zeros(envs, jnp.int32))


def mc_push(
    rec: MCRecords,
    ended: jax.Array,
    k: jax.Array,
    o: jax.Array,
    ret: jax.Array,
    disc: jax.Array,
) -> MCRecords:
    """Append (k, o, R, gamma^tau) for the envs whose skill ``ended``."""
    rows = jnp.arange(rec.n.shape[0])
    slot = rec.n % rec.key.shape[1]

    def put(a: jax.Array, x: jax.Array) -> jax.Array:
        return a.at[rows, slot].set(jnp.where(ended, x, a[rows, slot]))

    return MCRecords(
        put(rec.key, k),
        put(rec.skill, o),
        put(rec.ret, ret.astype(jnp.float32)),
        put(rec.disc, disc.astype(jnp.float32)),
        rec.n + ended,
    )


def mc_update(
    q: jax.Array, rec: MCRecords, episode_end: jax.Array, alpha: float
) -> tuple[jax.Array, MCRecords]:
    """Update every record of the envs whose episode ended toward its return.

    Return of record i: ``G_i = R_i + gamma^tau_i G_{i+1}``, 0 after the last; the
    mean TD step per (k, o) as in `smdp_update`. Push the last skill end first;
    ended envs restart with no records. A reverse scan over ``length``: call it
    under ``lax.cond`` on any episode end.
    """
    q, rec, _ = _mc_flush(q, rec, episode_end, alpha)
    return q, rec


def _mc_flush(
    q: jax.Array, rec: MCRecords, episode_end: jax.Array, alpha: float
) -> tuple[jax.Array, MCRecords, tuple[jax.Array, ...]]:
    """`mc_update` and the `_update_stats` of its records."""
    k, o, g, mask = _mc_targets(rec, episode_end)
    td = g - q[k, o]
    q = _mean_step(q, k, o, td, mask, alpha)
    rec = rec.replace(n=jnp.where(episode_end, 0, rec.n))
    return q, rec, _update_stats(td, q[k, o], mask)


def _mc_targets(
    rec: MCRecords, episode_end: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Rows, skills, returns and mask ``[envs, length]`` of the records to update."""
    length = rec.key.shape[1]
    m = jnp.minimum(rec.n, length)
    start = jnp.where(rec.n > length, rec.n % length, 0)
    order = (start[:, None] + jnp.arange(length)) % length  # oldest first

    def chrono(a: jax.Array) -> jax.Array:
        return jnp.take_along_axis(a, order, axis=1)

    valid = jnp.arange(length) < m[:, None]
    ret, disc = chrono(rec.ret), chrono(rec.disc)

    def back(g: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
        g = jnp.where(valid[:, i], ret[:, i] + disc[:, i] * g, 0.0)
        return g, g

    _, gs = jax.lax.scan(back, jnp.zeros(m.shape), jnp.arange(length), reverse=True)
    return chrono(rec.key), chrono(rec.skill), gs.T, valid & episode_end[:, None]


@dataclass(frozen=True, slots=True)
class SMDPConfig:
    """Top-level settings of the rollout (the skill network's are `PQNRNNConfig`)."""

    gamma_top: float = 0.999
    """Per-game-step discount of the top level; a skill of tau steps gets gamma^tau."""
    alpha: float = 0.2
    granularity: Granularity = "recipe"
    """Key buckets of `top_key`, `skills.available` and `skills.skill_step` alike."""
    timeout_mult: float = 1.0
    log2_rows: int = 20
    q_init: float = 10.0
    low_flags: bool = True
    """Low-stat flags in the key and their skill interruption (off: the ablation)."""
    monte_carlo: bool = False
    """Monte Carlo top level (ablation) instead of the SMDP backup."""
    mc_length: int = 1024
    """Skill ends per env and episode the Monte Carlo records keep."""
    eps_top_end: float = 0.05
    """Top-level ε after the decay (1: the random-top-level ablation)."""
    eps_top_decay: float = 0.1
    """Share of the updates over which the top-level ε falls linearly from 1."""

    def __post_init__(self) -> None:
        """Refuse settings the update cannot use.

        Raises:
            ValueError: On ``tiers`` granularity (no materials, so collect skills
                could never succeed) or a value out of range.
        """
        checks = {
            f"granularity must be recipe or log2, got {self.granularity!r}": (
                self.granularity in ("recipe", "log2")
            ),
            f"need 0 < alpha <= 1, got {self.alpha}": 0 < self.alpha <= 1,
            f"need 0 < gamma_top <= 1, got {self.gamma_top}": 0 < self.gamma_top <= 1,
            f"need timeout_mult > 0, got {self.timeout_mult}": self.timeout_mult > 0,
            f"need mc_length >= 1, got {self.mc_length}": self.mc_length >= 1,
            f"need 0 <= eps_top_end <= 1, got {self.eps_top_end}": (
                0 <= self.eps_top_end <= 1
            ),
            f"need 0 < eps_top_decay <= 1, got {self.eps_top_decay}": (
                0 < self.eps_top_decay <= 1
            ),
        }
        if failed := [msg for msg, ok in checks.items() if not ok]:
            raise ValueError("; ".join(failed))


class SkillState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Running skill per env, int32 or float32 ``[envs]``."""

    skill: jax.Array
    k_start: jax.Array
    """Table row where the skill started."""
    ret: jax.Array
    """Discounted native reward (plus any ``r+``) since the start."""
    disc: jax.Array
    """``gamma_top^steps``."""
    steps: jax.Array
    """Steps taken by the skill before the current one."""


class PerEnv(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Leaves with the env count as leading dim (split over an env mesh)."""

    obs: jax.Array
    env_state: Any
    h: Carry
    """LSTM carry, kept across skill ends, reset after an episode end."""
    last_done: jax.Array
    """Episode end of the previous step."""
    last_act: jax.Array
    skill: SkillState
    episode: EpisodeAcc
    """Return, length and unlocks of the running episode, for logging only."""
    mc: MCRecords | None
    """Monte Carlo records, ``None`` unless ``monte_carlo``."""


class Shared(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Leaves shared by all envs (replicated), whatever their leading dim."""

    variables: Any
    """Skill-network params and BatchRenorm statistics."""
    q: jax.Array
    seen: jax.Array
    """`key_vector` of a decision made at each row, int8 ``[rows, fields]``, -1 for
    rows never decided at (one of the keys when several share a row)."""
    ends: jax.Array
    """Skill ends per skill and `skills.REASONS` index, int32 ``[25, 7]``."""
    agreement: jax.Array
    """Per skill: ends the events judged, and of those the ones that agreed, int32
    ``[25, 2]`` (`skills.test_agreement`; episode ends are not judged)."""
    method_state: Any
    """State of the optional ``intrinsic_reward_fn``."""
    t: jax.Array
    """Env steps so far (all envs), int32; the hooks' ``t``."""
    key: jax.Array
    eps: jax.Array
    """Skill-network ε."""
    eps_top: jax.Array
    """Top-level ε."""


class RolloutCarry(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """`rollout_step` carry: per-env and shared parts are placed separately."""

    per_env: PerEnv
    shared: Shared
    cfg: SMDPConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def _views(obs: jax.Array) -> ObsView:
    """Views of a batch of observations."""
    return jax.vmap(decode_obs)(obs)


def _keys(cfg: SMDPConfig, v: ObsView) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Table rows, start-condition masks and `key_vector` of a batch of views."""
    g = cfg.granularity
    f = jax.vmap(lambda x: key_vector(x, g, low_flags=cfg.low_flags))(v)
    return hash_fields(f, cfg.log2_rows), jax.vmap(lambda x: available(x, g))(v), f


def _pick(
    q: jax.Array, k: jax.Array, avail: jax.Array, eps: jax.Array, key: jax.Array
) -> jax.Array:
    """`select_skill` of every env, each with its own key."""
    keys = jax.random.split(key, k.shape[0])
    return jax.vmap(select_skill, in_axes=(None, 0, 0, None, 0))(q, k, avail, eps, keys)


def _visit(seen: jax.Array, k: jax.Array, f: jax.Array, mask: jax.Array) -> jax.Array:
    """Write the key vectors of the masked decisions into their rows."""
    rows = jnp.where(mask, k, seen.shape[0])  # out of range: dropped
    return seen.at[rows].set(f.astype(seen.dtype), mode="drop")


def init_rollout(
    cfg: SMDPConfig,
    env: Any,
    network: RNNQNetwork,
    variables: Any,
    key: jax.Array,
    *,
    env_params: Any = None,
    eps: float | jax.Array = 1.0,
    eps_top: float | jax.Array = 1.0,
    method_state: Any = None,
) -> RolloutCarry:
    """Reset the batched ``env`` and choose every env's first skill.

    Raises:
        ValueError: If the table rows equal the env count (`init_table`), or the
            network does not take the observation plus `EXTRA_DIM` inputs.
    """
    params = env.default_params if env_params is None else env_params
    envs = env.num_envs
    q = init_table(cfg.log2_rows, NUM_SKILLS, cfg.q_init, envs=envs)
    key, k_reset, k_pick = jax.random.split(key, 3)
    obs, st = env.reset(k_reset, params)
    if network.obs_dim != obs.shape[-1] + EXTRA_DIM:
        msg = (
            f"the skill network takes {network.obs_dim} inputs, not the observation "
            f"({obs.shape[-1]}) plus EXTRA_DIM ({EXTRA_DIM}): set "
            f"extra_obs_dim={EXTRA_DIM}"
        )
        raise ValueError(msg)
    k, avail, f = _keys(cfg, _views(obs))
    f0, f1 = jnp.zeros(envs, jnp.float32), jnp.ones(envs, jnp.float32)
    i0 = jnp.zeros(envs, jnp.int32)
    skill = SkillState(_pick(q, k, avail, jnp.float32(eps_top), k_pick), k, f0, f1, i0)
    per_env = PerEnv(
        obs=obs,
        env_state=st,
        h=network.init_carry(envs),
        last_done=jnp.zeros(envs, bool),
        last_act=jnp.full(envs, NO_ACTION, jnp.int32),
        skill=skill,
        episode=init_acc(envs, len(achievement_names(env.name))),
        mc=mc_init(envs, cfg.mc_length) if cfg.monte_carlo else None,
    )
    seen = jnp.full((q.shape[0], f.shape[-1]), -1, jnp.int8)
    shared = Shared(
        variables=variables,
        q=q,
        seen=_visit(seen, k, f, jnp.ones(envs, bool)),
        ends=jnp.zeros((NUM_SKILLS, len(REASONS)), jnp.int32),
        agreement=jnp.zeros((NUM_SKILLS, 2), jnp.int32),
        method_state=method_state,
        t=jnp.zeros((), jnp.int32),
        key=key,
        eps=jnp.float32(eps),
        eps_top=jnp.float32(eps_top),
    )
    return RolloutCarry(per_env, shared, cfg)


def _fresh(h: Carry, reset: jax.Array) -> Carry:
    """The LSTM carry with the envs where ``reset`` zeroed."""
    return cast(Carry, tuple(jnp.where(reset[:, None], 0.0, part) for part in h))


def _skill_actions(
    network: RNNQNetwork,
    variables: Any,
    h: Carry,
    raw: jax.Array,
    sk: tuple[jax.Array, jax.Array],
    la: jax.Array,
    view: ObsView,
    eps: jax.Array,
    key: jax.Array,
    timeout_mult: float,
) -> tuple[jax.Array, jax.Array, Carry, jax.Array, jax.Array]:
    """Network input, action, next LSTM carry, primitive flag and max Q per env.

    ``sk`` is the running skill and its steps; the input appends the skill one-hot
    and ``steps / step_limit`` to the observation ``raw``.
    """
    skill, steps = sk
    frac = steps / step_limit(skill, timeout_mult)
    extra = jnp.concatenate([jax.nn.one_hot(skill, NUM_SKILLS), frac[:, None]], -1)
    x = jnp.concatenate([raw, extra.astype(raw.dtype)], -1)
    none = jnp.zeros((1, raw.shape[0]), bool)  # the caller already reset ``h``
    h2, qv = apply_network(network, variables, h, x[None], none, la[None])
    prim = jnp.asarray(PRIMITIVE)[skill]
    fixed = jax.vmap(primitive_action)(skill, view)
    act = jnp.where(prim, fixed, eps_greedy(key, qv[0], eps)).astype(jnp.int32)
    return x, act, h2, prim, qv[0].max(-1)


def _update_stats(
    td: jax.Array, new: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per env: |TD| sum, updates, largest finite new value, a non-finite one.

    The largest value is -inf where the env updated nothing.
    """
    ax = tuple(range(1, td.ndim))
    ok = jnp.isfinite(new)
    return (
        jnp.sum(jnp.where(mask, jnp.abs(td), 0.0), ax),
        jnp.sum(mask, ax, dtype=jnp.int32),
        jnp.max(jnp.where(mask & ok, new, -jnp.inf), ax),
        jnp.any(mask & ~ok, ax),
    )


Stats = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
"""`_update_stats` of a step."""


def _top_update(
    cfg: SMDPConfig,
    q: jax.Array,
    mc: MCRecords | None,
    sk: SkillState,
    ret_disc: tuple[jax.Array, jax.Array],
    nxt: tuple[jax.Array, jax.Array],
    done: jax.Array,
    ended: jax.Array,
) -> tuple[jax.Array, MCRecords | None, Stats]:
    """SMDP backup of the ended skills, or the Monte Carlo records and update."""
    ret, disc = ret_disc
    k, o = sk.k_start, sk.skill
    if mc is None:
        q, td = _smdp_step(q, k, o, ret, disc, *nxt, done, ended, cfg.alpha)
        return q, mc, _update_stats(td, q[k, o], ended)
    mc = mc_push(mc, ended, k, o, ret, disc)
    none = jnp.zeros(done.shape)
    idle = (none, none.astype(jnp.int32), none - jnp.inf, none.astype(bool))
    return cast(
        tuple[jax.Array, MCRecords, Stats],
        jax.lax.cond(
            jnp.any(done),
            lambda: _mc_flush(q, mc, done, cfg.alpha),
            lambda: (q, mc, idle),
        ),
    )


def rollout_step(
    carry: RolloutCarry,
    env: Any,
    network: RNNQNetwork,
    intrinsic_reward_fn: Any = None,
    *,
    env_params: Any = None,
) -> tuple[RolloutCarry, dict[str, jax.Array]]:
    """One env step of every env, for a ``lax.scan`` over the rollout.

    Acting, skill ends and keys read only decoded observations, the action, the native
    reward and ``done``; the step's ``StepEvents`` reach only `skills.test_agreement`
    and the episode totals (logging). ``intrinsic_reward_fn`` has the
    ``hooks.HookFields`` signature; its ``r+`` joins the native reward in the
    top-level return, never the skill pseudo-reward; ``None`` skips it at trace time.
    Outputs per env: the skill-network transition (``obs`` widened by `EXTRA_DIM`,
    ``action``, pseudo-reward ``reward``, skill-end ``done``, ``loss_mask``,
    ``last_action``), ``episode_done``, ``native_reward``, ``skill``,
    ``steps_in_skill``, ``success``, ``reason``, ``next_skill``, ``q_max``, the
    top-level update's ``td_abs``, ``td_n``, ``q_upd``, ``q_bad`` (`_update_stats`)
    and the episode totals ``ach``, ``ret``, ``length`` (logging only).
    """
    carry, out, _ = _rollout_step(carry, env, network, intrinsic_reward_fn, env_params)
    return carry, out


def _rollout_step(
    carry: RolloutCarry,
    env: Any,
    network: RNNQNetwork,
    intrinsic_reward_fn: Any,
    env_params: Any,
) -> tuple[RolloutCarry, dict[str, jax.Array], dict[str, jax.Array]]:
    """`rollout_step` and its real skill outcomes (Dyna's model reads them).

    The outcomes per env: start key ``k``, skill ``o``, return ``ret``, ``disc``
    (``gamma^tau``), next key ``k_next``, ``done``, next mask ``avail`` and
    ``ended``, the values of the step's top-level backup.
    """
    cfg, pe, sh = carry.cfg, carry.per_env, carry.shared
    params = env.default_params if env_params is None else env_params
    key, k_act, k_env, k_top = jax.random.split(sh.key, 4)
    sk, view = pe.skill, _views(pe.obs)
    la = jnp.where(pe.last_done, NO_ACTION, pe.last_act)
    h = _fresh(pe.h, pe.last_done)
    x, act, h2, prim, q_max = _skill_actions(
        network, sh.variables, h, pe.obs, (sk.skill, sk.steps), la,
        view, sh.eps, k_act, cfg.timeout_mult,
    )  # fmt: skip
    obs2, st2, r, done, info = env.step(k_env, pe.env_state, act, params)
    view2 = _views(obs2)
    r_top, ms = r, sh.method_state
    if intrinsic_reward_fn is not None:
        r_plus, ms = intrinsic_reward_fn(ms, pe.env_state, st2, obs2, done, sh.t)
        r_top = r + r_plus

    def end(k: Any, a: Any, act: Any, r: Any, b: Any, d: Any, n: Any) -> Any:
        return skill_step(
            k, a, act, r, b, d, n, cfg.timeout_mult,
            granularity=cfg.granularity, low_interrupt=cfg.low_flags,
        )  # fmt: skip

    success, ended, reason = jax.vmap(end)(
        sk.skill, view, act, r, view2, done, sk.steps
    )
    ended = ended | done
    ret, disc = accumulate(sk.ret, sk.disc, r_top, cfg.gamma_top)
    k2, avail2, f2 = _keys(cfg, view2)
    q, mc, stats = _top_update(
        cfg, sh.q, pe.mc, sk, (ret, disc), (k2, avail2), done, ended
    )
    new_skill = SkillState(
        jnp.where(ended, _pick(q, k2, avail2, sh.eps_top, k_top), sk.skill),
        jnp.where(ended, k2, sk.k_start),
        jnp.where(ended, 0.0, ret),
        jnp.where(ended, 1.0, disc),
        jnp.where(ended, 0, sk.steps + 1),
    )
    events = info["events"]
    checked, agree = jax.vmap(test_agreement)(sk.skill, success, events)
    judged = checked & ended & ~done  # a done step's events may unlock, never succeed
    episode, total = episode_step(pe.episode, r, done, events.unlocked)
    shared = sh.replace(
        q=q,
        seen=_visit(sh.seen, k2, f2, ended),
        ends=sh.ends.at[sk.skill, reason].add(ended.astype(jnp.int32)),
        agreement=sh.agreement.at[sk.skill].add(
            jnp.stack([judged, judged & agree], -1).astype(jnp.int32)
        ),
        method_state=ms,
        t=sh.t + obs2.shape[0],
        key=key,
    )
    per_env = PerEnv(obs2, st2, h2, done, act, new_skill, episode, mc)
    out = {
        "obs": x, "action": act, "reward": success.astype(jnp.float32),
        "done": ended, "loss_mask": ~prim, "last_action": la, "episode_done": done,
        "native_reward": r, "skill": sk.skill, "steps_in_skill": sk.steps,
        "success": success, "reason": reason, "next_skill": new_skill.skill,
        "q_max": q_max, "td_abs": stats[0], "td_n": stats[1], "q_upd": stats[2],
        "q_bad": stats[3], "ach": total.ach, "ret": total.ret, "length": total.length,
    }  # fmt: skip
    ends = {
        "k": sk.k_start, "o": sk.skill, "ret": ret, "disc": disc, "k_next": k2,
        "done": done, "avail": avail2, "ended": ended,
    }  # fmt: skip
    return carry.replace(per_env=per_env, shared=shared), out, ends


def to_traj(out: dict[str, jax.Array]) -> tuple[Traj, jax.Array]:
    """`pqn_rnn.learn` inputs from stacked `rollout_step` outputs: Traj, loss mask.

    ``dones`` are skill ends (no Q(lambda) target crosses one);
    ``extra["episode_dones"]`` keeps the episode ends the acting LSTM reset on.
    ``hidden`` is empty: pass the carry step 0 acted from as ``h0``, the rollout's
    start carry reset where ``last_done``.
    """
    traj = Traj(
        out["obs"],
        out["action"],
        out["reward"],
        out["done"],
        cast(Any, None),
        cast(Any, None),
        {"last_actions": out["last_action"], "episode_dones": out["episode_done"]},
    )
    return traj, out["loss_mask"]


TOP_KEYS = frozenset({f.name for f in fields(SMDPConfig)} - {"gamma_top"}) | {
    "top_horizon"
}
"""Run-config keys of the top level; ``top_horizon`` H sets ``gamma_top = 1 - 1/H``
(the fine grid scales the horizon). The other keys configure the skill network."""
MAX_OCCUPANCY = 0.25
"""Share of the table rows decided at above which ``smdp/table_full`` aborts."""
MAX_Q = 300.0
"""Table value above which ``smdp/divergence`` aborts: no discounted return exceeds
the 226 achievement reward plus the bounded health term."""
SHARD_ROLLOUT = False
"""Split the envs over the CPU mesh (shared leaves replicated); off: one actor device.
Off: every step's table update gathers all envs, and each CPU keeps its own table.
Craftax-Classic, 256 x 16, 2^20 rows (2026-10-21): 6,869 rollout steps/s and 1.9 GB on
one device, 5,823 steps/s and 4.9 GB split over 8."""


@dataclass(frozen=True, slots=True)
class SMDPTrainConfig:
    """Run config: skill network `PQNRNNConfig`, top level `SMDPConfig`."""

    net: PQNRNNConfig
    top: SMDPConfig

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Split a run config: `TOP_KEYS` to the top level, the rest to the network.

        Raises:
            ValueError: On ``gamma_top`` (set ``top_horizon``), an ``extra_obs_dim``
                other than `EXTRA_DIM`, or a key neither config takes.
        """
        if "gamma_top" in config:
            msg = "set the top-level discount as top_horizon, not gamma_top"
            raise ValueError(msg)
        if config.get("extra_obs_dim", EXTRA_DIM) != EXTRA_DIM:
            msg = f"extra_obs_dim must be {EXTRA_DIM} (skill one-hot, step fraction)"
            raise ValueError(msg)
        top = {k: v for k, v in config.items() if k in TOP_KEYS}
        if "top_horizon" in top:
            top["gamma_top"] = 1 - 1 / top.pop("top_horizon")
        net = {k: v for k, v in config.items() if k not in TOP_KEYS}
        net["extra_obs_dim"] = EXTRA_DIM
        return cls(PQNRNNConfig.from_config(net), SMDPConfig(**top))


def eps_top(cfg: SMDPTrainConfig, update: jax.Array | int) -> jax.Array:
    """Top-level ε of update ``update``: 1 down to ``eps_top_end`` linearly."""
    top = cfg.top
    sched = optax.linear_schedule(
        1.0, top.eps_top_end, top.eps_top_decay * cfg.net.n_updates
    )
    return jnp.asarray(sched(update), jnp.float32)


class SMDPState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Full training state: one pytree that `save_state` round-trips."""

    train_state: QTrainState
    carry: RolloutCarry
    """Envs, skills, table and counters; ``shared.variables`` is None (each rollout
    gets the train state's) and ``shared.eps``, ``eps_top`` are the next rollout's."""
    updates: jax.Array
    diamond: jax.Array
    """Finished episodes with ``collect_diamond``, and the env steps at which the
    first and the tenth ended (-1 before), int32[3]."""
    start_key: jax.Array
    start_avail: jax.Array
    """Table row and start-condition mask of a reset observation (``smdp/q_start``)."""
    key: jax.Array
    """Learner key (minibatch shuffles)."""
    cfg: SMDPTrainConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


@dataclass(frozen=True, slots=True)
class _Setup:
    """Everything built once per config: network, env, devices, jitted rollout."""

    net: RNNQNetwork
    env: Any
    env_params: Any
    collect: Callable[..., Any]
    learner: jax.Device
    actor: jax.Device
    mesh: Mesh | None
    shared_at: jax.Device | NamedSharding


@functools.cache
def _setup(cfg: SMDPTrainConfig) -> _Setup:
    """Cached per config, so every chunk reuses the compiled rollout.

    Raises:
        ValueError: If the env batch does not split over the CPU devices.
    """
    # checks that the envs split over the CPUs even unsharded (pqn_rnn.learn does too)
    roles, mesh, shared_at = env_placement(cfg.net.envs)
    if not SHARD_ROLLOUT:
        mesh, shared_at = None, roles.actor
    env, env_params = make_train_env(cfg.net)
    width = env.observation_space(env_params).shape[0]
    net = build_network(cfg.net, env.action_space(env_params).n, width)
    names = achievement_names(cfg.net.env)
    collect = _make_collect(cfg, net, env, env_params, names.index("collect_diamond"))
    return _Setup(
        net, env, env_params, collect, roles.learner, roles.actor, mesh, shared_at
    )


def _place(carry: RolloutCarry, s: _Setup) -> RolloutCarry:
    """Per-env leaves split over the mesh, shared ones replicated; else the actor."""
    if s.mesh is None:
        return cast(RolloutCarry, jax.device_put(carry, s.actor))
    per_env = shard_envs(carry.per_env, s.mesh)
    shared = jax.device_put(carry.shared, s.shared_at)
    return carry.replace(per_env=per_env, shared=shared)


def _make_collect(
    cfg: SMDPTrainConfig,
    net: RNNQNetwork,
    env: Any,
    env_params: Any,
    diamond: int,
    observe: Callable[[Any, dict[str, jax.Array], jax.Array], Any] | None = None,
) -> Callable[..., Any]:
    """Jitted rollout of ``num_steps`` steps: carry, outputs, ``h0``, statistics, aux.

    ``observe(aux, ends, t)`` folds every step's real skill outcomes (`_rollout_step`)
    at env-step count ``t`` into the extra carry ``aux`` (Dyna-Q's model); without it
    ``aux`` passes through. The carry and ``aux`` are donated (`train_chunk` hands
    them copies of the state's).
    """
    envs, steps = cfg.net.envs, cfg.net.num_steps

    def body(c: tuple[RolloutCarry, Any], _: None) -> Any:
        carry, aux = c
        carry, out, ends = _rollout_step(carry, env, net, None, env_params)
        if observe is not None:
            aux = observe(aux, ends, carry.shared.t)
        return (carry, aux), out

    @functools.partial(jax.jit, donate_argnums=(0, 4))
    def collect(
        carry: RolloutCarry, variables: Any, found: jax.Array, start: Any, aux: Any
    ) -> Any:
        h0 = _fresh(carry.per_env.h, carry.per_env.last_done)
        c = carry.replace(shared=carry.shared.replace(variables=variables))
        (c, aux), out = jax.lax.scan(body, (c, aux), None, steps)
        stats = _rollout_stats(carry.shared, c.shared, out, start)
        hits = (out["episode_done"] & out["ach"][..., diamond]).sum(1)
        found = _diamonds(found, hits, carry.shared.t + envs * (jnp.arange(steps) + 1))
        c = c.replace(shared=c.shared.replace(variables=None))
        return c, out, h0, stats, found, aux

    return cast(Callable[..., Any], collect)


def _diamonds(found: jax.Array, hits: jax.Array, at: jax.Array) -> jax.Array:
    """`SMDPState.diamond` after a rollout with ``hits`` such episodes per step."""
    count = found[0] + jnp.cumsum(hits)

    def reach(prev: jax.Array, n: int) -> jax.Array:
        now = jnp.where(count[-1] >= n, at[jnp.argmax(count >= n)], -1)
        return jnp.where(prev >= 0, prev, now)

    return jnp.stack([count[-1], reach(found[1], 1), reach(found[2], 10)])


def _rollout_stats(
    before: Shared, after: Shared, out: dict[str, jax.Array], start: Any
) -> dict[str, jax.Array]:
    """Counts and sums of one rollout that the update's metrics are made of."""
    done = out["done"]
    lm = out["loss_mask"]
    row = after.q[start[0]]
    return {
        "ends": after.ends - before.ends,
        "agreement": after.agreement - before.agreement,
        "chosen": jnp.zeros(NUM_SKILLS, jnp.int32).at[out["next_skill"]].add(done),
        "length": jnp
        .zeros(NUM_SKILLS, jnp.int32)
        .at[out["skill"]]
        .add(done * (out["steps_in_skill"] + 1)),
        "td_abs": out["td_abs"].sum(),
        "td_n": out["td_n"].sum(),
        "q_upd": out["q_upd"].max(),
        "q_bad": out["q_bad"].any(),
        "q_mean": jnp.sum(out["q_max"] * lm) / jnp.maximum(lm.sum(), 1),
        "episodes": out["episode_done"].sum(),
        "visited": jnp.sum(after.seen[:, 0] >= 0),
        "q_start": jnp.max(jnp.where(start[1], row, -jnp.inf)),
    }


def init_state(cfg: SMDPTrainConfig, key: jax.Array) -> SMDPState:
    """Fresh network on the learner; envs, first skills and table on the actor."""
    s = _setup(cfg)
    key, k_init, k_roll = jax.random.split(key, 3)
    with jax.default_device(s.learner):
        v = s.net.init_variables(k_init)
        ts = QTrainState.create(  # type: ignore[no-untyped-call]  # flax API
            apply_fn=s.net.apply,
            params=v["params"],
            tx=optimizer(cfg.net),
            batch_stats=v["batch_stats"],
            cfg=cfg.net,
        )
        # an array from the start, as after an update (see `ppo.init_state`)
        ts = ts.replace(step=jnp.zeros((), jnp.int32))
    with jax.default_device(s.actor):
        zero = jnp.zeros((), jnp.int32)
        eps0, top0 = epsilon(cfg.net, zero), eps_top(cfg, zero)
        carry = init_rollout(
            cfg.top, s.env, s.net, None, k_roll, env_params=s.env_params,
            eps=eps0, eps_top=top0,
        )  # fmt: skip
        first = decode_obs(carry.per_env.obs[0])
        found = jnp.array([0, -1, -1], jnp.int32)
    at = s.shared_at
    return SMDPState(
        train_state=cast(QTrainState, ts),
        carry=_place(carry, s),
        updates=jax.device_put(jnp.zeros((), jnp.int32), at),
        diamond=jax.device_put(found, at),
        start_key=jax.device_put(carry.per_env.skill.k_start[0], at),
        start_avail=jax.device_put(available(first, cfg.top.granularity), at),
        key=jax.device_put(key, s.learner),
        cfg=cfg,
    )


@jax.jit
def _finite(tree: Any) -> jax.Array:
    """Whether every leaf of ``tree`` is finite."""
    return jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(tree)]))


def _one_update(
    state: SMDPState, s: _Setup, carry: RolloutCarry
) -> tuple[SMDPState, RolloutCarry, dict[str, Any], dict[str, jax.Array]]:
    """Rollout on the actor, `pqn_rnn.learn` on the learner, next ε values.

    Returns the state (its carry not yet set), the donated-on carry, the update's
    record and the chunk-metric arrays of its rollout.
    """
    variables, start = _rollout_inputs(state, s)
    rollout = s.collect(carry, variables, state.diamond, start, None)
    return _learn_after_rollout(state, s, rollout[0], rollout[1:5])


def _rollout_inputs(state: SMDPState, s: _Setup) -> tuple[Any, tuple[Any, Any]]:
    """The network variables on the actor and the start row and mask of a rollout."""
    variables = jax.device_put(variables_of(state.train_state), s.shared_at)
    return variables, (state.start_key, state.start_avail)


def _learn_after_rollout(
    state: SMDPState, s: _Setup, carry: RolloutCarry, rollout: tuple[Any, ...]
) -> tuple[SMDPState, RolloutCarry, dict[str, Any], dict[str, jax.Array]]:
    """`pqn_rnn.learn` on a rollout's ``(out, h0, stats, found)``, next ε values.

    The rest of `_one_update`; Dyna-Q runs it after its own rollout.
    """
    out, h0, stats, found = rollout
    traj, mask = to_traj(out)
    key, k_learn = jax.random.split(state.key)
    put = functools.partial(jax.device_put, device=s.learner)
    ts, m = learn(state.train_state, put(traj), k_learn, put(mask), put(h0))
    updates = state.updates + 1
    sh = carry.shared
    record = stats | m | {"finite": _finite(ts.params), "found": found}
    record |= {"eps_top": sh.eps_top, "eps": sh.eps}
    nxt = (epsilon(state.cfg.net, updates), eps_top(state.cfg, updates))
    eps, top = jax.device_put(nxt, s.shared_at)
    carry = carry.replace(shared=sh.replace(eps=eps, eps_top=top))
    state = state.replace(train_state=ts, updates=updates, diamond=found, key=key)
    chunk = {k: out[k] for k in ("episode_done", "ach", "ret", "length")}
    return state, carry, record, chunk


def train_chunk(state: SMDPState, n_updates: int) -> tuple[SMDPState, "ChunkMetrics"]:
    """Run ``n_updates`` rollout-and-update steps; metrics stacked per update.

    The input state stays usable: the rollouts donate a copy of its carry.
    """
    s = _setup(state.cfg)
    carry = jax.tree.map(jnp.copy, state.carry)
    records, chunks = [], []
    for _ in range(n_updates):
        state, carry, record, chunk = _one_update(state, s, carry)
        records.append(record)
        chunks.append(chunk)
    state = state.replace(carry=carry)
    return state, _chunk_metrics(records, chunks, state.carry.shared.q.shape[0])


def _chunk_metrics(
    records: list[dict[str, Any]],
    chunks: list[dict[str, jax.Array]],
    rows: int,
    max_q: float | None = None,
) -> "ChunkMetrics":
    """The `ChunkMetrics` and ``smdp/...`` metrics of the updates' records.

    ``max_q`` is the divergence bound (`method_metrics`).
    """
    host = jax.device_get(records)
    stacked = {k: np.stack([r[k] for r in host]) for k in host[0]}
    cols = {"value_loss": stacked["td_loss"], "grad_norm": stacked["grad_norm"]}
    nan = np.full(len(records), np.nan, np.float32)
    metrics: dict[str, Any] = {
        "losses": np.stack([cols.get(c, nan) for c in LOSS_COLUMNS], -1),
        "done": jnp.stack([c["episode_done"] for c in chunks]),
        "ach": jnp.stack([c["ach"] for c in chunks]),
        "ret": jnp.stack([c["ret"] for c in chunks]),
        "length": jnp.stack([c["length"] for c in chunks]),
    }
    metrics |= method_metrics(stacked, rows, max_q)
    return cast("ChunkMetrics", metrics)


END_SHARES = {
    "success": (SUCCESS,),
    "failure": (FAILURE,),
    "timeout": (TIMEOUT,),
    "episode_end": (EPISODE_END,),
    "interrupt": (FLOOR_CHANGE, LOW_STAT),
}
"""``smdp/skill/<name>/end/<reason>``: `skills.REASONS` indices pooled per name."""


def _share(n: np.ndarray, total: np.ndarray, empty: float = 0.0) -> np.ndarray:
    """``n / total``, ``empty`` where ``total`` is 0."""
    return np.where(total > 0, n / np.maximum(total, 1), empty)


def method_metrics(
    r: Mapping[str, np.ndarray], rows: int, max_q: float | None = None
) -> dict[str, np.ndarray]:
    """The ``smdp/...`` metrics of stacked update records, ``(updates,)`` each.

    Per skill, over the update's skill ends: ``chosen_share`` (of the decisions),
    ``success_rate``, ``mean_len`` (steps), ``end/<reason>`` (`END_SHARES`; the
    success share is the success rate) and ``test_agreement`` (of the judged ends);
    a skill without ends has rates 0 and agreement 1, so every value is finite.
    ``decisions_per_episode`` is the update's skill ends per episode it finished. The
    guards ``smdp/nan`` (a non-finite table update, loss or parameter),
    ``smdp/table_full`` (occupancy above `MAX_OCCUPANCY`) and ``smdp/divergence``
    (an updated entry above ``max_q``, default `MAX_Q`; Dyna-Q+ raises it) are 0, or
    NaN once broken, for ``abort.finite``.
    """
    ends, agree, chosen = r["ends"], r["agreement"], r["chosen"]
    n_ends = ends.sum(-1)
    per = {
        "chosen_share": _share(chosen, chosen.sum(-1, keepdims=True)),
        "success_rate": _share(ends[..., SUCCESS], n_ends),
        "mean_len": _share(r["length"], n_ends),
        **{
            f"end/{name}": _share(ends[..., list(ix)].sum(-1), n_ends)
            for name, ix in END_SHARES.items()
        },
        "test_agreement": _share(agree[..., 1], agree[..., 0], 1.0),
    }
    out = {
        f"smdp/skill/{s.name}/{k}": v[:, i]
        for k, v in per.items()
        for i, s in enumerate(SKILLS)
    }
    visited, q_upd = r["visited"], r["q_upd"]
    bad = r["q_bad"] | ~r["finite"] | ~np.isfinite(r["td_loss"] + r["grad_norm"])

    def guard(broken: np.ndarray) -> np.ndarray:
        return np.where(broken, np.nan, 0.0)

    return out | {
        "smdp/eps": r["eps_top"],
        "smdp/td_abs_mean": _share(r["td_abs"], r["td_n"]),
        "smdp/decisions_per_episode": _share(n_ends.sum(-1), r["episodes"]),
        "smdp/keys_visited": visited,
        "smdp/occupancy": visited / rows,
        "smdp/collision_est": visited / (2 * rows),  # share of keys sharing a row
        "smdp/q_start": r["q_start"],
        "smdp/q_max_updated": np.where(np.isfinite(q_upd), q_upd, 0.0),
        "smdp/low/loss": r["td_loss"],
        "smdp/low/q_mean": r["q_mean"],
        "smdp/low/eps": r["eps"],
        "smdp/diamond_episodes": r["found"][:, 0],
        "smdp/diamond_first_step": r["found"][:, 1],
        "smdp/diamond_tenth_step": r["found"][:, 2],
        "smdp/nan": guard(bad),
        "smdp/table_full": guard(visited > MAX_OCCUPANCY * rows),
        "smdp/divergence": guard(q_upd > (MAX_Q if max_q is None else max_q)),
    }


GREEDY_TABLE_FILE = "smdp_greedy_table.parquet"
"""File in the run directory's ``eval/``: `greedy_table` rows of each checkpoint."""
FLAG_NAMES = ("low_food", "low_drink", "low_energy", "low_health")
"""Columns of the low-stat flags in `key_vector` order."""


def field_names(width: int) -> list[str]:
    """Names of a `key_vector` of ``width``: progress-key fields, then `FLAG_NAMES`.

    Classic keys are 12 wide (16 with flags), Craftax keys 23 (27).
    """
    base = CLASSIC_FIELDS if width in (12, 16) else FIELDS
    return [*base, *FLAG_NAMES][:width]


def greedy_table(params: Mapping[str, Any], q_init: float) -> dict[str, np.ndarray]:
    """Columns per stored key: ``key``, fields, greedy and greedy tried skill.

    Columns: ``key``, the fields, ``greedy_skill``, ``value``, ``tried``,
    ``greedy_tried_skill``, ``greedy_tried_value``.
    The stored keys are the rows training moved off ``q_init`` (`sparse_table`). The
    greedy skill is the row's unmasked argmax: start conditions need the observation,
    which a row does not keep, so an untried optimistic entry can win. Tried entries
    are derived as ``q_rows != q_init``: ``tried`` counts them, and the greedy tried
    skill is the argmax over them ("" and NaN for none). Fields are
    `progress_key.FIELDS` (Classic `CLASSIC_FIELDS`), then `FLAG_NAMES` unless the
    run left the flags out.
    """
    real = np.asarray(params["rows"]) != SENTINEL_ROW
    fields_ = np.asarray(params["fields"])[real]
    q = np.asarray(params["q_rows"])[real]
    names = field_names(fields_.shape[1])
    cols = {"key": np.asarray(params["rows"], np.int32)[real]}
    cols |= {n: fields_[:, i] for i, n in enumerate(names)}
    skill_names = np.array([s.name for s in SKILLS], object)
    cols["greedy_skill"] = skill_names[q.argmax(1)]
    cols["value"] = q.max(1) if len(q) else np.zeros(0, q.dtype)
    tried = q != q.dtype.type(q_init)
    best = np.where(tried, q, -np.inf).argmax(1)
    any_tried = tried.any(1)
    cols["tried"] = tried.sum(1).astype(np.int32)
    cols["greedy_tried_skill"] = np.where(any_tried, skill_names[best], "").astype(
        object
    )
    picked = np.take_along_axis(q, best[:, None], 1)[:, 0]
    cols["greedy_tried_value"] = np.where(any_tried, picked, np.nan)
    return cols


SENTINEL_ROW = int(np.iinfo(np.int32).max)
"""Row index of padding in `sparse_table`: sorts last and matches no key."""
MIN_STORED_ROWS = 1024
"""Fewest rows `sparse_table` stores, so early checkpoints share one shape."""


def _padded(n: int) -> int:
    """Stored rows for ``n`` real ones: the next power of two, at least the minimum."""
    return max(MIN_STORED_ROWS, 1 << max(n - 1, 0).bit_length())


def sparse_table(q: Any, seen: Any, q_init: float) -> dict[str, Any]:
    """The table rows that differ from ``q_init``, with values and key fields.

    ``rows`` int32 ascending, ``q_rows``, and ``fields`` from ``seen`` (a hashed key
    keeps no fields, so they are stored). Padded to `_padded` rows with
    `SENTINEL_ROW` (values ``q_init``, fields -1), so evaluation compiles once per
    power of two, not per checkpoint. The abstract state of evaluation's template
    gets empty arrays: loading keeps the stored shapes.
    """
    if isinstance(q, jax.ShapeDtypeStruct):
        q, seen = (np.zeros((0, x.shape[1]), x.dtype) for x in (q, seen))
    q, seen = np.asarray(q), np.asarray(seen)
    rows = np.flatnonzero((q != q.dtype.type(q_init)).any(1))
    n = len(rows)
    if not q.shape[0]:  # the template
        return {"rows": rows.astype(np.int32), "q_rows": q, "fields": seen}
    m = _padded(n)
    out = {
        "rows": np.full(m, SENTINEL_ROW, np.int32),
        "q_rows": np.full((m, q.shape[1]), q_init, q.dtype),
        "fields": np.full((m, seen.shape[1]), -1, seen.dtype),
    }
    out["rows"][:n], out["q_rows"][:n], out["fields"][:n] = rows, q[rows], seen[rows]
    return out


def dense_table(params: Mapping[str, Any], cfg: SMDPConfig) -> np.ndarray:
    """The full table of a `sparse_table` payload, bit for bit."""
    q = np.full((2**cfg.log2_rows, NUM_SKILLS), cfg.q_init, np.float32)
    rows = np.asarray(params["rows"])
    real = rows != SENTINEL_ROW
    q[rows[real]] = np.asarray(params["q_rows"])[real]
    return q


def _lookup(params: Mapping[str, Any], k: jax.Array, q_init: float) -> jax.Array:
    """Table rows ``k`` of a `sparse_table` payload, ``[envs, skills]``."""
    rows, vals = jnp.asarray(params["rows"]), jnp.asarray(params["q_rows"])
    if rows.shape[0] == 0:
        return jnp.full((k.shape[0], NUM_SKILLS), q_init, jnp.float32)
    i = jnp.clip(jnp.searchsorted(rows, k), 0, rows.shape[0] - 1)
    return jnp.where((rows[i] == k)[:, None], vals[i], jnp.float32(q_init))


def write_greedy_table(
    run_dir: Path, update: int, params: Mapping[str, Any], q_init: float
) -> Path:
    """Put `greedy_table` of ``update`` into the run's ``eval/`` file, atomically.

    Rows of an earlier call for the same update are replaced.
    """
    path = Path(run_dir) / "eval" / GREEDY_TABLE_FILE
    return write_update_rows(path, update, greedy_table(params, q_init))


def write_update_rows(path: Path, update: int, cols: Mapping[str, Any]) -> Path:
    """Put the rows ``cols`` of ``update`` into the parquet file ``path``, atomically.

    An ``update`` column comes first; rows of an earlier call for the same update
    are replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(cols.values())))
    new = pa.table({"update": np.full(n, update, np.int32), **cols})
    if path.is_file():
        old = pq.read_table(path)
        keep = pc.not_equal(old["update"], pa.scalar(update, pa.int32()))
        new = pa.concat_tables([old.filter(keep), new])
    tmp = path.with_suffix(".tmp")
    pq.write_table(new, tmp)
    tmp.replace(path)
    return path


PolicyCarry = tuple[Carry, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
"""Per env: LSTM carry, last action, running skill, its steps, the previous
observation and whether a skill was chosen yet."""


@dataclass(frozen=True, slots=True)
class SMDPPolicy:
    """The two-level actor: table rows and skill network from `SMDPTrainer.params`."""

    net: RNNQNetwork
    cfg: SMDPConfig

    def init_carry(self, n: int) -> PolicyCarry:
        """Zero LSTM carry, no last action and no skill yet for ``n`` envs."""
        zero = jnp.zeros(n, jnp.int32)
        last = jnp.full(n, NO_ACTION, jnp.int32)
        prev = jnp.zeros((n, self.net.obs_dim - EXTRA_DIM))
        return self.net.init_carry(n), last, zero, zero, prev, jnp.zeros(n, bool)

    def step(
        self,
        params: Mapping[str, Any],
        carry: PolicyCarry,
        obs: jax.Array,
        key: jax.Array,
        greedy: bool | jax.Array,
        reward: jax.Array | None = None,
        done: jax.Array | None = None,
    ) -> tuple[jax.Array, PolicyCarry]:
        """End the last step's skill if its test says so, choose anew, then act.

        ``reward`` and ``done`` are those of the step that produced ``obs``, as in the
        training rollout; pass them: without ``reward`` (taken as 0) ``unlock_any``
        and ``fight``'s kill reward never succeed. A done starts a new episode (fresh
        skill, LSTM and last action). Greedy acts at ε 0 on both levels (greedy skill
        ties stay random; the random-top-level ablation keeps ``eps_top`` 1), else at
        the checkpoint's ``eps_top`` and ``eps``.
        """
        lstm, last, skill, steps, prev, started = carry
        n = obs.shape[0]
        r = jnp.zeros(n) if reward is None else reward
        new = ~started if done is None else ~started | done
        cfg, view2 = self.cfg, _views(obs)

        def end(k: Any, a: Any, act: Any, r: Any, b: Any, n: Any) -> Any:
            return skill_step(
                k, a, act, r, b, False, n, cfg.timeout_mult,
                granularity=cfg.granularity, low_interrupt=cfg.low_flags,
            )  # fmt: skip

        _, ended = jax.vmap(end)(skill, _views(prev), last, r, view2, steps)[:2]
        ended = ended | new
        k_top, k_act = jax.random.split(key)
        k, avail, _ = _keys(cfg, view2)
        # the random-top-level ablation stays random when greedy: no learned top level
        top = (
            params["eps_top"]
            if cfg.eps_top_end >= 1
            else jnp.where(greedy, 0.0, params["eps_top"])
        )
        q = _lookup(params, k, cfg.q_init)  # row i holds env i's table row
        picks = _pick(q, jnp.arange(n), avail, top, k_top)
        skill = jnp.where(ended, picks, skill)
        steps = jnp.where(ended, 0, steps + 1)
        last = jnp.where(new, NO_ACTION, last)
        eps = jnp.where(greedy, 0.0, params["eps"])
        _, act, lstm, _, _ = _skill_actions(
            self.net, params["variables"], _fresh(lstm, new), obs, (skill, steps),
            last, view2, eps, k_act, cfg.timeout_mult,
        )  # fmt: skip
        return act, (lstm, act, skill, steps, obs, jnp.ones_like(started))

    def extras(self, carry: PolicyCarry) -> dict[str, jax.Array]:
        """Per step for the eval file: the running skill and its step count."""
        return {
            "skill": carry[2].astype(jnp.uint8),
            "steps_in_skill": carry[3].astype(jnp.uint16),
        }


@dataclass(frozen=True, slots=True)
class SMDPTrainer:
    """The ``smdp-skills`` registry entry (`orchestration.runner.Trainer`).

    No random warm-up rollout as the PQN-RNN control's: the top level's ε = 1 phase
    is this method's warm-up, and every primitive step counts toward the budget.
    """

    loss_columns: ClassVar[tuple[str, ...]] = FILLED_LOSS_COLUMNS
    """The skill network's TD loss and gradient norm (the others are NaN)."""

    def config(self, config: Mapping[str, Any]) -> SMDPTrainConfig:
        """See `SMDPTrainConfig.from_config`."""
        return SMDPTrainConfig.from_config(config)

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Rollout batch: ``envs * num_steps``."""
        return self.config(config).net.batch

    def init(self, config: Mapping[str, Any], key: jax.Array) -> SMDPState:
        """Validated config to the initial state.

        Raises:
            ValueError: On an unknown key, or no ``steps`` (the ε and learning-rate
                schedules span the budget).
        """
        if "steps" not in config:
            msg = "SMDP schedules span the budget: pass `steps` in the config"
            raise ValueError(msg)
        return init_state(self.config(config), key)

    def train_chunk(
        self, state: SMDPState, n_updates: int
    ) -> tuple[SMDPState, "ChunkMetrics"]:
        """See `train_chunk`."""
        return train_chunk(state, n_updates)

    def params(self, state: SMDPState) -> dict[str, Any]:
        """The ``ckpt/`` payload: network variables, both ε, the sparse table."""
        sh = state.carry.shared
        base = {
            "variables": variables_of(state.train_state),
            "eps": sh.eps,
            "eps_top": sh.eps_top,
        }
        return base | sparse_table(sh.q, sh.seen, state.cfg.top.q_init)

    def policy(self, state: SMDPState) -> SMDPPolicy:
        """The two-level policy of this config (reads no state values)."""
        return SMDPPolicy(_setup(state.cfg).net, state.cfg.top)

    def env_wrapper(self, state: SMDPState) -> None:  # protocol
        """No env wrapper: the rollout is the method's own."""
