"""Dyna-Q over the SMDP skills method: a skill-outcome model, planning and trainer.

The model maps each visited (key, skill) pair to a slot (``pair_id``, first visits in
order) holding a ring of its last ``M`` real outcomes ``(R, gamma^tau, k', done,
mask')``; `write` stores one env step's skill ends, `model_accuracy` counts outcomes
whose next key the pair already held. `plan` backs up the top-level table with
`smdp.smdp_update` (Dyna-Q+ sample backups, prioritized sweeping, value iteration);
the bonus is paid on simulated backups only. `DynaTrainer` (registry ``dyna-q``) is
the `smdp-skills` trainer with model writes in the rollout and `plan` after it; it
acts through `smdp.SMDPPolicy`. Imported after ``configure_runtime`` (via ``smdp``).
Design: [[Dyna-Q over a Skill-Level Progress Model design]].
"""

import functools
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .pqn_rnn import FILLED_LOSS_COLUMNS
from .records import achievement_names
from .skills import NUM_SKILLS, SKILLS
from .smdp import (
    RolloutCarry,
    SMDPPolicy,
    SMDPState,
    SMDPTrainConfig,
    SMDPTrainer,
    _chunk_metrics,
    _learn_after_rollout,
    _make_collect,
    _rollout_inputs,
    _setup,
    _share,
    _smdp_step,
    _update_stats,
    field_names,
    smdp_target,
    write_update_rows,
)
from .smdp import init_state as smdp_init_state

if TYPE_CHECKING:  # the runner imports mlflow; tests of this module do not need it
    from ..orchestration.runner import ChunkMetrics

type Variant = Literal["uniform", "prioritized", "value_iteration"]
"""Planning variant: sample backups (Dyna-Q, Dyna-Q+), sweeping, full sweeps."""


class Outcome(NamedTuple):
    """Finished real skills of one env step, one row per env."""

    k: jax.Array
    o: jax.Array
    """Start key and skill, int32 ``[envs]``."""
    ret: jax.Array
    disc: jax.Array
    """Discounted return and ``gamma^tau``, float32 ``[envs]``."""
    k_next: jax.Array
    done: jax.Array
    avail: jax.Array
    """Start-condition mask at the next key, bool ``[envs, n_skills]``."""


class ModelState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Lookup-table model of skill outcomes; P visited pairs, M ring slots each."""

    pair_id: jax.Array
    """Slot of each (key, skill), int32 ``[2^log2_rows, n_skills]``; -1 unvisited."""
    pair_k: jax.Array
    pair_o: jax.Array
    """Key and skill of each slot, int32 ``[P]``."""
    n_pairs: jax.Array
    """Slots in use, int32."""
    m_R: jax.Array  # noqa: N815  # the design's name
    m_disc: jax.Array
    """Stored ``R`` and ``gamma^tau``, float32 ``[P, M]``."""
    m_next: jax.Array
    m_done: jax.Array
    m_avail: jax.Array
    """Next key, done and packed next mask (`pack_mask`), ``[P, M]``."""
    count: jax.Array
    """Outcomes written per pair, int32 ``[P]`` (the ring holds the last M)."""
    last_tried: jax.Array
    """Time of the latest write per pair, float32 ``[P]`` (Dyna-Q+ bonus)."""
    full: jax.Array
    """All P slots taken: later new pairs are dropped (an abort signal), bool."""


def init_model(
    log2_rows: int, n_skills: int, pairs: int, slots: int, *, envs: int | None = None
) -> ModelState:
    """Empty model: ``2^log2_rows`` keys x ``n_skills``, ``pairs`` x ``slots``.

    Raises:
        ValueError: If a table's leading dim equals ``envs``: the trainers would
            split it over the env mesh as a per-env leaf.
    """
    rows = 2**log2_rows
    if envs in (rows, pairs):
        msg = f"model rows {rows} or pairs {pairs} equal envs {envs}; resize them"
        raise ValueError(msg)
    zi, zf = jnp.zeros((pairs, slots), jnp.int32), jnp.zeros((pairs, slots))
    return ModelState(
        pair_id=jnp.full((rows, n_skills), -1, jnp.int32),
        pair_k=jnp.zeros(pairs, jnp.int32),
        pair_o=jnp.zeros(pairs, jnp.int32),
        n_pairs=jnp.zeros((), jnp.int32),
        m_R=zf,
        m_disc=zf,
        m_next=zi,
        m_done=jnp.zeros((pairs, slots), bool),
        m_avail=zi,
        count=jnp.zeros(pairs, jnp.int32),
        last_tried=jnp.zeros(pairs),
        full=jnp.zeros((), bool),
    )


def pack_mask(avail: jax.Array) -> jax.Array:
    """Bool masks on the last axis as int32 bit sets.

    Raises:
        ValueError: On more than 31 skills (the sign bit is not used).
    """
    if avail.shape[-1] > 31:
        msg = f"pack_mask holds at most 31 skills, got {avail.shape[-1]}"
        raise ValueError(msg)
    bits = jnp.left_shift(1, jnp.arange(avail.shape[-1], dtype=jnp.int32))
    return jnp.sum(jnp.where(avail, bits, 0), -1, dtype=jnp.int32)


def unpack_mask(packed: jax.Array, n_skills: int) -> jax.Array:
    """Inverse of `pack_mask`: bool ``[..., n_skills]``."""
    shifted = jnp.right_shift(packed[..., None], jnp.arange(n_skills, dtype=jnp.int32))
    return (shifted & 1).astype(bool)


def write(
    model: ModelState, rec: Outcome, ended: jax.Array, t: jax.Array
) -> ModelState:
    """Store the outcomes of the envs whose skill ``ended`` at time ``t``.

    A new pair takes the next free slot, envs in order; envs ending one pair in the
    same step share its slot and take consecutive ring slots. Of more than M such
    envs only the last M are written (as M sequential writes would leave), so no two
    scatters hit one ring slot and every row comes from one env; ``count`` counts
    them all. With all P slots taken, new pairs are dropped.
    """
    pairs, slots = model.m_R.shape
    n_skills = model.pair_id.shape[1]
    flat = rec.k * n_skills + rec.o
    same = (flat[:, None] == flat[None, :]) & ended[:, None] & ended[None, :]
    first = jnp.argmax(same, axis=1)  # lowest env index on the same pair
    rank = jnp.sum(jnp.tril(same, -1), axis=1)  # earlier envs on the same pair
    last = rank >= jnp.sum(same, axis=1) - slots  # the last M of each pair group
    known = model.pair_id[rec.k, rec.o]
    opens = ended & (known < 0) & (first == jnp.arange(flat.shape[0]))
    new_id = model.n_pairs + jnp.cumsum(opens) - 1
    pid = jnp.where(known >= 0, known, new_id[first])
    ok = ended & (pid < pairs)
    pid = jnp.where(ok, pid, pairs)  # out of bounds: dropped by every scatter
    slot = (model.count.at[pid].get(mode="fill", fill_value=0) + rank) % slots
    rows = jnp.where(opens & ok, rec.k, model.pair_id.shape[0])
    rid = jnp.where(last, pid, pairs)

    def ring(a: jax.Array, x: jax.Array) -> jax.Array:
        return a.at[rid, slot].set(x.astype(a.dtype), mode="drop")

    n_new = model.n_pairs + jnp.sum(opens)
    return ModelState(
        pair_id=model.pair_id.at[rows, rec.o].set(pid, mode="drop"),
        pair_k=model.pair_k.at[pid].set(rec.k, mode="drop"),
        pair_o=model.pair_o.at[pid].set(rec.o, mode="drop"),
        n_pairs=jnp.minimum(n_new, pairs),
        m_R=ring(model.m_R, rec.ret),
        m_disc=ring(model.m_disc, rec.disc),
        m_next=ring(model.m_next, rec.k_next),
        m_done=ring(model.m_done, rec.done),
        m_avail=ring(model.m_avail, pack_mask(rec.avail)),
        count=model.count.at[pid].add(1, mode="drop"),
        last_tried=model.last_tried.at[pid].set(t, mode="drop"),
        full=model.full | (n_new >= pairs),
    )


def model_accuracy(
    model: ModelState, rec: Outcome, ended: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """``(hits, ended)``: ended outcomes whose next key the pair already holds.

    Call before `write`; a pair never seen counts as a miss.
    """
    slots = model.m_R.shape[1]
    pid = model.pair_id[rec.k, rec.o]
    seen = pid >= 0
    p = jnp.where(seen, pid, 0)
    filled = jnp.arange(slots) < jnp.minimum(model.count[p], slots)[:, None]
    hit = jnp.any(filled & (model.m_next[p] == rec.k_next[:, None]), axis=1)
    return jnp.sum(ended & seen & hit), jnp.sum(ended)


def _bonus(model: ModelState, p: jax.Array, kappa: float, t: jax.Array) -> jax.Array:
    """Dyna-Q+ bonus ``kappa sqrt(t - last_tried)`` of pair slots ``p``."""
    return kappa * jnp.sqrt(jnp.maximum(t - model.last_tried[p], 0.0))


def _records(
    model: ModelState, p: jax.Array, m: jax.Array, kappa: float, t: jax.Array
) -> Outcome:
    """Stored outcomes at pair slots ``p`` and ring slots ``m``, bonus in ``ret``."""
    return Outcome(
        k=model.pair_k[p],
        o=model.pair_o[p],
        ret=model.m_R[p, m] + _bonus(model, p, kappa, t),
        disc=model.m_disc[p, m],
        k_next=model.m_next[p, m],
        done=model.m_done[p, m],
        avail=unpack_mask(model.m_avail[p, m], model.pair_id.shape[1]),
    )


PlanStats = dict[str, jax.Array]
"""Backups' ``td_abs`` (sum of ``|y - Q|``), ``td_n`` (outcomes), ``bonus`` (sum),
``q_upd`` (largest finite new value, -inf for none), ``q_bad`` (a non-finite one)."""


def _no_stats() -> PlanStats:
    """`PlanStats` of no backup."""
    zero = jnp.float32(0.0)
    return {
        "td_abs": zero,
        "td_n": jnp.int32(0),
        "bonus": zero,
        "q_upd": jnp.float32(-jnp.inf),
        "q_bad": jnp.bool_(False),
    }


def _merge(a: PlanStats, b: PlanStats) -> PlanStats:
    """`PlanStats` of the backups of ``a`` and ``b``."""
    return {
        **{k: a[k] + b[k] for k in ("td_abs", "td_n", "bonus")},
        "q_upd": jnp.maximum(a["q_upd"], b["q_upd"]),
        "q_bad": a["q_bad"] | b["q_bad"],
    }


def _backup(
    q: jax.Array,
    model: ModelState,
    pm: tuple[jax.Array, jax.Array],
    valid: jax.Array,
    kappa: float,
    t: jax.Array | float,
    alpha: float,
) -> tuple[jax.Array, PlanStats]:
    """`sample_backup` of the outcomes ``pm = (p, m)`` and its `PlanStats`."""
    p, m = pm
    t = jnp.asarray(t, jnp.float32)
    r = _records(model, p, m, kappa, t)
    q, td = _smdp_step(
        q, r.k, r.o, r.ret, r.disc, r.k_next, r.avail, r.done, valid, alpha
    )
    per_batch = _update_stats(td[None], q[r.k, r.o][None], valid[None])
    td_abs, n, q_upd, q_bad = (x[0] for x in per_batch)
    bonus = jnp.sum(jnp.where(valid, _bonus(model, p, kappa, t), 0.0))
    stats = {"td_abs": td_abs, "td_n": n, "bonus": bonus, "q_upd": q_upd}
    return q, stats | {"q_bad": q_bad}


def sample_backup(
    q: jax.Array,
    model: ModelState,
    p: jax.Array,
    m: jax.Array,
    valid: jax.Array,
    kappa: float,
    t: jax.Array | float,
    alpha: float,
) -> jax.Array:
    """Backups of the stored outcomes ``(p, m)`` where ``valid``.

    The SMDP target of `smdp.smdp_update` plus ``kappa sqrt(t - last_tried)``;
    duplicates averaged.
    """
    return _backup(q, model, (p, m), valid, kappa, t, alpha)[0]


def _filled(model: ModelState, p: jax.Array) -> jax.Array:
    """Ring slots holding an outcome, bool ``[len(p), M]``."""
    slots = model.m_R.shape[1]
    return jnp.arange(slots) < jnp.minimum(model.count[p], slots)[:, None]


def _full(
    q: jax.Array,
    model: ModelState,
    p: jax.Array,
    valid: jax.Array,
    kappa: float,
    t: jax.Array | float,
    alpha: float,
) -> tuple[jax.Array, PlanStats]:
    """`full_backup` and its `PlanStats` (over the stored outcomes)."""
    slots = model.m_R.shape[1]
    pp = jnp.repeat(p, slots)
    mm = jnp.tile(jnp.arange(slots), p.shape[0])
    mask = (valid[:, None] & _filled(model, p)).ravel()
    return _backup(q, model, (pp, mm), mask, kappa, t, alpha)


def full_backup(
    q: jax.Array,
    model: ModelState,
    p: jax.Array,
    valid: jax.Array,
    kappa: float,
    t: jax.Array | float,
    alpha: float,
) -> jax.Array:
    """``Q += alpha (mean stored-outcome target - Q)`` for pairs ``p`` where ``valid``.

    A `sample_backup` of every filled slot: the duplicate-mean rule of
    `smdp.smdp_update` averages them.
    """
    return _full(q, model, p, valid, kappa, t, alpha)[0]


def uniform_batch(
    model: ModelState, key: jax.Array, size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``size`` uniform draws of a visited pair and one of its filled ring slots.

    ``valid`` is false while the model is empty.
    """
    k_p, k_m = jax.random.split(key)
    p = jax.random.randint(k_p, (size,), 0, jnp.maximum(model.n_pairs, 1))
    n = jnp.minimum(model.count[p], model.m_R.shape[1])
    m = jnp.floor(jax.random.uniform(k_m, (size,)) * n).astype(jnp.int32)
    m = jnp.minimum(m, jnp.maximum(n - 1, 0))
    return p, m, jnp.full(size, model.n_pairs > 0)


def priorities(
    q: jax.Array, model: ModelState, kappa: float, t: jax.Array | float
) -> jax.Array:
    """``|full-backup target - Q|`` of every pair slot, float32 ``[P]``; -1 unused."""
    pairs, slots = model.m_R.shape
    p = jnp.arange(pairs)
    r = _records(model, p[:, None], jnp.arange(slots)[None], kappa, jnp.float32(t))
    y = smdp_target(q, r.ret, r.disc, r.k_next, r.avail, r.done)
    filled = _filled(model, p)
    mean = jnp.sum(jnp.where(filled, y, 0.0), 1) / jnp.maximum(filled.sum(1), 1)
    prio = jnp.abs(mean - q[model.pair_k, model.pair_o])
    return jnp.where(p < model.n_pairs, prio, -1.0)


def _top(prio: jax.Array, size: int, theta: float) -> tuple[jax.Array, jax.Array]:
    """The ``size`` pairs of highest priority and which of them exceed ``theta``."""
    top, p = jax.lax.top_k(prio, size)
    return p, top > theta


def prioritized_batch(
    q: jax.Array,
    model: ModelState,
    kappa: float,
    t: jax.Array | float,
    size: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    """The ``size`` pairs of highest priority and which of them exceed ``theta``."""
    return _top(priorities(q, model, kappa, t), size, theta)


def _value_iteration(
    q: jax.Array,
    model: ModelState,
    kappa: float,
    t: jax.Array | float,
    tol: float,
    max_sweeps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, PlanStats]:
    """`value_iteration`, the last sweep's largest change and all sweeps' stats."""
    p = jnp.arange(model.m_R.shape[0])
    visited = p < model.n_pairs

    def body(c: tuple[jax.Array, jax.Array, jax.Array, PlanStats]) -> Any:
        q, _, n, stats = c
        q2, s = _full(q, model, p, visited, kappa, t, 1.0)
        return q2, jnp.max(jnp.abs(q2 - q)), n + 1, _merge(stats, s)

    def more(c: tuple[jax.Array, jax.Array, jax.Array, PlanStats]) -> jax.Array:
        _, change, n, _ = c
        return (change >= tol) & (n < max_sweeps)

    start = (q, jnp.float32(jnp.inf), jnp.int32(0), _no_stats())
    q, change, sweeps, stats = jax.lax.while_loop(more, body, start)
    return q, sweeps, change, stats


def value_iteration(
    q: jax.Array,
    model: ModelState,
    kappa: float,
    t: jax.Array | float,
    tol: float,
    max_sweeps: int,
) -> tuple[jax.Array, jax.Array]:
    """Synchronous full backups (step 1) of all visited pairs, and the sweeps run.

    Stops when the largest change of a sweep is below ``tol`` (a change, not an
    error bound) or after ``max_sweeps``.
    """
    return _value_iteration(q, model, kappa, t, tol, max_sweeps)[:2]


def plan(
    q: jax.Array,
    model: ModelState,
    n_backups: jax.Array | int,
    variant: Variant,
    kappa: float,
    t: jax.Array | float,
    key: jax.Array,
    *,
    alpha: float,
    batch: int,
    theta: float,
    tol: float,
    max_sweeps: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Planning on ``q`` after a rollout: ``n_backups`` backups in batches.

    ``uniform``: sample backups of random stored outcomes; ``prioritized``: full
    backups of the top pairs by recomputed priority (above ``theta``), priorities
    recomputed per batch; ``value_iteration``: sweeps to ``tol`` (``n_backups``
    unused). ``kappa`` is the Dyna-Q+ bonus per ``sqrt`` time unit. The keyword
    settings come from the method config (no defaults to drift from it). Returns
    the table and statistics: ``backups`` (outcomes, pairs or sweeps x visited
    pairs), ``sweeps`` and ``residual`` (value iteration's last change),
    ``above_theta`` (pairs above ``theta`` before planning, prioritized) and the
    `PlanStats` of every backup.
    """
    n = jnp.asarray(n_backups, jnp.int32)
    zero = jnp.int32(0)
    none = {"sweeps": zero, "above_theta": zero, "residual": jnp.float32(0.0)}
    if variant == "prioritized":  # top_k cannot take more than the pair slots
        batch = min(batch, model.m_R.shape[0])
    if variant == "value_iteration":
        q, sweeps, change, stats = _value_iteration(q, model, kappa, t, tol, max_sweeps)
        backups = sweeps * model.n_pairs
        out = {"backups": backups, "sweeps": sweeps, "residual": change}
        return q, stats | none | out

    def body(i: jax.Array, c: tuple[jax.Array, jax.Array, Any, jax.Array]) -> Any:
        q, done, stats, above = c
        left = jnp.arange(batch) < n - i * batch
        if variant == "uniform":
            p, m, valid = uniform_batch(model, jax.random.fold_in(key, i), batch)
            valid &= left
            q, s = _backup(q, model, (p, m), valid, kappa, t, alpha)
        else:
            prio = priorities(q, model, kappa, t)
            above = jnp.where(i == 0, jnp.sum(prio > theta), above)
            p, valid = _top(prio, batch, theta)
            valid &= left
            q, s = _full(q, model, p, valid, kappa, t, alpha)
        return q, done + jnp.sum(valid), _merge(stats, s), above

    batches = (n + batch - 1) // batch
    start = (q, zero, _no_stats(), zero)
    q, backups, stats, above = jax.lax.fori_loop(0, batches, body, start)
    return q, stats | none | {"backups": backups, "above_theta": above}


VARIANTS = ("uniform", "prioritized", "value_iteration")
"""The `Variant` values."""
TIME_UNITS = 1000
"""Dyna-Q+ time units per run budget: κ transfers between budgets."""
STALE_UNITS = 100.0
"""Time units without a real outcome after which ``dyna/stale_share`` counts a pair."""


@dataclass(frozen=True, slots=True)
class DynaConfig:
    """Planning settings; no defaults, so the method YAML decides every run."""

    plan_variant: Variant
    plan_n: int
    """Planning backups per real skill end of the rollout (value iteration: unused)."""
    plan_kappa: float
    """Dyna-Q+ bonus per ``sqrt`` time unit (`TIME_UNITS` per budget); 0 is Dyna-Q."""
    model_slots: int
    model_pairs: int
    """Ring slots per pair (M) and visited-pair capacity (P)."""
    plan_batch: int
    ps_theta: float
    vi_tol: float
    vi_max_sweeps: int

    def __post_init__(self) -> None:
        """Refuse settings planning cannot use.

        Raises:
            ValueError: On an unknown variant or a value out of range.
        """
        checks = {
            f"plan_variant must be one of {VARIANTS}, got {self.plan_variant!r}": (
                self.plan_variant in VARIANTS
            ),
            f"need plan_n >= 0, got {self.plan_n}": self.plan_n >= 0,
            f"need plan_kappa >= 0, got {self.plan_kappa}": self.plan_kappa >= 0,
            f"need model_slots >= 1, got {self.model_slots}": self.model_slots >= 1,
            f"need model_pairs >= 1, got {self.model_pairs}": self.model_pairs >= 1,
            f"need plan_batch >= 1, got {self.plan_batch}": self.plan_batch >= 1,
            f"need ps_theta >= 0, got {self.ps_theta}": self.ps_theta >= 0,
            f"need vi_tol > 0, got {self.vi_tol}": self.vi_tol > 0,
            f"need vi_max_sweeps >= 1, got {self.vi_max_sweeps}": (
                self.vi_max_sweeps >= 1
            ),
        }
        if failed := [msg for msg, ok in checks.items() if not ok]:
            raise ValueError("; ".join(failed))


DYNA_KEYS = frozenset(f.name for f in fields(DynaConfig))
"""Run-config keys of the planning; the other keys configure `smdp-skills`."""


@dataclass(frozen=True, slots=True)
class DynaTrainConfig:
    """Run config: `smdp.SMDPTrainConfig` plus the planning `DynaConfig`."""

    smdp: SMDPTrainConfig
    dyna: DynaConfig

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Split a run config: `DYNA_KEYS` to the planning, the rest to `smdp-skills`.

        Raises:
            ValueError: On a missing planning key or a value `DynaConfig` refuses.
        """
        if missing := sorted(DYNA_KEYS - config.keys()):
            msg = f"set every planning key: missing {missing}"
            raise ValueError(msg)
        rest = {k: v for k, v in config.items() if k not in DYNA_KEYS}
        dyna = DynaConfig(**{k: config[k] for k in DYNA_KEYS})
        return cls(SMDPTrainConfig.from_config(rest), dyna)


class DynaState(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Training state: the `smdp-skills` state, the model and the planning key.

    The model and key sit on the actor device with the table; `save_state` keeps
    them, the ``ckpt/`` params do not (acting reads only the table).
    """

    smdp: SMDPState
    model: ModelState
    plan_key: jax.Array
    cfg: DynaConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]  # flax API


def _now(t_env: jax.Array, steps: int) -> jax.Array:
    """Env-step count ``t_env`` in time units (`TIME_UNITS` per budget)."""
    return jnp.asarray(t_env, jnp.float32) * (TIME_UNITS / steps)


@functools.cache
def _collect(cfg: SMDPTrainConfig) -> Callable[..., Any]:
    """`smdp-skills` rollout that also writes every real skill end to the model.

    ``aux`` is (model, [hits, ends]) with the `model_accuracy` counts; cached per
    config, so runs that differ only in planning share the compiled rollout.
    """
    s = _setup(cfg)

    def observe(aux: Any, e: dict[str, jax.Array], t: jax.Array) -> Any:
        model, acc = aux
        rec = Outcome(*(e[k] for k in Outcome._fields))
        hits, ended = model_accuracy(model, rec, e["ended"])
        model = write(model, rec, e["ended"], _now(t, cfg.net.steps))
        return model, acc + jnp.stack([hits, ended]).astype(jnp.int32)

    diamond = achievement_names(cfg.net.env).index("collect_diamond")
    return _make_collect(cfg, s.net, s.env, s.env_params, diamond, observe)


@functools.cache
def _planner(cfg: DynaConfig, alpha: float, steps: int) -> Callable[..., Any]:
    """Jitted `plan` with every setting from ``cfg``; the table is donated.

    ``run(q, model, ended, t_env, key)`` plans ``plan_n x ended`` backups at the
    env-step count ``t_env`` and returns the table and the statistics.
    """

    @functools.partial(jax.jit, donate_argnums=0)
    def run(
        q: jax.Array, model: ModelState, ended: jax.Array, t_env: jax.Array, key: Any
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        t = _now(t_env, steps)
        q, stats = plan(
            q, model, cfg.plan_n * ended, cfg.plan_variant, cfg.plan_kappa, t, key,
            alpha=alpha, batch=cfg.plan_batch, theta=cfg.ps_theta, tol=cfg.vi_tol,
            max_sweeps=cfg.vi_max_sweeps,
        )  # fmt: skip
        return q, stats | _model_stats(model, t)

    return cast(Callable[..., Any], run)


def _model_stats(model: ModelState, t: jax.Array) -> dict[str, jax.Array]:
    """Visited, full (M outcomes) and stale pairs at time ``t``, and ``full``.

    New arrays: the model itself is donated to the next rollout.
    """
    pairs, slots = model.m_R.shape
    visited = jnp.arange(pairs) < model.n_pairs
    stale = visited & (t - model.last_tried > STALE_UNITS)
    return {
        "n_pairs": model.n_pairs.astype(jnp.float32),
        "full_pairs": jnp.sum(visited & (model.count >= slots)),
        "stale": jnp.sum(stale),
        "full": model.full | False,
    }


def init_state(cfg: DynaTrainConfig, key: jax.Array) -> DynaState:
    """The `smdp-skills` initial state of ``key``, an empty model, a planning key.

    Raises:
        ValueError: If the model's rows or pairs equal the env count (`init_model`).
    """
    s, d = _setup(cfg.smdp), cfg.dyna
    with jax.default_device(s.actor):
        model = init_model(
            cfg.smdp.top.log2_rows, NUM_SKILLS, d.model_pairs, d.model_slots,
            envs=cfg.smdp.net.envs,
        )  # fmt: skip
    state = smdp_init_state(cfg.smdp, key)
    # a stream of its own: the smdp-skills state of the same key stays unchanged
    plan_key = jax.random.fold_in(key, 1)
    at = s.shared_at
    return DynaState(
        smdp=state,
        model=jax.device_put(model, at),
        plan_key=jax.device_put(plan_key, at),
        cfg=d,
    )


def _run_plan(planner: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """``planner(*args)`` to completion and its wall time in seconds."""
    start = time.perf_counter()
    out = jax.block_until_ready(planner(*args))  # type: ignore[no-untyped-call]  # jax API
    return out, time.perf_counter() - start


def _one_update(
    state: DynaState, carry: RolloutCarry, model: ModelState
) -> tuple[DynaState, RolloutCarry, ModelState, tuple[Any, ...]]:
    """Rollout with model writes, planning on the table, then `pqn_rnn.learn`.

    Planning runs (and is timed) before the learner's update, which does not read
    the table. Returns the state (carry and model not yet set), the donated-on
    carry and model, and the update's smdp record, chunk arrays and Dyna record.
    """
    begin = time.perf_counter()
    cfg = state.smdp.cfg
    s = _setup(cfg)
    variables, start = _rollout_inputs(state.smdp, s)
    acc = jax.device_put(jnp.zeros(2, jnp.int32), s.shared_at)
    *rollout, (model, acc) = _collect(cfg)(
        carry, variables, state.smdp.diamond, start, (model, acc)
    )
    carry = rollout[0]
    jax.block_until_ready(carry)  # type: ignore[no-untyped-call]  # jax API
    key, k_plan = jax.random.split(state.plan_key)
    planner = _planner(state.cfg, cfg.top.alpha, cfg.net.steps)
    (q, stats), seconds = _run_plan(
        planner, carry.shared.q, model, acc[1], carry.shared.t, k_plan
    )
    carry = carry.replace(shared=carry.shared.replace(q=q))
    sm, carry, record, chunk = _learn_after_rollout(
        state.smdp, s, carry, tuple(rollout[1:])
    )
    jax.block_until_ready(sm.train_state)  # type: ignore[no-untyped-call]  # jax API
    dyna = {**stats, "hits": acc[0], "ends": acc[1]}
    dyna |= {"plan_seconds": seconds, "update_seconds": time.perf_counter() - begin}
    state = state.replace(smdp=sm, plan_key=key)
    return state, carry, model, (record, chunk, dyna)


def train_chunk(state: DynaState, n_updates: int) -> tuple[DynaState, "ChunkMetrics"]:
    """Run ``n_updates`` updates; the `smdp-skills` metrics plus ``dyna/...``.

    The input state stays usable: the rollouts donate copies of its carry and model.
    """
    carry = jax.tree.map(jnp.copy, state.smdp.carry)
    model = jax.tree.map(jnp.copy, state.model)
    records, chunks, dyna = [], [], []
    for _ in range(n_updates):
        state, carry, model, (record, chunk, d) = _one_update(state, carry, model)
        records.append(record)
        chunks.append(chunk)
        dyna.append(d)
    state = state.replace(smdp=state.smdp.replace(carry=carry), model=model)
    metrics = _chunk_metrics(records, chunks, carry.shared.q.shape[0])
    host = jax.device_get(dyna)
    stacked = {k: np.stack([r[k] for r in host]) for k in host[0]}
    return state, cast("ChunkMetrics", metrics | method_metrics(stacked))


def method_metrics(r: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The ``dyna/...`` metrics of stacked update records, ``(updates,)`` each.

    Model after the rollout: ``pairs_visited``, ``pairs_full_share`` (pairs with M
    outcomes), ``stale_share`` (no real outcome for `STALE_UNITS`) and
    ``model_accuracy`` (the rollout's real outcomes whose next key the pair already
    held). Planning: ``backups``, ``plan_td_abs`` and ``bonus_mean`` per backed-up
    outcome, ``ps_above_theta``, ``vi_sweeps``, ``vi_residual`` (0 for the other
    variants), ``plan_seconds`` and ``plan_wall_share`` (of the update's wall time).
    Empty shares are 0, so every value is finite.
    """
    pairs, td_n = r["n_pairs"], r["td_n"]
    return {
        "dyna/pairs_visited": pairs,
        "dyna/pairs_full_share": _share(r["full_pairs"], pairs),
        "dyna/model_accuracy": _share(r["hits"], r["ends"]),
        "dyna/backups": r["backups"],
        "dyna/plan_td_abs": _share(r["td_abs"], td_n),
        "dyna/bonus_mean": _share(r["bonus"], td_n),
        "dyna/stale_share": _share(r["stale"], pairs),
        "dyna/ps_above_theta": r["above_theta"],
        "dyna/vi_sweeps": r["sweeps"],
        "dyna/vi_residual": r["residual"],
        "dyna/plan_seconds": r["plan_seconds"],
        "dyna/plan_wall_share": _share(r["plan_seconds"], r["update_seconds"]),
    }


MODEL_GRAPH_FILE = "dyna_model_graph.parquet"
"""File in the run directory's ``eval/``: `model_graph` rows of each checkpoint."""


def model_graph(state: DynaState) -> dict[str, np.ndarray]:
    """The learned skill graph: one row per visited pair (key, skill) -> next key.

    Columns: ``pair``, ``key`` and its fields, ``skill``, ``next_key`` (the most
    frequent next key of the stored outcomes, the lowest slot on ties) and its
    fields as ``next_<field>``, ``mean_return``, ``mean_tau`` (from ``gamma^tau``)
    and ``count`` (real outcomes, the ring keeps the last M). Fields as in
    `smdp.greedy_table`, read from the table's decided keys.
    """
    m = jax.device_get(state.model)
    seen = np.asarray(state.smdp.carry.shared.seen)
    n, slots = int(m.n_pairs), m.m_R.shape[1]
    count = np.asarray(m.count[:n])
    filled = np.arange(slots) < np.minimum(count, slots)[:, None]
    n_filled = filled.sum(1)
    nxt = np.asarray(m.m_next[:n])
    votes = ((nxt[:, :, None] == nxt[:, None, :]) & filled[:, None, :]).sum(-1)
    pick = np.where(filled, votes, -1).argmax(1)
    next_key = np.take_along_axis(nxt, pick[:, None], 1)[:, 0]
    k = np.asarray(m.pair_k[:n])
    disc = np.where(filled, np.asarray(m.m_disc[:n], np.float64), 1.0)  # no log 0
    tau = np.log(disc) / np.log(state.smdp.cfg.top.gamma_top)

    def mean(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.where(filled, x, 0.0).sum(1) / np.maximum(n_filled, 1))

    names = field_names(seen.shape[1])
    skill_names = np.array([s.name for s in SKILLS], object)
    return {
        "pair": np.arange(n, dtype=np.int32),
        "key": k,
        **{f: seen[k, i] for i, f in enumerate(names)},
        "skill": skill_names[np.asarray(m.pair_o[:n])],
        "next_key": next_key,
        **{f"next_{f}": seen[next_key, i] for i, f in enumerate(names)},
        "mean_return": mean(np.asarray(m.m_R[:n])),
        "mean_tau": mean(tau),
        "count": count,
    }


def write_model_graph(run_dir: Path, update: int, state: DynaState) -> Path:
    """Put `model_graph` of ``update`` into the run's ``eval/`` file, atomically.

    Rows of an earlier call for the same update are replaced. The model is in the
    training state only (``state/``, not ``ckpt/``), so this runs at training time.
    """
    path = Path(run_dir) / "eval" / MODEL_GRAPH_FILE
    return write_update_rows(path, update, model_graph(state))


_SMDP = SMDPTrainer()
"""The `smdp-skills` entry: checkpoint payload and policy are its own."""


@dataclass(frozen=True, slots=True)
class DynaTrainer:
    """The ``dyna-q`` registry entry (`orchestration.runner.Trainer`)."""

    loss_columns: ClassVar[tuple[str, ...]] = FILLED_LOSS_COLUMNS
    """The skill network's TD loss and gradient norm (the others are NaN)."""

    def config(self, config: Mapping[str, Any]) -> DynaTrainConfig:
        """See `DynaTrainConfig.from_config`."""
        return DynaTrainConfig.from_config(config)

    def steps_per_update(self, config: Mapping[str, Any]) -> int:
        """Rollout batch: ``envs * num_steps``; planning takes no env steps."""
        return self.config(config).smdp.net.batch

    def init(self, config: Mapping[str, Any], key: jax.Array) -> DynaState:
        """Validated config to the initial state.

        Raises:
            ValueError: On an unknown or missing key, or no ``steps`` (the schedules
                and the Dyna-Q+ time unit span the budget).
        """
        if "steps" not in config:
            msg = "Dyna-Q schedules span the budget: pass `steps` in the config"
            raise ValueError(msg)
        return init_state(self.config(config), key)

    def train_chunk(
        self, state: DynaState, n_updates: int
    ) -> tuple[DynaState, "ChunkMetrics"]:
        """See `train_chunk`."""
        return train_chunk(state, n_updates)

    def params(self, state: DynaState) -> dict[str, Any]:
        """The ``ckpt/`` payload of `smdp-skills` (sparse table); no model."""
        return _SMDP.params(state.smdp)

    def policy(self, state: DynaState) -> SMDPPolicy:
        """The two-level policy of `smdp-skills`."""
        return _SMDP.policy(state.smdp)

    def env_wrapper(self, state: DynaState) -> None:  # protocol
        """No env wrapper: the rollout is the method's own."""
