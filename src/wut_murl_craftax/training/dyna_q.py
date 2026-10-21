"""Dyna-Q over the SMDP skills method: a skill-outcome model and planning updates.

The model maps each visited (key, skill) pair to a compact slot (``pair_id``, first
visits in order) holding a ring of its last ``M`` real outcomes ``(R, gamma^tau, k',
done, mask')``; `write` stores the skill ends of one env step, `model_accuracy`
counts outcomes whose next key the pair had already seen. `plan` runs the planning
backups on the top-level table with `smdp.smdp_update` (sample backups with the
Dyna-Q+ bonus, prioritized sweeping, value iteration); the bonus is paid on
simulated backups only. Pure, for ``jit``. Design: [[Dyna-Q over a Skill-Level
Progress Model design]].
"""

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from .smdp import smdp_target, smdp_update

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
    """Bool masks on the last axis (at most 31 skills) as int32 bit sets."""
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
    same step share its slot and take consecutive ring slots (more than M of them
    overwrite each other). With all P slots taken, new pairs are dropped.
    """
    pairs, slots = model.m_R.shape
    n_skills = model.pair_id.shape[1]
    flat = rec.k * n_skills + rec.o
    same = (flat[:, None] == flat[None, :]) & ended[:, None] & ended[None, :]
    first = jnp.argmax(same, axis=1)  # lowest env index on the same pair
    rank = jnp.sum(jnp.tril(same, -1), axis=1)  # earlier envs on the same pair
    known = model.pair_id[rec.k, rec.o]
    opens = ended & (known < 0) & (first == jnp.arange(flat.shape[0]))
    new_id = model.n_pairs + jnp.cumsum(opens) - 1
    pid = jnp.where(known >= 0, known, new_id[first])
    ok = ended & (pid < pairs)
    pid = jnp.where(ok, pid, pairs)  # out of bounds: dropped by every scatter
    slot = (model.count.at[pid].get(mode="fill", fill_value=0) + rank) % slots
    rows = jnp.where(opens & ok, rec.k, model.pair_id.shape[0])

    def ring(a: jax.Array, x: jax.Array) -> jax.Array:
        return a.at[pid, slot].set(x.astype(a.dtype), mode="drop")

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


def _records(
    model: ModelState, p: jax.Array, m: jax.Array, kappa: float, t: jax.Array
) -> Outcome:
    """Stored outcomes at pair slots ``p`` and ring slots ``m``, bonus in ``ret``."""
    age = jnp.maximum(t - model.last_tried[p], 0.0)
    return Outcome(
        k=model.pair_k[p],
        o=model.pair_o[p],
        ret=model.m_R[p, m] + kappa * jnp.sqrt(age),
        disc=model.m_disc[p, m],
        k_next=model.m_next[p, m],
        done=model.m_done[p, m],
        avail=unpack_mask(model.m_avail[p, m], model.pair_id.shape[1]),
    )


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
    r = _records(model, p, m, kappa, jnp.asarray(t, jnp.float32))
    return smdp_update(
        q, r.k, r.o, r.ret, r.disc, r.k_next, r.avail, r.done, valid, alpha
    )


def _filled(model: ModelState, p: jax.Array) -> jax.Array:
    """Ring slots holding an outcome, bool ``[len(p), M]``."""
    slots = model.m_R.shape[1]
    return jnp.arange(slots) < jnp.minimum(model.count[p], slots)[:, None]


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
    slots = model.m_R.shape[1]
    pp = jnp.repeat(p, slots)
    mm = jnp.tile(jnp.arange(slots), p.shape[0])
    mask = (valid[:, None] & _filled(model, p)).ravel()
    return sample_backup(q, model, pp, mm, mask, kappa, t, alpha)


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


def prioritized_batch(
    q: jax.Array,
    model: ModelState,
    kappa: float,
    t: jax.Array | float,
    size: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    """The ``size`` pairs of highest priority and which of them exceed ``theta``."""
    prio, p = jax.lax.top_k(priorities(q, model, kappa, t), size)
    return p, prio > theta


def value_iteration(
    q: jax.Array,
    model: ModelState,
    kappa: float,
    t: jax.Array | float,
    tol: float,
    max_sweeps: int,
) -> tuple[jax.Array, jax.Array]:
    """Synchronous full backups (step 1) of all visited pairs, and the sweeps run.

    Stops when the largest change is below ``tol`` or after ``max_sweeps``.
    """
    p = jnp.arange(model.m_R.shape[0])
    visited = p < model.n_pairs

    def body(c: tuple[jax.Array, jax.Array, jax.Array]) -> Any:
        q, _, n = c
        q2 = full_backup(q, model, p, visited, kappa, t, 1.0)
        return q2, jnp.max(jnp.abs(q2 - q)), n + 1

    def more(c: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, change, n = c
        return (change >= tol) & (n < max_sweeps)

    start = (q, jnp.float32(jnp.inf), jnp.int32(0))
    q, _, sweeps = jax.lax.while_loop(more, body, start)
    return q, sweeps


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
    batch: int = 1024,
    theta: float = 1e-4,
    tol: float = 1e-3,
    max_sweeps: int = 50,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Planning on ``q`` after a rollout: ``n_backups`` backups in batches.

    ``uniform``: sample backups of random stored outcomes; ``prioritized``: full
    backups of the top pairs by recomputed priority (above ``theta``), priorities
    recomputed per batch; ``value_iteration``: sweeps to ``tol`` (``n_backups``
    unused). ``kappa`` is the Dyna-Q+ bonus per ``sqrt`` time unit. Returns the table
    and ``backups`` (pairs or outcomes backed up) and ``sweeps``.
    """
    n = jnp.asarray(n_backups, jnp.int32)
    zero = jnp.int32(0)
    if variant == "prioritized":  # top_k cannot take more than the pair slots
        batch = min(batch, model.m_R.shape[0])
    if variant == "value_iteration":
        q, sweeps = value_iteration(q, model, kappa, t, tol, max_sweeps)
        return q, {"backups": sweeps * model.n_pairs, "sweeps": sweeps}

    def body(i: jax.Array, c: tuple[jax.Array, jax.Array]) -> Any:
        q, done = c
        left = jnp.arange(batch) < n - i * batch
        if variant == "uniform":
            p, m, valid = uniform_batch(model, jax.random.fold_in(key, i), batch)
            valid &= left
            q = sample_backup(q, model, p, m, valid, kappa, t, alpha)
        else:
            p, valid = prioritized_batch(q, model, kappa, t, batch, theta)
            valid &= left
            q = full_backup(q, model, p, valid, kappa, t, alpha)
        return q, done + jnp.sum(valid)

    batches = (n + batch - 1) // batch
    q, backups = jax.lax.fori_loop(0, batches, body, (q, zero))
    return q, {"backups": backups, "sweeps": zero}
