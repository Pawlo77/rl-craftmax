"""Dyna-Q over the SMDP skills method: a skill-outcome model and planning updates.

The model maps each visited (key, skill) pair to a compact slot (``pair_id``, first
visits in order) holding a ring of its last ``M`` real outcomes ``(R, gamma^tau, k',
done, mask')``; `write` stores the skill ends of one env step, `model_accuracy`
counts outcomes whose next key the pair had already seen. Pure, for ``jit`` and
``vmap``. Design: [[Dyna-Q over a Skill-Level Progress Model design]].
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


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
