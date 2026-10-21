"""Top level of the SMDP skills method: the table ``Q[2^bits, 25]`` and its updates.

`top_key` hashes `skills.key_fields` plus the 4 low-stat flags to a row;
`select_skill` is masked ε-greedy over `skills.available`; `accumulate` keeps a
running skill's discounted return and ``gamma^tau``; `smdp_update` is the
semi-Markov Q-learning backup, one mean TD step per (key, skill) pair ended in the
same env step; `mc_init`, `mc_push`, `mc_update` are the Monte Carlo ablation. Pass
one granularity to `top_key`, `skills.available` and `skills.skill_step`. Pure, for
``jit``; no Craftax import. Design: [[SMDP Q-Learning over Hand-Defined Skills design]].
"""

import jax
import jax.numpy as jnp
from flax import struct

from .progress_key import Granularity, hash_fields
from .skills import ObsView, key_fields, low_stats


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
    f = key_fields(view, granularity)
    if low_flags:
        f = jnp.concatenate([f, low_stats(view).astype(jnp.int32)])
    return hash_fields(f, bits)


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
    best = jnp.max(jnp.where(avail_next, q[k_next], -jnp.inf), axis=-1)
    target = ret + jnp.where(done, 0.0, disc * best)
    return _mean_step(q, k, o, target - q[k, o], ended, alpha)


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
    k, o = chrono(rec.key), chrono(rec.skill)
    mask = valid & episode_end[:, None]
    q = _mean_step(q, k, o, gs.T - q[k, o], mask, alpha)
    return q, rec.replace(n=jnp.where(episode_end, 0, rec.n))
