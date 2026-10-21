"""SMDP skills method: the top-level table ``Q[2^bits, 25]``, its updates, the rollout.

`top_key` hashes `skills.key_fields` plus the 4 low-stat flags to a row;
`select_skill` is masked ε-greedy over `skills.available`; `accumulate` keeps a
running skill's discounted return and ``gamma^tau``; `smdp_update` is the
semi-Markov Q-learning backup, one mean TD step per (key, skill) pair ended in the
same env step; `mc_init`, `mc_push`, `mc_update` are the Monte Carlo ablation;
`rollout_step` is one env step of every env (action, skill ends, top-level update,
re-decision); `SMDPConfig` holds the one granularity they all use. A trainer module
(Craftax via ``pqn_rnn``), imported after ``configure_runtime``. Design:
[[SMDP Q-Learning over Hand-Defined Skills design]].
"""

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import struct

from .hooks import Traj
from .pqn_rnn import NO_ACTION, Carry, RNNQNetwork, apply_network, eps_greedy
from .progress_key import Granularity, hash_fields
from .skills import (
    NUM_SKILLS,
    PRIMITIVE,
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
    """Low-stat flags in the key (off: the no-flags ablation)."""
    monte_carlo: bool = False
    """Monte Carlo top level (ablation) instead of the SMDP backup."""
    mc_length: int = 1024
    """Skill ends per env and episode the Monte Carlo records keep."""

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
    mc: MCRecords | None
    """Monte Carlo records, ``None`` unless ``monte_carlo``."""


class Shared(struct.PyTreeNode):  # type: ignore[no-untyped-call]  # flax API
    """Leaves shared by all envs (replicated), whatever their leading dim."""

    variables: Any
    """Skill-network params and BatchRenorm statistics."""
    q: jax.Array
    ends: jax.Array
    """Skill ends per skill and `skills.REASONS` index, int32 ``[25, 7]``."""
    agreement: jax.Array
    """Per skill: ends the events judged, and of those the ones that agreed, int32
    ``[25, 2]`` (`skills.test_agreement`)."""
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


def _keys(cfg: SMDPConfig, v: ObsView) -> tuple[jax.Array, jax.Array]:
    """Table rows and start-condition masks of a batch of views."""
    g = cfg.granularity
    k = jax.vmap(lambda x: top_key(x, g, cfg.log2_rows, low_flags=cfg.low_flags))(v)
    return k, jax.vmap(lambda x: available(x, g))(v)


def _pick(
    q: jax.Array, k: jax.Array, avail: jax.Array, eps: jax.Array, key: jax.Array
) -> jax.Array:
    keys = jax.random.split(key, k.shape[0])
    return jax.vmap(select_skill, in_axes=(None, 0, 0, None, 0))(q, k, avail, eps, keys)


def init_rollout(
    cfg: SMDPConfig,
    env: Any,
    network: RNNQNetwork,
    variables: Any,
    key: jax.Array,
    *,
    env_params: Any = None,
    eps: float = 1.0,
    eps_top: float = 1.0,
    method_state: Any = None,
) -> RolloutCarry:
    """Reset the batched ``env`` and choose every env's first skill.

    Raises:
        ValueError: If the table rows equal the env count (`init_table`).
    """
    params = env.default_params if env_params is None else env_params
    envs = env.num_envs
    q = init_table(cfg.log2_rows, NUM_SKILLS, cfg.q_init, envs=envs)
    key, k_reset, k_pick = jax.random.split(key, 3)
    obs, st = env.reset(k_reset, params)
    k, avail = _keys(cfg, jax.vmap(decode_obs)(obs))
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
        mc=mc_init(envs, cfg.mc_length) if cfg.monte_carlo else None,
    )
    reasons = 7  # len(skills.REASONS)
    shared = Shared(
        variables=variables,
        q=q,
        ends=jnp.zeros((NUM_SKILLS, reasons), jnp.int32),
        agreement=jnp.zeros((NUM_SKILLS, 2), jnp.int32),
        method_state=method_state,
        t=jnp.zeros((), jnp.int32),
        key=key,
        eps=jnp.float32(eps),
        eps_top=jnp.float32(eps_top),
    )
    return RolloutCarry(per_env, shared, cfg)


def _act(
    carry: RolloutCarry, network: RNNQNetwork, view: ObsView, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, Carry, jax.Array]:
    """Network input, action, last action, next LSTM carry and primitive flag."""
    pe, sh, cfg = carry.per_env, carry.shared, carry.cfg
    sk = pe.skill
    frac = sk.steps / step_limit(sk.skill, cfg.timeout_mult)
    extra = jnp.concatenate([jax.nn.one_hot(sk.skill, NUM_SKILLS), frac[:, None]], -1)
    x = jnp.concatenate([pe.obs, extra.astype(pe.obs.dtype)], -1)
    reset = pe.last_done[:, None]
    h = cast(Carry, tuple(jnp.where(reset, 0.0, part) for part in pe.h))
    la = jnp.where(pe.last_done, NO_ACTION, pe.last_act)
    h2, qv = apply_network(
        network, sh.variables, h, x[None], pe.last_done[None], la[None]
    )
    prim = jnp.asarray(PRIMITIVE)[sk.skill]
    fixed = jax.vmap(primitive_action)(sk.skill, view)
    act = jnp.where(prim, fixed, eps_greedy(key, qv[0], sh.eps)).astype(jnp.int32)
    return x, act, la, h2, prim


def _top_update(
    cfg: SMDPConfig,
    q: jax.Array,
    mc: MCRecords | None,
    sk: SkillState,
    ret_disc: tuple[jax.Array, jax.Array],
    nxt: tuple[jax.Array, jax.Array],
    done: jax.Array,
    ended: jax.Array,
) -> tuple[jax.Array, MCRecords | None]:
    """SMDP backup of the ended skills, or the Monte Carlo records and update."""
    ret, disc = ret_disc
    if mc is None:
        q = smdp_update(
            q, sk.k_start, sk.skill, ret, disc, *nxt, done, ended, cfg.alpha
        )
        return q, mc
    mc = mc_push(mc, ended, sk.k_start, sk.skill, ret, disc)
    return cast(
        tuple[jax.Array, MCRecords],
        jax.lax.cond(
            jnp.any(done),
            lambda: mc_update(q, mc, done, cfg.alpha),
            lambda: (q, mc),
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
    reward and ``done``; the step's ``StepEvents`` reach only `skills.test_agreement`.
    ``intrinsic_reward_fn`` has the ``hooks.HookFields`` signature; its ``r+`` joins
    the native reward in the top-level return, never the skill pseudo-reward, and
    ``None`` skips it at trace time. Outputs per env: the skill-network transition
    (``obs`` widened by `EXTRA_DIM`, ``action``, pseudo-reward ``reward``, skill-end
    ``done``, ``loss_mask``, ``last_action``), plus ``episode_done``,
    ``native_reward``, ``skill``, ``steps_in_skill``, ``success``, ``reason`` and the
    step's ``unlocked`` achievements.
    """
    cfg, pe, sh = carry.cfg, carry.per_env, carry.shared
    params = env.default_params if env_params is None else env_params
    key, k_act, k_env, k_top = jax.random.split(sh.key, 4)
    sk, view = pe.skill, jax.vmap(decode_obs)(pe.obs)
    x, act, la, h2, prim = _act(carry, network, view, k_act)
    obs2, st2, r, done, info = env.step(k_env, pe.env_state, act, params)
    view2 = jax.vmap(decode_obs)(obs2)
    r_top, ms = r, sh.method_state
    if intrinsic_reward_fn is not None:
        r_plus, ms = intrinsic_reward_fn(ms, pe.env_state, st2, obs2, done, sh.t)
        r_top = r + r_plus

    def end(k: Any, a: Any, act: Any, r: Any, b: Any, d: Any, n: Any) -> Any:
        return skill_step(
            k, a, act, r, b, d, n, cfg.timeout_mult, granularity=cfg.granularity
        )

    success, ended, reason = jax.vmap(end)(
        sk.skill, view, act, r, view2, done, sk.steps
    )
    ended = ended | done
    ret, disc = accumulate(sk.ret, sk.disc, r_top, cfg.gamma_top)
    nxt = _keys(cfg, view2)
    q, mc = _top_update(cfg, sh.q, pe.mc, sk, (ret, disc), nxt, done, ended)
    picks = _pick(q, *nxt, sh.eps_top, k_top)
    new_skill = SkillState(
        jnp.where(ended, picks, sk.skill),
        jnp.where(ended, nxt[0], sk.k_start),
        jnp.where(ended, 0.0, ret),
        jnp.where(ended, 1.0, disc),
        jnp.where(ended, 0, sk.steps + 1),
    )
    checked, agree = jax.vmap(test_agreement)(sk.skill, success, info["events"])
    judged = jnp.stack([checked & ended, checked & agree & ended], -1)
    shared = sh.replace(
        q=q,
        ends=sh.ends.at[sk.skill, reason].add(ended.astype(jnp.int32)),
        agreement=sh.agreement.at[sk.skill].add(judged.astype(jnp.int32)),
        method_state=ms,
        t=sh.t + obs2.shape[0],
        key=key,
    )
    per_env = PerEnv(obs2, st2, h2, done, act, new_skill, mc)
    out = {
        "obs": x, "action": act, "reward": success.astype(jnp.float32),
        "done": ended, "loss_mask": ~prim, "last_action": la, "episode_done": done,
        "native_reward": r, "skill": sk.skill, "steps_in_skill": sk.steps,
        "success": success, "reason": reason, "unlocked": info["events"].unlocked,
    }  # fmt: skip
    return carry.replace(per_env=per_env, shared=shared), out


def to_traj(out: dict[str, jax.Array]) -> tuple[Traj, jax.Array]:
    """`pqn_rnn.learn` inputs from stacked `rollout_step` outputs: Traj, loss mask.

    ``dones`` are skill ends (no Q(lambda) target crosses one);
    ``extra["episode_dones"]`` keeps the episode ends the acting LSTM reset on.
    ``hidden`` is empty: pass the rollout's start carry as ``h0``.
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
