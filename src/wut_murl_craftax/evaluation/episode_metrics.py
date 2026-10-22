"""Online metrics of training episodes, accumulated from step events inside the rollout.

``episode_step`` adds one transition per env: return, length, unlocks and the
per-episode sums of the online families (kills, damage, needs, actions, stalls, close
calls, loss window for the cause of death), all from ``StepEvents`` of the pre-reset
next state, so a done step is kept and no state is compared across it. A done step
adds its whole episode to the update's sums, which ``close_update`` hands over;
``update_metrics`` turns them into the ``train/...`` names of `metric_keys`. Episode
shares (funnel, end, death causes, ``adv``) are % of finished episodes, as
``train/ach/*``; step shares (survival, actions, wasted, forced) are fractions.
Definitions: "Scoreboard and basics", "In-game metric families" of the evaluation plan.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .events import DAMAGE_SOURCES, LAVA, UNKNOWN, StepEvents
from .milestones import (
    ADVANCED_ACHIEVEMENTS,
    CREATURES,
    FLOORS,
    Game,
    achievement_names,
    action_names,
    ladder,
    reward_weights,
)

N_ACTIONS = 43
"""Action histogram width: Craftax's actions; Classic uses the first 17."""
NOOP = 0
"""``Action.NOOP.value`` of both games."""
STATS = ("food", "drink", "energy")
"""Needs in ``StepEvents.stat_low`` order."""
NEEDS = ("starvation", "dehydration", "exhaustion")
"""Cause of a needs drain, by the first of `STATS` at 0."""
CAUSES: tuple[str, ...] = (*DAMAGE_SOURCES, *NEEDS)
"""Cause of death ids: ``damage_source`` ids, then `NEEDS`."""
START_HEALTH = 9.0
"""Health at the start of an episode in both games."""
STALL_STEPS, CLOSE_HEALTH, CLOSE_STEPS, DEATH_WINDOW = 300, 2.0, 50, 20
"""Stall gap; close-call health and the steps the episode must go on; cause window."""
_TOL = 1e-3
"""Slack on health summed from float deltas (Craftax armour scales damage)."""
_COUNTS = (
    ("episodes", 1), ("steps", 1), ("kills", len(CREATURES)), ("passive_kills", 1),
    ("damage_dealt", 1), ("damage_taken", 1), ("low", 3), ("zero", 3),
    ("actions", N_ACTIONS), ("wasted", 1), ("forced", 1), ("stalls", 1),
    ("close_calls", 1),
)  # fmt: skip
"""Per-step additive columns of an episode; ``episodes`` is 1 on its done step."""
_ENDS = (("floor", len(FLOORS)), ("death", 1), ("cause", len(CAUSES)))
"""Columns set once per finished episode: max floor >= f, death, cause one-hot."""


def _layout(blocks: Sequence[tuple[str, int]]) -> dict[str, slice]:
    edges = np.cumsum([0, *(n for _, n in blocks)]).tolist()
    return {
        name: slice(a, b)
        for (name, _), a, b in zip(blocks, edges[:-1], edges[1:], strict=True)
    }


ONLINE_LAYOUT = _layout((*_COUNTS, *_ENDS))
"""Column slices of the ``online`` sums; ``EpisodeAcc.counts`` holds the first ones."""
ONLINE_KEYS = ("online", "first_unlock")
"""`ChunkMetrics` keys of `close_update`, stacked per update by `stack_online`."""
_N_COUNTS = sum(n for _, n in _COUNTS)
_N_ONLINE = _N_COUNTS + sum(n for _, n in _ENDS)


@struct.dataclass
class EpisodeAcc:
    """Running episode of every env and the update's sums; leaves lead with envs."""

    ret: jax.Array
    """Return so far, float32."""
    length: jax.Array
    """Steps so far, int32."""
    ach: jax.Array
    """Achievements unlocked so far, bool (envs, achievements) by ``.value``."""
    counts: jax.Array
    """Episode sums of the ``_COUNTS`` columns, float32."""
    floor: jax.Array
    """Deepest floor so far, int8."""
    health: jax.Array
    """``START_HEALTH`` plus the health deltas so far, float32."""
    since: jax.Array
    """Steps since the last unlock (or the start), int32."""
    close_at: jax.Array
    """Step a close call opened at, not yet confirmed; -1 when none, int32."""
    win_cause: jax.Array
    win_loss: jax.Array
    """Last `DEATH_WINDOW` steps, oldest first: (needs, hit) cause ids and losses."""
    win_sum: jax.Array
    """Loss per `CAUSES` id over the window, float32."""
    t: jax.Array
    """Steps this env took since the run start (also across episodes), int32."""
    first: jax.Array
    """Env steps of the batch when this env first unlocked each achievement; -1."""
    sums: jax.Array
    """Episodes finished this update, summed over ``ONLINE_LAYOUT``, float32."""


class UpdateSummary(NamedTuple):
    """Means over the episodes that finished in one update; NaN when none did."""

    n_episodes: int
    ret: float
    length: float
    success: np.ndarray
    """Per achievement, % of the finished episodes that unlocked it."""
    ach: np.ndarray
    """Flags of the finished episodes, bool (n_episodes, achievements)."""


def _blank(envs: int, n_ach: int) -> dict[str, jax.Array]:
    """The per-episode fields of fresh episodes."""
    window = (envs, DEATH_WINDOW, 2)
    return {
        "ret": jnp.zeros(envs),
        "length": jnp.zeros(envs, jnp.int32),
        "ach": jnp.zeros((envs, n_ach), bool),
        "counts": jnp.zeros((envs, _N_COUNTS)),
        "floor": jnp.zeros(envs, jnp.int8),
        "health": jnp.full(envs, START_HEALTH),
        "since": jnp.zeros(envs, jnp.int32),
        "close_at": jnp.full(envs, -1, jnp.int32),
        "win_cause": jnp.zeros(window, jnp.int8),
        "win_loss": jnp.zeros(window),
        "win_sum": jnp.zeros((envs, len(CAUSES))),
    }


def init_acc(envs: int, n_ach: int) -> EpisodeAcc:
    """Empty episodes and sums for ``envs`` envs and ``n_ach`` achievements."""
    return EpisodeAcc(
        **_blank(envs, n_ach),
        t=jnp.zeros(envs, jnp.int32),
        first=jnp.full((envs, n_ach), -1, jnp.int32),
        sums=jnp.zeros((envs, _N_ONLINE)),
    )


def events_of(info: Mapping[str, Any]) -> StepEvents:
    """The ``StepEvents`` of a wrapper step's ``info``.

    Raises:
        ValueError: Without ``info["events"]`` (Craftax's own auto-reset), where the
            achievements of a done step are not available: better no metrics than
            silent zeros.
    """
    events = info.get("events")
    if events is None:
        msg = (
            'step info has no "events": episode metrics need the optimistic-reset '
            "wrapper (resets='optimistic'); auto-reset runs cannot log them"
        )
        raise ValueError(msg)
    return events  # type: ignore[no-any-return]  # info is untyped


def _close_calls(
    acc: EpisodeAcc, length: jax.Array, health: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Close calls confirmed at this step and the start of the open one.

    One opens when health falls to `CLOSE_HEALTH` and counts once the episode has gone
    on `CLOSE_STEPS` more steps; a new fall while one is open joins it.
    """
    low = CLOSE_HEALTH + _TOL
    confirmed = (acc.close_at >= 0) & (length - acc.close_at >= CLOSE_STEPS)
    entered = (health <= low) & (acc.health > low)
    open_at = jnp.where(confirmed, -1, acc.close_at)
    return confirmed, jnp.where(entered & (open_at < 0), length, open_at)


def _window(acc: EpisodeAcc, ev: StepEvents) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The loss window shifted by this step: causes, losses, loss per cause.

    A step adds its needs drain (cause: the first need at 0) and the rest of its
    health loss (cause: ``damage_source``; a harmful potion without a hit is unknown).
    """
    drain = ev.drain.astype(jnp.float32)
    needs = len(DAMAGE_SOURCES) + jnp.argmax(ev.stat_zero, -1)
    cause = jnp.stack(
        [
            jnp.where(ev.stat_zero.any(-1), needs, UNKNOWN),
            jnp.where(ev.damage_source >= 0, ev.damage_source, UNKNOWN),
        ],
        -1,
    ).astype(jnp.int8)
    rest = jnp.maximum(ev.health_loss.astype(jnp.float32) - drain, 0.0)
    loss = jnp.stack([drain, rest], -1)

    def per_cause(c: jax.Array, x: jax.Array) -> jax.Array:
        return (jax.nn.one_hot(c, len(CAUSES)) * x[..., None]).sum(-2)

    # running sums: O(1) per step instead of re-summing the whole window
    total = acc.win_sum + per_cause(cause, loss)
    total -= per_cause(acc.win_cause[:, 0], acc.win_loss[:, 0])
    causes = jnp.concatenate([acc.win_cause[:, 1:], cause[:, None]], 1)
    losses = jnp.concatenate([acc.win_loss[:, 1:], loss[:, None]], 1)
    return causes, losses, total


def _step_counts(
    ev: StepEvents, action: jax.Array, done: jax.Array, extra: dict[str, jax.Array]
) -> jax.Array:
    """This step's ``_COUNTS`` columns; ``extra`` holds stalls and close calls."""
    free = ~ev.forced
    parts = extra | {
        "episodes": done,
        "steps": jnp.ones_like(done),
        "kills": ev.kills,
        "passive_kills": ev.passive_kills,
        "damage_dealt": ev.damage_dealt,
        "damage_taken": ev.combat_damage,
        "low": ev.stat_low,
        "zero": ev.stat_zero,
        "actions": jax.nn.one_hot(action, N_ACTIONS) * free[:, None],
        "wasted": free & ~ev.effective & (action != NOOP),
        "forced": ev.forced,
    }
    envs = done.shape[0]
    cols = [parts[k].reshape(envs, -1).astype(jnp.float32) for k, _ in _COUNTS]
    return jnp.concatenate(cols, -1)


def _ending(total: EpisodeAcc, ev: StepEvents) -> jax.Array:
    """The ``ONLINE_LAYOUT`` row of the episode ``total`` if it ended at this step.

    Death: health at 0, or lava under the player (Classic ends the game there); its
    cause is the largest loss of the window, lava first, unknown without a loss.
    """
    lava = ev.damage_source == LAVA
    died = (total.health <= _TOL) | lava
    largest = jnp.argmax(total.win_sum, -1)
    cause = jnp.where(total.win_sum.max(-1) > _TOL, largest, UNKNOWN)
    cause = jnp.where(lava, LAVA, cause)
    floors = jnp.arange(len(FLOORS)) <= total.floor[:, None]
    died_col = died[:, None]
    end = [floors, died_col, jax.nn.one_hot(cause, len(CAUSES)) * died_col]
    return jnp.concatenate([total.counts, *(x.astype(jnp.float32) for x in end)], -1)


def _restart(total: EpisodeAcc, live: jax.Array) -> EpisodeAcc:
    """``total`` with the per-episode fields of ended envs blank."""
    envs, n_ach = total.ach.shape

    def keep(x: jax.Array, blank: jax.Array) -> jax.Array:
        return jnp.where(live.reshape(-1, *(1,) * (x.ndim - 1)), x, blank)

    blank = _blank(envs, n_ach)
    kept = {k: keep(getattr(total, k), v) for k, v in blank.items()}
    return dataclasses.replace(total, **kept)


def episode_step(
    acc: EpisodeAcc,
    reward: jax.Array,
    done: jax.Array,
    events: StepEvents,
    action: jax.Array,
) -> tuple[EpisodeAcc, EpisodeAcc]:
    """Add one transition per env; return ``(next accumulators, totals)``.

    ``totals`` include this step and are the finished episode where ``done``; there
    the episode is added to ``sums`` and the next accumulators restart empty.
    ``events`` are the step's (batched) `StepEvents`, ``action`` the chosen action.
    """
    ev = events
    unlock = ev.unlocked.any(-1)
    length, health, gap = acc.length + 1, acc.health + ev.health_delta, acc.since + 1
    stall = (unlock | done) & (gap >= STALL_STEPS)
    confirmed, close_at = _close_calls(acc, length, health)
    win_cause, win_loss, win_sum = _window(acc, ev)
    extra = {"stalls": stall, "close_calls": confirmed}
    t = acc.t + 1
    stamp = t * t.shape[0]  # env steps of the whole batch: every env steps at once
    total = dataclasses.replace(
        acc,
        ret=acc.ret + reward,
        length=length,
        ach=acc.ach | ev.unlocked,
        counts=acc.counts + _step_counts(ev, action, done, extra),
        floor=jnp.maximum(acc.floor, ev.floor),
        health=health,
        since=jnp.where(unlock, 0, gap),
        close_at=close_at,
        win_cause=win_cause,
        win_loss=win_loss,
        win_sum=win_sum,
        t=t,
        first=jnp.where((acc.first < 0) & ev.unlocked, stamp[:, None], acc.first),
    )
    ended = jnp.where(done[:, None], _ending(total, ev), 0.0)
    total = dataclasses.replace(total, sums=acc.sums + ended)
    return _restart(total, ~done), total


def close_update(acc: EpisodeAcc) -> tuple[EpisodeAcc, dict[str, jax.Array]]:
    """End an update: its ``online`` sums and the run's ``first_unlock`` env steps.

    ``first_unlock`` is per achievement the earliest env step any env unlocked it at
    (-1: never); the sums restart at 0, the episodes and first unlocks go on.
    """
    big = 2**31 - 1  # int32 max: above every env step
    first = jnp.where(acc.first >= 0, acc.first, big).min(0)
    out = {
        "online": acc.sums.sum(0),
        "first_unlock": jnp.where(first == big, -1, first),
    }
    # derived from the sums, not zeros_like: keeps their env-mesh split (a constant
    # comes back replicated on every device)
    return dataclasses.replace(acc, sums=acc.sums * 0.0), out


def stack_online(updates: Sequence[Mapping[str, jax.Array]]) -> dict[str, jax.Array]:
    """The `ONLINE_KEYS` of each update's `close_update` output, stacked."""
    return {k: jnp.stack([u[k] for u in updates]) for k in ONLINE_KEYS}


ADDED_FIELDS: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(EpisodeAcc)
    if f.name not in {"ret", "length", "ach"}
)
"""`EpisodeAcc` fields the online families added; states saved before lack them."""


def _accs(tree: Any) -> list[tuple[str, EpisodeAcc]]:
    """(tree path, node) of every `EpisodeAcc` in ``tree``."""
    found = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: isinstance(x, EpisodeAcc)
    )[0]
    return [(jax.tree_util.keystr(p), x) for p, x in found if isinstance(x, EpisodeAcc)]


def added_paths(tree: Any) -> set[str]:
    """Tree paths of the `ADDED_FIELDS` leaves of every `EpisodeAcc` in ``tree``."""
    return {f"{path}.{f}" for path, _ in _accs(tree) for f in ADDED_FIELDS}


def resume_from(tree: Any, env_steps: int) -> Any:
    """``tree`` with every `EpisodeAcc` counting env steps from ``env_steps`` on.

    For accumulators filled in fresh into a run that already took ``env_steps``
    steps: first unlocks then carry the run's env steps, not the resume's.
    """

    def move(x: Any) -> Any:
        if not isinstance(x, EpisodeAcc):
            return x
        # from x.t, not a new array: keeps its env-mesh placement
        return dataclasses.replace(x, t=x.t * 0 + env_steps // x.t.shape[0])

    return jax.tree.map(move, tree, is_leaf=lambda x: isinstance(x, EpisodeAcc))


def first_unlock_offsets(unlocked: jax.Array) -> jax.Array:
    """Env steps into an update at which any env first unlocked each achievement.

    ``unlocked`` is ``(steps, envs, achievements)``; env steps count the whole batch
    (step ``t`` of ``envs`` envs ends at ``(t + 1) * envs``); -1 where none unlocked.
    """
    envs = unlocked.shape[1]
    any_env = unlocked.any(1)
    first = jnp.argmax(any_env, 0)
    return jnp.where(any_env.any(0), (first + 1) * envs, -1).astype(jnp.int32)


def update_summary(
    done: np.ndarray, ret: np.ndarray, length: np.ndarray, ach: np.ndarray
) -> UpdateSummary:
    """Finished-episode statistics of one update from its ``(steps, envs)`` arrays.

    ``ret``, ``length`` and ``ach`` (steps, envs, achievements) are read on done
    steps only, where `episode_step` holds the whole episode.
    """
    d = np.asarray(done, bool)
    n = int(d.sum())
    got = np.asarray(ach)[d]
    nan = float("nan")
    if not n:
        return UpdateSummary(0, nan, nan, np.full(got.shape[-1], nan), got)
    ret_mean = float(np.asarray(ret)[d].mean())
    mean_len = float(np.asarray(length)[d].mean())
    return UpdateSummary(n, ret_mean, mean_len, got.mean(0) * 100, got)


@overload
def score_pct(ret: float, max_reward: float) -> float: ...
@overload
def score_pct(ret: np.ndarray, max_reward: float) -> np.ndarray: ...
def score_pct(ret: Any, max_reward: float) -> Any:
    """Return as % of the maximum reward (226 on Craftax, 22 on Classic)."""
    return ret / max_reward * 100


def _creatures(game: Game, classes: Sequence[str]) -> list[tuple[int, str]]:
    """(``CREATURES`` index, name) of ``classes`` in ``game`` (Classic: type 0)."""
    return [
        (i, name)
        for i, (cls, type_id, name) in enumerate(CREATURES)
        if cls in classes and (game == "craftax" or type_id == 0)
    ]


def _causes(game: Game) -> list[tuple[int, str]]:
    """(`CAUSES` id, name) of the causes of death ``game`` logs.

    Hits come from melee mobs and projectiles (a ranged mob's shot is its projectile).
    """
    hits = {name for _, name in _creatures(game, ("melee", "projectile"))}
    keep = hits | {"lava", "unknown", *NEEDS}
    return [(i, c) for i, c in enumerate(CAUSES) if c in keep]


@functools.cache
def metric_keys(game: Game) -> tuple[str, ...]:
    """Names of `update_metrics` of ``game``, in logging order."""
    floors = len(FLOORS) if game == "craftax" else 1
    adv = ADVANCED_ACHIEVEMENTS if game == "craftax" else ()
    return (
        "train/ach_per_episode",
        "train/ach_reward_share",
        *(f"train/funnel/depth_{k}" for k in range(1, len(ladder(game)) + 1)),
        *(f"train/funnel/floor_{f}" for f in range(floors)),
        "train/end/death",
        "train/end/timeout",
        *(f"train/death/{c}" for _, c in _causes(game)),
        "train/stalls_per_episode",
        *(f"train/survival/{s}_{k}" for s in STATS for k in ("low", "zero")),
        "train/close_calls",
        *(f"train/kills/{n}" for _, n in _creatures(game, ("melee", "ranged"))),
        "train/kills/passive",
        "train/kills_per_1k",
        "train/damage_dealt",
        "train/damage_taken",
        *(f"train/action/{a}" for a in action_names(game)),
        "train/action_entropy",
        "train/wasted_share",
        "train/forced_share",
        *(f"train/adv/{a.lower()}" for a in adv),
    )


ONLINE_PREFIX = "online/"
"""Prefix of the `sum_columns` series in the runner's per-update extras."""


@functools.cache
def sum_columns(game: Game) -> tuple[str, ...]:
    """Names of `update_sums`, in order.

    The ``online`` columns, then the finished episodes' achievement count, tier
    reward, return, depth >= k counts and unlocks per achievement.
    """
    online = [
        name if b.stop - b.start == 1 else f"{name}/{i}"
        for name, b in ONLINE_LAYOUT.items()
        for i in range(b.stop - b.start)
    ]
    depth = (f"depth/{k}" for k in range(1, len(ladder(game)) + 1))
    names = (f"ach/{a}" for a in achievement_names(game))
    return (*online, "ach_count", "ach_reward", "return", *depth, *names)


def update_sums(online: np.ndarray, s: UpdateSummary, game: Game) -> np.ndarray:
    """One update's `sum_columns`; additive over updates, so a window adds them up.

    Tech depth is the highest ladder rung an episode reached.
    """
    rungs = s.ach[:, list(ladder(game))]
    depth = np.where(rungs.any(1), rungs.shape[1] - rungs[:, ::-1].argmax(1), 0)
    ret = s.ret * s.n_episodes if s.n_episodes else 0.0
    host = [s.ach.sum(), (s.ach @ reward_weights(game)).sum(), ret]
    host += [(depth >= k).sum() for k in range(1, rungs.shape[1] + 1)]
    return np.concatenate([np.asarray(online, np.float64), host, s.ach.sum(0)])


def _at(game: Game) -> dict[str, int]:
    return {c: i for i, c in enumerate(sum_columns(game))}


def _per_episode(v: Any, game: Game) -> dict[str, Any]:
    """Episode means and % of episodes (death causes: % of deaths)."""
    n, deaths, ret = v("episodes"), v("death"), v("return")
    hostile = _creatures(game, ("melee", "ranged"))
    floors = len(FLOORS) if game == "craftax" else 1
    return {
        "train/ach_per_episode": v("ach_count") / n,
        "train/ach_reward_share": v("ach_reward") / ret if ret > 0 else math.nan,
        **{
            f"train/funnel/depth_{k}": v(f"depth/{k}") / n * 100
            for k in range(1, len(ladder(game)) + 1)
        },
        **{f"train/funnel/floor_{f}": v(f"floor/{f}") / n * 100 for f in range(floors)},
        "train/end/death": deaths / n * 100,
        "train/end/timeout": (n - deaths) / n * 100,
        **{
            f"train/death/{c}": v(f"cause/{i}") / deaths * 100 for i, c in _causes(game)
        },
        "train/stalls_per_episode": v("stalls") / n,
        "train/close_calls": v("close_calls") / n,
        **{f"train/kills/{name}": v(f"kills/{i}") / n for i, name in hostile},
        "train/kills/passive": v("passive_kills") / n,
        "train/damage_dealt": v("damage_dealt") / n,
        "train/damage_taken": v("damage_taken") / n,
        **{
            f"train/adv/{a.lower()}": v(f"ach/{a.lower()}") / n * 100
            for a in (ADVANCED_ACHIEVEMENTS if game == "craftax" else ())
        },
    }


def _per_step(v: Any, game: Game) -> dict[str, Any]:
    """Shares of the finished episodes' steps (actions: of the steps not forced)."""
    steps, actions = v("steps"), action_names(game)
    hist = np.array([v(f"actions/{i}") for i in range(len(actions))])
    p = hist / hist.sum()
    hostile = sum(v(f"kills/{i}") for i, _ in _creatures(game, ("melee", "ranged")))
    return {
        **{
            f"train/survival/{s}_low": v(f"low/{i}") / steps
            for i, s in enumerate(STATS)
        },
        **{
            f"train/survival/{s}_zero": v(f"zero/{i}") / steps
            for i, s in enumerate(STATS)
        },
        "train/kills_per_1k": hostile * 1000 / steps,
        **{f"train/action/{a}": p[i] for i, a in enumerate(actions)},
        "train/action_entropy": -(p * np.log(np.where(p > 0, p, 1.0))).sum(),
        "train/wasted_share": v("wasted") / hist.sum(),
        "train/forced_share": v("forced") / steps,
    }


def window_metrics(sums: np.ndarray, game: Game) -> dict[str, float]:
    """The `metric_keys` of the episodes summed in ``sums`` (`sum_columns`).

    One update's sums give its own values; a window's sums give the values of all
    its episodes pooled. All NaN without episodes (death causes: without deaths),
    so MLflow windows skip them as they skip the score.
    """
    keys, at = metric_keys(game), _at(game)
    sums = np.asarray(sums, np.float64)

    def v(name: str) -> Any:
        return sums[at[name]]

    if not v("episodes"):
        return dict.fromkeys(keys, math.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        got = _per_episode(v, game) | _per_step(v, game)
    return {k: float(got[k]) for k in keys}


def update_metrics(
    online: np.ndarray, summary: UpdateSummary, game: Game
) -> dict[str, float]:
    """The `metric_keys` of one update from its ``online`` sums and `update_summary`."""
    return window_metrics(update_sums(online, summary, game), game)


def chunk_sums(metrics: Mapping[str, Any], game: Game) -> dict[str, np.ndarray]:
    """`update_sums` of a chunk's updates, column -> values, `ONLINE_PREFIX` names."""
    m = {k: np.asarray(metrics[k]) for k in ("online", "done", "ret", "length", "ach")}
    rows = np.stack([
        update_sums(
            m["online"][i],
            update_summary(m["done"][i], m["ret"][i], m["length"][i], m["ach"][i]),
            game,
        )
        for i in range(len(m["online"]))
    ])
    return {f"{ONLINE_PREFIX}{c}": rows[:, j] for j, c in enumerate(sum_columns(game))}


def new_first_unlocks(
    stamps: np.ndarray, first: int, spu: int
) -> list[tuple[int, int, int]]:
    """First unlocks made in each update of a chunk after update ``first``.

    ``stamps`` is the chunk's ``first_unlock`` (updates, achievements); an env step in
    ``((first + i) * spu, (first + i + 1) * spu]`` was first reached in update ``i``.
    Returns ``(achievement, env step, i)``; stateless, so a resumed run agrees.
    """
    found = []
    for i, row in enumerate(np.asarray(stamps)):
        lo, hi = (first + i) * spu, (first + i + 1) * spu
        hits = np.flatnonzero((row > lo) & (row <= hi))
        found += [(int(a), int(row[a]), i) for a in hits]
    return found
