"""Records of eval episodes: one row of the eval-episode table per episode.

`episode_record` reads an episode's stored ``StepEvents`` and actions
(``eval/<mode>/<update>.npz``, `stored_records`) and, when a replay rebuilt it, its
`Trace` (`trace_of`). Steps count actions: an unlock in action ``t`` (0-based) has
unlock step ``t + 1``, the frame that first shows it; stall and fight bounds are frames
too. Resources, wasted crafts and stats at stalls and refills need the trace and are
None (NaN in `EpisodeRecord.row`) without it. Rules shared with the online metrics
come from `episode_metrics`. Definitions: "In-game metric families" of the evaluation
plan.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np

from .episode_metrics import (
    CAUSES,
    CLOSE_HEALTH,
    CLOSE_STEPS,
    DEATH_WINDOW,
    NOOP,
    STALL_STEPS,
    START_HEALTH,
    STATS,
    TOL,
    creatures,
    score_pct,
)
from .events import DAMAGE_SOURCES, LAVA, UNKNOWN, StepEvents
from .milestones import (
    ADVANCED_ACTIONS,
    FLOORS,
    Game,
    achievement_names,
    action_names,
    constants,
    ladder,
    reward_weights,
)

RESOURCES: tuple[str, ...] = (
    "wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling",
)  # fmt: skip
"""Inventory items of the gathered and used totals; Classic has no sapphire or ruby."""
FIGHT_GAP = 5
"""Fight steps at most this far apart join one fight (gaps under 5 steps bridged)."""
RECOVER_HEALTH = 5.0
"""A close call is recovered once health later reaches this."""


def resources(game: Game) -> tuple[str, ...]:
    """`RESOURCES` of ``game``, the column order of `Trace.inventory`."""
    return (
        RESOURCES
        if game == "craftax"
        else tuple(r for r in RESOURCES if r not in {"sapphire", "ruby"})
    )


class Trace(NamedTuple):
    """Replayed state fields, ``n + 1`` rows: before each action, then the last."""

    position: np.ndarray
    """``player_position``, (n + 1, 2)."""
    floor: np.ndarray
    """``player_level`` (Classic 0)."""
    intrinsics: np.ndarray
    """Health, food, drink, energy, (n + 1, 4)."""
    inventory: np.ndarray
    """`resources` counts, (n + 1, len(resources(game)))."""
    levels: np.ndarray
    """Owned levels a craft can raise: Craftax pickaxe, sword, bow, arrows, torches and
    the 4 armour slots; Classic pickaxe and sword tier (0 none, 1 wood to 3 iron)."""
    table_near: np.ndarray
    """A crafting table on one of the 8 tiles around the player (crafting distance)."""


class Stall(NamedTuple):
    """At least `STALL_STEPS` steps without an unlock, frames ``start`` to ``end``."""

    start: int
    """The frame of the previous unlock, 0 at the episode start."""
    end: int
    length: int
    floor: int
    """Floor at ``start``."""
    ending: str
    """``unlock``, ``death`` or ``timeout``."""
    ended_by: str
    """The achievement unlocked at ``end`` (the lowest ``.value`` of several); ""."""
    actions: np.ndarray
    """Histogram of the actions inside the stall, forced steps excepted
    (`action_names` order)."""
    intrinsics: np.ndarray | None
    """Health, food, drink, energy at ``start``; None without a trace."""
    inventory: np.ndarray | None
    """`resources` at ``start``; None without a trace."""


class Fight(NamedTuple):
    """Steps with >= 2 hostiles on screen and >= 1 within 2 tiles, gaps bridged."""

    start: int
    end: int
    """First and last frame; a death within `FIGHT_GAP` steps extends it to the end."""
    max_enemies: int
    damage_taken: float
    """Combat damage over the fight, bridged steps included."""
    outcome: str
    """``survived`` or ``died``."""


class _Ids(NamedTuple):
    crafts: np.ndarray
    place_table: int
    wood: int
    diamond: int
    advanced: np.ndarray
    milestones: tuple[int, ...]
    """Tech ladder, then the ``ENTER_*`` achievements (Craftax)."""


@functools.cache
def _ids(game: Game) -> _Ids:
    c = constants(game)
    act, ach = c.Action, c.Achievement
    advanced = ADVANCED_ACTIONS if game == "craftax" else ()
    enter = tuple(ach[e].value for _, e in FLOORS if e) if game == "craftax" else ()
    return _Ids(
        np.array([a.value for a in act if a.name.startswith("MAKE_")]),
        act.PLACE_TABLE.value,
        ach.COLLECT_WOOD.value,
        ach.COLLECT_DIAMOND.value,
        np.array([act[a].value for a in advanced], int),
        (*ladder(game), *enter),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class EpisodeRecord:
    """One eval episode; `row` gives its eval-episode table columns."""

    game: Game
    length: int
    ret: float
    score_pct: float
    end_type: str
    """``death`` or ``timeout``."""
    cause_of_death: str
    """A `CAUSES` id; "" for a timeout."""
    max_floor: int
    tech_depth: int
    unlock_steps: np.ndarray
    """First unlock step per achievement by ``.value``, -1 if never."""
    stalls: tuple[Stall, ...]
    fights: tuple[Fight, ...]
    kills: np.ndarray
    """Hostile kills per ``CREATURES`` index."""
    passive_kills: int
    damage_dealt: float
    damage_taken: float
    projectile_damage: float
    close_calls: int
    close_calls_recovered: int
    low_share: np.ndarray
    zero_share: np.ndarray
    """Share of steps with food, drink, energy ``<= 2`` and ``== 0``."""
    action_counts: np.ndarray
    """Actions of the steps not forced."""
    action_entropy: float
    wasted_share: float
    noop_share: float
    """Of the steps not forced."""
    forced_share: float
    adv: np.ndarray
    """Effective uses per ``ADVANCED_ACTIONS`` (Craftax; empty on Classic)."""
    steps_per_achievement: float
    """NaN without an unlock."""
    wood_to_diamond: int
    """Steps from the first wood to the first diamond, -1 without a diamond."""
    gathered: np.ndarray | None
    used: np.ndarray | None
    """Positive and negative inventory changes per `resources`; None without a trace."""
    use_ratio: float
    wasted_crafts: int | None
    """Crafts that raised no owned level, plus tables placed beside a table."""
    need_counts: np.ndarray
    need_gaps: np.ndarray
    """Per `STATS`: eats, drinks, sleep starts and the mean steps between consecutive
    ones (NaN with fewer than 2)."""
    need_levels: np.ndarray | None
    """Mean food before an eat, drink before a drink, energy before a sleep start;
    None without a trace."""

    @property
    def n_achievements(self) -> int:
        """Achievements unlocked in the episode."""
        return int((self.unlock_steps >= 0).sum())

    @property
    def unlocks(self) -> list[tuple[int, str]]:
        """(unlock step, achievement) in unlock order."""
        names = achievement_names(self.game)
        got = np.flatnonzero(self.unlock_steps >= 0)
        return sorted((int(self.unlock_steps[a]), names[a]) for a in got)

    @property
    def splits(self) -> dict[str, int]:
        """Unlock step of each funnel milestone and ``ENTER_*`` achievement; -1."""
        names = achievement_names(self.game)
        return {names[a]: int(self.unlock_steps[a]) for a in _ids(self.game).milestones}

    def row(self) -> dict[str, Any]:
        """The eval-episode table columns (identity columns excepted).

        Plain Python values, except ``unlock_step/*`` (``np.int32``, the contract
        dtype); replay-only columns are floats, NaN without a trace.
        """
        g, res = self.game, resources(self.game)
        levels = np.full(3, math.nan) if self.need_levels is None else self.need_levels
        names, missing = achievement_names(g), np.full(len(res), math.nan)
        hostile = creatures(g, _HOSTILE)
        adv = [a.lower() for a in ADVANCED_ACTIONS] if g == "craftax" else []
        return {
            "length": self.length,
            "return": self.ret,
            "score_pct": self.score_pct,
            "end_type": self.end_type,
            "cause_of_death": self.cause_of_death,
            "max_floor": self.max_floor,
            "tech_depth": self.tech_depth,
            "n_achievements": self.n_achievements,
            **_cols("ach", names, self.unlock_steps >= 0),
            **{
                f"unlock_step/{a}": np.int32(u)
                for a, u in zip(names, self.unlock_steps, strict=True)
            },
            "stalls": [_plain(s) for s in self.stalls],
            "n_stalls": len(self.stalls),
            "max_stall": max((s.length for s in self.stalls), default=0),
            "fights": [f._asdict() for f in self.fights],
            **_cols(
                "kills", [c for _, c in hostile], self.kills[[i for i, _ in hostile]]
            ),
            "kills/passive": self.passive_kills,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "projectile_damage": self.projectile_damage,
            "close_calls": self.close_calls,
            "close_calls_recovered": self.close_calls_recovered,
            **_cols("low_share", STATS, self.low_share),
            **_cols("zero_share", STATS, self.zero_share),
            **_cols("action_count", action_names(g), self.action_counts),
            "action_entropy": self.action_entropy,
            "wasted_share": self.wasted_share,
            "noop_share": self.noop_share,
            "forced_share": self.forced_share,
            **_cols(
                "gathered",
                res,
                missing if self.gathered is None else 1.0 * self.gathered,
            ),
            **_cols("used", res, missing if self.used is None else 1.0 * self.used),
            "use_ratio": self.use_ratio,
            "steps_per_achievement": self.steps_per_achievement,
            "wasted_crafts": math.nan
            if self.wasted_crafts is None
            else float(self.wasted_crafts),
            "wood_to_diamond": self.wood_to_diamond,
            **_cols("adv", adv, self.adv),
            **_cols("survival/count", STATS, self.need_counts),
            **_cols("survival/gap", STATS, self.need_gaps),
            **_cols("survival/level", STATS, levels),
            **{f"split/{m}": u for m, u in self.splits.items()},
        }


_HOSTILE = ("melee", "ranged")


def _cols(prefix: str, names: Sequence[str], values: Any) -> dict[str, Any]:
    """``prefix/name`` -> plain value, one per name."""
    plain = np.asarray(values).tolist()
    return {f"{prefix}/{k}": v for k, v in zip(names, plain, strict=True)}


def _plain(stall: Stall) -> dict[str, Any]:
    """``stall`` with lists for arrays, for parquet and JSON."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in stall._asdict().items()
    }


def _death(ev: StepEvents, health: np.ndarray) -> tuple[str, str]:
    """End type and cause: the largest loss of the last `DEATH_WINDOW` steps.

    As the online `episode_metrics` rule: a needs drain is caused by the first stat at
    0 (unknown if none), the rest of a step's loss by its ``damage_source``; lava wins.
    """
    lava = ev.damage_source[-1] == LAVA
    if health[-1] > TOL and not lava:
        return "timeout", ""
    if lava:
        return "death", "lava"
    w = slice(max(len(health) - DEATH_WINDOW, 0), None)
    drain, zero, src = (
        ev.drain[w].astype(np.float64),
        ev.stat_zero[w],
        ev.damage_source[w],
    )
    needs = np.where(zero.any(-1), len(DAMAGE_SOURCES) + zero.argmax(-1), UNKNOWN)
    hits = np.where(src >= 0, src, UNKNOWN)
    rest = np.maximum(ev.health_loss[w].astype(np.float64) - drain, 0.0)
    loss = np.bincount(
        np.concatenate([needs, hits]), np.concatenate([drain, rest]), len(CAUSES)
    )
    return "death", CAUSES[int(loss.argmax()) if loss.max() > TOL else UNKNOWN]


def _stalls(
    ev: StepEvents, actions: np.ndarray, trace: Trace | None, end_type: str, game: Game
) -> tuple[Stall, ...]:
    """Gaps of at least `STALL_STEPS` between unlocks, the start and the end."""
    n, names = len(actions), achievement_names(game)
    at = np.flatnonzero(ev.unlocked.any(-1)) + 1
    out = []
    for a, b in zip(np.r_[0, at], np.r_[at, n], strict=True):
        if b - a < STALL_STEPS:
            continue
        got = np.flatnonzero(ev.unlocked[b - 1])
        out.append(
            Stall(
                int(a),
                int(b),
                int(b - a),
                int(ev.floor[a - 1]) if a else 0,
                "unlock" if len(got) else end_type,
                names[got[0]] if len(got) else "",
                np.bincount(
                    actions[a:b][~ev.forced[a:b]], minlength=len(action_names(game))
                ),
                None if trace is None else trace.intrinsics[a],
                None if trace is None else trace.inventory[a],
            )
        )
    return tuple(out)


def _fights(ev: StepEvents, died: bool) -> tuple[Fight, ...]:
    """Fight steps split where more than `FIGHT_GAP` steps apart."""
    on = np.flatnonzero((ev.hostiles_on_screen >= 2) & (ev.hostiles_close >= 1))
    last, out = len(ev.floor) - 1, []
    for run in (
        np.split(on, np.flatnonzero(np.diff(on) > FIGHT_GAP) + 1) if len(on) else ()
    ):
        a, b = int(run[0]), int(run[-1])
        dies = died and last - b <= FIGHT_GAP
        b = last if dies else b
        damage = float(ev.combat_damage[a : b + 1].astype(np.float64).sum())
        enemies = int(ev.hostiles_on_screen[a : b + 1].max())
        out.append(Fight(a + 1, b + 1, enemies, damage, "died" if dies else "survived"))
    return tuple(out)


def _close_calls(health: np.ndarray) -> tuple[int, int]:
    """Close calls and how many recovered, as the online rule.

    One opens when health falls to `CLOSE_HEALTH` and counts once the episode goes on
    `CLOSE_STEPS` more steps; a fall while one is open joins it.
    """
    low = CLOSE_HEALTH + TOL
    prev = np.r_[START_HEALTH, health[:-1]]
    opened, at = [], -1
    for length in np.flatnonzero((health <= low) & (prev > low)) + 1:
        if at >= 0 and length - at >= CLOSE_STEPS:
            opened.append(at)
            at = -1
        at = int(length) if at < 0 else at
    if at >= 0 and len(health) - at >= CLOSE_STEPS:
        opened.append(at)
    back = sum(bool((health[s:] >= RECOVER_HEALTH - TOL).any()) for s in opened)
    return len(opened), back


def _action_usage(ev: StepEvents, actions: np.ndarray, game: Game) -> dict[str, Any]:
    """Histogram, entropy and shares of the steps not forced; advanced uses."""
    free = ~ev.forced
    counts = np.bincount(actions[free], minlength=len(action_names(game)))
    total, nan = counts.sum(), math.nan
    p = counts / total if total else np.full(len(counts), nan)
    wasted = (free & ~ev.effective & (actions != NOOP)).sum()
    used = free & ev.effective
    return {
        "action_counts": counts,
        "action_entropy": float(-(p * np.log(np.where(p > 0, p, 1.0))).sum()),
        "wasted_share": float(wasted / total) if total else nan,
        "noop_share": float(p[NOOP]),
        "forced_share": float(ev.forced.mean()),
        "adv": np.array(
            [(used & (actions == i)).sum() for i in _ids(game).advanced], int
        ),
    }


def _crafting(trace: Trace | None, actions: np.ndarray, game: Game) -> dict[str, Any]:
    """Gathered and used resources, use ratio and wasted crafts from the trace."""
    if trace is None:
        return {
            "gathered": None,
            "used": None,
            "use_ratio": math.nan,
            "wasted_crafts": None,
        }
    ids = _ids(game)
    d = np.diff(np.asarray(trace.inventory, np.int64), axis=0)
    gathered, used = np.clip(d, 0, None).sum(0), np.clip(-d, 0, None).sum(0)
    consumed = (d < 0).any(-1)
    raised = (np.diff(np.asarray(trace.levels, np.int64), axis=0) > 0).any(-1)
    crafts = np.isin(actions, ids.crafts) & ~raised
    tables = (actions == ids.place_table) & trace.table_near[:-1]
    return {
        "gathered": gathered,
        "used": used,
        "use_ratio": float(used.sum() / gathered.sum()) if gathered.sum() else math.nan,
        "wasted_crafts": int((consumed & (crafts | tables)).sum()),
    }


def _needs(ev: StepEvents, trace: Trace | None) -> dict[str, Any]:
    """Survival timing per `STATS` from the eat, drink and sleep-start events."""
    marks = np.stack([ev.ate, ev.drank, ev.slept])
    counts = marks.sum(1)
    gaps = [
        np.diff(np.flatnonzero(m)).mean() if c > 1 else math.nan
        for m, c in zip(marks, counts, strict=True)
    ]
    levels = None
    if trace is not None:  # the stat in the state before the step that refilled it
        before = np.asarray(trace.intrinsics, np.float64)[:-1, 1:]
        levels = np.array([
            before[m, k].mean() if m.any() else math.nan for k, m in enumerate(marks)
        ])
    return {
        "need_counts": counts,
        "need_gaps": np.array(gaps, float),
        "need_levels": levels,
    }


def episode_record(
    events: StepEvents, actions: Any, trace: Trace | None = None, *, game: Game
) -> EpisodeRecord:
    """The record of one episode from its ``n`` steps of events and actions.

    ``trace`` (`trace_of` of the replayed states) adds resources, wasted crafts, the
    stats at a stall's start and the stat levels at eats, drinks and sleeps.
    """
    ev = jax.tree.map(np.asarray, events)
    actions = np.asarray(actions, np.int64)
    n = len(actions)
    health = START_HEALTH + np.cumsum(ev.health_delta, dtype=np.float64)
    end_type, cause = _death(ev, health)
    unlocked = ev.unlocked.any(0)
    unlock_steps = np.where(unlocked, ev.unlocked.argmax(0) + 1, -1).astype(np.int32)
    rungs = unlocked[list(ladder(game))]
    ids = _ids(game)
    wood, diamond = unlock_steps[ids.wood], unlock_steps[ids.diamond]
    ret = ev.unlocked.sum(0) @ reward_weights(game) + 0.1 * ev.health_delta.sum(
        dtype=np.float64
    )
    close, recovered = _close_calls(health)
    return EpisodeRecord(
        game=game,
        length=n,
        ret=float(ret),
        score_pct=float(score_pct(ret, float(reward_weights(game).sum()))),
        end_type=end_type,
        cause_of_death=cause,
        max_floor=int(ev.floor.max(initial=0)),
        tech_depth=int(len(rungs) - rungs[::-1].argmax()) if rungs.any() else 0,
        unlock_steps=unlock_steps,
        stalls=_stalls(ev, actions, trace, end_type, game),
        fights=_fights(ev, end_type == "death"),
        kills=ev.kills.sum(0, dtype=np.int64),
        passive_kills=int(ev.passive_kills.sum()),
        damage_dealt=float(ev.damage_dealt.sum(dtype=np.float64)),
        damage_taken=float(ev.combat_damage.sum(dtype=np.float64)),
        projectile_damage=float(ev.projectile_damage.sum(dtype=np.float64)),
        close_calls=close,
        close_calls_recovered=recovered,
        low_share=ev.stat_low.mean(0),
        zero_share=ev.stat_zero.mean(0),
        steps_per_achievement=n / unlocked.sum() if unlocked.any() else math.nan,
        wood_to_diamond=int(diamond - wood) if diamond >= 0 and wood >= 0 else -1,
        **_action_usage(ev, actions, game),
        **_crafting(trace, actions, game),
        **_needs(ev, trace),
    )


def stored_records(arrays: Mapping[str, np.ndarray], game: Game) -> list[EpisodeRecord]:
    """Records without a trace of every seed of an eval file's arrays (an npz)."""
    names = [f.name for f in dataclasses.fields(StepEvents)]
    events = {k: np.asarray(arrays[f"events/{k}"]) for k in names}
    actions = np.asarray(arrays["actions"])
    return [
        episode_record(
            # numpy rows, not jax arrays: the record reads them with numpy
            StepEvents(**{k: v[i, :n] for k, v in events.items()}),  # type: ignore[arg-type]
            actions[i, :n],
            game=game,
        )
        for i, n in enumerate(np.asarray(arrays["length"]))
    ]


def _tier(inventory: Any, tool: str) -> np.ndarray:
    """Classic tool tier: 1 wood, 2 stone, 3 iron by the best one owned, else 0."""
    owned = [
        np.asarray(getattr(inventory, f"{m}_{tool}")) > 0
        for m in ("wood", "stone", "iron")
    ]
    return np.max([(k + 1) * o for k, o in enumerate(owned)], axis=0)


def trace_of(states: Any, game: Game) -> Trace:
    """The `Trace` of replayed raw ``EnvState`` rows (``Replay.states``, numpy)."""
    c = constants(game)
    inv, pos = states.inventory, np.asarray(states.player_position)
    rows = np.arange(len(pos))
    grid = np.asarray(states.map)
    if game == "classic":
        floor = np.zeros(len(pos), np.int32)
        levels = np.stack([_tier(inv, "pickaxe"), _tier(inv, "sword")], -1)
    else:
        floor = np.asarray(states.player_level)
        grid = grid[rows, floor]
        tools = (inv.pickaxe, inv.sword, inv.bow, inv.arrows, inv.torches, inv.armour)
        levels = np.column_stack([np.asarray(x) for x in tools])
    near = pos[:, None] + np.asarray(c.CLOSE_BLOCKS)
    inside = ((near >= 0) & (near < grid.shape[1:])).all(-1)
    r, col = (np.clip(near[..., i], 0, grid.shape[1 + i] - 1) for i in (0, 1))
    table = grid[rows[:, None], r, col] == c.BlockType.CRAFTING_TABLE.value
    stats = (
        states.player_health,
        states.player_food,
        states.player_drink,
        states.player_energy,
    )
    return Trace(
        position=pos,
        floor=floor,
        intrinsics=np.stack([np.asarray(x, np.float32) for x in stats], -1),
        inventory=np.stack([np.asarray(getattr(inv, k)) for k in resources(game)], -1),
        levels=levels,
        table_near=(table & inside).any(-1),
    )
