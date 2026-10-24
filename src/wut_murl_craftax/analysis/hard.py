"""Hard scenarios of the training analysis: floors, achievements, deaths and stalls.

Each summary reads the ``eval_episodes`` table (contracts.md section 1) and returns one
block of rows per (``method``, ``config_ref``, ``budget``); a unit is never pooled with
another (R77), and the comparison track's death and stall figures (F9) import these
functions. Defaults: final checkpoint, stochastic eval. Intervals are
`stats.stratified_bootstrap_ci` over seeds. Only the stored event stream is read, never
the replay-only fields (``gathered/*``, ``coverage/*``, a stall's ``intrinsics``); a
value that cannot be known is NaN or a status, never 0.

@see [[2026-10-20-craftax-training-analysis]]
"""

import json
import logging
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from ..evaluation.milestones import FLOORS
from ..evaluation.stats import BOOTSTRAP_N, Interval, stratified_bootstrap_ci
from ..visualisation import charts
from .load import UNIT_COLUMNS, final_checkpoint

logger = logging.getLogger(__name__)

EVAL_MODE = "stochastic"
"""Eval mode of the thresholds (final stochastic eval episodes)."""
GROUP = (*UNIT_COLUMNS, "budget")
"""Key of one block of rows: the statistics unit and the budget."""
RARE_RATE = 0.05
"""Rare: seed-mean rate below this (a floor reached, an achievement unlocked)."""
NEVER, NEVER_EVAL, RARE, COMMON = "never", "never in eval", "rare", "common"
"""Statuses: ``never in eval`` is a rate of 0 in every eval of every seed with the
training episodes not logged for every seed, so "never" cannot be asserted."""
NO_DEATHS, NOT_RECORDED = "(no deaths)", "not recorded"
"""`death_causes` row of a checkpoint whose episodes all timed out, and the cause of a
death with an empty ``cause_of_death`` (not the real ``unknown`` damage source)."""
BASIS = ("mode", "scope")
"""Columns that say what a row summarises: the eval mode and the checkpoints (``final``
= each run's last, ``all`` = every one, ``checkpoint`` = one row set per checkpoint)."""
SCOPE_TEXT = {
    "final": "final checkpoint",
    "all": "all checkpoints",
    "checkpoint": "every checkpoint",
}
"""Plot-title words of a ``scope`` value."""
START, UNKNOWN = "(episode start)", "(unknown)"
"""``started_by`` of a stall that starts with the episode, and of one whose first frame
matches no unlock step."""
N_FLOORS = len(FLOORS)
"""Floors 0-8 of Craftax."""
TIERS = ("basic", "intermediate", "advanced", "very_advanced")
"""Achievement tiers (reward 1, 3, 5, 8) in order."""
TRAIN_PREFIX = "train/ach/"
"""Curves metric of an achievement's training success rate (% of episodes)."""


class Achievement(NamedTuple):
    """Reward tier and home floor of one achievement."""

    tier: str
    floor: int
    """Home floor: where the mob lives or the floor entered; for ore, the floor with
    the ore (sapphire and ruby 2, though a floor-1 chest can hold them); for chest
    loot (bow, potion, books and so the spells and enchanting) the first floor whose
    chests hold it. Crafting at a table has no floor (floor 0). Checked against
    craftax 1.6.1."""


_BASIC = (
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword", "make_arrow", "make_torch", "place_torch",
)  # fmt: skip
_REST: tuple[tuple[str, str, int], ...] = (
    ("make_diamond_sword", "intermediate", 0), ("make_iron_armour", "intermediate", 0),
    ("make_diamond_armour", "intermediate", 0),
    ("enter_gnomish_mines", "intermediate", 2), ("enter_dungeon", "intermediate", 1),
    ("enter_sewers", "advanced", 3), ("enter_vault", "advanced", 4),
    ("enter_troll_mines", "advanced", 5), ("enter_fire_realm", "very_advanced", 6),
    ("enter_ice_realm", "very_advanced", 7), ("enter_graveyard", "very_advanced", 8),
    ("defeat_gnome_warrior", "intermediate", 2),
    ("defeat_gnome_archer", "intermediate", 2),
    ("defeat_orc_solider", "intermediate", 1), ("defeat_orc_mage", "intermediate", 1),
    ("defeat_lizard", "advanced", 3), ("defeat_kobold", "advanced", 3),
    ("defeat_troll", "advanced", 5), ("defeat_deep_thing", "advanced", 5),
    ("defeat_pigman", "very_advanced", 6),
    ("defeat_fire_elemental", "very_advanced", 6),
    ("defeat_frost_troll", "very_advanced", 7),
    ("defeat_ice_elemental", "very_advanced", 7),
    ("damage_necromancer", "very_advanced", 8),
    ("defeat_necromancer", "very_advanced", 8),
    ("eat_bat", "intermediate", 2), ("eat_snail", "intermediate", 1),
    ("find_bow", "intermediate", 1), ("fire_bow", "intermediate", 1),
    ("collect_sapphire", "intermediate", 2),
    ("learn_fireball", "advanced", 3), ("cast_fireball", "advanced", 3),
    ("learn_iceball", "advanced", 3), ("cast_iceball", "advanced", 3),
    ("collect_ruby", "intermediate", 2), ("make_diamond_pickaxe", "intermediate", 0),
    ("open_chest", "intermediate", 1), ("drink_potion", "intermediate", 1),
    ("enchant_sword", "advanced", 3), ("enchant_armour", "advanced", 3),
    ("defeat_knight", "advanced", 4), ("defeat_archer", "advanced", 4),
)  # fmt: skip
ACHIEVEMENTS: dict[str, Achievement] = {n: Achievement("basic", 0) for n in _BASIC} | {
    n: Achievement(tier, floor) for n, tier, floor in _REST
}
"""Craftax achievements in ``.value`` order (IDs 0-66), spelled out because
`milestones.achievement_names` imports Craftax; the tests check the names and tiers
against the game. Classic's 22 are the first 22 names (all basic, floor 0)."""
_VALUE = {name: i for i, name in enumerate(ACHIEVEMENTS)}

_STALL_COLUMNS = (
    "run_id", "seed", "update", "env_steps", "mode", "eval_seed", "start", "end",
    "length", "floor", "ending", "ended_by", "started_by",
)  # fmt: skip
STALL_COLUMNS = (*GROUP, "scope", *_STALL_COLUMNS)
"""Columns of `stall_table`."""

_FLOOR_COLUMNS = (
    "floor", "floor_name", "n_seeds", "n_episodes", "rate", "lo", "hi",
    "reached_seeds", "status",
)  # fmt: skip
_ACH_COLUMNS = (
    "achievement", "tier", "floor", "n_seeds", "final_rate", "best_rate",
    "best_update", "eval_seeds", "train_seeds", "train_logged", "seeds_ever",
    "status",
)  # fmt: skip
_DEATH_COLUMNS = (
    "update", "env_steps", "cause", "n_deaths", "n_episodes", "n_seeds", "share",
)  # fmt: skip
STAT_FLOORS = tuple(f"start_floor/{f}" for f in range(N_FLOORS))
_STALL_STAT_COLUMNS = (
    "n_seeds", "n_episodes", "n_stalls", "stalls_per_episode", "stalls_lo",
    "stalls_hi", "median_length", "escape_rate", *STAT_FLOORS,
)  # fmt: skip


def _info(name: str) -> Achievement:
    """Tier and floor of an achievement name.

    Raises:
        ValueError: If the name is not a Craftax achievement.
    """
    if name not in ACHIEVEMENTS:
        msg = f"unknown achievement {name!r}: not in hard.ACHIEVEMENTS"
        raise ValueError(msg)
    return ACHIEVEMENTS[name]


def _empty(keys: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(keys))


def _ci(per_seed: np.ndarray, n: int) -> Interval:
    """Mean over seeds with its percentile bootstrap interval (shared bootstrap)."""
    if not per_seed.size:
        return Interval(float("nan"), float("nan"), float("nan"))
    if np.all(per_seed == per_seed[0]):  # every resample has this mean
        return Interval(float(per_seed[0]), float(per_seed[0]), float(per_seed[0]))
    return stratified_bootstrap_ci(
        {"seeds": per_seed.tolist()}, lambda d: float(d["seeds"].mean()), n=n
    )


def _selected(episodes: pd.DataFrame, mode: str, *, final: bool) -> pd.DataFrame:
    """The ``mode`` episodes, of the final checkpoint of each run if ``final``."""
    rows = episodes[episodes["mode"] == mode]
    return final_checkpoint(rows) if final and len(rows) else rows


def _blocks(episodes: pd.DataFrame) -> list[tuple[tuple[Hashable, ...], pd.DataFrame]]:
    """Each (method, config_ref, budget) of ``episodes`` with its rows, sorted."""
    return [(tuple(k), g) for k, g in episodes.groupby(list(GROUP), sort=True)]


def _lead(key: tuple[Hashable, ...], mode: str, scope: str) -> dict[str, Hashable]:
    """The leading columns of a summary row: the unit, then what it summarises."""
    return {**dict(zip(GROUP, key, strict=True)), "mode": mode, "scope": scope}


def _by_basis(frame: pd.DataFrame) -> list[tuple[dict[str, Hashable], pd.DataFrame]]:
    """Each unit, mode and scope of a stall table, as leading columns and rows."""
    return [
        (_lead(k[:3], k[3], k[4]), g)
        for k, g in frame.groupby([*GROUP, *BASIS], sort=True)
    ]


# ---------------------------------------------------------------------- floors


def _floor_status(reached: int, rate: float) -> str:
    if not reached:
        return NEVER
    return RARE if rate < RARE_RATE else COMMON


def floor_reach(
    episodes: pd.DataFrame,
    mode: str = EVAL_MODE,
    n: int = BOOTSTRAP_N,
    floors: int = N_FLOORS,
) -> pd.DataFrame:
    """Share of final eval episodes whose ``max_floor`` is at least ``f``, per floor.

    ``rate`` is the mean over seeds of each seed's share (fractions, not %), with the
    bootstrap interval ``lo`` and ``hi`` over seeds; ``reached_seeds`` is the number of
    seeds with at least one such episode. ``status`` is ``never`` (no final episode of
    any seed reaches the floor), ``rare`` (``rate`` below `RARE_RATE`) or ``common``.
    ``mode`` and ``scope`` (``final``) columns record the basis.
    """
    final = _selected(episodes, mode, final=True)
    rows = []
    for key, g in _blocks(final):
        run = g["run_id"]
        for f in range(floors):
            hit = g["max_floor"] >= f
            per_seed = hit.groupby(run).mean().to_numpy()
            est = _ci(per_seed, n)
            rows.append({
                **_lead(key, mode, "final"),
                "floor": f,
                "floor_name": FLOORS[f][0],
                "n_seeds": int(run.nunique()),
                "n_episodes": len(g),
                "rate": est.estimate,
                "lo": est.lo,
                "hi": est.hi,
                "reached_seeds": int(hit.groupby(run).any().sum()),
                "status": _floor_status(int(hit.sum()), est.estimate),
            })
    return pd.DataFrame(rows) if rows else _empty((*GROUP, *BASIS, *_FLOOR_COLUMNS))


# --------------------------------------------------------------- achievements


def _achievement_order(cols: Sequence[str]) -> list[str]:
    """Achievement ``ach/`` columns by tier, floor and ``.value``; names validated."""
    names = [c.removeprefix("ach/") for c in cols]
    infos = {n: _info(n) for n in names}
    return sorted(
        names, key=lambda n: (TIERS.index(infos[n].tier), infos[n].floor, _VALUE[n])
    )


def _training(curves: pd.DataFrame | None, runs: pd.Index) -> pd.DataFrame:
    """Peak ``train/ach/<name>`` per (run, achievement); NaN where not logged."""
    if curves is None:
        return pd.DataFrame(index=runs)
    sub = curves[
        curves["run_id"].isin(runs) & curves["metric"].str.startswith(TRAIN_PREFIX)
    ]
    peak = sub.groupby(["run_id", "metric"])["value"].max().unstack("metric")
    peak.columns = [c.removeprefix(TRAIN_PREFIX) for c in peak.columns]
    return peak.reindex(runs)


def _achievement_block(
    g: pd.DataFrame, names: list[str], peak: pd.DataFrame, mode: str
) -> list[dict[str, Any]]:
    cols = [f"ach/{n}" for n in names]
    runs = pd.Index(sorted(g["run_id"].unique()))
    n_seeds = len(runs)
    # any mode, any checkpoint: "never unlocked" covers every eval of every seed
    evals = g.groupby("run_id")[cols].any().reindex(runs).set_axis(names, axis=1)
    rows = g[g["mode"] == mode]
    final = final_checkpoint(rows).groupby("run_id")[cols].mean().mean()
    per_update = (
        rows.groupby(["update", "run_id"])[cols].mean().groupby("update").mean()
    )
    best, best_at = per_update.max(), per_update.idxmax()
    out = []
    for name, col in zip(names, cols, strict=True):
        series = peak[name].dropna() if name in peak else pd.Series(dtype=float)
        logged = len(series)
        trained = peak[name] > 0 if name in peak else pd.Series(False, index=runs)
        ever = int((evals[name] | trained).sum())
        info = _info(name)
        out.append({
            "achievement": name,
            "tier": info.tier,
            "floor": info.floor,
            "n_seeds": n_seeds,
            "final_rate": float(final[col]),
            "best_rate": float(best[col]),
            "best_update": int(best_at[col]),
            "eval_seeds": int(evals[name].sum()),
            "train_seeds": int((series > 0).sum()) if logged else float("nan"),
            "train_logged": logged,
            "seeds_ever": ever,
            "status": _achievement_status(ever, logged, n_seeds, float(final[col])),
        })
    return out


def _achievement_status(ever: int, logged: int, n_seeds: int, final: float) -> str:
    if not ever:
        return NEVER if logged == n_seeds else NEVER_EVAL
    return RARE if final < RARE_RATE else COMMON


def achievement_status(
    episodes: pd.DataFrame, curves: pd.DataFrame | None = None, mode: str = EVAL_MODE
) -> pd.DataFrame:
    """Per achievement: final rate, best rate over training, seeds that ever unlocked.

    Rates are fractions: the mean over seeds of each seed's share of ``mode`` eval
    episodes with the achievement, at the final checkpoint (``final_rate``) and at the
    checkpoint where that mean peaks (``best_rate``, ``best_update``). ``seeds_ever`` is
    the k of ``n_seeds`` that unlocked it in any eval (either mode, any checkpoint) or,
    given ``curves``, in training (``train/ach/<name>`` above 0); ``eval_seeds`` and
    ``train_seeds`` split it. ``train_seeds`` is NaN and ``train_logged`` counts the
    seeds with the series when ``curves`` lacks it (the reused runs, R78): a series
    with no finished episode is not logged. ``status``: ``never`` (no unlock anywhere,
    training series logged for every seed), ``never in eval`` (no eval unlock, training
    not logged for every seed), ``rare`` (final rate below `RARE_RATE`), ``common``.
    Rows are ordered by tier, floor and ``.value`` of the achievement; a ``mode`` column
    records the eval mode of the rates.

    Raises:
        ValueError: If an ``ach/<name>`` column is not a Craftax achievement.
    """
    cols = [c for c in episodes.columns if c.startswith("ach/")]
    names = _achievement_order(cols)
    rows = []
    for key, g in _blocks(episodes):
        if not (g["mode"] == mode).any():
            continue
        peak = _training(curves, pd.Index(sorted(g["run_id"].unique())))
        rows += [
            {**dict(zip(GROUP, key, strict=True)), "mode": mode, **r}
            for r in _achievement_block(g, names, peak, mode)
        ]
    return pd.DataFrame(rows) if rows else _empty((*GROUP, "mode", *_ACH_COLUMNS))


# ---------------------------------------------------------------------- deaths


def _death_rows(
    u: pd.DataFrame, lead: dict[str, Hashable], update: int
) -> list[dict[str, Any]]:
    """Cause rows of the episodes ``u`` (one checkpoint, or each run's last)."""
    dead = u[u["end_type"] == "death"]
    cause = dead["cause_of_death"].fillna("").replace("", NOT_RECORDED)
    counts = cause.value_counts().sort_index()
    counts = counts.sort_values(ascending=False, kind="stable")
    row = {
        **lead,
        "update": update,
        "env_steps": int(u["env_steps"].max()),
        "n_episodes": len(u),
        "n_seeds": int(u["run_id"].nunique()),
    }
    if counts.empty:
        return [{**row, "cause": NO_DEATHS, "n_deaths": 0, "share": np.nan}]
    return [
        {**row, "cause": c, "n_deaths": int(k), "share": k / counts.sum()}
        for c, k in counts.items()
    ]


def death_causes(episodes: pd.DataFrame, mode: str = EVAL_MODE) -> pd.DataFrame:
    """Cause-of-death shares per eval checkpoint (``scope`` ``checkpoint``) and final.

    One row per (unit, checkpoint, cause) over the ``mode`` episodes of all seeds:
    ``share`` is the cause's deaths over all deaths there, so the shares of a
    checkpoint sum to 1 (a timeout is not a death; a death with an empty cause is
    `NOT_RECORDED`, distinct from the real ``unknown`` cause). The rows with scope
    ``final`` use each run's own last checkpoint, as the other final tables do; their
    ``update`` and ``env_steps`` are the largest of those. A set with no death has one
    row, cause `NO_DEATHS`, ``share`` NaN (undefined, not 0). ``n_episodes`` counts all
    its episodes, timeouts included.
    """
    rows = []
    sel = episodes[episodes["mode"] == mode]
    for key, g in _blocks(sel):
        for update, u in g.groupby("update", sort=True):
            rows += _death_rows(u, _lead(key, mode, "checkpoint"), int(update))
        final = final_checkpoint(g)
        rows += _death_rows(
            final, _lead(key, mode, "final"), int(final["update"].max())
        )
    return pd.DataFrame(rows) if rows else _empty((*GROUP, *BASIS, *_DEATH_COLUMNS))


# ---------------------------------------------------------------------- stalls


def _cells(cell: Any) -> list[dict[str, Any]]:
    """The stalls of one episode: an array of structs (parquet) or JSON (CSV)."""
    if isinstance(cell, str):
        return list(json.loads(cell))
    if cell is None or isinstance(cell, float):  # a missing list reads as no stalls
        return []
    return list(cell)


def _by_value(names: Sequence[str]) -> list[str]:
    """Achievement names in ``.value`` order; the order that breaks unlock ties."""
    for n in names:
        _info(n)
    return sorted(names, key=_VALUE.__getitem__)


def _started_by(start: int, steps: np.ndarray | None, names: list[str]) -> str:
    """The achievement unlocked at the stall's first frame (ties: lowest value)."""
    if start == 0:
        return START
    if steps is None:
        return UNKNOWN
    hit = np.flatnonzero(steps == start)
    return names[hit[0]] if hit.size else UNKNOWN


def stall_table(
    episodes: pd.DataFrame, mode: str = EVAL_MODE, *, final: bool = True
) -> pd.DataFrame:
    """One row per stall of the ``mode`` episodes (final checkpoint only if ``final``).

    Columns: the unit, ``scope`` (``final`` or ``all``), the episode (``run_id``,
    ``seed``, ``update``, ``env_steps``, ``mode``, ``eval_seed``) and the stall frames
    (``start``, ``end``, ``length``, ``floor``, ``ending``, ``ended_by``) as stored,
    plus ``started_by``: the achievement whose ``unlock_step/<name>`` is ``start``
    (lowest ``.value`` of several), `START` at 0 and `UNKNOWN` without a match or
    without the ``unlock_step`` columns.
    The replay-only fields of a stall (``intrinsics``, ``inventory``) are dropped.
    """
    sel = _selected(episodes, mode, final=final).reset_index(drop=True)
    unlock = _by_value([
        c.removeprefix("unlock_step/")
        for c in sel.columns
        if c.startswith("unlock_step/")
    ])
    steps = sel[[f"unlock_step/{n}" for n in unlock]].to_numpy() if unlock else None
    rows = []
    for i in range(len(sel)):
        episode = sel.iloc[i]
        for s in _cells(episode["stalls"]):
            by = _started_by(
                int(s["start"]), None if steps is None else steps[i], unlock
            )
            rows.append({
                **{c: episode[c] for c in GROUP},
                "scope": "final" if final else "all",
                **{c: episode[c] for c in _STALL_COLUMNS[:6]},
                **{c: s[c] for c in _STALL_COLUMNS[6:12]},
                "started_by": by,
            })
    return pd.DataFrame(rows, columns=list(STALL_COLUMNS))


def stall_stats(
    episodes: pd.DataFrame,
    mode: str = EVAL_MODE,
    n: int = BOOTSTRAP_N,
    *,
    final: bool = True,
) -> pd.DataFrame:
    """Stall statistics per unit over the ``mode`` episodes (final checkpoint default).

    ``stalls_per_episode`` is the mean over seeds of each seed's mean ``n_stalls``,
    with the bootstrap interval ``stalls_lo`` and ``stalls_hi``; ``median_length`` is
    over all stalls of the unit; ``escape_rate`` is the share of stalls that ended in
    an unlock (not death or timeout). Both are NaN when the unit has no stall.
    ``start_floor/<f>`` counts the stalls that start on floor ``f``.
    """
    sel = _selected(episodes, mode, final=final)
    table = stall_table(episodes, mode, final=final)
    by_unit = {tuple(k): g for k, g in table.groupby(list(GROUP), sort=False)}
    rows = []
    scope = "final" if final else "all"
    for key, g in _blocks(sel):
        stalls = by_unit.get(key, table.iloc[:0])
        est = _ci(g.groupby("run_id")["n_stalls"].mean().to_numpy(), n)
        floors = stalls["floor"].value_counts()
        rows.append({
            **_lead(key, mode, scope),
            "n_seeds": int(g["run_id"].nunique()),
            "n_episodes": len(g),
            "n_stalls": len(stalls),
            "stalls_per_episode": est.estimate,
            "stalls_lo": est.lo,
            "stalls_hi": est.hi,
            "median_length": float(stalls["length"].median())
            if len(stalls)
            else np.nan,
            "escape_rate": float((stalls["ending"] == "unlock").mean())
            if len(stalls)
            else np.nan,
            **{f"start_floor/{f}": int(floors.get(f, 0)) for f in range(N_FLOORS)},
        })
    return (
        pd.DataFrame(rows) if rows else _empty((*GROUP, *BASIS, *_STALL_STAT_COLUMNS))
    )


def stall_starts(stalls: pd.DataFrame) -> pd.DataFrame:
    """Where stalls start: count and share of a unit's stalls per floor and unlock."""
    keys = [*GROUP, *BASIS, "floor", "started_by"]
    if stalls.empty:
        return _empty([*keys, "n_stalls", "share"])
    out = stalls.groupby(keys, sort=True).size().rename("n_stalls").reset_index()
    by_unit = out.groupby([*GROUP, *BASIS])["n_stalls"].transform("sum")
    out["share"] = out["n_stalls"] / by_unit
    order = [*GROUP, *BASIS, "n_stalls"]
    return out.sort_values(order, ascending=[True] * 5 + [False]).reset_index(drop=True)


def stall_ccdf(stalls: pd.DataFrame) -> pd.DataFrame:
    """Per unit and stall length, the share of stalls at least that long."""
    keys = [*GROUP, *BASIS, "length", "ccdf"]
    rows: list[dict[str, Any]] = []
    for lead, g in _by_basis(stalls) if len(stalls) else []:
        lengths = np.sort(g["length"].to_numpy())
        rows += [
            {**lead, "length": int(x), "ccdf": float((lengths >= x).mean())}
            for x in np.unique(lengths)
        ]
    return pd.DataFrame(rows) if rows else _empty(keys)


# ----------------------------------------------------------------------- plots

PERCENT_TICKS = (0, 1, 10, 100)
"""Ticks of the symlog percent axes (the 5% limit is a dashed line)."""
MARGIN_PX = {"left": 90, "right": 30, "top": 50, "bottom": 70}
"""Figure margins in pixels, turned into fractions per figure size."""


def _label(method: str, ref: str) -> str:
    return method if ref == "tuned" else f"{method} ({ref})"


def _unit_keys(table: pd.DataFrame) -> list[tuple[str, str]]:
    """The (method, config_ref) of a plotted table, tuned first, one budget only.

    Raises:
        ValueError: If the table holds several budgets (their axes must not mix).
    """
    if table["budget"].nunique() > 1:
        msg = f"one budget per plot, got {sorted(table['budget'].unique())}"
        raise ValueError(msg)
    pairs = table[list(UNIT_COLUMNS)].drop_duplicates()
    keys = [(str(m), str(r)) for m, r in pairs.itertuples(index=False, name=None)]
    return sorted(keys, key=lambda k: (k[0], k[1] != "tuned", k[1]))


def _basis(table: pd.DataFrame) -> str:
    """Title words for the eval mode and checkpoints of a plotted table.

    Raises:
        ValueError: If the table mixes modes or scopes (one basis per plot).
    """
    parts = []
    for column in BASIS:
        if column not in table:
            continue
        values = table[column].unique()
        if len(values) != 1:
            msg = f"one {column} per plot, got {sorted(values)}"
            raise ValueError(msg)
        parts.append(f"{values[0]} eval" if column == "mode" else SCOPE_TEXT[values[0]])
    return ", ".join(parts)


def _rows_of(table: pd.DataFrame, key: tuple[str, str]) -> pd.DataFrame:
    return table[(table["method"] == key[0]) & (table["config_ref"] == key[1])]


def _figure(width: int, height: int, **margins: int) -> Figure:
    """A dark figure with pixel margins (``MARGIN_PX`` overridden by ``margins``)."""
    fig = charts.figure(width, height)
    m = MARGIN_PX | margins
    fig.subplots_adjust(
        left=m["left"] / width,
        right=1 - m["right"] / width,
        top=1 - m["top"] / height,
        bottom=m["bottom"] / height,
    )
    return fig


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9, color=charts.INK)


def _save(fig: Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path


def _nothing(out_dir: Path, name: str, what: str) -> Path:
    """A placeholder plot that says there is nothing to show."""
    fig = charts.figure(640, 300)
    ax = charts.style(fig.add_subplot())
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"no {what}", ha="center", va="center", color=charts.MUTED)
    return _save(fig, out_dir, name)


def _handle(label: str, marker: str = "o", **style: str) -> Line2D:
    """A legend marker without a line."""
    return Line2D([], [], ls="", marker=marker, label=label, **style)  # type: ignore[arg-type]  # matplotlib kwargs are typed one by one


def _legend(fig: Figure, handles: Sequence[Artist], **kw: Any) -> None:
    legend = fig.legend(handles=handles, **kw)
    legend.get_frame().set_facecolor(charts.SURFACE)
    legend.get_frame().set_edgecolor(charts.GRID)
    for text in legend.get_texts():
        text.set_color(charts.INK2)


def steps_text(steps: float) -> str:
    """Env steps as ``1.5M``, ``820k`` or ``40``, for titles, ticks and captions."""
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if steps >= scale:
            return f"{steps / scale:.3g}{unit}"
    return f"{steps:.0f}"


def _plain_log(axis: Axis) -> None:
    """Log ticks at 1, 2, 3 and 5 times a power of ten, labelled as plain numbers."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(NullLocator())


def _percent_axis(ax: Axes, which: str) -> None:
    """Symlog percent axis from 0 to 100 (rates of a few % stay readable)."""
    getattr(ax, f"set_{which}scale")("symlog", linthresh=1.0)
    getattr(ax, f"set_{which}lim")(-0.3, 100)  # room for the 0 marks
    getattr(ax, f"set_{which}ticks")(PERCENT_TICKS, [f"{p}%" for p in PERCENT_TICKS])


def plot_floor_reach(table: pd.DataFrame, out_dir: Path) -> Path:
    """``floor_reach.png``: reach rate per floor (%, symlog) with the bootstrap band.

    One line per unit of ``table`` (one budget); the dashed line is the 5% rare limit.
    """
    name = "floor_reach.png"
    if table.empty:
        return _nothing(out_dir, name, "floor reach rates")
    keys = _unit_keys(table)
    fig = _figure(900, 520)
    ax = charts.style(fig.add_subplot())
    for i, (method, ref) in enumerate(keys):
        t = _rows_of(table, (method, ref)).sort_values("floor")
        color = charts.SERIES[i % len(charts.SERIES)]
        ax.fill_between(
            t["floor"], t["lo"] * 100, t["hi"] * 100, color=color, alpha=0.18
        )
        ax.plot(
            t["floor"], t["rate"] * 100, "-o", color=color, ms=4,
            label=_label(method, ref),
        )  # fmt: skip
    ax.axhline(RARE_RATE * 100, color=charts.MUTED, ls="--", lw=1)
    _percent_axis(ax, "y")
    floors = sorted(table["floor"].unique())
    names = table.drop_duplicates("floor").set_index("floor")["floor_name"]
    ax.set_xticks(floors, [f"{f}\n{names[f]}" for f in floors])
    ax.set_xlabel("floor")
    ax.set_ylabel("final eval episodes reaching the floor")
    budget = table["budget"].iloc[0]
    _title(ax, f"Floor reach, {budget}, {_basis(table)} (dashed: 5% rare limit)")
    _legend(fig, ax.get_legend_handles_labels()[0], loc="upper right")
    return _save(fig, out_dir, name)


MARK_LABELS = {
    NEVER: "never unlocked",
    NEVER_EVAL: "no eval unlock (training not logged)",
}
"""Legend and scatter labels of the two zero marks: ``never`` is asserted, ``never in
eval`` is not (R78), so they are drawn apart."""


def _achievement_marks(ax: Axes, t: pd.DataFrame, y: np.ndarray) -> None:
    """Final-rate dots (blue common, amber rare), zero marks for never, hollow best.

    ``never``: a solid cross; ``never in eval``: a hollow diamond (both at 0).
    """
    final, best = t["final_rate"] * 100, t["best_rate"] * 100
    for status, color in ((COMMON, charts.SERIES[0]), (RARE, charts.SERIES[3])):
        m = (t["status"] == status).to_numpy()
        ax.hlines(y[m], final[m], best[m], color=charts.GRID)
        ax.scatter(final[m], y[m], s=16, color=color, zorder=3)
    never = (t["status"] == NEVER).to_numpy()
    ax.scatter(
        np.zeros(never.sum()), y[never], marker="x", s=18, color=charts.SERIES[1],
        label=MARK_LABELS[NEVER],
    )  # fmt: skip
    unknown = (t["status"] == NEVER_EVAL).to_numpy()
    ax.scatter(
        np.zeros(unknown.sum()), y[unknown], marker="D", s=14, facecolors="none",
        edgecolors=charts.SERIES[1], linewidths=0.9, label=MARK_LABELS[NEVER_EVAL],
    )  # fmt: skip
    ax.scatter(best, y, s=22, facecolors="none", edgecolors=charts.INK2, linewidths=0.8)


def plot_achievement_status(table: pd.DataFrame, out_dir: Path) -> Path:
    """``achievement_status.png``: a dot row per achievement, grouped by tier and floor.

    Filled dot: final rate; hollow: best rate over training; a solid cross at 0 marks an
    achievement never unlocked (``never``), a hollow diamond one with no eval unlock
    whose training episodes are not logged for every seed (``never in eval``, which
    does not assert "never", R78); rare ones are drawn in the second colour. One
    panel per unit, the rate axis symlog in %; the group labels (tier and floor) stand
    right of the last panel.
    """
    name = "achievement_status.png"
    if table.empty:
        return _nothing(out_dir, name, "achievement rates")
    keys = _unit_keys(table)
    first = _rows_of(table, keys[0]).reset_index(drop=True)
    group = (first["tier"] + " f" + first["floor"].astype(str)).to_numpy()
    y = np.cumsum(np.r_[0.0, (group[1:] != group[:-1]) * 0.7 + 1.0])
    fig = _figure(
        260 * len(keys) + 330,
        int(15 * (y[-1] + 1) + 160),
        left=170,
        right=150,
        bottom=80,
    )
    axes = fig.subplots(1, len(keys), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(wspace=0.08)
    for ax, key in zip(axes, keys, strict=True):
        charts.style(ax, size=7)
        t = _rows_of(table, key).set_index("achievement").reindex(first["achievement"])
        _achievement_marks(ax, t, y)
        ax.axvline(RARE_RATE * 100, color=charts.MUTED, ls="--", lw=1)
        _percent_axis(ax, "x")
        ax.set_yticks(y, first["achievement"])
        ax.set_ylim(y[-1] + 1, -1)
        _title(ax, _label(*key))
    for g in dict.fromkeys(group):
        axes[-1].text(
            1.03, y[group == g].mean(), g.replace("_", " "),
            transform=axes[-1].get_yaxis_transform(), fontsize=7, color=charts.MUTED,
            va="center",
        )  # fmt: skip
    _legend(
        fig,
        [
            _handle("final rate", color=charts.SERIES[0]),
            _handle("best over training", mec=charts.INK2, mfc="none"),
            _handle("rare (< 5%)", color=charts.SERIES[3]),
            _handle(MARK_LABELS[NEVER], color=charts.SERIES[1], marker="x"),
            _handle(
                MARK_LABELS[NEVER_EVAL], marker="D", mec=charts.SERIES[1], mfc="none"
            ),
        ],
        loc="lower center",
        ncol=3,
    )
    basis = f"{_basis(table)}: final rate, best over all checkpoints"
    fig.suptitle(basis, x=0.01, ha="left", fontsize=9, color=charts.INK)
    return _save(fig, out_dir, name)


def plot_death_causes(table: pd.DataFrame, out_dir: Path) -> Path:
    """``death_causes_over_training.png``: stacked cause shares per checkpoint.

    The four most frequent causes of the table get a colour, the rest are ``other``; a
    checkpoint without deaths has no bar. One panel per unit.
    """
    name = "death_causes_over_training.png"
    table = table[table["scope"] == "checkpoint"]
    deaths = table[table["n_deaths"] > 0] if len(table) else table
    if deaths.empty:
        return _nothing(out_dir, name, "deaths")
    keys = _unit_keys(table)
    top = list(deaths.groupby("cause")["n_deaths"].sum().nlargest(4).index)
    colors = dict(zip(top, charts.SERIES, strict=False)) | {"other": charts.MUTED}
    fig = _figure(330 * len(keys) + 260, 470, right=170, bottom=90)
    axes = fig.subplots(1, len(keys), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(wspace=0.08)
    for ax, key in zip(axes, keys, strict=True):
        charts.style(ax)
        t = _rows_of(table, key)
        steps = t.groupby("update")["env_steps"].max()
        d = t[t["n_deaths"] > 0].assign(
            group=lambda x: x["cause"].where(x["cause"].isin(top), "other")
        )
        shares = (
            d
            .pivot_table(index="update", columns="group", values="share", aggfunc="sum")
            .reindex(steps.index)
            .fillna(0.0)
        )
        bottom = np.zeros(len(steps))
        for cause in [c for c in (*top, "other") if c in shares]:
            ax.bar(
                range(len(steps)), shares[cause] * 100, bottom=bottom * 100,
                color=colors[cause], width=0.8,
            )  # fmt: skip
            bottom += shares[cause].to_numpy()
        ticks = [steps_text(s) for s in steps]
        ax.set_xticks(range(len(steps)), ticks, rotation=60, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlabel("env steps")
        _title(ax, _label(*key))
    axes[0].set_ylabel("share of deaths (%)")
    fig.suptitle(_basis(table), x=0.01, ha="left", fontsize=9, color=charts.INK)
    _legend(
        fig,
        [
            Line2D([], [], marker="s", ls="", color=c, label=n)
            for n, c in colors.items()
        ],
        loc="center right",
    )
    return _save(fig, out_dir, name)


def plot_stall_lengths(ccdf: pd.DataFrame, out_dir: Path) -> Path:
    """``stall_lengths.png``: CCDF of stall length (log-log), one line per unit."""
    name = "stall_lengths.png"
    if ccdf.empty:
        return _nothing(out_dir, name, "stalls")
    fig = _figure(820, 520)
    ax = charts.style(fig.add_subplot())
    for i, key in enumerate(_unit_keys(ccdf)):
        t = _rows_of(ccdf, key).sort_values("length")
        ax.plot(
            t["length"], t["ccdf"], drawstyle="steps-post", marker="o", ms=2.5,
            color=charts.SERIES[i % len(charts.SERIES)], label=_label(*key),
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax.xaxis)
    _plain_log(ax.yaxis)
    ax.set_xlabel("stall length (steps without an unlock)")
    ax.set_ylabel("share of stalls at least this long")
    _title(ax, f"Stall lengths, {ccdf['budget'].iloc[0]}, {_basis(ccdf)}")
    _legend(fig, ax.get_legend_handles_labels()[0], loc="upper right")
    return _save(fig, out_dir, name)
