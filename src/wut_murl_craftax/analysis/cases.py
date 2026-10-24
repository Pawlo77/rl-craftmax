"""Case episodes of the training analysis: find, export and describe them.

`find_cases` picks the plan's cases of one (method, `config_ref`) unit from the
``eval_episodes`` table (long escaped stalls, deepest floors, rare unlocks, deaths) as
``cases.csv`` rows (contracts section 4); `write_cases` writes the budget table and the
per-method copies. `export_case` replays a case from its eval file into a rollout, a
16 px master, a GIF of the window and, for a stall, an agent-internals clip;
`describe_case` gives the facts the write-up needs. Finding imports neither JAX nor
MLflow: the replay side imports JAX when called.

@see [[2026-10-20-craftax-training-analysis]]
"""

import functools
import json
import logging
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

from ..evaluation.milestones import FLOORS, Game, game_of
from .hard import (
    NO_DEATHS,
    NOT_RECORDED,
    RARE,
    achievement_status,
    death_causes,
    stall_table,
    steps_text,
)
from .load import Frames, select

if TYPE_CHECKING:
    from ..visualisation.eval_rollouts import EvalRun

logger = logging.getLogger(__name__)

OUTPUTS = Path("outputs")
"""Root that the file columns of a case are relative to."""
STORE_TIMEOUT_S = 300.0
"""SQLite busy timeout of `run_dirs` when the URI names none (R103)."""
EVAL_MODE = "stochastic"
"""Eval mode the cases are searched in (the thresholds' basis)."""
TOP_STALLS, EARLIER_CHECKPOINTS, DEEP_CASES, TOP_CAUSES = 5, 3, 3, 3
"""Longest escaped final-checkpoint stalls, earlier checkpoints with their longest one,
deepest-floor episodes, most frequent causes of death with one episode each."""
STALL_SPEED = 8.0
"""Playback speed of a stall case; the other kinds play at 1x."""
STALL_TAIL = 20
"""Steps a stall case runs on after its unlock (VIS:72)."""
BEFORE, AFTER = 150, 49
"""Steps before and after the peak of the other kinds: 200 frames, 20 s at 1x."""
GIF_SECONDS = 20
"""GIF length cap: a longer window keeps its last `GIF_SECONDS`, which hold the peak."""
OVERLAY_STEPS = 100
"""Steps before the unlock shown in a stall case's agent-internals clip."""
LAST_ACTIONS = 50
"""Actions before the peak that `describe_case` lists."""
CASE_COLUMNS = (
    "id", "method", "config_ref", "budget", "run_id", "run_name", "update", "mode",
    "seed", "kind", "start", "end", "peak", "unlock_step", "unlocked", "speed",
    "game", "caption", "rollout", "master", "gif", "overlay", "video",
    "env_steps", "length", "detail",
)  # fmt: skip
"""``cases.csv`` columns: contracts section 4, then the checkpoint's env steps, the
episode length and the case's subject (stall length, floor, achievement or cause)."""
FILES = ("rollout", "master", "gif", "overlay")
"""File columns `export_case` fills, paths relative to `OUTPUTS`."""
STATS = ("health", "food", "drink", "energy")
"""`Trace.intrinsics` columns."""
_KEY = ["run_id", "update", "eval_seed"]


class _Pick(NamedTuple):
    """One case episode before it becomes a row."""

    kind: str
    episode: pd.Series
    peak: int
    start: int
    """Stall cases only (the stall's first frame); -1 for the window around the peak."""
    unlocked: str
    detail: str
    what: str
    """Caption verb phrase."""


def _words(name: str) -> str:
    return name.replace("_", " ")


def _stall_picks(eps: pd.DataFrame, mode: str) -> list[_Pick]:
    """The longest escaped final stalls, then the longest of 3 earlier checkpoints.

    The earlier checkpoints are spread evenly over those with an escaped stall.
    """
    table = stall_table(eps, mode, final=False)
    table = table[table["ending"] == "unlock"].merge(eps[[*_KEY, "final"]], on=_KEY)
    order = {"by": ["length", *_KEY, "start"], "ascending": [False] + [True] * 4}
    final = table[table["final"]].sort_values(**order).head(TOP_STALLS)
    earlier = table[~table["final"]]
    ups = np.sort(earlier["update"].unique())
    k = min(EARLIER_CHECKPOINTS, len(ups))
    chosen = ups[np.round(np.linspace(0, len(ups) - 1, k)).astype(int)]
    best = earlier[earlier["update"].isin(chosen)].sort_values(**order)
    best = best.drop_duplicates("update").sort_values("update")
    episodes = eps.set_index(_KEY, drop=False)
    return [
        _Pick(
            "stall",
            episodes.loc[tuple(s[c] for c in _KEY)],
            int(s["end"]),
            int(s["start"]),
            s["ended_by"],
            f"{s['length']} steps",
            f"breaks a {s['length']}-step stall with {_words(s['ended_by'])}",
        )
        for s in pd.concat([final, best]).to_dict("records")
    ]


def _deep_picks(eps: pd.DataFrame) -> list[_Pick]:
    """The deepest-floor episodes, ties broken by achievements, then later checkpoint.

    The peak is the first step on the deepest floor (its ``enter_*`` unlock); on floor
    0, where every episode starts, the last unlock.

    Raises:
        ValueError: If a floor above 0 has no ``unlock_step`` of its ``enter_*``.
    """
    by = ["max_floor", "n_achievements", "update", "run_id", "eval_seed"]
    top = eps.sort_values(by, ascending=[False, False, False, True, True])
    unlocks = [c for c in eps.columns if c.startswith("unlock_step/")]
    picks = []
    for _, ep in top.head(DEEP_CASES).iterrows():
        f = int(ep["max_floor"])
        if f:
            peak = int(ep.get(f"unlock_step/{str(FLOORS[f][1]).lower()}", -1))
            if peak < 0:
                msg = f"episode {ep['run_id']}/{ep['eval_seed']}: no step of floor {f}"
                raise ValueError(msg)
            what = f"reaches floor {f} ({_words(FLOORS[f][0])})"
        else:
            peak = max([0, *(int(ep[c]) for c in unlocks)])
            what = f"unlocks {ep['n_achievements']} achievements on floor 0"
        picks.append(_Pick("deep_floor", ep, peak, -1, "", f"floor {f}", what))
    return picks


def _rare_picks(
    eps: pd.DataFrame, every_mode: pd.DataFrame, curves: pd.DataFrame, mode: str
) -> list[_Pick]:
    """One episode per rare achievement that unlocked it.

    Final checkpoint first, then the latest checkpoint, then the earliest unlock.
    """
    status = achievement_status(every_mode, curves if len(curves) else None, mode)
    picks = []
    for name in status.loc[status["status"] == RARE, "achievement"]:
        hits, step = eps[eps[f"ach/{name}"].astype(bool)], f"unlock_step/{name}"
        if hits.empty:  # rare by training or the other mode only: nothing to replay
            logger.info("rare %s: no %s eval episode unlocked it", name, mode)
            continue
        by = ["final", "update", step, "run_id", "eval_seed"]
        ep = hits.sort_values(by, ascending=[False, False, True, True, True]).iloc[0]
        what = f"unlocks {_words(name)} (rare)"
        picks.append(_Pick("rare_unlock", ep, int(ep[step]), -1, "", name, what))
    return picks


def _death_picks(eps: pd.DataFrame, mode: str) -> list[_Pick]:
    """The median-length final death of each of the most frequent causes."""
    causes = death_causes(eps, mode)
    final = causes["scope"] == "final"
    causes = causes[final & ~causes["cause"].isin([NO_DEATHS, NOT_RECORDED])]
    dead = eps[eps["final"] & (eps["end_type"] == "death")]
    picks = []
    for cause in causes["cause"].head(TOP_CAUSES):
        d = dead[dead["cause_of_death"] == cause]
        d = d.assign(dist=(d["length"] - d["length"].median()).abs())
        ep = d.sort_values(["dist", "run_id", "eval_seed"]).iloc[0]
        what = f"dies of {_words(cause)}"
        picks.append(_Pick("death", ep, int(ep["length"]), -1, "", cause, what))
    return picks


def _row(pick: _Pick, n: int, run: pd.Series, ref: str) -> dict[str, Any]:
    """The `CASE_COLUMNS` of one pick, the ``n``-th of its kind."""
    ep, kind, peak = pick.episode, pick.kind, pick.peak
    length = int(ep["length"])
    if kind == "stall":
        start, end = pick.start, min(peak + STALL_TAIL, length)
    else:
        start, end = max(0, peak - BEFORE), min(length, peak + AFTER)
    method = str(ep["method"])
    who, stem = (
        (method, method) if ref == "tuned" else (f"{method} ({ref})", f"{method}-{ref}")
    )
    steps, seed = steps_text(int(ep["env_steps"])), int(ep["eval_seed"])
    return {
        "id": f"{ep['budget']}-{stem}-{kind}-{n}",
        "method": method,
        "config_ref": ref,
        "budget": ep["budget"],
        "run_id": ep["run_id"],
        "run_name": run["run_name"],
        "update": int(ep["update"]),
        "mode": ep["mode"],
        "seed": seed,
        "kind": kind,
        "start": start,
        "end": end,
        "peak": peak,
        "unlock_step": peak if kind == "stall" else -1,
        "unlocked": pick.unlocked,
        "speed": STALL_SPEED if kind == "stall" else 1.0,
        "game": game_of(str(json.loads(run["config_json"])["env"])),
        "caption": f"{who} {pick.what} after {steps} steps, eval seed {seed}",
        **dict.fromkeys((*FILES, "video"), ""),
        "env_steps": int(ep["env_steps"]),
        "length": length,
        "detail": pick.detail,
    }


def find_cases(
    frames: Frames,
    method: str,
    budget: str,
    config_ref: str = "tuned",
    mode: str = EVAL_MODE,
) -> pd.DataFrame:
    """The case episodes of one (method, ``config_ref``) unit at ``budget`` (R77).

    Thresholds of the plan, over the ``mode`` eval episodes: the `TOP_STALLS` longest
    stalls escaped by an unlock at each run's final checkpoint, then the longest one at
    each of `EARLIER_CHECKPOINTS` earlier checkpoints; the `DEEP_CASES` deepest-floor
    episodes (ties by achievements); one episode per rare achievement
    (`hard.achievement_status`); the median-length death of each of the `TOP_CAUSES`
    most frequent final causes. Rows run stall, deep_floor, rare_unlock, death, ranked
    within a kind; ``id`` is ``<budget>-<method>-<kind>-<n>`` (an ablation adds
    ``-<config_ref>`` to the method). The file columns and ``video`` are "" until
    `export_case` and TA7.
    """
    unit = select(frames, method=method, budget=budget, config_ref=config_ref)
    every_mode = unit["eval_episodes"]
    eps = every_mode[every_mode["mode"] == mode].reset_index(drop=True)
    if eps.empty:
        return pd.DataFrame(columns=list(CASE_COLUMNS))
    eps["final"] = eps["update"] == eps.groupby("run_id")["update"].transform("max")
    picks = [
        *_stall_picks(eps, mode),
        *_deep_picks(eps),
        *_rare_picks(eps, every_mode, unit["curves"], mode),
        *_death_picks(eps, mode),
    ]
    runs = unit["runs"].set_index("run_id")
    seen: dict[str, int] = {}
    rows = []
    for p in picks:
        seen[p.kind] = n = seen.get(p.kind, 0) + 1
        rows.append(_row(p, n, runs.loc[p.episode["run_id"]], config_ref))
    return pd.DataFrame(rows, columns=list(CASE_COLUMNS))


def write_cases(cases: pd.DataFrame, root: Path = OUTPUTS) -> list[Path]:
    """``analysis/<budget>/cases.csv`` and the per-method copies under ``root``.

    The budget table keeps the rows of the methods ``cases`` does not hold, so a
    per-method loop builds it up; the passed methods' rows are replaced.
    """
    paths = []
    for budget, b in cases.groupby("budget", sort=True):
        table = root / "analysis" / str(budget) / "cases.csv"
        old = (
            pd.read_csv(table, keep_default_na=False) if table.is_file() else b.iloc[:0]
        )
        kept = old[~old["method"].isin(b["method"])]
        merged = pd.concat([kept, b[list(CASE_COLUMNS)]], ignore_index=True)
        groups = [(table, merged.sort_values("method", kind="stable"))]
        groups += [
            (table.parent / str(m) / "cases.csv", g) for m, g in b.groupby("method")
        ]
        for path, rows in groups:
            path.parent.mkdir(parents=True, exist_ok=True)
            rows[list(CASE_COLUMNS)].to_csv(path, index=False)
            paths.append(path)
    return paths


def run_dirs(run_ids: Iterable[str], tracking_uri: str) -> dict[str, Path]:
    """Artifact directory of each run (``mlruns/<exp>/<run_id>/artifacts``).

    Reads the ``runs`` table of a SQLite store read-only (R85) in one short query, so
    no read transaction outlives the call (R103). ``?timeout=<s>`` in the URI sets the
    busy timeout (default `STORE_TIMEOUT_S`); the query is not part of the file name.

    Raises:
        ValueError: If the store is not a SQLite file.
        KeyError: If a run is not in the store.
    """
    # imported here: tracking pulls in the experiment config and reproducibility code
    from ..orchestration.tracking import artifact_path

    base, _, query = tracking_uri.partition("?")
    if not base.startswith("sqlite:///"):
        msg = f"run_dirs reads a sqlite:/// store, got {tracking_uri}"
        raise ValueError(msg)
    timeout = parse_qs(query).get("timeout", [str(STORE_TIMEOUT_S)])[0]
    ids = list(dict.fromkeys(run_ids))
    db = base.removeprefix("sqlite:///")
    sql = "select run_uuid, artifact_uri from runs where run_uuid in ({})"
    with closing(
        sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=float(timeout))
    ) as con:
        found = dict(con.execute(sql.format(",".join("?" * len(ids))), ids).fetchall())
    if missing := [r for r in ids if r not in found]:
        msg = f"runs not in the store: {', '.join(missing)}"
        raise KeyError(msg)
    return {r: artifact_path(found[r]) for r in ids}


def _open(
    case: Mapping[str, Any], run: "EvalRun | Path | None", tracking_uri: str | None
) -> "EvalRun":
    """The case's run, opened once by the caller or here.

    Raises:
        ValueError: Without ``run`` or ``tracking_uri``, or for a run of another method.
    """
    from ..visualisation.eval_rollouts import EvalRun  # JAX: see the module docstring

    if not isinstance(run, EvalRun):
        if run is None and tracking_uri is None:
            msg = (
                f"case {case['id']}: pass its run (or run directory) or a tracking_uri"
            )
            raise ValueError(msg)
        rid = str(case["run_id"])
        run = EvalRun.load(
            Path(run) if run else run_dirs([rid], str(tracking_uri))[rid]
        )
    if run.method != case["method"]:
        msg = f"case {case['id']} is of {case['method']}, the run of {run.method}"
        raise ValueError(msg)
    return run


def _gif_row(case: Mapping[str, Any], fps: int) -> dict[str, str]:
    """`cuts.gif`'s row: the last `GIF_SECONDS` of the window, so the peak stays in."""
    speed, end = float(case["speed"]), int(case["end"])
    start = max(int(case["start"]), end - int(GIF_SECONDS * fps * speed) + 1)
    return {
        "speed": f"{speed:g}",
        "start": str(start),
        "end": str(end),
        "game": str(case["game"]),
        "caption": str(case["caption"]),
    }


def _rel(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` (the contract's file columns).

    Raises:
        ValueError: If ``path`` is outside ``root``.
    """
    return str(path.relative_to(root))


def export_case(
    case: Mapping[str, Any],
    out_dir: Path,
    run: "EvalRun | Path | None" = None,
    *,
    tracking_uri: str | None = None,
    root: Path = OUTPUTS,
    overlay_kw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a case: rollout, 16 px master, GIF and (stall) internals clip.

    ``run`` is the case's `EvalRun` (pass one to share its compiled replay across
    cases), its directory, or None to look the directory up in ``tracking_uri``.
    Files under ``out_dir``: ``rollouts/case-<id>.npz`` (`eval_rollouts.as_rollout`),
    copied flat to ``<root>/visualisation/runs/`` for the video glob;
    ``masters/case-<id>.mkv`` (the whole episode, frame = step); ``gifs/<id>.gif``
    (`cuts.gif`, at most `GIF_SECONDS`); for a stall ``overlays/<id>.mp4``
    (`render_overlay` of the `OVERLAY_STEPS` steps before the unlock; ``overlay_kw``
    sets its sizes).

    Returns:
        The case with ``rollout`` (the flat copy), ``master``, ``gif`` and ``overlay``
        relative to ``root``; ``overlay`` stays "" for other kinds.

    Raises:
        ValueError: If the master's replay diverges from the stored episode, or
            ``out_dir`` is outside ``root``.
    """
    # imported here: replay and rendering pull in JAX (module docstring)
    from ..visualisation import cuts, overlay
    from ..visualisation.eval_rollouts import as_rollout
    from ..visualisation.render import render_master
    from ..visualisation.rollout import DEFAULT_RUNS

    cid, update, seed = str(case["id"]), int(case["update"]), int(case["seed"])
    paths = {
        "rollout": root / DEFAULT_RUNS.relative_to(OUTPUTS) / f"case-{cid}.npz",
        "master": out_dir / "masters" / f"case-{cid}.mkv",
        "gif": out_dir / "gifs" / f"{cid}.gif",
    }
    if case["kind"] == "stall":
        paths["overlay"] = out_dir / "overlays" / f"{cid}.mp4"
    files = {k: _rel(p, root) for k, p in paths.items()}  # raises before any work
    opened = _open(case, run, tracking_uri)
    rollout = as_rollout(opened, update, seed, case["mode"] == "greedy")
    kept = rollout.save(out_dir / "rollouts" / f"case-{cid}.npz")
    paths["rollout"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kept, paths["rollout"])
    if not render_master(rollout, paths["master"]).features_match:
        msg = f"case {cid}: the replay diverged from the stored episode"
        raise ValueError(msg)
    cuts.gif(_gif_row(case, cuts.FPS_MASTER), paths["master"], paths["gif"])
    if "overlay" in paths:
        peak = int(case["peak"])
        paths["overlay"].parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:  # its work files stay in tmp
            clip = overlay.render_overlay(
                opened.dir,
                update,
                seed,
                (max(0, peak - OVERLAY_STEPS), peak),
                out=Path(tmp) / f"{cid}.mp4",
                mode=case["mode"],
                **(overlay_kw or {}),
            )
            shutil.move(clip, paths["overlay"])
    return {**dict(case), "overlay": "", **files}


@functools.cache
def _seen_scan(game: Game) -> Callable[..., Any]:
    """Jitted ``(states, n) -> seen``: `coverage.update_seen` over n states."""
    import jax  # JAX: see the module docstring

    from ..evaluation.coverage import init_seen, update_seen

    def scan(states: Any, n: Any) -> Any:
        rest = jax.tree.map(lambda x: x[1:], states)
        steps = jax.numpy.arange(1, jax.tree.leaves(states)[0].shape[0])

        def step(seen: Any, x: tuple[Any, Any]) -> tuple[Any, None]:
            return update_seen(seen, x[1], game, x[0] < n), None

        first = init_seen(jax.tree.map(lambda x: x[0], states), game)
        return jax.lax.scan(step, first, (steps, rest))[0]

    return jax.jit(scan)


def _seen(states: Any, game: Game, chunk: int) -> np.ndarray:
    """First-seen timestep per tile (`coverage`), states padded to a ``chunk`` multiple.

    The padding repeats the last state and is masked, so one compile serves every
    episode of a length bucket.
    """
    import jax  # JAX: see the module docstring

    from ..orchestration import device_roles

    n = len(jax.tree.leaves(states)[0])
    pad = -(-n // chunk) * chunk - n
    padded = jax.tree.map(
        lambda x: np.concatenate([x, np.repeat(x[-1:], pad, 0)]), states
    )
    with jax.default_device(device_roles().actor):
        return np.asarray(_seen_scan(game)(padded, n))


def _histogram(
    actions: np.ndarray, forced: np.ndarray, names: tuple[str, ...]
) -> dict[str, int]:
    """Counts of the actions not forced, by name, in action order; zeros left out."""
    counts = np.bincount(actions[~forced], minlength=len(names))
    return {names[a]: int(k) for a, k in enumerate(counts) if k}


def _close_calls_in(health: np.ndarray, a: int, p: int) -> int:
    """Close calls of the episode (`episode_records` rule) opening at frames ``a..p``.

    ``health`` is per action (frame ``t + 1``). Openings depend only on the falls
    before them, and one counts once ``CLOSE_STEPS`` more steps follow, so the rule
    on a prefix ending ``CLOSE_STEPS`` after frame ``x`` counts the openings up to
    ``x``; the window is the difference of two such prefixes.
    """
    # the record's own rule, not a copy: counts match the exported close_calls
    from ..evaluation.episode_metrics import CLOSE_STEPS
    from ..evaluation.episode_records import _close_calls

    n = len(health)
    upto = [_close_calls(health[: min(x + CLOSE_STEPS, n)])[0] for x in (a - 1, p)]
    return upto[1] - upto[0]


def describe_case(
    case: Mapping[str, Any],
    run: "EvalRun | Path | None" = None,
    *,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """The facts of a case for the write-up, from one replay of its episode.

    The window runs from ``start`` to ``peak`` (for a stall, the stall up to the
    unlock). Keys: ``floor``, ``stats`` (health, food, drink, energy) and ``inventory``
    (`episode_records.resources`) at ``start`` and at ``peak`` (the frame that shows
    the unlock); ``actions``: counts of the steps not forced in the window and in the
    episode; ``coverage_gained``: per floor visited in the window, the share of its
    tiles first seen inside it (`evaluation.coverage`); ``last_actions``: the
    `LAST_ACTIONS` actions before the peak; ``combat_damage`` (needs drain and
    potions excluded, as the record's ``damage_taken``), ``close_calls`` (the
    `episode_records` rule: opened at a frame of the window, with `CLOSE_STEPS` more
    steps in the episode) and ``min_health`` inside the window. ``run`` as in
    `export_case`.
    """
    import jax  # JAX: see the module docstring

    from ..evaluation.episode_metrics import START_HEALTH
    from ..evaluation.episode_records import resources, trace_of
    from ..evaluation.milestones import action_names
    from ..visualisation.eval_rollouts import CHUNK_STEPS, stored_episode

    opened = _open(case, run, tracking_uri)
    greedy = case["mode"] == "greedy"
    ep, actions = stored_episode(opened, int(case["update"]), int(case["seed"]), greedy)
    game, states = opened.game, jax.device_get(ep.states)
    trace, names = trace_of(states, game), action_names(game)
    a, p = int(case["start"]), int(case["peak"])
    at = {"start": a, "peak": p}
    seen = _seen(states, game, CHUNK_STEPS)
    gained = (seen > a) & (seen <= p)
    events = jax.tree.map(np.asarray, ep.events)
    health = START_HEALTH + np.cumsum(events.health_delta, dtype=np.float64)
    return {
        **{
            k: _plain(case[k])
            for k in ("id", "kind", "run_id", "update", "mode", "seed")
        },
        **{k: int(case[k]) for k in ("start", "peak", "end")},
        "length": len(actions),
        "floor": {w: int(trace.floor[i]) for w, i in at.items()},
        "stats": {
            w: dict(zip(STATS, map(float, trace.intrinsics[i]), strict=True))
            for w, i in at.items()
        },
        "inventory": {
            w: dict(zip(resources(game), map(int, trace.inventory[i]), strict=True))
            for w, i in at.items()
        },
        "actions": {
            "window": _histogram(actions[a:p], events.forced[a:p], names),
            "episode": _histogram(actions, events.forced, names),
        },
        "coverage_gained": {
            FLOORS[f][0]: float(gained[f].mean())
            for f in np.unique(trace.floor[a : p + 1])
        },
        "last_actions": [names[x] for x in actions[max(0, p - LAST_ACTIONS) : p]],
        "combat_damage": float(events.combat_damage[a:p].sum(dtype=np.float64)),
        "close_calls": _close_calls_in(health, a, p),
        "min_health": float(trace.intrinsics[a : p + 1, 0].min()),
    }


def _plain(value: Any) -> Any:
    """A numpy scalar as its Python value (JSON-ready facts)."""
    return value.item() if isinstance(value, np.generic) else value
