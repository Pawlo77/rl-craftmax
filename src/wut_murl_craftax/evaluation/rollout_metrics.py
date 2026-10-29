"""Offline metrics of stored eval episodes: every seed of an eval file replayed at once.

`replay_file` replays ``eval/<mode>/<update>.npz`` from seed + actions with
`EvalRun.env_step` (64 seeds vmapped, split over the CPU devices, jitted `CHUNK`-step
scans, each seed to its stored ``length``), keeping per step the `Trace` row, the
capped-inventory flag and, for skill policies, the decision key and skill end reason;
`update_seen` runs inside. Events or skill ends that differ from the file raise
`ReplayMismatchError`; ``recompute`` takes the replayed events and lists the
differences. `file_tables` builds the eval-episode rows and the TA4 frames (contracts.md
sections 1, 9), `process_file` writes and logs them. CLI: ``python -m
wut_murl_craftax.evaluation.rollout_metrics --help`` (``--recompute-events``).
"""

import argparse
import functools
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..analysis.obstacles import INVENTORY_COLUMNS, SKILL_ENDS_COLUMNS
from ..orchestration.runtime import configure_runtime
from ..training.progress_key import CAPS
from .coverage import coverage_summary, init_seen, update_seen
from .episode_records import Trace, episode_record, trace_of
from .events import StepEvents
from .hardware import open_store
from .milestones import Game

if TYPE_CHECKING:  # the CLI splits the CPU first; these imports start JAX's backend
    from ..training.smdp import SMDPConfig
    from ..visualisation.eval_rollouts import EvalRun

logger = logging.getLogger(__name__)

CHUNK = 256
"""Steps per jitted replay scan; every chunk of a file has this one shape."""
MODES = ("stochastic", "greedy")
OUT_ROOT = Path("outputs/metrics")
"""Tables go to ``<OUT_ROOT>/<run_id>/``; EV10 concatenates them per phase."""
TABLES_ARTIFACT = "eval_tables"
"""Artifact folder of the run that receives the tables."""
IDENTITY = (
    "run_id", "method", "config_ref", "budget", "seed", "update", "env_steps", "mode",
    "eval_seed",
)  # fmt: skip
"""Leading columns of ``eval_episodes``. ``config_ref`` depends on the exported phase
(a run can be in a tuning and a final phase), so the per-run files leave it out and
the phase export inserts it (`RUN_IDENTITY`)."""
RUN_IDENTITY = tuple(c for c in IDENTITY if c != "config_ref")
"""Leading columns of the per-run eval tables written here."""
EPISODE_COLUMNS = (
    *IDENTITY, "length", "return", "score_pct", "end_type", "cause_of_death",
    "max_floor", "tech_depth", "n_achievements", "ach/", "unlock_step/", "stalls",
    "n_stalls", "max_stall", "fights", "kills/", "kills/passive", "damage_dealt",
    "damage_taken", "projectile_damage", "close_calls", "close_calls_recovered",
    "coverage/", "new_tiles_per_100", "revisit_rate", "unique_positions", "low_share/",
    "zero_share/", "survival/count/", "survival/gap/", "survival/level/",
    "action_count/", "action_entropy", "wasted_share", "noop_share", "forced_share",
    "gathered/", "used/", "use_ratio", "steps_per_achievement", "wasted_crafts",
    "wood_to_diamond", "adv/", "split/",
)  # fmt: skip
"""Columns of ``eval_episodes`` in contracts.md section 1 order; a name ending in
``/`` is a family the game expands (`episode_columns`; ``kills/`` holds the hostile
creatures, ``kills/passive`` follows them). `SKILL_ENDS_COLUMNS` and
`INVENTORY_COLUMNS` (section 9) are the training analysis's own constants."""
HOARD_CAPS = {m: c for m, c in CAPS.items() if c > 1}
"""Materials whose `progress_key.CAPS` cap is above 1, the ones ``capped_steps``
counts (R98): holding one sapling, gem, arrow, torch, book or potion is no hoarding."""
SUMMARY_PREFIXES = ("coverage/", "gathered/", "used/", "survival/level/")
SUMMARY_COLUMNS = (
    "new_tiles_per_100", "revisit_rate", "unique_positions", "use_ratio",
    "wasted_crafts",
)  # fmt: skip
"""Columns only the replay produces; their episode means go to ``eval/<mode>/``."""
_STALL = pa.struct([
    ("start", pa.int64()), ("end", pa.int64()), ("length", pa.int64()),
    ("floor", pa.int64()), ("ending", pa.string()), ("ended_by", pa.string()),
    ("actions", pa.list_(pa.int64())), ("intrinsics", pa.list_(pa.float64())),
    ("inventory", pa.list_(pa.int64())),
])  # fmt: skip
_FIGHT = pa.struct([
    ("start", pa.int64()), ("end", pa.int64()), ("max_enemies", pa.int64()),
    ("damage_taken", pa.float64()), ("outcome", pa.string()),
])  # fmt: skip
_TYPES = {"stalls": pa.list_(_STALL), "fights": pa.list_(_FIGHT)}
"""Arrow types numpy cannot infer: an all-empty column would be typed null and refuse
to concatenate with the other files (the R81 bug class)."""
_TEXT = {"skill": pa.string(), "reason": pa.string(), "mode": pa.string()}
MISMATCH_COLUMNS = (
    "run_id", "update", "mode", "eval_seed", "field", "steps", "first_step", "old",
    "new",
)  # fmt: skip
"""Columns of ``event_mismatches_<update>_<mode>.csv``: a row per seed and `StepEvents`
field the replay computes differently; ``steps`` of the episode differ, the first at
``first_step``; ``old`` and ``new`` are the episode sums of the stored and replayed
values."""


class ReplayMismatchError(RuntimeError):
    """A replay disagrees with what the eval file stored."""


class Identity(NamedTuple):
    """Store facts of a run that its directory lacks (tags)."""

    run_id: str
    budget: str
    seed: int


class FileReplay(NamedTuple):
    """Replay of every seed of an eval file; arrays lead with the seed."""

    traces: list[Trace]
    """Per seed, ``length + 1`` rows."""
    seen: np.ndarray
    """`update_seen` after each seed's last state."""
    capped: list[np.ndarray]
    """Per seed, whether each state before an action has a `HOARD_CAPS` material at
    its cap (`at_cap`)."""
    keys: np.ndarray | None
    """Skill policies: `smdp.top_key` of the observation at each step."""
    reasons: np.ndarray | None
    """Skill policies: `skills.REASONS` index of the running skill after each step."""
    events: StepEvents
    """The replayed events, seeds first (numpy); equal to the stored ones unless
    ``recompute``."""
    mismatches: list[dict[str, Any]]
    """With ``recompute``: a row of `MISMATCH_COLUMNS` after ``run_id``, ``update`` and
    ``mode`` for every seed and field whose replayed events differ; else empty."""


class Tables(NamedTuple):
    """The frames of one eval file."""

    episodes: pd.DataFrame
    skill_ends: pd.DataFrame | None
    """None for policies without skills."""
    inventory: pd.DataFrame
    mismatches: pd.DataFrame
    """`MISMATCH_COLUMNS`; empty unless the replay recomputed differing events."""


def at_cap(s: Any) -> jax.Array:
    """A `HOARD_CAPS` material of ``s.inventory`` at or above its cap (R98)."""
    inv = s.inventory
    hits = [
        jnp.sum(getattr(inv, m)) >= c for m, c in HOARD_CAPS.items() if hasattr(inv, m)
    ]
    return jnp.any(jnp.stack(hits))


def _row(s: Any, game: Game) -> Trace:
    """The `Trace` row of one raw state, traced."""
    one = trace_of(jax.tree.map(lambda x: x[None], s), game, xp=jnp)
    return Trace(*(x[0] for x in one))


class Replayer:
    """Jitted replay of one run's eval files (a compile per run, then per file)."""

    def __init__(self, run: "EvalRun", chunk: int = CHUNK) -> None:
        """Build the jitted start and chunk functions of ``run``."""
        from ..training.smdp import SMDPPolicy

        self.run, self.chunk = run, chunk
        self.skills = run.policy.cfg if isinstance(run.policy, SMDPPolicy) else None
        self._start = jax.jit(jax.vmap(self._seed_start))
        self._scan = jax.jit(jax.vmap(self._seed_chunk, (0, 0, 0, None, 0, 0, 0)))

    def _seed_start(self, seed: jax.Array) -> Any:
        from ..training.wrappers import unwrap_state
        from ..visualisation.rollout import keys

        run = self.run
        obs, st = run.env.reset(keys(seed)[0], run.env_params)  # type: ignore[arg-type]  # traced seed
        s0 = unwrap_state(st)
        return (obs, st, init_seen(s0, run.game)), (_row(s0, run.game), at_cap(s0))

    def _skill_end(self, x: tuple[Any, ...]) -> dict[str, jax.Array]:
        """Key at the observation and end reason of the step, as `SMDPPolicy.step`."""
        from ..training.skills import decode_obs, skill_step
        from ..training.smdp import top_key

        cfg, (obs, a, r, obs2, skill, steps) = cast("SMDPConfig", self.skills), x
        view, view2 = decode_obs(obs), decode_obs(obs2)
        key = top_key(view, cfg.granularity, cfg.log2_rows, low_flags=cfg.low_flags)
        reason = skill_step(
            skill, view, a, r, view2, False, steps, cfg.timeout_mult,
            granularity=cfg.granularity, low_interrupt=cfg.low_flags,
        )[2]  # fmt: skip
        return {"key": key, "reason": reason.astype(jnp.int8)}

    def _seed_chunk(
        self,
        carry: Any,
        seed: jax.Array,
        actions: jax.Array,
        t0: jax.Array,
        length: jax.Array,
        skill: jax.Array,
        steps: jax.Array,
    ) -> Any:
        from ..training.wrappers import unwrap_state
        from ..visualisation.rollout import keys

        run, game = self.run, self.run.game
        k_env = keys(seed)[1]  # type: ignore[arg-type]  # traced seed

        def step(c: Any, x: Any) -> Any:
            obs, st, seen = c
            t, a, sk, n = x
            obs2, st2, r, _, ev = run.env_step(st, a, jax.random.fold_in(k_env, t))
            s2 = unwrap_state(st2)
            out = {"events": ev, "trace": _row(s2, game), "capped": at_cap(s2)}
            if self.skills is not None:
                out |= self._skill_end((obs, a, r, obs2, sk, n))
            seen = update_seen(seen, s2, game, t < length)  # past the end: unchanged
            return (obs2, st2, seen), out

        xs = (t0 + jnp.arange(actions.shape[0]), actions, skill, steps)
        return jax.lax.scan(step, carry, xs)


def _place(tree: Any) -> Any:
    """``tree`` split over the CPU devices when its seeds divide evenly."""
    from ..orchestration import env_mesh, shard_envs

    cpus = jax.devices("cpu")
    n = jax.tree.leaves(tree)[0].shape[0]
    return (
        shard_envs(tree, env_mesh()) if len(cpus) > 1 and n % len(cpus) == 0 else tree
    )


def _padded(x: np.ndarray, width: int, dtype: Any) -> np.ndarray:
    """``x`` (seeds, T) clipped at 0 and zero-padded to ``width`` columns."""
    out = np.zeros((x.shape[0], width), dtype)
    out[:, : min(width, x.shape[1])] = np.maximum(x[:, :width], 0)
    return out


def _run_chunks(rp: Replayer, arrays: Mapping[str, np.ndarray]) -> tuple[Any, ...]:
    """Start state rows, stacked per-step outputs and the final carry, on the host."""
    from ..orchestration.runtime import device_roles

    seeds, length = np.asarray(arrays["seeds"]), np.asarray(arrays["length"])
    width = -(-max(int(length.max()), 1) // rp.chunk) * rp.chunk
    skill = rp.skills is not None
    inputs = (
        _padded(np.asarray(arrays["actions"]), width, np.int32),
        _padded(arrays["extras/skill"], width, np.int32) if skill else None,
        _padded(arrays["extras/steps_in_skill"], width, np.int32) if skill else None,
    )
    zeros = np.zeros_like(inputs[0])
    acts, sk, st = (zeros if x is None else x for x in inputs)
    outs = []
    with jax.default_device(device_roles().actor):
        seed_arr, n_arr = _place((jnp.asarray(seeds, jnp.int32), jnp.asarray(length)))
        carry, first = rp._start(seed_arr)
        for t0 in range(0, width, rp.chunk):
            part = _place(
                tuple(jnp.asarray(x[:, t0 : t0 + rp.chunk]) for x in (acts, sk, st))
            )
            carry, out = rp._scan(carry, seed_arr, part[0], t0, n_arr, *part[1:])
            outs.append(jax.device_get(out))
        first, seen = jax.device_get((first, carry[2]))
    out = jax.tree.map(lambda *xs: np.concatenate(xs, 1), *outs)
    return first, out, np.asarray(seen)


def _event_diffs(
    arrays: Mapping[str, np.ndarray], replayed: Any
) -> list[dict[str, Any]]:
    """A row per seed and field whose replayed events differ from the stored ones.

    Compared up to each seed's ``length``; the row holds ``eval_seed``, ``field``,
    ``steps`` (how many differ), ``first_step`` and the episode sums ``old`` (stored)
    and ``new`` (replayed).
    """
    rows = []
    for i, (seed, n) in enumerate(zip(arrays["seeds"], arrays["length"], strict=True)):
        for f in fields(StepEvents):
            old = np.asarray(arrays[f"events/{f.name}"])[i, :n]
            new = np.asarray(getattr(replayed, f.name))[i, :n]
            differ = (old != new).reshape(len(old), -1).any(-1) if n else old[:0]
            if differ.any():
                rows.append({
                    "eval_seed": int(seed),
                    "field": f.name,
                    "steps": int(differ.sum()),
                    "first_step": int(differ.argmax()),
                    "old": float(old.sum(dtype=np.float64)),
                    "new": float(new.sum(dtype=np.float64)),
                })
    return rows


def _check_events(diffs: Sequence[Mapping[str, Any]], where: str) -> None:
    """Raise `ReplayMismatchError` for the first seed with a differing field."""
    if diffs:
        seed = diffs[0]["eval_seed"]
        bad = [d["field"] for d in diffs if d["eval_seed"] == seed]
        msg = f"replay of {where} seed {seed} differs from the stored {bad}"
        raise ReplayMismatchError(msg)


def _check_skills(
    arrays: Mapping[str, np.ndarray], reasons: np.ndarray, where: str
) -> None:
    """Raise unless a replayed skill end comes exactly before each stored new skill."""
    from ..training.skills import RUNNING

    steps = np.asarray(arrays["extras/steps_in_skill"])
    for i, (seed, n) in enumerate(zip(arrays["seeds"], arrays["length"], strict=True)):
        starts = steps[i, :n] == 0
        ends = reasons[i, : n - 1] != RUNNING
        if n and (not starts[0] or not np.array_equal(starts[1:], ends)):
            at = int(np.flatnonzero(np.r_[~starts[:1], starts[1:] != ends])[0])
            msg = (
                f"replay of {where} seed {int(seed)} ends skills where the stored "
                f"extras do not (step {at})"
            )
            raise ReplayMismatchError(msg)


def replay_file(
    rp: Replayer,
    arrays: Mapping[str, np.ndarray],
    where: str = "the eval file",
    recompute: bool = False,
) -> FileReplay:
    """Replay every seed of an eval file's arrays and check them against the file.

    ``recompute`` accepts events that differ (the code that computes them changed since
    the file was written): they are listed in ``mismatches``, and ``events`` holds the
    replayed ones. Skill ends are checked either way.

    Raises:
        ReplayMismatchError: If a seed's replayed events (unless ``recompute``) or
            skill ends differ from the stored ones; the message names ``where`` (run,
            update, mode) and seed.
    """
    first, out, seen = _run_chunks(rp, arrays)
    diffs = _event_diffs(arrays, out["events"])
    if not recompute:
        _check_events(diffs, where)
    length = np.asarray(arrays["length"])
    traces = [
        Trace(
            *(
                np.concatenate([a[i][None], b[i, :n]])
                for a, b in zip(first[0], out["trace"], strict=True)
            )
        )
        for i, n in enumerate(length)
    ]
    capped = [
        np.r_[first[1][i], out["capped"][i, : n - 1]] for i, n in enumerate(length)
    ]
    reasons = out.get("reason")
    if reasons is not None:
        _check_skills(arrays, reasons, where)
    return FileReplay(
        traces, seen, capped, out.get("key"), reasons, out["events"], diffs
    )


def pair_counts(run_dir: Path, update: int) -> dict[tuple[int, str], int] | None:
    """(key, skill) -> real outcomes in the Dyna-Q model graph of ``update``.

    None without a graph (not a Dyna-Q run); {} for a checkpoint without pairs.
    """
    from ..training.dyna_q import MODEL_GRAPH_FILE

    path = Path(run_dir) / "eval" / MODEL_GRAPH_FILE
    if not path.is_file():
        return None
    g = pq.read_table(path, columns=["update", "key", "skill", "count"]).to_pandas()
    g = g[g["update"] == update]
    return {
        (int(k), str(s)): int(c)
        for k, s, c in zip(g["key"], g["skill"], g["count"], strict=True)
    }


def _skill_ends(
    arrays: Mapping[str, np.ndarray],
    keys: np.ndarray,
    reasons: np.ndarray,
    head: Mapping[str, Any],
    pairs: dict[tuple[int, str], int] | None,
) -> pd.DataFrame:
    """One row per skill of each episode: a skill runs from its decision to the next."""
    from ..training.skills import EPISODE_END, REASONS, SKILLS

    names = np.array([s.name for s in SKILLS], object)
    parts = []
    for i, (seed, n) in enumerate(zip(arrays["seeds"], arrays["length"], strict=True)):
        start = np.flatnonzero(np.asarray(arrays["extras/steps_in_skill"])[i, :n] == 0)
        end = np.r_[start[1:], n]
        why = np.r_[reasons[i, end[:-1] - 1], EPISODE_END].astype(int)
        parts.append(
            pd.DataFrame({
                "eval_seed": int(seed),
                "start": start,
                "end": end,
                "skill": names[np.asarray(arrays["extras/skill"])[i, start]],
                "key": keys[i, start].astype(np.int64),
                "reason": np.array(REASONS, object)[why],
            })
        )
    df = pd.concat(parts, ignore_index=True)
    for k, v in head.items():
        df[k] = v
    df = df[list(SKILL_ENDS_COLUMNS)]
    if pairs is not None:
        got = zip(df["key"], df["skill"], strict=True)
        df["pair_count"] = np.array(
            [pairs.get((int(k), s), 0) for k, s in got], np.int64
        )
    return df


@functools.cache
def episode_columns(game: Game) -> tuple[str, ...]:
    """`EPISODE_COLUMNS` with each family expanded to the names of ``game``."""
    from .episode_metrics import STATS, creatures
    from .episode_records import HOSTILE, milestone_ids, resources
    from .milestones import ADVANCED_ACTIONS, FLOORS, achievement_names, action_names

    ach = achievement_names(game)
    floors = len(FLOORS) if game == "craftax" else 1
    adv = [a.lower() for a in ADVANCED_ACTIONS] if game == "craftax" else []
    res, stats = resources(game), (
        "low_share/", "zero_share/", "survival/count/", "survival/gap/",
        "survival/level/",
    )  # fmt: skip
    names: dict[str, Sequence[str]] = {
        "ach/": ach,
        "unlock_step/": ach,
        "kills/": [c for _, c in creatures(game, HOSTILE)],
        "coverage/": [f for f, _ in FLOORS[:floors]],
        "action_count/": action_names(game),
        "gathered/": res,
        "used/": res,
        "adv/": adv,
        "split/": [ach[a] for a in milestone_ids(game)],
    } | dict.fromkeys(stats, STATS)
    return tuple(f"{c}{n}" for c in EPISODE_COLUMNS for n in names.get(c, [""]))


def file_tables(
    rp: Replayer,
    arrays: Mapping[str, np.ndarray],
    update: int,
    mode: str,
    ident: Identity,
    recompute: bool = False,
) -> Tables:
    """The eval-episode rows, ``skill_ends`` and ``eval_inventory`` of one eval file.

    With ``recompute`` the rows come from the replayed events and ``mismatches`` lists
    where they differ from the stored ones (`replay_file`).

    Raises:
        ReplayMismatchError: If the replay disagrees with the file (`replay_file`).
    """
    run = rp.run
    where = f"run {ident.run_id} update {update} mode {mode}"
    if "extras/skill" in arrays and rp.skills is None:  # a wrapper of the skill policy
        msg = (
            f"{where} has extras/skill but {run.method}'s policy is not a skills policy"
        )
        raise ReplayMismatchError(msg)
    fr = replay_file(rp, arrays, where, recompute)
    actions, steps = np.asarray(arrays["actions"]), run.env_steps(update)
    names = [f.name for f in fields(StepEvents)]
    rows, inv = [], []
    for i, (seed, n) in enumerate(zip(arrays["seeds"], arrays["length"], strict=True)):
        ev = StepEvents(**{k: np.asarray(getattr(fr.events, k))[i, :n] for k in names})  # type: ignore[arg-type]  # numpy rows
        trace = fr.traces[i]
        record = episode_record(ev, actions[i, :n], trace, game=run.game)
        head = {
            "run_id": ident.run_id, "method": run.method, "budget": ident.budget,
            "seed": ident.seed, "update": update, "env_steps": steps, "mode": mode,
            "eval_seed": int(seed),
        }  # fmt: skip
        cover = coverage_summary(fr.seen[i], trace.position, trace.floor)
        rows.append(head | record.row() | cover)
        inv.append((int(seed), int(n), int(fr.capped[i].sum())))
    ends = None
    if fr.keys is not None and fr.reasons is not None:
        head = {
            "run_id": ident.run_id,
            "update": update,
            "env_steps": steps,
            "mode": mode,
        }
        pairs = pair_counts(run.dir, update)
        ends = _skill_ends(arrays, fr.keys, fr.reasons, head, pairs)
    inventory = pd.DataFrame(inv, columns=["eval_seed", "steps", "capped_steps"])
    for k, v in (("run_id", ident.run_id), ("update", update), ("mode", mode)):
        inventory[k] = v
    episodes = pd.DataFrame(rows)
    order = [
        c for c in episode_columns(run.game) if c in RUN_IDENTITY or c not in IDENTITY
    ]
    if set(order) != set(episodes.columns):
        odd = set(order) ^ set(episodes.columns)
        msg = f"eval-episode columns differ from EPISODE_COLUMNS: {sorted(odd)}"
        raise ValueError(msg)
    head = {"run_id": ident.run_id, "update": update, "mode": mode}
    wrong = pd.DataFrame([head | d for d in fr.mismatches], columns=MISMATCH_COLUMNS)
    return Tables(episodes[order], ends, inventory[list(INVENTORY_COLUMNS)], wrong)


def out_paths(out: Path, run_id: str, update: int, mode: str) -> dict[str, Path]:
    """Files of one (run, update, mode); ``episodes`` is written last (done marker)."""
    d = Path(out) / run_id
    return {
        "skill_ends": d / f"skill_ends_{update}_{mode}.parquet",
        "inventory": d / f"inventory_{update}_{mode}.parquet",
        "episodes": d / f"eval_{update}_{mode}.parquet",
    }


def mismatch_path(out: Path, run_id: str, update: int, mode: str) -> Path:
    """The event-mismatch CSV of (run, update, mode); written only if events differ."""
    return Path(out) / run_id / f"event_mismatches_{update}_{mode}.csv"


def arrow_table(df: pd.DataFrame) -> pa.Table:
    """``df`` as an Arrow table with the fixed types of `_TYPES` and `_TEXT`."""
    typed = {k: v for k, v in (_TYPES | _TEXT).items() if k in df.columns}
    t = pa.Table.from_pandas(df.drop(columns=list(typed)), preserve_index=False)
    at = sorted((df.columns.get_loc(k), k) for k in typed)  # ascending: indices hold
    for i, name in at:
        t = t.add_column(i, name, pa.array(df[name].tolist(), typed[name]))
    return t


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically: a file that exists is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(arrow_table(df), tmp)
    tmp.replace(path)


def summaries(episodes: pd.DataFrame, mode: str) -> dict[str, float]:
    """``eval/<mode>/<column>`` -> episode mean of each replay-only column (no NaN)."""
    cols = [
        c
        for c in episodes.columns
        if c.startswith(SUMMARY_PREFIXES) or c in SUMMARY_COLUMNS
    ]
    means = episodes[cols].astype(float).mean()
    return {f"eval/{mode}/{c}": float(v) for c, v in means.items() if not math.isnan(v)}


def _marker(paths: Mapping[str, Path]) -> Path:
    return paths["episodes"].with_suffix(".logged")


def pending(out: Path, run_id: str, update: int, mode: str, log: bool) -> bool:
    """Whether (run, update, mode) still needs its tables, or its logging."""
    paths = out_paths(out, run_id, update, mode)
    return not paths["episodes"].is_file() or (log and not _marker(paths).is_file())


def _log(
    client: Any,
    run_id: str,
    files: Iterable[Path],
    metrics: Mapping[str, float],
    step: int,
    recover: bool,
) -> None:
    """Tables under `TABLES_ARTIFACT`, summaries at ``step`` (checkpoint steps).

    ``recover`` (a cut earlier call) skips the summaries already logged at ``step``;
    the artifacts are overwritten.
    """
    from mlflow.entities import Metric

    for f in files:
        client.log_artifact(run_id, str(f), TABLES_ARTIFACT)
    if recover:
        metrics = {
            k: v
            for k, v in metrics.items()
            if all(m.step != step for m in client.get_metric_history(run_id, k))
        }
    now = int(time.time() * 1000)
    client.log_batch(
        run_id, metrics=[Metric(k, v, now, step) for k, v in metrics.items()]
    )


def process_file(
    rp: Replayer,
    ident: Identity,
    update: int,
    mode: str,
    out: Path = OUT_ROOT,
    client: Any = None,
    recompute: bool = False,
) -> bool:
    """Write the tables of one eval file and log them to the run; False if done.

    ``recompute`` is `file_tables`'s: differing events go to `mismatch_path`, written
    before the tables. Idempotent: existing tables are not recomputed (so recompute
    into a fresh ``out``). A ``.logging`` marker is written
    before the logging and replaced by ``.logged`` after it, so a rerun never logs
    twice: after a cut it logs only the summaries the store lacks.
    """
    from ..visualisation.eval_rollouts import load_eval

    if not pending(out, ident.run_id, update, mode, client is not None):
        return False
    paths = out_paths(out, ident.run_id, update, mode)
    if paths["episodes"].is_file():
        episodes = pd.read_parquet(paths["episodes"])
    else:
        arrays = load_eval(rp.run.dir, update, mode == "greedy")
        t = file_tables(rp, arrays, update, mode, ident, recompute)
        if not t.mismatches.empty:
            csv = mismatch_path(out, ident.run_id, update, mode)
            csv.parent.mkdir(parents=True, exist_ok=True)
            t.mismatches.to_csv(csv, index=False)
        if t.skill_ends is not None:
            _write(t.skill_ends, paths["skill_ends"])
        _write(t.inventory, paths["inventory"])
        _write(t.episodes, paths["episodes"])
        episodes = t.episodes
    if client is not None:
        done = _marker(paths)
        cut = done.with_suffix(".logging")
        recover = cut.is_file()
        cut.write_text(f"{ident.run_id} {update} {mode}\n")
        files = [p for p in paths.values() if p.is_file()]
        step = rp.run.env_steps(update)
        _log(client, ident.run_id, files, summaries(episodes, mode), step, recover)
        cut.replace(done)
    return True


class RunRef(NamedTuple):
    """A run of the store: id, name, the tags EV6 needs and its artifact directory."""

    run_id: str
    name: str
    method: str
    budget: str
    seed: int
    dir: Path
    status: str = ""
    """MLflow status."""
    experiment: str = ""
    """Name of the run's experiment (its first phase; context only, R77)."""
    tags: Mapping[str, str] = MappingProxyType({})
    """Every tag of the run."""


def phase_runs(tracking_uri: str, phases: Sequence[str] | None) -> list[RunRef]:
    """Active runs whose tag ``phases`` holds one of ``phases`` (all if None), by name.

    Reads the store read-only in one short connection (tag membership, never the
    experiment name, R77; short reads, R103). A missing ``seed`` tag reads -1.
    """
    from ..orchestration.tracking import artifact_path

    with open_store(tracking_uri) as con:
        rows = con.execute("SELECT run_uuid, key, value FROM tags").fetchall()
        runs = con.execute(
            "SELECT r.run_uuid, r.name, r.status, e.name, r.artifact_uri FROM runs r "
            "JOIN experiments e ON e.experiment_id = r.experiment_id "
            "WHERE r.lifecycle_stage = 'active' ORDER BY r.name"
        ).fetchall()
    tags: dict[str, dict[str, str]] = {}
    for rid, key, value in rows:
        tags.setdefault(rid, {})[key] = value
    wanted = set(phases or ())
    out = []
    for rid, name, status, exp, uri in runs:
        t = tags.get(rid, {})
        if wanted and not wanted & set(t.get("phases", "").split(",")):
            continue
        out.append(
            RunRef(
                rid,
                name,
                t.get("method", ""),
                t.get("budget", ""),
                int(t.get("seed", -1)),
                artifact_path(uri),
                status,
                exp,
                t,
            )
        )
    return out


def _updates(ref: RunRef, mode: str) -> list[int]:
    return sorted(int(p.stem) for p in (ref.dir / "eval" / mode).glob("*.npz"))


def plan_jobs(
    runs: Sequence[RunRef], updates: str | Sequence[int], modes: Sequence[str]
) -> list[tuple[RunRef, int, str]]:
    """(run, update, mode) jobs with an eval file: final checkpoints of every run first.

    ``updates`` is ``final``, ``all`` (the finals, then each run's other checkpoints,
    latest first) or a list of updates.
    """
    have = {(r.run_id, m): _updates(r, m) for r in runs for m in modes}

    def jobs(r: RunRef, us: Iterable[int]) -> list[tuple[RunRef, int, str]]:
        return [(r, u, m) for u in us for m in modes if u in have[r.run_id, m]]

    any_mode = {r.run_id: {u for m in modes for u in have[r.run_id, m]} for r in runs}
    final = {k: sorted(v)[-1:] for k, v in any_mode.items()}  # [] without files
    if not isinstance(updates, str):
        return [j for r in runs for j in jobs(r, sorted(updates, reverse=True))]
    out = [j for r in runs for j in jobs(r, final[r.run_id])]
    if updates == "all":
        for r in runs:
            rest = any_mode[r.run_id] - set(final[r.run_id])
            out += jobs(r, sorted(rest, reverse=True))
    return out


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", action="append", help="tag phases value; repeatable")
    p.add_argument("--run", action="append", help="run id or name; repeatable")
    p.add_argument("--updates", default="all", help="all, final or a comma list")
    p.add_argument("--mode", choices=(*MODES, "both"), default="both")
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--tracking-uri", required=True)
    p.add_argument("--no-log", action="store_true", help="write tables only")
    p.add_argument(
        "--recompute-events",
        action="store_true",
        help="use the replayed events where they differ from the stored ones and list "
        "them in event_mismatches_*.csv, instead of failing the file",
    )
    p.add_argument("--host-devices", type=int, default=8, help="CPU shards of seeds")
    a = p.parse_args(argv)
    if not a.phase and not a.run:
        p.error("give --phase or --run")
    return a


def main(argv: list[str] | None = None) -> None:
    """Replay the eval files of a phase's runs (or of named runs), finals first.

    Raises:
        SystemExit: If some file's replay disagreed with it (the rest still ran).
    """
    a = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    configure_runtime(host_devices=a.host_devices)  # before JAX starts its backend
    from ..orchestration.tracking import resolve_store_uri
    from ..visualisation.eval_rollouts import EvalRun

    runs = phase_runs(a.tracking_uri, a.phase)
    if a.run:
        runs = [r for r in runs if r.run_id in a.run or r.name in a.run]
    modes = MODES if a.mode == "both" else (a.mode,)
    ups = (
        a.updates
        if a.updates in ("all", "final")
        else [int(u) for u in a.updates.split(",")]
    )
    client = None
    if not a.no_log:
        import mlflow

        client = mlflow.MlflowClient(resolve_store_uri(a.tracking_uri))
    todo = [
        j
        for j in plan_jobs(runs, ups, modes)
        if pending(a.out, j[0].run_id, j[1], j[2], client is not None)
    ]
    logger.info("%d runs, %d files to do", len(runs), len(todo))
    rp, failed, t0 = None, [], time.time()
    for k, (ref, u, m) in enumerate(todo, 1):
        if rp is None or rp.run.dir != ref.dir:
            rp = Replayer(EvalRun.load(ref.dir))
        try:
            process_file(
                rp,
                Identity(ref.run_id, ref.budget, ref.seed),
                u,
                m,
                a.out,
                client,
                a.recompute_events,
            )
        except ReplayMismatchError as e:
            failed.append(str(e))
            logger.exception("skipped")
        logger.info(
            "%d/%d %s update %d %s (%.0fs)",
            k,
            len(todo),
            ref.name,
            u,
            m,
            time.time() - t0,
        )
    if a.recompute_events:
        n = len(list(Path(a.out).glob("*/event_mismatches_*.csv")))
        logger.info("%d files with recomputed events under %s", n, a.out)
    if failed:
        msg = f"{len(failed)} files failed: {failed}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
