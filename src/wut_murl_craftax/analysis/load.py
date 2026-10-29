"""Loaders of the export tables (contracts.md section 1) and the run selectors on them.

``outputs/metrics/<phase>/{runs,curves,eval_episodes}.parquet`` is read as a dict of
frames keyed by those stems. Runs belong to a phase by their ``phases`` tag (R77), and
statistics are per (``method``, ``config_ref``): `select` refuses a result that mixes
two ``config_ref`` values of a method unless asked to pool, and `units` walks the
units. Base controls come from `select_pair.controls_of`; a combo run's control comes
from its own pair (R80a). Parquet-only use imports neither JAX nor MLflow: the registry
is imported by the functions that need it.

@see [[2026-10-20-craftax-training-analysis]]
"""

import contextlib
import functools
import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

Frames = dict[str, pd.DataFrame]
"""The tables of one or more phases, keyed by the file stems in `TABLES`."""
METRICS_ROOT = Path("outputs/metrics")
"""Where the exporter writes ``<phase>/<table>.parquet``."""
TABLES = ("runs", "curves", "eval_episodes")
FINAL_PHASES = ("final-1m", "final-1b", "combo-1m", "combo-1b")
"""The phases of the final analysis; the 1B ones do not exist before R75 is lifted."""
UNIT_COLUMNS = ("method", "config_ref")
"""The statistics unit: never pool two `config_ref` values of one method."""
RUNS_COLUMNS = [
    "run_id",
    "run_name",
    "experiment",
    "phase",
    "phases",
    "method",
    "budget",
    "config_hash",
    "seed",
    "config_ref",
    "best",
    "profile",
    "device_kind",
    "actor_device",
    "learner_device",
    "status",
    "config_json",
    "online_families",
    "final_score",
    "eval_stochastic_score",
    "eval_greedy_score",
    "eval_stochastic_return",
    "eval_greedy_return",
    "env_steps",
    "wall_clock_s",
    "steady_sps",
    "overall_sps",
    "compile_s",
    "eval_s",
    "cpu_util_mean",
    "cpu_util_p95",
    "gpu_util_mean",
    "gpu_util_p95",
    "peak_device_mem_mb",
    "energy_j",
    "steps_per_joule",
    "peak_rss_mb",
    "peak_host_mem_mb",
]
"""Columns of ``runs.parquet``."""
CURVES_COLUMNS = ["run_id", "env_steps", "metric", "value"]
"""Columns of ``curves.parquet``."""
EPISODE_REQUIRED = [
    "run_id",
    "method",
    "config_ref",
    "budget",
    "seed",
    "update",
    "env_steps",
    "mode",
    "eval_seed",
    "length",
    "return",
    "score_pct",
    "end_type",
    "cause_of_death",
    "max_floor",
    "tech_depth",
    "n_achievements",
    "stalls",
    "n_stalls",
    "max_stall",
    "fights",
    "damage_dealt",
    "damage_taken",
    "close_calls",
    "close_calls_recovered",
    "new_tiles_per_100",
    "revisit_rate",
    "unique_positions",
    "action_entropy",
    "wasted_share",
    "noop_share",
    "forced_share",
    "use_ratio",
    "steps_per_achievement",
    "wasted_crafts",
    "wood_to_diamond",
]
"""The fixed columns of ``eval_episodes.parquet``; the ``<family>/<name>`` ones vary.
Not required: ``any_zero_share``, which older exports lack (the radar's Survivor axis
falls back to a bound without it, `figures_game.SURVIVOR_COLUMN`)."""
_REQUIRED = {
    "runs": RUNS_COLUMNS,
    "curves": CURVES_COLUMNS,
    "eval_episodes": EPISODE_REQUIRED,
}
_EPISODE_KEYS = ["method", "config_ref", "budget"]

_COMBO = "combo"
"""`select_pair.COMBO`, repeated so that this module imports without the registry."""
CONTROLS: tuple[str, ...]
"""The registry methods that are their own base control (combo never is)."""
BASE_CONTROL: dict[str, str]
"""Non-combo method -> its control; a combo run's control: `base_control` of its row."""


@functools.cache
def _control_maps() -> tuple[tuple[str, ...], dict[str, str]]:
    """`CONTROLS` and `BASE_CONTROL` from the registry's ``base_from`` chains."""
    # imported here: select_pair pulls in MLflow and the orchestration code
    from ..evaluation.select_pair import COMBO, controls_of

    # controls_of also reads combo.yaml; its "combo" entry is dropped, never used (R80a)
    of = {r: c for r, c in controls_of().items() if r != COMBO}
    controls = tuple(sorted(r for r, c in of.items() if r == c))
    return controls, {r: c for r, c in of.items() if r != c}


def __getattr__(name: str) -> object:
    """Build `CONTROLS` and `BASE_CONTROL` on first access (they read the registry)."""
    if name in ("CONTROLS", "BASE_CONTROL"):
        return _control_maps()[name == "BASE_CONTROL"]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


class MissingPhaseError(FileNotFoundError):
    """A phase has no export, or an incomplete one; ``phases`` names them."""

    def __init__(self, phases: Sequence[str], detail: str) -> None:
        """Keep the phase names; the message adds where and what is missing."""
        super().__init__(f"missing export of phase {', '.join(phases)}: {detail}")
        self.phases = list(phases)


class SchemaError(ValueError):
    """An export table breaks the contract (a column, the run key, the phase tag)."""


@contextlib.contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """A temporary file of this process beside ``path``, which it replaces on success.

    A reader (or a rerun after a kill) sees the old file or the whole new one; the
    process id keeps two writers from sharing a temporary file. An error in the block
    (a full disk, say) removes the temporary file; a kill cannot, and leaves it.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through `atomic_path`."""
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def in_phase(runs: pd.DataFrame, phase: str) -> pd.Series:
    """Mask of the runs whose ``phases`` tag lists ``phase`` (R77)."""
    return ("," + runs["phases"] + ",").str.contains(f",{phase},", regex=False)


def _check(frames: Frames, phase: str) -> None:
    """Raise `SchemaError` unless the loaded tables fit the contract and each other.

    ``frames`` holds ``runs`` and any of the others (a runs-only load, `load_phases`).
    """
    for name, required in _REQUIRED.items():
        if name in frames and (
            missing := [c for c in required if c not in frames[name].columns]
        ):
            msg = f"{name} of {phase}: missing columns {', '.join(missing)}"
            raise SchemaError(msg)
    runs = frames["runs"]
    if not runs["run_id"].is_unique:
        msg = f"runs of {phase}: run_id is not unique"
        raise SchemaError(msg)
    if not (outside := runs.loc[~in_phase(runs, phase), "run_id"]).empty:
        msg = (
            f"runs of {phase}: tag `phases` lacks the phase for {len(outside)} runs, "
            f"e.g. {outside.iloc[0]}"
        )
        raise SchemaError(msg)
    for name in ("curves", "eval_episodes"):
        if name in frames and (
            n := (~frames[name]["run_id"].isin(runs["run_id"])).sum()
        ):
            msg = f"{name} of {phase}: {n} rows with a run id not in runs"
            raise SchemaError(msg)
    if "eval_episodes" not in frames:
        return
    episodes = frames["eval_episodes"]
    for column in _EPISODE_KEYS:
        owner = runs.set_index("run_id")[column]
        if (episodes[column] != episodes["run_id"].map(owner)).any():
            msg = f"eval_episodes of {phase}: {column} disagrees with runs"
            raise SchemaError(msg)


def _gaps(phase_dir: Path, tables: Sequence[str] = TABLES) -> list[str]:
    """The files of ``tables`` that a phase directory lacks."""
    return [
        f"{t}.parquet" for t in tables if not (phase_dir / f"{t}.parquet").is_file()
    ]


def _tables(tables: Sequence[str]) -> tuple[str, ...]:
    """``tables`` checked: known stems that include ``runs``, in `TABLES` order."""
    if set(tables) - set(TABLES) or "runs" not in tables:
        msg = f"tables must be stems of {TABLES} and include runs, got {list(tables)}"
        raise ValueError(msg)
    return tuple(t for t in TABLES if t in tables)


def load_phase(phase_dir: Path, tables: Sequence[str] = TABLES) -> Frames:
    """The tables of one phase directory, checked against the contract.

    ``tables`` limits what is read (``runs`` is always needed); a table left out is an
    empty frame with the contract columns.

    Raises:
        MissingPhaseError: If a table file is absent.
        SchemaError: If a column, the run key, the ``phases`` tag or the agreement of
            ``eval_episodes`` with ``runs`` is wrong.
        ValueError: If ``tables`` names an unknown table or leaves out ``runs``.
    """
    tables = _tables(tables)
    if gaps := _gaps(phase_dir, tables):
        raise MissingPhaseError(
            [phase_dir.name], f"{phase_dir} lacks {', '.join(gaps)}"
        )
    frames = {t: pd.read_parquet(phase_dir / f"{t}.parquet") for t in tables}
    _check(frames, phase_dir.name)
    return {
        t: frames[t] if t in frames else pd.DataFrame(columns=_REQUIRED[t])
        for t in TABLES
    }


def _warn_dropped(dropped: pd.DataFrame, kept: dict[str, tuple[str, str]]) -> None:
    """Warn about duplicate runs whose kept copy has another phase or `config_ref`."""
    differ = [
        r.run_id
        for r in dropped.itertuples()
        if kept[r.run_id] != (r.phase, r.config_ref)
    ]
    if differ:
        was = kept[differ[0]]
        logger.warning(
            "%d runs listed in two phases keep their first copy, e.g. %s (%s, %s)",
            len(differ),
            differ[0],
            *was,
        )


def load_phases(
    phases: Sequence[str],
    root: Path = METRICS_ROOT,
    *,
    skip_missing: bool = False,
    tables: Sequence[str] = TABLES,
) -> Frames:
    """The tables of several phases in one frame each.

    A run listed in two phases (its ``phases`` tag) is kept once, with the rows of the
    first phase of ``phases`` that holds it, so ``phase`` and ``config_ref`` come from
    that phase in all three frames; a warning names the run count when the dropped
    copy differs in either. Pass ``final-*`` and ``combo-*`` phases before a tuning
    phase: the other way round the control runs get their hash as ``config_ref``,
    which splits their (method, config_ref) unit. ``skip_missing`` leaves out phases
    with no export at all (``final-1b`` before 1B exists, R86); an incomplete export
    still fails. ``tables=("runs",)`` reads the run tables only (the others come back
    empty): the tuning phases' curves and episodes are far larger than anything the
    tuning tables need.

    Raises:
        MissingPhaseError: Naming every phase that lacks an export (those left out by
            ``skip_missing`` excepted, unless none is left).
        ValueError: As `load_phase` for ``tables``; for no ``phases``.
    """
    if not phases:
        msg = "no phases given"
        raise ValueError(msg)
    tables = _tables(tables)
    gaps = {p: _gaps(root / p, tables) for p in phases}
    absent = [p for p, g in gaps.items() if len(g) == len(tables)]
    bad = [p for p, g in gaps.items() if g and not (skip_missing and p in absent)]
    todo = [p for p in phases if not gaps[p]]
    if bad or not todo:
        detail = "; ".join(
            f"{p}: {root / p} lacks {', '.join(gaps[p])}" for p in bad or absent
        )
        raise MissingPhaseError(bad or absent, detail)
    if absent:
        logger.info("phases without an export, skipped: %s", ", ".join(absent))
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    kept: dict[str, tuple[str, str]] = {}
    for phase in todo:
        frames = load_phase(root / phase, tables)
        runs = frames["runs"]
        fresh = set(runs["run_id"]) - kept.keys()
        _warn_dropped(runs[~runs["run_id"].isin(fresh)], kept)
        kept |= {
            r.run_id: (phase, r.config_ref)
            for r in runs.itertuples()
            if r.run_id in fresh
        }
        for name in TABLES:
            parts[name].append(frames[name][frames[name]["run_id"].isin(fresh)])
    # an empty part would only add a dtype-warning to the concatenation
    return {
        name: pd.concat([p for p in ps if len(p)] or ps[:1], ignore_index=True)
        for name, ps in parts.items()
    }


def select(
    frames: Frames,
    *,
    method: str | Sequence[str] | None = None,
    config_ref: str | Sequence[str] | None = None,
    budget: str | Sequence[str] | None = None,
    phase: str | None = None,
    pool: bool = False,
) -> Frames:
    """The runs that match every given filter and their rows of every other frame.

    ``phase`` matches the ``phases`` tag (R77), not the ``phase`` column. The input is
    left alone; no match gives empty frames with the same columns.

    Raises:
        ValueError: If no ``config_ref`` is given, ``pool`` is False and some method in
            the result has more than one ``config_ref`` (tuned and ablation runs must
            not be pooled, R77): name the ``config_ref``, walk `units`, or pass
            ``pool=True``.
    """
    runs = frames["runs"]
    keep = pd.Series(True, index=runs.index)
    for column, want in (
        ("method", method),
        ("config_ref", config_ref),
        ("budget", budget),
    ):
        if want is not None:
            keep &= runs[column].isin([want] if isinstance(want, str) else list(want))
    if phase is not None:
        keep &= in_phase(runs, phase)
    if config_ref is None and not pool:
        refs = runs.loc[keep].groupby("method")["config_ref"].nunique()
        if mixed := sorted(refs[refs > 1].index):
            msg = (
                f"select would pool several config_ref values of {', '.join(mixed)}; "
                "pass config_ref, iterate units(frames) or pass pool=True"
            )
            raise ValueError(msg)
    ids = runs.loc[keep, "run_id"]
    return {
        name: frame[frame["run_id"].isin(ids)].reset_index(drop=True)
        for name, frame in frames.items()
    }


def units(frames: Frames) -> Iterator[tuple[tuple[str, str], Frames]]:
    """Each (``method``, ``config_ref``) of ``frames``, sorted, with its own frames."""
    pairs = frames["runs"][list(UNIT_COLUMNS)].drop_duplicates()
    for method, ref in sorted(pairs.itertuples(index=False, name=None)):
        yield (method, ref), select(frames, method=method, config_ref=ref)


def pair_of(run: Mapping[str, Any] | pd.Series) -> dict[str, Any] | None:
    """A combo run's ``{base, methods}`` from its own ``config_json``; None otherwise.

    Never reads combo.yaml: a recorded ``combine`` block wins, else the pair is
    inferred from the config's keys (R80a).
    """
    if run["method"] != _COMBO:
        return None
    # imported here: the registry pulls in JAX and MLflow
    from ..training.interface import METHODS, combo_pair

    return combo_pair(json.loads(run["config_json"]), METHODS)


def base_control(run: Mapping[str, Any] | pd.Series) -> str:
    """The control a ``runs`` row is compared with: itself for a control.

    Raises:
        ValueError: If the method is not in the registry.
    """
    if (pair := pair_of(run)) is not None:
        return str(pair["base"])
    method = str(run["method"])
    controls, base = _control_maps()
    if method not in base and method not in controls:
        msg = f"unknown method {method!r}: not in the registry"
        raise ValueError(msg)
    return base.get(method, method)


def final_checkpoint(episodes: pd.DataFrame, run: str | None = None) -> pd.DataFrame:
    """The eval episodes of the last checkpoint (largest ``update``) of each run.

    ``run`` (a ``run_id``) restricts to that run.

    Raises:
        ValueError: If ``run`` has no episodes.
    """
    if run is not None:
        episodes = episodes[episodes["run_id"] == run]
        if episodes.empty:
            msg = f"no eval episodes for run {run!r}"
            raise ValueError(msg)
    last = episodes.groupby("run_id")["update"].transform("max")
    return episodes[episodes["update"] == last].reset_index(drop=True)
