"""Loaders of the export tables (contracts.md section 1) and the run selectors on them.

``outputs/metrics/<phase>/{runs,curves,eval_episodes}.parquet`` is read as a dict of
frames keyed by those stems. Runs belong to a phase by their ``phases`` tag (R77), and
statistics are per (``method``, ``config_ref``): select a ``config_ref`` whenever tuned
and ablation runs of one method share a frame. Base controls come from
`select_pair.controls_of`; a combo run's control comes from its own pair (R80a).

@see [[2026-10-20-craftax-training-analysis]]
"""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..evaluation.select_pair import COMBO, controls_of
from ..training.interface import METHODS, combo_pair

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
"""The fixed columns of ``eval_episodes.parquet``; the ``<family>/<name>`` ones vary."""
_REQUIRED = {
    "runs": RUNS_COLUMNS,
    "curves": CURVES_COLUMNS,
    "eval_episodes": EPISODE_REQUIRED,
}
_EPISODE_KEYS = ["method", "config_ref", "budget"]

# controls_of also reads combo.yaml; its "combo" entry is dropped, never used (R80a)
_CONTROL_OF = controls_of()
CONTROLS = tuple(sorted(r for r, c in _CONTROL_OF.items() if r == c))
"""The registry methods that are their own base control."""
BASE_CONTROL = {r: c for r, c in _CONTROL_OF.items() if r != c and r != COMBO}
"""Non-combo method -> its control; a combo run's control: `base_control` of its row."""


class MissingPhaseError(FileNotFoundError):
    """A phase has no export, or an incomplete one; ``phases`` names them."""

    def __init__(self, phases: Sequence[str], detail: str) -> None:
        """Keep the phase names; the message adds where and what is missing."""
        super().__init__(f"missing export of phase {', '.join(phases)}: {detail}")
        self.phases = list(phases)


class SchemaError(ValueError):
    """An export table breaks the contract (a column, the run key, the phase tag)."""


def in_phase(runs: pd.DataFrame, phase: str) -> pd.Series:
    """Mask of the runs whose ``phases`` tag lists ``phase`` (R77)."""
    return ("," + runs["phases"] + ",").str.contains(f",{phase},", regex=False)


def _check(frames: Frames, phase: str) -> None:
    """Raise `SchemaError` unless the three tables fit the contract and each other."""
    for name, required in _REQUIRED.items():
        if missing := [c for c in required if c not in frames[name].columns]:
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
        if n := (~frames[name]["run_id"].isin(runs["run_id"])).sum():
            msg = f"{name} of {phase}: {n} rows with a run id not in runs"
            raise SchemaError(msg)
    episodes = frames["eval_episodes"]
    for column in _EPISODE_KEYS:
        owner = runs.set_index("run_id")[column]
        if (episodes[column] != episodes["run_id"].map(owner)).any():
            msg = f"eval_episodes of {phase}: {column} disagrees with runs"
            raise SchemaError(msg)


def _gaps(phase_dir: Path) -> list[str]:
    """The table files a phase directory lacks."""
    return [
        f"{t}.parquet" for t in TABLES if not (phase_dir / f"{t}.parquet").is_file()
    ]


def load_phase(phase_dir: Path) -> Frames:
    """The three tables of one phase directory, checked against the contract.

    Raises:
        MissingPhaseError: If a table file is absent.
        SchemaError: If a column, the run key, the ``phases`` tag or the agreement of
            ``eval_episodes`` with ``runs`` is wrong.
    """
    if gaps := _gaps(phase_dir):
        raise MissingPhaseError(
            [phase_dir.name], f"{phase_dir} lacks {', '.join(gaps)}"
        )
    frames = {t: pd.read_parquet(phase_dir / f"{t}.parquet") for t in TABLES}
    _check(frames, phase_dir.name)
    return frames


def load_phases(
    phases: Sequence[str], root: Path = METRICS_ROOT, *, skip_missing: bool = False
) -> Frames:
    """The tables of several phases in one frame each.

    A run listed in two phases (its ``phases`` tag) is kept once, with the rows of the
    first phase of ``phases`` that holds it, so ``phase`` and ``config_ref`` come from
    that phase in all three frames. ``skip_missing`` leaves out phases with no export
    at all (``final-1b`` before 1B exists, R86); an incomplete export still fails.

    Raises:
        MissingPhaseError: Naming every phase that lacks an export (those left out by
            ``skip_missing`` excepted, unless none is left).
    """
    if not phases:
        msg = "no phases given"
        raise ValueError(msg)
    gaps = {p: _gaps(root / p) for p in phases}
    absent = [p for p, g in gaps.items() if len(g) == len(TABLES)]
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
    seen: set[str] = set()
    for phase in todo:
        frames = load_phase(root / phase)
        fresh = set(frames["runs"]["run_id"]) - seen
        seen |= fresh
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
) -> Frames:
    """The runs that match every given filter and their rows of every other frame.

    ``phase`` matches the ``phases`` tag (R77), not the ``phase`` column. The input is
    left alone; no match gives empty frames with the same columns.
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
    ids = runs.loc[keep, "run_id"]
    return {
        name: frame[frame["run_id"].isin(ids)].reset_index(drop=True)
        for name, frame in frames.items()
    }


def pair_of(run: Mapping[str, Any] | pd.Series) -> dict[str, Any] | None:
    """A combo run's ``{base, methods}`` from its own ``config_json``; None otherwise.

    Never reads combo.yaml: a recorded ``combine`` block wins, else the pair is
    inferred from the config's keys (R80a).
    """
    if run["method"] != COMBO:
        return None
    return combo_pair(json.loads(run["config_json"]), METHODS)


def base_control(run: Mapping[str, Any] | pd.Series) -> str:
    """The control a ``runs`` row is compared with: itself for a control.

    Raises:
        ValueError: If the method is not in the registry.
    """
    if (pair := pair_of(run)) is not None:
        return str(pair["base"])
    method = str(run["method"])
    if method not in BASE_CONTROL and method not in CONTROLS:
        msg = f"unknown method {method!r}: not in the registry"
        raise ValueError(msg)
    return BASE_CONTROL.get(method, method)


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
