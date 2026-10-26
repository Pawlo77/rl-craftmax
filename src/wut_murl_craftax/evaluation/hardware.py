"""Hardware usage of one run, summarised from its stored MLflow series.

``hardware_summary(store, run_id)`` reads one run's rows of the SQLite store (read-only,
by ``run_uuid``): wall-clock, SPS, compile and eval time, CPU and GPU utilisation, peak
memory, energy. A series the store lacks gives NaN, which `show_metric` writes "n.m.".
Runs compare only inside one `profile_of`. Every laptop run trained with 2 queue workers
(R97): its wall-clock, SPS and the machine-wide cpu, host-memory and RSS series are
context only. `reused_from_tuning` marks runs reused from tuning. CLI: ``--run <id>``.
@see [[2026-10-20-craftax-evaluation-and-metrics]]
@tested-by [[tests/evaluation/test_hardware.py]]
"""

import argparse
import contextlib
import fnmatch
import math
import re
import sqlite3
import sys
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

type Store = str | Path | sqlite3.Connection
"""A SQLite MLflow store: a file, a ``sqlite:///`` URI or an open connection."""

NOT_MEASURED = "n.m."
"""How reports write a metric whose series the store does not hold."""
WALL = "train/wall_s"
"""Cumulative seconds at an update; the runner carries it across resume segments."""
COMPILE = "compile_s"
EVAL_TIME = "eval_s"
# placeholder: no runner code logs eval time yet, so it reads NaN until one does
RSS = "peak_rss_mb"
DEVICE_PEAK = "peak_device_mb"
"""The runner's peak device memory (``memory_stats``); no Metal device logs it."""
SYSTEM_LO, SYSTEM_HI = "system/", "system0"
"""Key range of the mlflow-sysmetrics series (``'0'`` follows ``'/'``)."""
_TRAIN_KEYS = (WALL, COMPILE, EVAL_TIME, RSS, DEVICE_PEAK)
_LOAD_SQL = (
    "SELECT key, step, timestamp, value FROM metrics WHERE run_uuid = ? "
    "AND is_nan = 0 AND (key IN (?, ?, ?, ?, ?) OR (key >= ? AND key < ?)) "
    "ORDER BY timestamp"  # one ? per `_TRAIN_KEYS` entry, then the system key range
)
WARMUP_UPDATES = 10
"""Updates left out of the steady SPS (compilation and the first slow chunks)."""
GAP_FACTOR = 10.0
"""A power interval this many times the median one is a resume gap, not energy."""
_CPU = "system/cpu_utilization_percentage"
_HOST_MEM = "system/system_memory_usage_megabytes"
HARDWARE_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "laptop": {
        "cpu_util": (_CPU,),
        "host_mem": (_HOST_MEM,),
        # none of these is logged on the laptop (Metal exposes no counters; preflight
        # fact 4): the metrics read "n.m."
        "gpu_util": (),
        "device_mem": ("peak_device_mb",),
        "power": (),
    },
    "cuda": {
        "cpu_util": (_CPU,),
        "host_mem": (_HOST_MEM,),
        # names of mlflow's NVIDIA and ROCm monitors; no cluster run has logged them
        # yet, so check them with ``--list-keys`` on the first smoke run
        "gpu_util": ("system/gpu_*_utilization_percentage",),
        "device_mem": (
            "peak_device_mb",
            "system/gpu_*_memory_usage_megabytes",
            "system/gpu_*_memory_usage_gigabytes",
        ),
        "power": ("system/gpu_*_power_usage_watts",),
    },
}
"""Per profile, per metric, the stored names it is read from (``fnmatch`` patterns)."""
TUNING_PHASES = frozenset({"tune-coarse", "tune-fine"})
"""Phases whose runs `final-*` and `combo-*` phases reuse."""
_BUDGET = re.compile(r"(\d+)([mb])")
_BUDGET_UNIT = {"m": 10**6, "b": 10**9}
_NAN = math.nan
STORE_TIMEOUT_S = 300.0
"""SQLite busy timeout of store reads (R103); ``?timeout=`` in a URI overrides it."""


class Series(NamedTuple):
    """One stored metric: MLflow step, sample time in seconds and value."""

    step: np.ndarray
    time: np.ndarray
    value: np.ndarray


def profile_of(learner_device: str) -> str:
    """The comparison group of a run, from its ``learner_device`` tag (contracts.md)."""
    if learner_device.startswith("MetalDevice"):
        return "laptop"
    if learner_device.startswith(("cuda", "CudaDevice")):
        return "cuda"
    return "unknown"


def reused_from_tuning(phases: str) -> bool:
    """True for a run whose ``phases`` tag lists a tuning phase (it was reused)."""
    return not TUNING_PHASES.isdisjoint(phases.split(","))


def show_metric(value: float | None, spec: str = "{:.1f}") -> str:
    """A metric for a report: "n.m." for NaN (not measured), else ``spec``-formatted."""
    if value is None or math.isnan(value):
        return NOT_MEASURED
    return spec.format(value)


@contextlib.contextmanager
def open_store(store: Store) -> Generator[sqlite3.Connection]:
    """A read-only connection to the store (an open one is passed through).

    The store runs in rollback-journal mode, so a read blocks the writers' commits
    while it lasts (R103): callers keep connections short and fetch every row. The
    busy timeout is `STORE_TIMEOUT_S`, or ``?timeout=<s>`` of a ``sqlite:///`` URI.

    Raises:
        ValueError: For a URI of another kind of store.
    """
    if isinstance(store, sqlite3.Connection):
        yield store
        return
    text, _, query = str(store).partition("?")
    if "://" in text and not text.startswith("sqlite:///"):
        msg = f"expected a sqlite:/// tracking URI, got {store}"
        raise ValueError(msg)
    timeout = float(parse_qs(query).get("timeout", [str(STORE_TIMEOUT_S)])[0])
    path = Path(text.removeprefix("sqlite:///")).expanduser().resolve()
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=timeout)
    try:
        yield con
    finally:
        con.close()


def system_keys(store: Store, run_id: str) -> list[str]:
    """The ``system/*`` metric names stored for one run, sorted.

    Raises:
        ValueError: If the run is not in the store (a typo is not "nothing logged").
    """
    with open_store(store) as con:
        _tags(con, run_id)
        rows = con.execute(
            "SELECT DISTINCT key FROM metrics WHERE run_uuid = ? AND key >= ? "
            "AND key < ? ORDER BY key",
            (run_id, SYSTEM_LO, SYSTEM_HI),
        ).fetchall()
    return [k for (k,) in rows]


def _tags(con: sqlite3.Connection, run_id: str) -> dict[str, str]:
    """The tags of a run."""
    if (
        con.execute("SELECT 1 FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
        is None
    ):
        msg = f"no run {run_id!r} in the store"
        raise ValueError(msg)
    rows = con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,))
    return dict(rows.fetchall())


def _load(con: sqlite3.Connection, run_id: str) -> dict[str, Series]:
    """The runner's hardware series and every ``system/*`` series of one run.

    A training update logged twice (a resume replays the updates since its checkpoint)
    keeps its latest value; sampled system series are told apart by their time.
    """
    rows = con.execute(_LOAD_SQL, (run_id, *_TRAIN_KEYS, SYSTEM_LO, SYSTEM_HI))
    frame = pd.DataFrame(rows.fetchall(), columns=["key", "step", "timestamp", "value"])
    system = frame["key"].str.startswith(SYSTEM_LO)
    frame = pd.concat([
        frame[~system].drop_duplicates(["key", "step"], keep="last"),
        frame[system].drop_duplicates(["key", "timestamp"], keep="last"),
    ])
    out = {}
    for key, rows_of in frame.groupby("key"):
        by = "timestamp" if str(key).startswith(SYSTEM_LO) else "step"
        part = rows_of.sort_values(by, kind="stable")
        out[str(key)] = Series(
            part["step"].to_numpy(),
            part["timestamp"].to_numpy() / 1000.0,
            part["value"].to_numpy(float),
        )
    return out


def _matching(
    series: Mapping[str, Series], profile: str, metric: str
) -> dict[str, Series]:
    """The series a metric is read from; an unknown profile reads every profile's."""
    known = profile in HARDWARE_KEYS
    groups = [HARDWARE_KEYS[profile]] if known else list(HARDWARE_KEYS.values())
    patterns = {p for g in groups for p in g[metric]}
    return {
        k: s
        for k, s in series.items()
        if any(fnmatch.fnmatchcase(k, p) for p in patterns)
    }


def _budget_steps(tag: str | None) -> int | None:
    """Env steps of a ``budget`` tag such as ``1m``; None when absent or unreadable."""
    match = _BUDGET.fullmatch(tag or "")
    return int(match[1]) * _BUDGET_UNIT[match[2]] if match else None


def _timing(
    series: Mapping[str, Series], budget: int | None, run_id: str
) -> dict[str, float]:
    """Wall-clock, steady and overall SPS, compile and eval time.

    ``train/wall_s`` is cumulative over the resume segments (the runner restores it), so
    the last value is the wall-clock; the MLflow step of its points is the env steps.
    Steady SPS runs from update `WARMUP_UPDATES` (counted from the first logged point,
    one update at 1M and a window of updates at 1B) to the last point. Compile time is
    the stored ``compile_s`` when the runner logged it, else the plan's estimate: the
    first point's ``wall_s`` minus the median time of the steps it covers.
    """
    out = dict.fromkeys(
        ("wall_clock_s", "steady_sps", "overall_sps", "compile_s"), _NAN
    )
    out["eval_s"] = float(series[EVAL_TIME].value[-1]) if EVAL_TIME in series else _NAN
    if COMPILE in series:
        out["compile_s"] = float(series[COMPILE].value[-1])
    if WALL not in series:
        return out
    steps, wall = series[WALL].step.astype(float), series[WALL].value
    if (np.diff(wall) < 0).any():
        msg = f"{WALL} of run {run_id} runs backwards: resume segments not accumulated"
        raise ValueError(msg)
    out["wall_clock_s"] = float(wall[-1])
    if wall[-1] > 0:
        out["overall_sps"] = float((budget or steps[-1]) / wall[-1])
    ref = int(np.searchsorted(steps, WARMUP_UPDATES * steps[0]))
    if ref < len(steps) - 1 and wall[-1] > wall[ref]:
        out["steady_sps"] = float((steps[-1] - steps[ref]) / (wall[-1] - wall[ref]))
    if COMPILE not in series and len(steps) > 1:
        per_step = np.median(np.diff(wall) / np.diff(steps))
        out["compile_s"] = float(max(0.0, wall[0] - steps[0] * per_step))
    return out


def _utilisation(series: Mapping[str, Series]) -> tuple[float, float]:
    """Mean and p95 of the samples of all series pooled; NaN without samples."""
    pooled = [s.value for s in series.values()]
    values = np.concatenate(pooled) if pooled else np.empty(0)
    if not values.size:
        return _NAN, _NAN
    return float(values.mean()), float(np.percentile(values, 95))


def _peak(series: Mapping[str, Series], mb_per_unit: float = 1.0) -> float:
    """Largest value of any series, scaled; NaN without samples."""
    peaks = [s.value.max() for s in series.values() if s.value.size]
    return float(max(peaks)) * mb_per_unit if peaks else _NAN


def _device_peak_mb(series: Mapping[str, Series]) -> float:
    """Peak device memory in MB over the devices; ROCm series are in (decimal) GB."""
    peaks = [
        _peak({k: s}, 1000.0 if k.endswith("_gigabytes") else 1.0)
        for k, s in series.items()
    ]
    return max((p for p in peaks if not math.isnan(p)), default=_NAN)


def _energy_j(series: Mapping[str, Series]) -> float:
    """Trapezoid integral (J) of every power series, summed; NaN without any integral.

    An interval much longer than the median one is the gap between two resume segments
    (the machine drew power the run did not log) and is left out, so the energy is a
    lower bound for a resumed run. A series of fewer than 2 samples integrates nothing.
    """
    total, seen = 0.0, False
    for s in series.values():
        if s.value.size < 2:
            continue
        dt = np.diff(s.time)
        mean_power = (s.value[1:] + s.value[:-1]) / 2
        total += float((dt * mean_power)[dt <= GAP_FACTOR * np.median(dt)].sum())
        seen = True
    return total if seen else _NAN


def hardware_summary(
    store: Store, run_id: str, *, budget: int | None = None
) -> dict[str, Any]:
    """The hardware metrics and machine identity of one run, NaN where not measured.

    ``budget`` is the env steps of the budget (overall SPS and steps per joule); by
    default the run's ``budget`` tag (``1m``, ``1b``), else its last logged env step.
    Energy covers the logged GPU power only: mlflow-sysmetrics has no CPU or system
    power, so steps per joule overstates the efficiency of a machine that also burns
    CPU power. ``peak_rss_mb`` is the peak of the whole worker process, as the runner
    logs it (several runs of one worker share it). ``peak_device_mem_mb`` is the largest
    of the runner's ``peak_device_mb`` and the sampled device-memory series (the sampled
    ones see the whole node, so the runner's figure is the floor on a shared GPU).

    Raises:
        ValueError: If the run is not in the store or its ``train/wall_s`` decreases.
    """
    with open_store(store) as con:
        tags = _tags(con, run_id)
        series = _load(con, run_id)
    learner = tags.get("learner_device", "")
    profile = profile_of(learner)
    budget = budget or _budget_steps(tags.get("budget"))
    out: dict[str, Any] = {
        "profile": profile,
        "device_kind": tags.get("device_kind", ""),
        "actor_device": tags.get("actor_device", ""),
        "learner_device": learner,
        "reused_from_tuning": reused_from_tuning(tags.get("phases", "")),
        **_timing(series, budget, run_id),
    }
    out["cpu_util_mean"], out["cpu_util_p95"] = _utilisation(
        _matching(series, profile, "cpu_util")
    )
    out["gpu_util_mean"], out["gpu_util_p95"] = _utilisation(
        _matching(series, profile, "gpu_util")
    )
    out["peak_rss_mb"] = _peak({k: v for k, v in series.items() if k == RSS})
    out["peak_host_mem_mb"] = _peak(_matching(series, profile, "host_mem"))
    out["peak_device_mem_mb"] = _device_peak_mb(
        _matching(series, profile, "device_mem")
    )
    out["energy_j"] = _energy_j(_matching(series, profile, "power"))
    steps = budget or (series[WALL].step[-1] if WALL in series else None)
    out["steps_per_joule"] = (
        float(steps / out["energy_j"]) if steps and out["energy_j"] > 0 else _NAN
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: print a run's summary, or its stored system keys."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run", required=True, help="MLflow run id")
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlruns.db",
        help="SQLite store (opened read-only)",
    )
    parser.add_argument(
        "--list-keys", action="store_true", help="print the stored system/* names"
    )
    args = parser.parse_args(argv)
    try:
        if args.list_keys:
            print("\n".join(system_keys(args.tracking_uri, args.run)))
            return
        summary = hardware_summary(args.tracking_uri, args.run)
    except (ValueError, sqlite3.Error) as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        raise SystemExit(2) from refusal
    for name, value in summary.items():
        print(f"{name}: {show_metric(value) if isinstance(value, float) else value}")


if __name__ == "__main__":
    main()
