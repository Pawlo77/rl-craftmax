"""Tests for the hardware summary of one run, on a throwaway MLflow store."""

import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
from mlflow.entities import Metric

from wut_murl_craftax.evaluation import hardware
from wut_murl_craftax.evaluation.hardware import (
    NOT_MEASURED,
    hardware_summary,
    profile_of,
    reused_from_tuning,
    show_metric,
)

SPU = 1000
"""Env steps per update of the fake runs."""
T0 = 1_700_000_000_000
"""Wall-clock time (ms) of the first fake sample."""
LAPTOP = {
    "device_kind": "Apple GPU",
    "actor_device": "cpu:0",
    "learner_device": "MetalDevice(id=0)",
    "budget": "1m",
    "phases": "final-1m",
}
type Points = Sequence[tuple[int, int, float]]
"""``(step, timestamp ms, value)`` of one metric."""
type Series = Mapping[str, Points]


class Fake:
    """A real SQLite MLflow store that takes runs with given tags and series."""

    def __init__(self, root: Path) -> None:
        self.path = root / "mlflow.db"
        self.client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{self.path}")
        self.exp = self.client.create_experiment(
            "hardware", artifact_location=(root / "a").as_uri()
        )

    def __call__(self, tags: dict[str, str], series: Series) -> str:
        rid = self.client.create_run(self.exp, tags=tags).info.run_id
        batch = [Metric(k, v, t, s) for k, pts in series.items() for s, t, v in pts]
        for i in range(0, len(batch), 1000):
            self.client.log_batch(rid, metrics=batch[i : i + 1000])
        return str(rid)


type AddRun = Fake


@pytest.fixture(scope="module")
def add_run(tmp_path_factory: pytest.TempPathFactory) -> Fake:
    return Fake(tmp_path_factory.mktemp("store"))


def store_of(add: Fake) -> Path:
    return add.path


def train(
    first: int, last: int, wall: Callable[[int], float], t0: int = T0
) -> dict[str, Points]:
    """``train/env_steps`` and ``train/wall_s`` of updates ``first .. last``."""
    ups = range(first, last + 1)
    return {
        "train/env_steps": [(u * SPU, t0 + u * 1000, float(u * SPU)) for u in ups],
        "train/wall_s": [(u * SPU, t0 + u * 1000, wall(u)) for u in ups],
    }


def sys_series(
    key: str, values: Sequence[float], every_s: int = 10, t0: int = T0
) -> dict[str, Points]:
    """A sampled ``system/`` series; MLflow's step there counts samples."""
    return {
        f"system/{key}": [(i, t0 + i * every_s * 1000, v) for i, v in enumerate(values)]
    }


def run_with(add: Fake, series: Series, **tags: str) -> dict[str, Any]:
    """Summary of a run with the laptop tags (overridable) and the given series."""
    rid = add({**LAPTOP, **tags}, series)
    return hardware_summary(store_of(add), rid, budget=100 * SPU)


# --- time and throughput ------------------------------------------------------------


def steady(u: int) -> float:
    """Compile 10 s inside the first update, then 2 s per update."""
    return 10.0 + 2.0 * u


def test_wall_clock_steady_overall_sps_and_compile(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 100, steady))
    assert out["wall_clock_s"] == pytest.approx(210.0)
    # updates 10 -> 100: 90 updates in 180 s, compilation excluded
    assert out["steady_sps"] == pytest.approx(90 * SPU / 180.0)
    assert out["overall_sps"] == pytest.approx(100 * SPU / 210.0)
    assert out["compile_s"] == pytest.approx(10.0)


def test_stored_compile_time_wins_over_the_wall_clock_estimate(add_run: AddRun) -> None:
    series = {**train(1, 100, steady), "compile_s": [(100 * SPU, T0, 9.5)]}
    assert run_with(add_run, series)["compile_s"] == 9.5


def test_wall_clock_across_two_resume_segments(add_run: AddRun) -> None:
    # segment 1 logs updates 1-60, then dies; segment 2 resumes from the checkpoint at
    # update 50 (wall_s continues from the saved 110 s), is slower, and logs 51-100
    first = train(1, 60, steady)
    second = train(51, 100, lambda u: 110.0 + 3.0 * (u - 50), t0=T0 + 10**6)
    series = {k: [*first[k], *second[k]] for k in first}
    out = run_with(add_run, series)
    assert out["wall_clock_s"] == pytest.approx(260.0)
    # the re-logged updates 51-60 keep the later (segment 2) values
    assert out["steady_sps"] == pytest.approx(90 * SPU / (260.0 - 30.0))
    assert out["overall_sps"] == pytest.approx(100 * SPU / 260.0)


def test_a_wall_clock_that_runs_backwards_is_refused(add_run: AddRun) -> None:
    series = train(1, 20, lambda u: 100.0 - u)
    rid = add_run({**LAPTOP}, series)
    with pytest.raises(ValueError, match="wall_s"):
        hardware_summary(store_of(add_run), rid)


def test_overall_sps_takes_the_budget_from_the_tag(add_run: AddRun) -> None:
    rid = add_run({**LAPTOP, "budget": "1m"}, train(1, 100, steady))
    out = hardware_summary(store_of(add_run), rid)
    assert out["overall_sps"] == pytest.approx(1_000_000 / 210.0)


def test_a_run_that_logged_nothing_has_nothing_measured(add_run: AddRun) -> None:
    out = hardware_summary(store_of(add_run), add_run({**LAPTOP}, {}))
    for name in ("wall_clock_s", "steady_sps", "overall_sps", "compile_s"):
        assert math.isnan(out[name]), name
    assert out["profile"] == "laptop"


def test_a_run_shorter_than_the_warm_up_has_no_steady_sps(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 8, steady))
    assert math.isnan(out["steady_sps"])
    assert out["wall_clock_s"] == pytest.approx(26.0)


def test_logging_every_eighth_update_scales_by_steps(add_run: AddRun) -> None:
    # a 1B run logs windows of 8 updates; times and steps scale together
    ups = range(8, 801, 8)
    series = {
        "train/env_steps": [(u * SPU, T0 + u, float(u * SPU)) for u in ups],
        "train/wall_s": [(u * SPU, T0 + u, steady(u)) for u in ups],
    }
    out = run_with(add_run, series)
    assert out["steady_sps"] == pytest.approx(SPU / 2.0)
    assert out["compile_s"] == pytest.approx(10.0)


def test_eval_time_is_not_measured_without_its_series(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 100, steady))
    assert math.isnan(out["eval_s"])
    out = run_with(
        add_run, {**train(1, 100, steady), "eval_s": [(100 * SPU, T0, 31.0)]}
    )
    assert out["eval_s"] == 31.0


# --- utilisation and memory ---------------------------------------------------------


def test_cpu_utilisation_mean_and_p95(add_run: AddRun) -> None:
    values = [float(v) for v in range(1, 101)]
    out = run_with(add_run, sys_series("cpu_utilization_percentage", values))
    assert out["cpu_util_mean"] == pytest.approx(50.5)
    assert out["cpu_util_p95"] == pytest.approx(np.percentile(values, 95))


def test_laptop_without_gpu_series_reports_them_as_missing(add_run: AddRun) -> None:
    out = run_with(add_run, sys_series("cpu_utilization_percentage", [10.0, 20.0]))
    for name in ("gpu_util_mean", "gpu_util_p95", "peak_device_mem_mb"):
        assert math.isnan(out[name]), name
    assert out["cpu_util_mean"] == 15.0


def test_peak_host_memory_is_the_series_max_and_rss_the_stored_peak(
    add_run: AddRun,
) -> None:
    series = {
        **sys_series("system_memory_usage_megabytes", [9000.0, 12000.0, 10000.0]),
        "peak_rss_mb": [
            (1000, T0, 3000.0),
            (2000, T0 + 1, 4590.0),
            (3000, T0 + 2, 4590.0),
        ],
        # the other sampled series are never mixed in
        **sys_series("system_memory_usage_percentage", [99.0]),
        **sys_series("disk_usage_megabytes", [1e9]),
    }
    out = run_with(add_run, series)
    assert out["peak_host_mem_mb"] == 12000.0
    assert out["peak_rss_mb"] == 4590.0


def test_missing_memory_series_are_nan_not_zero(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 20, steady))
    assert math.isnan(out["peak_host_mem_mb"])
    assert math.isnan(out["peak_rss_mb"])
    assert math.isnan(out["cpu_util_mean"])


def test_cuda_profile_reads_gpu_utilisation_and_device_memory(add_run: AddRun) -> None:
    series = {
        **sys_series("gpu_0_utilization_percentage", [40.0, 80.0]),
        **sys_series("gpu_1_utilization_percentage", [60.0, 100.0]),
        **sys_series("gpu_0_memory_usage_megabytes", [1000.0, 2000.0]),
        **sys_series("gpu_1_memory_usage_gigabytes", [1.0, 3.0]),
    }
    out = run_with(add_run, series, learner_device="cuda:0", device_kind="NVIDIA A100")
    assert out["profile"] == "cuda"
    assert out["gpu_util_mean"] == pytest.approx(70.0)  # pooled over both devices
    assert out["gpu_util_p95"] == pytest.approx(np.percentile([40, 80, 60, 100], 95))
    assert out["peak_device_mem_mb"] == 3000.0  # 3 GB, in decimal MB like the others


def test_the_runners_device_peak_counts_and_the_larger_peak_wins(
    add_run: AddRun,
) -> None:
    sampled = sys_series("gpu_0_memory_usage_megabytes", [1000.0, 2000.0])
    above = {**sampled, "peak_device_mb": [(1000, T0, 9000.0)]}
    below = {**sampled, "peak_device_mb": [(1000, T0, 500.0)]}
    cuda = {"learner_device": "cuda:0"}
    assert run_with(add_run, above, **cuda)["peak_device_mem_mb"] == 9000.0
    assert run_with(add_run, below, **cuda)["peak_device_mem_mb"] == 2000.0
    only = {"peak_device_mb": [(1000, T0, 2500.0), (2000, T0 + 1, 3500.0)]}
    assert run_with(add_run, only, **cuda)["peak_device_mem_mb"] == 3500.0


def test_an_unknown_profile_reads_the_series_of_every_profile(add_run: AddRun) -> None:
    series = {
        **sys_series("cpu_utilization_percentage", [30.0, 50.0]),
        **sys_series("gpu_0_utilization_percentage", [70.0, 90.0]),
        **sys_series("gpu_0_power_usage_watts", [100.0, 100.0]),
        "peak_device_mb": [(1000, T0, 4000.0)],
    }
    out = run_with(add_run, series, learner_device="TFRT_CPU_0")
    assert out["profile"] == "unknown"
    assert out["cpu_util_mean"] == 40.0
    assert out["gpu_util_mean"] == 80.0
    assert out["energy_j"] == pytest.approx(1000.0)
    assert out["peak_device_mem_mb"] == 4000.0


# --- energy -------------------------------------------------------------------------


def test_constant_100_watts_for_10_seconds_is_1000_joules(add_run: AddRun) -> None:
    series = {
        **train(1, 100, steady),
        **sys_series("gpu_0_power_usage_watts", [100.0, 100.0, 100.0], every_s=5),
    }
    out = run_with(add_run, series, learner_device="cuda:0")
    assert out["energy_j"] == pytest.approx(1000.0)
    assert out["steps_per_joule"] == pytest.approx(100 * SPU / 1000.0)


def test_energy_trapezoid_sums_every_power_series(add_run: AddRun) -> None:
    series = {
        **sys_series("gpu_0_power_usage_watts", [0.0, 100.0], every_s=10),  # 500 J
        **sys_series(
            "gpu_1_power_usage_watts", [50.0, 50.0, 50.0], every_s=10
        ),  # 1000 J
        # a percentage series is not power and never integrated
        **sys_series("gpu_0_power_usage_percentage", [90.0, 90.0]),
    }
    out = run_with(add_run, series, learner_device="cuda:0")
    assert out["energy_j"] == pytest.approx(1500.0)


def test_the_gap_between_two_segments_is_not_integrated(add_run: AddRun) -> None:
    hour = 3_600_000
    first = sys_series("gpu_0_power_usage_watts", [100.0] * 4, every_s=10)
    second = sys_series(
        "gpu_0_power_usage_watts", [100.0] * 4, every_s=10, t0=T0 + hour
    )
    key = "system/gpu_0_power_usage_watts"
    series = {
        key: [*first[key], *[(i + 4, t, v) for i, (_, t, v) in enumerate(second[key])]]
    }
    out = run_with(add_run, series, learner_device="cuda:0")
    assert out["energy_j"] == pytest.approx(2 * 3 * 10 * 100.0)  # 2 segments x 30 s


def test_no_power_series_gives_nan_steps_per_joule_never_zero(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 100, steady))
    assert math.isnan(out["energy_j"])
    assert math.isnan(out["steps_per_joule"])


def test_a_single_power_sample_integrates_nothing_so_it_is_not_measured(
    add_run: AddRun,
) -> None:
    series = sys_series("gpu_0_power_usage_watts", [100.0])
    out = run_with(add_run, series, learner_device="cuda:0")
    assert math.isnan(out["energy_j"])


# --- identity of the machine --------------------------------------------------------


@pytest.mark.parametrize(
    ("learner", "profile"),
    [
        ("MetalDevice(id=0)", "laptop"),
        ("cuda:0", "cuda"),
        ("CudaDevice(id=0)", "cuda"),
        ("TFRT_CPU_0", "unknown"),
        ("", "unknown"),
    ],
)
def test_profile_comes_from_the_learner_device(learner: str, profile: str) -> None:
    assert profile_of(learner) == profile


def test_device_tags_are_passed_through_and_never_a_host_name(add_run: AddRun) -> None:
    rid = add_run({**LAPTOP, "host_name": "pawels-mac"}, train(1, 20, steady))
    out = hardware_summary(store_of(add_run), rid)
    assert out["profile"] == "laptop"
    assert out["device_kind"] == "Apple GPU"
    assert out["actor_device"] == "cpu:0"
    assert out["learner_device"] == "MetalDevice(id=0)"
    assert "pawels-mac" not in out.values()


@pytest.mark.parametrize(
    ("phases", "flag"),
    [
        ("tune-coarse,final-1m", True),
        ("tune-coarse,tune-fine,final-1m", True),
        ("tune-fine,final-1m", True),
        ("final-1m", False),
        ("combo-1m", False),
        ("final-1m,combo-1m", False),
    ],
)
def test_runs_whose_phases_list_a_tuning_phase_are_reused_from_tuning(
    phases: str, flag: bool
) -> None:
    assert reused_from_tuning(phases) is flag


def test_summary_carries_the_reused_from_tuning_flag(add_run: AddRun) -> None:
    out = run_with(add_run, train(1, 20, steady), phases="tune-coarse,final-1m")
    assert out["reused_from_tuning"] is True
    out = run_with(add_run, train(1, 20, steady), phases="final-1m")
    assert out["reused_from_tuning"] is False


# --- reading the store --------------------------------------------------------------


def test_unknown_run_is_refused(add_run: AddRun) -> None:
    add_run({**LAPTOP}, {})  # make sure the store exists
    with pytest.raises(ValueError, match="no run"):
        hardware_summary(store_of(add_run), "0" * 32)


def test_the_store_is_opened_read_only(add_run: AddRun) -> None:
    with (
        hardware.open_store(store_of(add_run)) as con,
        pytest.raises(sqlite3.OperationalError),
    ):
        con.execute("create table x (a)")


def test_a_busy_timeout_in_the_uri_is_not_part_of_the_path(add_run: AddRun) -> None:
    rid = add_run({**LAPTOP}, train(1, 20, steady))
    uri = f"sqlite:///{store_of(add_run)}?timeout=7"
    with hardware.open_store(uri) as con:
        assert con.execute("select 1 from runs where run_uuid = ?", (rid,)).fetchall()
    assert hardware_summary(uri, rid)["wall_clock_s"] == pytest.approx(steady(20))


def test_store_may_be_given_as_a_tracking_uri(add_run: AddRun) -> None:
    rid = add_run({**LAPTOP}, train(1, 20, steady))
    out = hardware_summary(f"sqlite:///{store_of(add_run)}", rid)
    assert out["wall_clock_s"] == pytest.approx(steady(20))


def test_list_keys_names_the_system_series_of_a_run(add_run: AddRun) -> None:
    series = {
        **sys_series("cpu_utilization_percentage", [1.0]),
        **sys_series("disk_usage_megabytes", [1.0]),
        "train/env_steps": [(1, T0, 1.0)],
    }
    rid = add_run({**LAPTOP}, series)
    keys = hardware.system_keys(store_of(add_run), rid)
    assert keys == [
        "system/cpu_utilization_percentage",
        "system/disk_usage_megabytes",
    ]


def test_list_keys_of_an_unknown_run_is_refused(add_run: AddRun) -> None:
    add_run({**LAPTOP}, {})
    with pytest.raises(ValueError, match="no run"):
        hardware.system_keys(store_of(add_run), "0" * 32)


def test_cli_lists_keys_and_prints_the_summary(
    add_run: AddRun, capsys: pytest.CaptureFixture[str]
) -> None:
    rid = add_run(
        {**LAPTOP},
        {**train(1, 100, steady), **sys_series("cpu_utilization_percentage", [5.0])},
    )
    base = ["--tracking-uri", f"sqlite:///{store_of(add_run)}", "--run", rid]
    hardware.main([*base, "--list-keys"])
    assert capsys.readouterr().out.split() == ["system/cpu_utilization_percentage"]
    hardware.main(base)
    shown = capsys.readouterr().out
    assert "energy_j" in shown
    assert NOT_MEASURED in shown


# --- display ------------------------------------------------------------------------


def test_show_writes_n_m_for_nan_and_formats_numbers() -> None:
    assert show_metric(float("nan")) == "n.m." == NOT_MEASURED
    assert show_metric(1234.5678) == "1234.6"
    assert show_metric(0.5, "{:.2f}") == "0.50"
    assert show_metric(0.0) == "0.0"  # a measured zero stays a zero
