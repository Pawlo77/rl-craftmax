"""Tests for the reproduction check: the acceptance band and the MLflow-backed CLI."""

import math
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
import yaml

from wut_murl_craftax.evaluation import reproduction
from wut_murl_craftax.evaluation.reproduction import band, check, main
from wut_murl_craftax.orchestration import queue
from wut_murl_craftax.orchestration.queue import (
    config_hash,
    load_method,
    resolve_config,
)

METHODS = (
    Path(__file__).parents[1] / "orchestration" / "fixtures" / "configs" / "methods"
)


def seeds(n: int, mean: float, sd: float) -> list[float]:
    """``n`` scores with exactly this mean and sample standard deviation."""
    z = np.linspace(-1, 1, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return list(mean + sd * z)


# --- band -----------------------------------------------------------------------


def test_planning_band_at_1m_is_0_124_wide_on_each_side() -> None:
    h, _ = band(seeds(10, 2.2, 0.083), 2.2, 0.083)
    assert h == pytest.approx(0.124, abs=1e-3)


def test_a_mean_off_by_more_than_the_half_width_fails() -> None:
    assert not band(seeds(10, 2.40, 0.083), 2.2, 0.083)[1]
    assert band(seeds(10, 2.25, 0.083), 2.2, 0.083)[1]
    assert band(seeds(10, 2.15, 0.083), 2.2, 0.083)[1]  # below passes symmetrically
    assert not band(seeds(10, 2.0, 0.083), 2.2, 0.083)[1]


def test_three_seeds_at_1b_widen_the_band() -> None:
    h3, _ = band(seeds(3, 11.9, 0.91), 11.9, 0.91)
    h10, _ = band(seeds(10, 11.9, 0.91), 11.9, 0.91)
    assert h3 == pytest.approx(1.25, abs=0.01)
    assert h10 == pytest.approx(0.86, abs=0.01)
    assert h3 > h10


def test_the_half_width_follows_the_formula() -> None:
    ours = [1.0, 2.0, 4.0, 5.0]
    se = float(np.std(ours, ddof=1)) / math.sqrt(4)
    want = 0.05 + 2 * math.sqrt(se**2 + (0.3 / math.sqrt(10)) ** 2)
    assert band(ours, 3.0, 0.3)[0] == pytest.approx(want)


def test_one_seed_has_no_standard_error() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        band([2.2], 2.2, 0.083)


# --- the CLI over an MLflow store ------------------------------------------------


class Store:
    """A throwaway MLflow store with finished runs tagged as the runner does."""

    def __init__(self, tmp_path: Path) -> None:
        self.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self.exp = self.client.create_experiment(
            "tune-coarse", artifact_location=(tmp_path / "art").as_uri()
        )

    def add(
        self,
        cfg: dict[str, Any],
        score: float,
        *,
        phases: str = "tune-coarse",
        status: str = "FINISHED",
        budget: str = "1m",
        method: str = "toy",
    ) -> str:
        tags = {
            queue.TAG_METHOD: method,
            queue.TAG_CONFIG_HASH: config_hash(cfg),
            queue.TAG_PHASES: phases,
            queue.TAG_BUDGET: budget,
        }
        rid = self.client.create_run(self.exp, tags=tags).info.run_id
        self.client.log_metric(rid, queue.SCORE_METRIC, score)
        self.client.set_terminated(rid, status)
        return rid

    def tags(self, rid: str) -> dict[str, str]:
        return dict(self.client.get_run(rid).data.tags)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "published.yaml"
    path.write_text(
        yaml.safe_dump({"toy": {"1m": {"score": 2.2, "s_pub": 0.083}}}),
    )
    return path


def published(budget: str = "1m") -> dict[str, Any]:
    method = load_method("toy", METHODS)
    return resolve_config(method, "toy", budget, ref="published", methods_dir=METHODS)


def run_check(store: Store, reference: Path, **kw: Any) -> Any:
    return check(
        "toy",
        "1m",
        "tune-coarse",
        tracking_uri=store.tracking_uri,
        methods_dir=METHODS,
        reference=reference,
        **kw,
    )


def test_check_reads_the_published_config_runs_and_tags_them(
    store: Store, reference: Path
) -> None:
    ids = [store.add(published(), s) for s in seeds(3, 2.25, 0.05)]
    other = store.add(
        resolve_config(
            load_method("toy", METHODS), "toy", "1m", ref="lr-hi", methods_dir=METHODS
        ),
        9.0,
    )
    failed = store.add(published(), 99.0, status="FAILED")
    elsewhere = store.add(published(), 99.0, phases="final-1m")

    result = run_check(store, reference)

    assert result.passed
    assert result.n == 3
    assert result.mean == pytest.approx(2.25)
    for rid in ids:
        tags = store.tags(rid)
        assert tags["repro/passed"] == "true"
        assert float(tags["repro/mean"]) == pytest.approx(2.25)
        assert float(tags["repro/half_width"]) == pytest.approx(result.half_width)
        assert float(tags["repro/published"]) == 2.2
        assert tags["repro/n"] == "3"
    for rid in (other, failed, elsewhere):
        assert not any(k.startswith("repro/") for k in store.tags(rid))


def test_check_fails_outside_the_band_and_main_exits_nonzero(
    store: Store, reference: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = [store.add(published(), s) for s in seeds(10, 2.6, 0.083)]
    assert not run_check(store, reference).passed
    assert store.tags(ids[0])["repro/passed"] == "false"
    argv = ["--method", "toy", "--budget", "1m", "--phase", "tune-coarse"]
    argv += ["--tracking-uri", store.tracking_uri, "--methods-dir", str(METHODS)]
    argv += ["--reference", str(reference)]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "2.6" in out


def test_check_needs_two_finished_runs(store: Store, reference: Path) -> None:
    store.add(published(), 2.2)
    with pytest.raises(ValueError, match="at least 2"):
        run_check(store, reference)


def test_check_needs_a_published_score_for_the_budget(
    store: Store, reference: Path
) -> None:
    with pytest.raises(ValueError, match="1b"):
        check(
            "toy",
            "1b",
            "tune-coarse",
            tracking_uri=store.tracking_uri,
            methods_dir=METHODS,
            reference=reference,
        )


def test_the_published_config_at_1b_is_the_base_with_the_budget_block(
    store: Store, reference: Path
) -> None:
    text = yaml.safe_load(reference.read_text())
    text["toy"]["1b"] = {"score": 11.9, "s_pub": 0.91}
    reference.write_text(yaml.safe_dump(text))
    for s in seeds(3, 11.9, 0.91):
        store.add(published("1b"), s, phases="final-1b", budget="1b")
    result = check(
        "toy",
        "1b",
        "final-1b",
        tracking_uri=store.tracking_uri,
        methods_dir=METHODS,
        reference=reference,
    )
    assert result.passed
    assert result.half_width == pytest.approx(1.25, abs=0.01)
    assert reproduction.SCORE_METRIC == queue.SCORE_METRIC
