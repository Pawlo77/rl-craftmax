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
        extra_tags: dict[str, str] | None = None,
    ) -> str:
        tags = {
            queue.TAG_METHOD: method,
            queue.TAG_CONFIG_HASH: config_hash(cfg),
            queue.TAG_PHASES: phases,
            queue.TAG_BUDGET: budget,
            **(extra_tags or {}),
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


def write_phase(
    root: Path, name: str, budget: str, method: str, ref: str, seeds_n: int
) -> Path:
    """A phases directory with one phase file listing ``seeds_n`` runs of ``ref``."""
    root.mkdir(exist_ok=True)
    body = {"budget": budget, "eval": "final", "seeds": seeds_n}
    body["runs"] = {method: {"configs": [ref]}}
    (root / f"{name}.yaml").write_text(yaml.safe_dump(body))
    return root


@pytest.fixture
def phases(tmp_path: Path) -> Path:
    """tune-coarse lists the published toy config with 3 seeds."""
    return write_phase(tmp_path / "phases", "tune-coarse", "1m", "toy", "published", 3)


def run_check(store: Store, reference: Path, phases: Path, **kw: Any) -> Any:
    return check(
        "toy",
        "1m",
        "tune-coarse",
        tracking_uri=store.tracking_uri,
        methods_dir=METHODS,
        reference=reference,
        phases_dir=phases,
        **kw,
    )


def test_check_reads_the_published_config_runs_and_tags_them(
    store: Store, reference: Path, phases: Path
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

    result = run_check(store, reference, phases)

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


def cli(store: Store, reference: Path, phases: Path, budget: str = "1m") -> list[str]:
    argv = ["--method", "toy", "--budget", budget, "--phase", "tune-coarse"]
    argv += ["--tracking-uri", store.tracking_uri, "--methods-dir", str(METHODS)]
    return [*argv, "--reference", str(reference), "--phases-dir", str(phases)]


def test_check_fails_outside_the_band_and_main_exits_nonzero(
    store: Store, reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phases = write_phase(tmp_path / "ten", "tune-coarse", "1m", "toy", "published", 10)
    ids = [store.add(published(), s) for s in seeds(10, 2.6, 0.083)]
    assert not run_check(store, reference, phases).passed
    assert store.tags(ids[0])["repro/passed"] == "false"
    with pytest.raises(SystemExit) as exit_info:
        main(cli(store, reference, phases))
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "2.6" in out


def test_main_refuses_while_the_phase_has_unfinished_published_runs(
    store: Store,
    reference: Path,
    phases: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ids = [store.add(published(), s) for s in seeds(2, 2.2, 0.05)]
    store.add(published(), 2.2, status="RUNNING")  # the third seed is not done
    with pytest.raises(ValueError, match="2 of 3"):
        run_check(store, reference, phases)
    with pytest.raises(SystemExit) as exit_info:
        main(cli(store, reference, phases))
    assert exit_info.value.code == 2
    assert "refused: 2 of 3" in capsys.readouterr().err
    assert not any(k.startswith("repro/") for k in store.tags(ids[0]))


def test_check_refuses_runs_whose_evaluation_is_pending(
    store: Store, reference: Path, phases: Path
) -> None:
    """A FINISHED run tagged eval/pending (a cut evaluation) is not complete yet."""
    for score in seeds(2, 2.2, 0.05):
        store.add(published(), score)
    pending = store.add(published(), 2.2, extra_tags={queue.TAG_EVAL_PENDING: "true"})
    with pytest.raises(ValueError, match=r"2 of 3.*1 with evaluation pending"):
        run_check(store, reference, phases)
    assert not any(k.startswith("repro/") for k in store.tags(pending))


def test_check_needs_at_least_two_finished_runs(
    store: Store, reference: Path, tmp_path: Path
) -> None:
    one = write_phase(tmp_path / "one", "tune-coarse", "1m", "toy", "published", 1)
    store.add(published(), 2.2)
    with pytest.raises(ValueError, match="at least 2"):
        run_check(store, reference, one)


def test_check_refuses_a_phase_that_lists_other_configs_only(
    store: Store, reference: Path, tmp_path: Path
) -> None:
    other = write_phase(tmp_path / "o", "tune-coarse", "1m", "toy", "lr-hi", 3)
    with pytest.raises(ValueError, match="lists no toy run"):
        run_check(store, reference, other)


def test_check_needs_a_published_score_for_the_budget(
    store: Store, reference: Path, phases: Path
) -> None:
    with pytest.raises(ValueError, match="1b"):
        check(
            "toy",
            "1b",
            "tune-coarse",
            tracking_uri=store.tracking_uri,
            methods_dir=METHODS,
            reference=reference,
            phases_dir=phases,
        )


def test_the_published_config_at_1b_is_the_base_with_the_budget_block(
    store: Store, tmp_path: Path
) -> None:
    """The real ppo method: a final-1b phase entry of its base is found at 1b."""
    method = load_method("ppo")
    base_1b = resolve_config(method, "ppo", "1b")
    assert (base_1b["envs"], base_1b["num_steps"], base_1b["lr"]) == (1024, 64, 2.0e-4)
    base_1m = resolve_config(method, "ppo", "1m")
    phases = write_phase(tmp_path / "p", "final-1b", "1b", "ppo", "base", 3)
    for s in seeds(3, 11.9, 0.91):
        store.add(base_1b, s, phases="final-1b", budget="1b", method="ppo")
    store.add(base_1m, 99.0, phases="final-1b", budget="1b", method="ppo")  # wrong cfg
    result = check(
        "ppo",
        "1b",
        "final-1b",
        tracking_uri=store.tracking_uri,
        phases_dir=phases,
    )
    assert (result.n, result.passed) == (3, True)
    assert result.half_width == pytest.approx(1.25, abs=0.01)
    assert reproduction.SCORE_METRIC == queue.SCORE_METRIC


def test_a_tuned_1b_phase_entry_would_find_no_published_runs(
    store: Store, tmp_path: Path
) -> None:
    """Guards the final-1b phase: it must run the base (Table 7), not `tuned`."""
    method = load_method("ppo")
    tuned = {**resolve_config(method, "ppo", "1b"), "lr": 1.0e-4}
    for s in seeds(3, 11.9, 0.91):
        store.add(tuned, s, phases="final-1b", budget="1b", method="ppo")
    phases = write_phase(tmp_path / "p", "final-1b", "1b", "ppo", "base", 3)
    with pytest.raises(ValueError, match="0 of 3"):
        check(
            "ppo", "1b", "final-1b", tracking_uri=store.tracking_uri, phases_dir=phases
        )
