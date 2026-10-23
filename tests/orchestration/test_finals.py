"""Tests for the dyna-q kappa ablation of final-1m once dyna-q is marked tuned."""

import re
import shutil
from pathlib import Path
from typing import Any

import mlflow
import pytest
import yaml

from wut_murl_craftax.orchestration import finals, queue
from wut_murl_craftax.orchestration.finals import kappa_ablation
from wut_murl_craftax.orchestration.queue import expand_phase

REPO = Path(__file__).resolve().parents[2]
SCORES = {0.0: 1.0, 0.003: 1.2, 0.03: 1.5}
"""Mean stage-1 score by kappa; the value-iteration k0.003 config beats them all."""


def _methods(out: Path, kappa: float) -> Path:
    out.mkdir()
    for path in (REPO / "configs" / "methods").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.stem == "dyna-q":
            raw["tuned"] = {"plan_n": 20, "plan_kappa": kappa}
        if path.stem == "smdp-skills":
            raw["tuned"] = raw["tuned"] or {"alpha": 0.05}
        (out / path.name).write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


@pytest.fixture
def phases(tmp_path: Path) -> Path:
    """tune-coarse as shipped; final-1m with the kappa-0 item, whatever the shipped
    file says now (the chain may have rewritten it)."""
    out = tmp_path / "phases"
    out.mkdir()
    shutil.copy(REPO / "configs" / "phases" / "tune-coarse.yaml", out)
    text = (REPO / "configs" / "phases" / "final-1m.yaml").read_text(encoding="utf-8")
    line = f"  dyna-q: {{configs: [tuned, {finals.ZERO_ITEM}]}}"
    text = re.sub(r"^  dyna-q: .*$", line, text, flags=re.MULTILINE)
    (out / "final-1m.yaml").write_text(text, encoding="utf-8")
    return out


def _store(tmp_path: Path, methods: Path, phases: Path) -> str:
    """A store with every dyna-q tune-coarse run finished at `SCORES`."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    exp = client.create_experiment("tune-coarse", (tmp_path / "art").as_uri())
    for spec in expand_phase(
        phases / "tune-coarse.yaml", method="dyna-q", methods_dir=methods
    ):
        kappa = spec.config["plan_kappa"]
        vi = spec.config["plan_variant"] == "value_iteration" and kappa == 0.003
        tags = {
            queue.TAG_METHOD: "dyna-q",
            queue.TAG_CONFIG_HASH: spec.config_hash,
            queue.TAG_PHASES: "tune-coarse",
        }
        run = client.create_run(exp, run_name=spec.name, tags=tags)
        score = 2.0 if vi else SCORES[kappa]
        client.log_metric(run.info.run_id, queue.SCORE_METRIC, score)
        client.set_terminated(run.info.run_id, "FINISHED")
    return uri


def _ablate(tmp_path: Path, kappa: float, phases: Path, **kw: Any) -> float | None:
    methods = _methods(tmp_path / "methods", kappa)
    uri = _store(tmp_path, methods, phases)
    return kappa_ablation(
        phases / "final-1m.yaml",
        tracking_uri=uri,
        methods_dir=methods,
        phases_dir=phases,
        **kw,
    )


def test_a_nonzero_tuned_kappa_keeps_the_zero_ablation(
    tmp_path: Path, phases: Path
) -> None:
    before = (phases / "final-1m.yaml").read_text(encoding="utf-8")
    assert _ablate(tmp_path, 0.003, phases) is None
    assert (phases / "final-1m.yaml").read_text(encoding="utf-8") == before


def test_a_zero_tuned_kappa_ablates_to_the_best_nonzero_stage1_kappa(
    tmp_path: Path, phases: Path
) -> None:
    path = phases / "final-1m.yaml"
    before = path.read_text(encoding="utf-8")
    with pytest.raises(queue.QueueConfigError, match="duplicate run"):
        expand_phase(path, methods_dir=_methods(tmp_path / "m0", 0.0))
    assert _ablate(tmp_path, 0.0, phases) == 0.003  # the vi config, not the 0.03 mean
    after = path.read_text(encoding="utf-8")
    assert after == before.replace(
        "set: {plan_kappa: 0.0}}]}", "set: {plan_kappa: 0.003}}]}"
    )
    specs = expand_phase(path, method="dyna-q", methods_dir=tmp_path / "methods")
    assert sorted({s.config["plan_kappa"] for s in specs}) == [0.0, 0.003]


def test_the_dry_run_leaves_the_phase_file(tmp_path: Path, phases: Path) -> None:
    before = (phases / "final-1m.yaml").read_text(encoding="utf-8")
    assert _ablate(tmp_path, 0.0, phases, dry_run=True) == 0.003
    assert (phases / "final-1m.yaml").read_text(encoding="utf-8") == before


def test_main_prints_the_ablation(
    tmp_path: Path, phases: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    methods = _methods(tmp_path / "methods", 0.0)
    uri = _store(tmp_path, methods, phases)
    finals.main([
        "--phase",
        str(phases / "final-1m.yaml"),
        "--tracking-uri",
        uri,
        "--methods-dir",
        str(methods),
        "--phases-dir",
        str(phases),
    ])
    assert "plan_kappa 0.003" in capsys.readouterr().out


def test_min_complete_leaves_out_a_coarse_config_aborted_for_good(
    tmp_path: Path, phases: Path
) -> None:
    methods = _methods(tmp_path / "methods", 0.0)
    uri = _store(tmp_path, methods, phases)
    client = mlflow.MlflowClient(tracking_uri=uri)
    vi = [
        r
        for r in client.search_runs([
            e.experiment_id for e in client.search_experiments()
        ])
        if r.data.metrics[queue.SCORE_METRIC] == 2.0
    ]
    client.set_terminated(vi[0].info.run_id, "FAILED")
    client.set_tag(vi[0].info.run_id, "abort/guard", "dyna/divergence")
    kw = {"tracking_uri": uri, "methods_dir": methods, "phases_dir": phases}
    with pytest.raises(queue.QueueConfigError, match="incomplete tune-coarse"):
        kappa_ablation(phases / "final-1m.yaml", dry_run=True, **kw)
    # the vi k0.003 config is out: the 0.03 configs (mean 1.5) win
    got = kappa_ablation(phases / "final-1m.yaml", min_complete=0.9, dry_run=True, **kw)
    assert got == 0.03
