"""Tests for the comparison CLI: every figure and table, resume, MLflow, partial runs.

The CLI runs on the synthetic export of all four final phases and both tuning phases
(`_analysis_export`; the 1B phases are listed in one test and absent in another, R86),
a fixture ``complexity.csv`` and a temporary MLflow store. The published scores of the
reproduction gate are replaced so that one control fails: the run must still write
everything.
"""

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from _analysis_export import write_export

from wut_murl_craftax.analysis import results
from wut_murl_craftax.analysis.results import (
    ResultsOptions,
    ResultsSummary,
    run_comparison,
)

PHASES = ("final-1m", "final-1b", "combo-1m", "combo-1b")
TUNING = ("tune-coarse", "tune-fine")
METHODS = (
    "count-bonus-pqn-rnn", "count-bonus-ppo-rnn", "combo", "dyna-q", "ppo",
    "ppo-rnn", "pqn-rnn", "sil", "smdp-skills",
)  # fmt: skip
STEMS = [
    "curves_score", "aggregate", "poi_matrix", "seed_spread", "success_heatmap",
    "sensitivity", "stability", "hardware", "race_steps", "race_wall",
    "order_agreement", "funnel", "funnel_training", "deaths", "stalls", "splits",
    "coverage", "survival", "combat", "actions", "resources", "advanced", "radar_1m",
    "radar_1b", *(f"order_{m}" for m in METHODS),
]  # fmt: skip
"""The figure file stems of the plan's list (F1 to F20), one per file."""
VARIANTS = ("", "_double", "_video")
"""Column report figure, double-width report figure, dark video figure."""
TABLES = [
    "main", "literature", "stability", "hardware", "game_summary", "sensitivity",
    "ablation", "tuning_optimism", "race", "poi_matrix", "reproduction_gate",
]  # fmt: skip
"""Table stems written as CSV."""
COMPLEXITY = {
    "budget": "1m", "method": "sil", "config_ref": "tuned", "variant": "base",
    "n_seeds": 3, "l80_median": 8.0, "l90_median": 32.0, "l95_median": float("inf"),
    "l90_lo": 16.0, "l90_hi": 64.0, "l90_spread": "iqr", "depth_at_l90_median": 5.0,
    "fidelity_16": 0.71234, "fidelity_64": 0.8, "fidelity_256": 0.9,
    "expected_fidelity_256": 0.95, "closed_loop_256": 0.5,
    "spearman_score_l90": -0.4321, "spearman_p": 0.2,
}  # fmt: skip
NO_MEMORY = {
    **COMPLEXITY, "variant": "no-memory", "l90_median": 64.0, "fidelity_256": 0.8,
}  # fmt: skip
"""The ``no-memory`` row every policy has in the real ``complexity.csv``."""
N_BOOT = 40
"""Bootstrap resamples in tests."""


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_export(tmp_path_factory.mktemp("metrics"), (*PHASES, *TUNING))


@pytest.fixture(scope="module")
def interpretation(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("interpretation")
    (root / "1m").mkdir()
    pd.DataFrame([COMPLEXITY, NO_MEMORY]).to_csv(
        root / "1m" / "complexity.csv", index=False
    )
    (root / "1m" / "fidelity_vs_leaves_1m.pdf").write_bytes(b"%PDF-1.4 fidelity")
    return root


@pytest.fixture(scope="module")
def published(export: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ppo-rnn at its own 1M mean (pass), ppo 3 points off (a fail)."""
    runs = pd.read_parquet(export / "final-1m" / "runs.parquet")
    mean = runs[runs["config_ref"] == "tuned"].groupby("method")["final_score"].mean()
    ref = {
        "ppo": {"1m": {"score": float(mean["ppo"]) + 3, "s_pub": 0.1}},
        "ppo-rnn": {"1m": {"score": float(mean["ppo-rnn"]), "s_pub": 0.1}},
        "pqn-rnn": {"1b": {"score": 16.0, "s_pub": 1.31}},
    }
    path = tmp_path_factory.mktemp("published") / "published.yaml"
    path.write_text(yaml.safe_dump(ref))
    return path


@pytest.fixture(scope="module")
def fixture_hash(export: Path) -> Callable[[str, str], str]:
    """The tuned `config_hash` of a control, standing in for the registry's."""
    runs = pd.concat([
        pd.read_parquet(export / p / "runs.parquet") for p in ("final-1m", "final-1b")
    ])
    tuned = runs[runs["config_ref"] == "tuned"]
    by = {(r.method, r.budget): r.config_hash for r in tuned.itertuples()}
    return lambda control, budget: by[control, budget]


@pytest.fixture(scope="module")
def base(
    export: Path,
    interpretation: Path,
    published: Path,
    fixture_hash: Callable[[str, str], str],
    tmp_path_factory: pytest.TempPathFactory,
) -> ResultsOptions:
    """ResultsOptions of a full run with a failing gate and a temporary MLflow store."""
    root = tmp_path_factory.mktemp("run")
    store = root / "store" / "mlruns.db"
    return ResultsOptions(
        phases=PHASES, tuning=TUNING, out=root / "results", report_dir=root / "report",
        metrics_root=export, complexity_root=interpretation,
        tracking_uri=f"sqlite:///{store}?timeout=300", n_boot=N_BOOT,
        published=published, published_hash=fixture_hash,
    )  # fmt: skip


@pytest.fixture(scope="module")
def full(base: ResultsOptions) -> ResultsSummary:
    """One full run, shared by the tests that only read its outputs."""
    return run_comparison(base)


def _files(folder: Path, stems: list[str], suffixes: tuple[str, ...]) -> list[Path]:
    return [
        folder / f"{s}{v}.{ext}"
        for s in stems
        for v in suffixes
        for ext in ("pdf", "png")
    ]


# --- outputs ----------------------------------------------------------------------


def test_a_full_run_writes_every_figure_in_both_themes_and_widths(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    missing = [
        p.name for p in _files(base.out / "figures", STEMS, VARIANTS) if not p.is_file()
    ]
    assert not missing
    assert not full.problems


def test_a_full_run_writes_every_table_as_csv_and_latex(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    tables = base.out / "tables"
    for stem in TABLES:
        assert (tables / f"{stem}.csv").is_file(), stem
    for stem in ("main", "literature", "game_summary", "stability_hardware"):
        assert (tables / f"{stem}.tex").is_file(), stem
    main = pd.read_csv(tables / "main.csv")
    assert set(main["block"]) == {"main", "skills"}
    assert set(main["benchmark"]) == {"1m", "1b"}
    assert set(pd.read_csv(tables / "sensitivity.csv")["phase"]) == set(TUNING)


def test_a_failing_reproduction_gate_still_writes_everything(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    gate = pd.read_csv(base.out / "tables" / "reproduction_gate.csv")
    row = gate.set_index(["control", "benchmark"])
    assert row.loc[("ppo", "1m"), "status"] == "fail"
    assert row.loc[("ppo-rnn", "1m"), "status"] == "pass"
    assert row.loc[("ppo", "1m"), "gap"] == pytest.approx(-3.0)
    assert "ppo" in " ".join(full.failed_gates)
    assert (base.out / "figures" / "curves_score.pdf").is_file()
    assert (base.report_dir / "numbers.tex").is_file()


def test_the_report_folder_gets_tables_numbers_and_report_figures(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    report = base.report_dir
    assert (report / "tables" / "main.tex").read_text() == (
        base.out / "tables" / "main.tex"
    ).read_text()
    numbers = (report / "numbers.tex").read_text()
    assert r"\scoreOneMPpo}" in numbers
    assert r"\closedLoopOneMSil}" in numbers
    assert r"\closedLoopOneMSilNoMemory}" in numbers
    figures = {p.name for p in (report / "figures").glob("*.pdf")}
    assert "curves_score_double.pdf" in figures
    assert "fidelity_vs_leaves_1m.pdf" in figures
    assert not [n for n in figures if n.endswith("_video.pdf")]


def test_the_fidelity_figure_is_copied_into_the_results_figures(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    got = base.out / "figures" / "fidelity_vs_leaves_1m.pdf"
    assert got.read_bytes() == b"%PDF-1.4 fidelity"


def test_no_text_output_has_the_long_dash(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    dash = chr(0x2014)
    texts = [
        *(base.out / "tables").glob("*"),
        *base.report_dir.rglob("*.tex"),
        Path(results.__file__),
    ]
    assert [p.name for p in texts if dash in p.read_text()] == []


# --- MLflow -----------------------------------------------------------------------


def test_the_run_is_logged_as_comparison_final_in_experiment_analysis(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    client = mlflow.MlflowClient(base.tracking_uri)
    exp = client.get_experiment_by_name("analysis")
    assert exp is not None
    (found,) = client.search_runs(
        [exp.experiment_id], "attributes.run_name = 'comparison-final'"
    )
    assert found.info.run_id == full.run_id
    assert found.data.tags["analysis/kind"] == "comparison"
    assert found.data.tags["analysis/phases"] == ",".join(PHASES)
    assert not {"phases", "method", "budget", "seed", "config_hash"} & set(
        found.data.tags
    )
    top = {a.path for a in client.list_artifacts(found.info.run_id)}
    assert top == {"tables", "figures"}
    tables = {a.path for a in client.list_artifacts(found.info.run_id, "tables")}
    assert "tables/main.csv" in tables
    figures = {a.path for a in client.list_artifacts(found.info.run_id, "figures")}
    assert "figures/curves_score.pdf" in figures


# --- resume and idempotence ---------------------------------------------------------


def test_a_rerun_skips_finished_stages_and_keeps_the_outputs_and_the_run(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    tables = base.out / "tables" / "main.csv"
    pdf = base.out / "figures" / "curves_score.pdf"
    before = (tables.read_bytes(), pdf.stat().st_mtime_ns)
    again = run_comparison(base)
    assert not again.done
    assert {"figures/F1", "log"} <= set(again.skipped)
    assert (tables.read_bytes(), pdf.stat().st_mtime_ns) == before
    assert again.run_id == full.run_id
    client = mlflow.MlflowClient(base.tracking_uri)
    exp = client.get_experiment_by_name("analysis")
    assert exp is not None
    assert len(client.search_runs([exp.experiment_id])) == 1
    assert (base.report_dir / "numbers.tex").is_file()


def test_a_stage_without_its_marker_is_redone_alone(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    (base.out / ".stages" / "figures" / "F5.json").unlink()
    again = run_comparison(base)
    assert again.done == ["figures/F5", "log"]
    assert "figures/F1" in again.skipped


def _two_stub_figures(
    monkeypatch: pytest.MonkeyPatch, base: ResultsOptions, tmp_path: Path
) -> ResultsOptions:
    """Options with their own folders; figure ids F1 and F6 draw nothing real."""

    def stub(fid: str) -> Callable[..., None]:
        def draw(frames: object, folder: Path, theme: str, *, width: str, **kw: object):
            (folder / f"{fid}_{theme}_{width}.pdf").write_bytes(b"x")

        return draw

    figures = {"F1": stub("F1"), "F6": stub("F6")}
    monkeypatch.setattr(results, "figure_functions", lambda: figures)
    return replace(base, out=tmp_path / "r", report_dir=tmp_path / "p", log=False)


def test_changed_inputs_redo_the_tables_and_the_figures(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _two_stub_figures(monkeypatch, base, tmp_path)
    assert run_comparison(opts).done == ["figures/F1", "figures/F6"]
    assert run_comparison(opts).skipped == ["figures/F1", "figures/F6"]
    again = run_comparison(replace(opts, n_boot=N_BOOT + 1))
    assert again.done == ["figures/F1", "figures/F6"]


def test_a_deleted_output_redoes_its_stage(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _two_stub_figures(monkeypatch, base, tmp_path)
    run_comparison(opts)
    (opts.out / "figures" / "F1_video_column.pdf").unlink()
    (opts.out / "tables" / "main.csv").unlink()
    again = run_comparison(opts)
    assert (opts.out / "tables" / "main.csv").is_file()  # tables are rewritten
    assert again.done == ["figures/F1"]
    assert again.skipped == ["figures/F6"]


def test_force_redoes_every_stage(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _two_stub_figures(monkeypatch, base, tmp_path)
    run_comparison(opts)
    again = run_comparison(replace(opts, force=True))
    assert not again.skipped
    assert again.done == ["figures/F1", "figures/F6"]


def test_a_changed_source_invalidates_the_stage_markers(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _two_stub_figures(monkeypatch, base, tmp_path)
    source = tmp_path / "figures_source.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(results, "code_files", lambda: [source])
    assert run_comparison(opts).done == ["figures/F1", "figures/F6"]
    assert run_comparison(opts).skipped == ["figures/F1", "figures/F6"]
    source.write_text("x = 2\n")
    assert run_comparison(opts).done == ["figures/F1", "figures/F6"]


def test_the_code_digest_covers_the_analysis_and_stats_sources() -> None:
    names = {p.name for p in results.code_files()}
    assert {"compare.py", "figures.py", "figures_game.py", "stats.py"} <= names


def test_the_default_published_scores_are_in_the_fingerprint(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "published.yaml"
    default.write_text("ppo: {}\n")
    monkeypatch.setattr(results, "default_published", lambda: default)
    opts = replace(base, published=None, out=tmp_path / "r", log=False)
    before = results._Inputs(opts).fingerprint()
    default.write_text("ppo: {1m: {score: 1.0}}\n")
    assert results._Inputs(opts).fingerprint() != before


# --- partial and failing runs -------------------------------------------------------


def test_a_missing_1m_complexity_csv_stops_the_cli_before_any_work(
    export: Path, tmp_path: Path
) -> None:
    out = tmp_path / "results"
    argv = [
        "--phases", "final-1m", "--tuning", *TUNING, "--out", str(out),
        "--report-dir", str(tmp_path / "report"), "--metrics-root", str(export),
        "--complexity-root", str(tmp_path / "none"), "--no-log",
    ]  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        results.main(argv)
    message = str(stop.value)
    assert str(tmp_path / "none" / "1m" / "complexity.csv") in message
    assert "--no-complexity" in message
    assert not out.exists()


def test_no_complexity_skips_only_what_depends_on_it(
    export: Path, base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_stub_figures(monkeypatch, base, tmp_path)
    out, report = tmp_path / "results", tmp_path / "report"
    results.main([
        "--phases", "final-1m", "final-1b", "--tuning", *TUNING, "--out", str(out),
        "--report-dir", str(report), "--metrics-root", str(export),
        "--complexity-root", str(tmp_path / "none"), "--no-complexity", "--no-log",
        "--n-boot", str(N_BOOT),
    ])  # fmt: skip
    numbers = (report / "numbers.tex").read_text()
    assert r"\scoreOneMPpo}" in numbers
    assert "closedLoop" not in numbers
    assert "spearman" not in numbers
    assert (out / "tables" / "main.csv").is_file()
    assert (out / "tables" / "sensitivity.csv").is_file()
    assert (out / "figures" / "F1_report_column.pdf").is_file()
    assert not list((out / "figures").glob("fidelity*"))
    assert not list((report / "figures").glob("fidelity*"))
    assert not (out / ".stages" / "log.json").exists()


def test_phases_never_exported_are_skipped_not_fatal(
    export: Path, tmp_path: Path
) -> None:
    out = tmp_path / "results"
    results.main([
        "--phases", "final-1m", "final-9b", "--tuning", "tune-coarse", "--out",
        str(out), "--report-dir", str(tmp_path / "report"), "--metrics-root",
        str(export), "--no-complexity", "--no-log", "--n-boot", str(N_BOOT),
    ])  # fmt: skip
    main = pd.read_csv(out / "tables" / "main.csv")
    assert set(main["benchmark"]) == {"1m"}


def test_missing_tuning_phases_are_an_error_naming_them(
    export: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="tune-never"):
        results.main([
            "--phases", "final-1m", "--tuning", "tune-never", "--out",
            str(tmp_path / "r"), "--report-dir", str(tmp_path / "p"),
            "--metrics-root", str(export), "--no-complexity", "--no-log",
        ])  # fmt: skip


def test_an_undefined_report_macro_fails_the_run_after_writing_and_logging(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_stub_figures(monkeypatch, base, tmp_path)
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.tex").write_text(r"\scoreOneMNoSuchMethod")
    opts = replace(
        base, out=tmp_path / "results", report_dir=report, phases=("final-1m",),
        tuning=("tune-coarse",), n_boot=N_BOOT,
    )  # fmt: skip
    summary = run_comparison(opts)
    assert any("scoreOneMNoSuchMethod" in p for p in summary.problems)
    assert (report / "numbers.tex").is_file()
    assert summary.run_id
    with pytest.raises(SystemExit):
        results.main([
            "--phases", "final-1m", "--tuning", "tune-coarse", "--out",
            str(opts.out), "--report-dir", str(report), "--metrics-root",
            str(opts.metrics_root), "--complexity-root", str(opts.complexity_root),
            "--no-log", "--n-boot", str(N_BOOT),
        ])  # fmt: skip


def test_logging_needs_a_sqlite_tracking_uri(export: Path, tmp_path: Path) -> None:
    common = ["--phases", "final-1m", "--tuning", "tune-coarse", "--no-complexity"]
    with pytest.raises(SystemExit):
        results.main([*common, "--metrics-root", str(export)])  # no URI, not --no-log
    with pytest.raises(SystemExit):
        results.main([
            *common, "--metrics-root", str(export), "--tracking-uri",
            "http://localhost:5000",
        ])  # fmt: skip


def test_importing_the_cli_imports_neither_jax_nor_mlflow() -> None:
    code = (
        "import sys, wut_murl_craftax.analysis.results\n"
        "bad = [m for m in ('jax', 'mlflow') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_stage_markers_hold_the_input_fingerprint(
    full: ResultsSummary, base: ResultsOptions
) -> None:
    marker = json.loads((base.out / ".stages" / "figures" / "F1.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", marker["fingerprint"])


def _stub_full_then_partial(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ResultsOptions, ResultsOptions]:
    """Options of a full run (stub figures) and the partial ones of its folders."""
    full_opts = replace(
        _two_stub_figures(monkeypatch, base, tmp_path),
        phases=("final-1m",), tuning=("tune-coarse",), n_boot=N_BOOT,
    )  # fmt: skip
    return full_opts, replace(full_opts, complexity=False)


def test_no_complexity_never_overwrites_a_numbers_file_that_has_the_macros(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_opts, partial = _stub_full_then_partial(base, tmp_path, monkeypatch)
    run_comparison(full_opts)
    numbers = full_opts.report_dir / "numbers.tex"
    before = numbers.read_text()
    assert r"\closedLoopOneMSil}" in before
    run_comparison(partial)
    assert numbers.read_text() == before
    side = (full_opts.report_dir / "numbers.partial.tex").read_text()
    assert r"\scoreOneMPpo}" in side
    assert "closedLoop" not in side


def test_no_complexity_removes_stale_fidelity_files_and_does_not_copy_them(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_opts, partial = _stub_full_then_partial(base, tmp_path, monkeypatch)
    run_comparison(full_opts)
    assert (full_opts.out / "figures" / "fidelity_vs_leaves_1m.pdf").is_file()
    (full_opts.report_dir / "figures" / "fidelity_vs_leaves_1m.pdf").unlink()
    run_comparison(partial)
    assert not list((partial.out / "figures").glob("fidelity*"))
    assert not list((partial.report_dir / "figures").glob("fidelity*"))


def test_no_complexity_still_checks_the_other_report_macros(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, partial = _stub_full_then_partial(base, tmp_path, monkeypatch)
    partial.report_dir.mkdir()
    report = partial.report_dir / "report.tex"
    report.write_text(r"\closedLoopOneMSil \scoreOneMPpo")
    assert not run_comparison(partial).problems  # complexity macros are left out
    report.write_text(r"\scoreOneMPpo \scoreOneMNoSuchMethod")
    problems = run_comparison(partial).problems
    assert len(problems) == 1
    assert "scoreOneMNoSuchMethod" in problems[0]


def test_the_log_stage_reruns_for_another_store_or_a_deleted_run(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = replace(
        _two_stub_figures(monkeypatch, base, tmp_path), log=True,
        phases=("final-1m",), tuning=("tune-coarse",), n_boot=N_BOOT,
    )  # fmt: skip
    first = run_comparison(opts)
    assert first.done[-1] == "log"
    assert run_comparison(opts).skipped[-1] == "log"
    mlflow.MlflowClient(opts.tracking_uri).delete_run(str(first.run_id))
    again = run_comparison(opts)
    assert again.done == ["log"]
    assert again.run_id != first.run_id
    elsewhere = replace(
        opts, tracking_uri=f"sqlite:///{tmp_path / 'other' / 'mlruns.db'}?timeout=300"
    )
    assert run_comparison(elsewhere).done == ["log"]


def test_tuning_only_runs_reach_the_tuning_tables_and_no_final_table(
    export_writer: Callable[..., Path],
    base: ResultsOptions,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = ("final-1m", "tune-coarse", "tune-fine")
    clean = export_writer(tmp_path / "clean", phases)
    dirty = export_writer(tmp_path / "dirty", phases)
    runs = pd.read_parquet(dirty / "tune-coarse" / "runs.parquet")
    extra = runs[runs["method"] == "ppo"].copy()
    extra["run_id"] = [f"only{i}".ljust(32, "0") for i in range(len(extra))]
    extra["phases"] = "tune-coarse"
    extra["config_hash"] = extra["config_ref"] = "deadbeef"
    extra["final_score"] = 99.0
    pd.concat([runs, extra]).to_parquet(dirty / "tune-coarse" / "runs.parquet")
    monkeypatch.setattr(results, "figure_functions", dict)
    out = {}
    for name, root in (("clean", clean), ("dirty", dirty)):
        opts = replace(
            base, phases=("final-1m",), tuning=("tune-coarse", "tune-fine"),
            out=tmp_path / name / "results", report_dir=tmp_path / name / "report",
            metrics_root=root, complexity=False, log=False, n_boot=N_BOOT,
        )  # fmt: skip
        run_comparison(opts)
        out[name] = opts.out / "tables"
    for stem in ("main", "ablation", "poi_matrix"):
        assert (out["dirty"] / f"{stem}.csv").read_text() == (
            out["clean"] / f"{stem}.csv"
        ).read_text()
    sens = {n: pd.read_csv(out[n] / "sensitivity.csv") for n in out}
    pick = lambda d: d[(d["method"] == "ppo") & (d["phase"] == "tune-coarse")]  # noqa: E731
    assert (
        pick(sens["dirty"])["configs"].iloc[0]
        == pick(sens["clean"])["configs"].iloc[0] + 1
    )
    assert (
        pick(sens["dirty"])["runs"].iloc[0] == pick(sens["clean"])["runs"].iloc[0] + 3
    )
    optimism = pd.read_csv(out["dirty"] / "tuning_optimism.csv")
    assert optimism["tuning_phase"].notna().any()


def test_an_unexpected_error_in_the_numbers_is_not_swallowed(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wut_murl_craftax.analysis import latex

    def boom(*args: object, **kwargs: object) -> None:
        raise KeyError("no-memory")

    monkeypatch.setattr(latex, "write_numbers", boom)
    opts = replace(
        _two_stub_figures(monkeypatch, base, tmp_path), phases=("final-1m",),
        tuning=("tune-coarse",), n_boot=N_BOOT,
    )  # fmt: skip
    with pytest.raises(KeyError, match="no-memory"):
        run_comparison(opts)


def test_markers_and_numbers_are_written_atomically_and_a_torn_marker_reruns(
    base: ResultsOptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opts = _two_stub_figures(monkeypatch, base, tmp_path)
    run_comparison(opts)
    assert not list(tmp_path.rglob("*.tmp"))
    marker = opts.out / ".stages" / "figures" / "F1.json"
    marker.write_text(marker.read_text()[:20])  # a kill mid-write, before this fix
    again = run_comparison(opts)
    assert again.done == ["figures/F1"]
    assert again.skipped == ["figures/F6"]
    assert json.loads(marker.read_text())["fingerprint"]


def test_the_code_digest_includes_the_pair_inference() -> None:
    assert "interface.py" in {p.name for p in results.code_files()}


def test_write_text_atomic_names_its_temporary_file_after_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from wut_murl_craftax.analysis.load import write_text_atomic

    seen = []
    real = Path.replace

    def spy(self: Path, target: Path) -> Path:
        seen.append(self.name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    write_text_atomic(tmp_path / "m.json", "{}")
    assert seen == [f"m.json.{os.getpid()}.tmp"]
    assert (tmp_path / "m.json").read_text() == "{}"
