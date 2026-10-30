"""Tests of the figure and table notes built from the captions of the stages."""

import csv
from pathlib import Path

import pytest

from wut_murl_craftax.analysis import captions
from wut_murl_craftax.visualisation import charts


def _write(folder: Path, fid: str, *rows: tuple[str, str, str]) -> None:
    wide = charts.WIDTHS
    caps = [charts.Caption(s, t, d, wide[charts.report_width(s)]) for s, t, d in rows]
    charts.write_captions(folder, fid, caps)


@pytest.fixture
def stages(tmp_path: Path) -> dict[str, Path]:
    results = tmp_path / "outputs" / "results"
    analysis = tmp_path / "outputs" / "analysis"
    report = tmp_path / "report"
    figures = results / "figures" / "1m"
    _write(
        figures,
        "F1",
        ("curves_score_main", "Training score", "Mean score."),
        ("curves_score_skills", "Training score, skills", "Mean score, skills."),
    )
    _write(figures, "F10", ("splits", "Speedrun splits", "Steps to a milestone."))
    _write(figures, "LEG", ("legend_methods", "Method legend", "Colour of a method."))
    _write(figures, "F27", ("distil_score", "Distillation", "Fidelity against score."))
    for stem in ("curves_score_main", "splits", "legend_methods"):
        (report / "figures" / "1m").mkdir(parents=True, exist_ok=True)
        (report / "figures" / "1m" / f"{stem}.png").write_bytes(b"x")
    tables = results / "tables" / "1m"
    _write(tables, "T1", ("main", "Main results", "Final score per method."))
    _write(tables, "T7", ("sensitivity", "Sensitivity", "Per tuning phase."))
    (tables / "main.csv").write_text("a\n1\n")
    (tables / "main.tex").write_text("x")
    (report / "tables" / "1m").mkdir(parents=True)
    (report / "tables" / "1m" / "main.tex").write_text("x")
    plots = analysis / "1m" / "ppo" / "plots"
    _write(plots, "HARD", ("floor_reach", "Floor reach", "Reach per floor."))
    (plots / "floor_reach.png").write_bytes(b"x")
    return {"results": results, "analysis": analysis, "repo": tmp_path}


def test_the_figure_note_groups_by_figure_id_in_natural_order(
    stages: dict[str, Path],
) -> None:
    text = captions.figures_note("1m", **stages, today="2026-10-28")
    assert text.startswith('---\ntitle: "Figures 1M"')
    order = [
        text.index(h)
        for h in ("### Method legend", "### Learning curves", "### Speedrun splits")
    ]
    assert order == sorted(order)  # LEG, F1, F10
    assert "## Comparison" in text
    assert "## Interpretation" in text
    assert text.index("## Interpretation") < text.index("### Distillation and score")
    assert "## Training plots" in text
    assert "### ppo" in text


def test_each_row_is_a_figure_link_and_the_title_with_the_description(
    stages: dict[str, Path],
) -> None:
    text = captions.figures_note("1m", **stages, today="2026-10-28")
    row = next(ln for ln in text.splitlines() if "curves_score_main" in ln)
    assert row == (
        "| [curves_score_main](../../report/figures/1m/curves_score_main.png) | "
        "80 mm | **Training score.** Mean score. |"
    )
    plot = next(ln for ln in text.splitlines() if "floor_reach" in ln)
    assert (
        "`outputs/analysis/1m/ppo/plots/floor_reach.png`" in plot
    )  # untracked: no link
    assert "](../../outputs/" not in text


def test_the_table_note_links_the_csv_and_the_tex_that_exist(
    stages: dict[str, Path],
) -> None:
    text = captions.tables_note(
        "1m", stages["results"], stages["repo"], today="2026-10-28"
    )
    main = next(ln for ln in text.splitlines() if "Final score per method" in ln)
    assert "`outputs/results/tables/1m/main.csv`" in main  # untracked: no link
    assert "[main.tex](../../report/tables/1m/main.tex)" in main
    sens = next(ln for ln in text.splitlines() if "Per tuning phase" in ln)
    assert "sensitivity.csv" not in sens  # no file: the stem alone, never a dead link
    assert "## Main results" in text
    assert "## Tuning" in text


def test_the_notes_have_no_long_dash_and_no_other_budget(
    stages: dict[str, Path],
) -> None:
    for note in (
        captions.figures_note("1m", **stages, today="2026-10-28"),
        captions.tables_note("1m", stages["results"], stages["repo"], "2026-10-28"),
    ):
        assert chr(0x2014) not in note
        assert "1B" not in note
        assert "pending" not in note


def test_write_notes_writes_both_files(stages: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "knowledge" / "results"
    paths = captions.write_notes("1m", **stages, out_dir=out, today="2026-10-28")
    assert {p.name for p in paths} == {"Figures 1M.md", "Tables 1M.md"}
    for p in paths:
        assert p.read_text().startswith("---\n")


def test_every_figure_links_its_one_png_and_names_its_width(
    stages: dict[str, Path],
) -> None:
    folder = stages["repo"] / "report" / "figures" / "1m"
    (folder / "order_ppo.png").write_bytes(b"x")
    _write(
        stages["results"] / "figures" / "1m",
        "F7G",
        ("order_ppo", "Order graph", "A graph."),
    )
    text = captions.figures_note("1m", **stages, today="2026-10-28")
    assert "_double" not in text
    assert "| Figure | Width | Description |" in text
    assert "(../../report/figures/1m/order_ppo.png) | 170 mm | **Order graph.**" in text
    assert "(../../report/figures/1m/splits.png) | 80 mm | **Speedrun splits.**" in text


def test_a_caption_without_a_recorded_width_shows_a_dash(
    stages: dict[str, Path],
) -> None:
    charts.write_captions(
        stages["results"] / "figures" / "1m",
        "TL",
        [charts.Caption("timeline", "T", "A.")],
    )
    text = captions.figures_note("1m", **stages, today="2026-10-28")
    assert "| `report/figures/1m/timeline.png` | - | **T.** A. |" in text


def test_a_missing_captions_file_is_an_empty_group(tmp_path: Path) -> None:
    assert captions.read_captions(tmp_path / "none.csv") == []
    text = captions.tables_note("1m", tmp_path, tmp_path, "2026-10-28")
    assert text.count("|") == 0
    with (tmp_path / "x.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["stem", "id", "title", "description"])
    assert captions.read_captions(tmp_path / "x.csv") == []


def test_a_file_that_does_not_exist_is_named_without_a_link(
    stages: dict[str, Path],
) -> None:
    """The vault check rejects dead links: the skills curves have no report PNG."""
    text = captions.figures_note("1m", **stages, today="2026-10-28")
    row = next(ln for ln in text.splitlines() if "Mean score, skills" in ln)
    assert row.startswith("| `report/figures/1m/curves_score_skills.png` |")
    assert "](" not in row
