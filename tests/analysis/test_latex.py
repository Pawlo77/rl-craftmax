"""Tests of the LaTeX export: booktabs tables, number macros and report figures.

The tables come from the comparison fixture (10 seeds at 1M, 3 at 1B); every macro is
read back and compared with the `compare` table cell it must expand to.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _compare_fixture import make_frames
from _game_fixture import game_frames

from wut_murl_craftax.analysis import compare, latex
from wut_murl_craftax.analysis.latex import (
    MACRO_NAMES,
    collect_numbers,
    copy_figures,
    escape,
    macro,
    main_tex,
    tabular,
    undefined_macros,
    write_numbers,
    write_tables,
)

N = 100
"""Bootstrap resamples in tests."""
COMPLEXITY = [
    {
        "budget": "1m", "method": "sil", "config_ref": "tuned", "variant": "base",
        "n_seeds": 3, "l80_median": 8.0, "l90_median": 32.0, "l95_median": np.inf,
        "l90_lo": 16.0, "l90_hi": 64.0, "l90_spread": "iqr", "depth_at_l90_median": 5.0,
        "fidelity_16": 0.71234, "fidelity_64": 0.8, "fidelity_256": 0.9,
        "expected_fidelity_256": 0.95, "closed_loop_256": 0.5,
        "spearman_score_l90": -0.4321, "spearman_p": 0.2,
    },
    {
        "budget": "1m", "method": "dyna-q", "config_ref": "kappa0", "variant": "dagger",
        "n_seeds": 3, "l80_median": 4.5, "l90_median": 16.0, "l95_median": 64.0,
        "l90_lo": 8.0, "l90_hi": 32.0, "l90_spread": "iqr", "depth_at_l90_median": 4.0,
        "fidelity_16": 0.6, "fidelity_64": 0.7, "fidelity_256": 0.8,
        "expected_fidelity_256": 0.85, "closed_loop_256": 0.4,
        "spearman_score_l90": 0.1, "spearman_p": 0.5,
    },
]  # fmt: skip


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return make_frames()


@pytest.fixture(scope="module")
def frames_1m() -> dict[str, pd.DataFrame]:
    return make_frames(("1m",))


@pytest.fixture(scope="module")
def tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """T1, the ablation table and the others, computed once with test bootstraps."""
    return {
        "main": compare.main_table(frames, n_boot=N),
        "ablation": compare.ablation_table(frames, n_boot=N),
        "literature": pd.DataFrame(columns=compare.LITERATURE_TABLE_COLUMNS),
        "stability": compare.stability_table(frames),
        "hardware": compare.hardware_table(frames),
        "game": compare.game_summary_table(game_frames()),
    }


@pytest.fixture
def complexity_root(tmp_path: Path) -> Path:
    root = tmp_path / "interpretation"
    (root / "1m").mkdir(parents=True)
    pd.DataFrame(COMPLEXITY).to_csv(root / "1m" / "complexity.csv", index=False)
    return root


def numbers(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    root: Path,
) -> dict[str, str]:
    return {
        n.name: n.text
        for n in collect_numbers(frames, tables=tables, complexity_root=root)
    }


# --- names ----------------------------------------------------------------------


def test_macro_names_are_hyphen_and_digit_free_camel_case() -> None:
    assert MACRO_NAMES["count-bonus-ppo-rnn"] == "CountBonusPpoRnn"
    assert MACRO_NAMES["dyna-q"] == "DynaQ"
    assert all(re.fullmatch(r"[A-Za-z]+", v) for v in MACRO_NAMES.values())
    assert set(MACRO_NAMES) == set(game_frames()["runs"]["method"])
    assert macro("score", "1b", "ppo-rnn") == "scoreOneBPpoRnn"
    assert (
        macro("iqm", "1m", "smdp-skills", "monte-carlo")
        == "iqmOneMSmdpSkillsMonteCarlo"
    )
    assert macro("poi", "1m", "sil", tail="PpoRnn") == "poiOneMSilPpoRnn"
    assert macro("score", "1m", "dyna-q", "kappa0") == "scoreOneMDynaQKappaZero"


def test_an_unknown_method_budget_or_ref_is_refused() -> None:
    for args in (
        ("score", "1m", "ppo-lstm"),
        ("score", "2m", "ppo"),
        ("score", "1m", "ppo", "wild"),
    ):
        with pytest.raises(ValueError, match="no macro word"):
            macro(*args)


def test_escape_covers_latex_specials() -> None:
    assert escape("a_b & 50% #1 {x} $y") == r"a\_b \& 50\% \#1 \{x\} \$y"
    assert escape("count-bonus-ppo-rnn") == "count-bonus-ppo-rnn"
    assert escape("\\") == r"\textbackslash{}"


# --- tables ---------------------------------------------------------------------


def test_a_table_is_a_booktabs_tabular_with_the_best_value_bold() -> None:
    cols = [
        latex.Column("Method", lambda r: escape(r["m"])),
        latex.Column("score", lambda r: f"{r['v']:.2f}", lambda r: r["v"]),
        latex.Column("p", lambda r: f"{r['p']:.2f}", lambda r: r["p"], higher=False),
    ]
    rows = [{"m": "a_b", "v": 1.0, "p": 0.5}, {"m": "c", "v": 2.0, "p": 0.1}]
    text = tabular(
        cols, [("main block", rows), ("skills", [{"m": "d", "v": 0.5, "p": 0.9}])]
    )
    assert text.startswith(r"\begin{tabular}{lrr}" + "\n" + r"\toprule")
    assert text.rstrip().endswith(r"\bottomrule" + "\n" + r"\end{tabular}")
    assert text.count(r"\midrule") == 2
    assert r"a\_b & 1.00 & 0.50 \\" in text  # escaped, not best
    assert r"c & \textbf{2.00} & \textbf{0.10} \\" in text
    assert r"d & \textbf{0.50} & \textbf{0.90} \\" in text  # best within its own group
    assert r"\multicolumn{3}{l}{\emph{skills}}" in text


def test_ties_are_all_bold_and_nan_never_is() -> None:
    cols = [latex.Column("v", lambda r: str(r["v"]), lambda r: r["v"])]
    text = tabular(cols, [("", [{"v": 3.0}, {"v": 3.0}, {"v": float("nan")}])])
    assert text.count(r"\textbf") == 2


def test_t1_per_benchmark_lists_every_row_with_method_names_escaped(
    tables: dict[str, pd.DataFrame],
) -> None:
    text = main_tex(tables["main"], "1m")
    t1 = tables["main"][tables["main"]["benchmark"] == "1m"]
    assert text.count(r"\toprule") == 1
    assert text.count(r"\bottomrule") == 1
    body = [
        ln for ln in text.splitlines() if ln.endswith(r"\\") and "multicolumn" not in ln
    ]
    assert len(body) == len(t1) + 1  # header
    assert "Uses hand-written skills block" in text
    assert r"mean $\pm$ s.e." in text  # headers are LaTeX, not escaped
    assert "IQM [95\\% CI]" in text
    assert "indicative" not in text
    one_b = main_tex(tables["main"], "1b")
    assert "indicative (3 seeds)" in one_b
    assert "median [95\\% CI]" in one_b
    assert "IQM" not in one_b


def test_the_best_mean_of_a_block_is_bold(tables: dict[str, pd.DataFrame]) -> None:
    t1 = tables["main"]
    text = main_tex(t1, "1m")
    main = t1[(t1["benchmark"] == "1m") & (t1["block"] == "main")]
    best = main.loc[main["mean"].idxmax()]
    line = next(ln for ln in text.splitlines() if ln.startswith(f"{best['method']} &"))
    assert rf"\textbf{{{best['mean']:.2f} $\pm$ {best['se']:.2f}}}" in line
    skills = t1[(t1["benchmark"] == "1m") & (t1["block"] == "skills")]
    top = skills.loc[skills["mean"].idxmax()]
    bold = [ln for ln in text.splitlines() if rf"\textbf{{{top['mean']:.2f} $" in ln]
    assert any(ln.startswith(f"{top['method']} &") for ln in bold)


def test_write_tables_writes_each_file_and_skips_a_missing_1b(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    t = {
        "main": compare.main_table(frames_1m, n_boot=N),
        "literature": pd.DataFrame(columns=compare.LITERATURE_TABLE_COLUMNS),
        "stability": compare.stability_table(frames_1m),
        "hardware": compare.hardware_table(frames_1m),
        "game": compare.game_summary_table(game_frames(("1m",))),
    }
    paths = write_tables(frames_1m, tmp_path, tables=t)
    names = {p.name for p in paths}
    assert names == {
        "main_1m.tex",
        "main_vs_1m.tex",
        "game_summary_1m.tex",
        "stability_hardware.tex",
    }  # fmt: skip
    assert not (tmp_path / "main_1b.tex").exists()
    for p in paths:
        text = p.read_text()
        assert r"\begin{tabular}" in text
        assert r"\toprule" in text
        assert r"\bottomrule" in text


def test_write_tables_with_1b_and_literature(
    frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    lit = pd.DataFrame([{
        "score_kind": "training", "benchmark": "1m", "name": "PPO (paper_x)",
        "protocol": "training-episode return", "seeds": 5, "score_pct": 11.5,
        "spread": 0.7, "spread_kind": "sd",
    }])  # fmt: skip
    paths = write_tables(frames, tmp_path, tables={**tables, "literature": lit})
    assert {"main_1b.tex", "game_summary_1b.tex", "literature.tex"} <= {
        p.name for p in paths
    }
    text = (tmp_path / "literature.tex").read_text()
    assert r"PPO (paper\_x)" in text
    assert r"11.50 $\pm$ 0.70 (sd)" in text
    vs = (tmp_path / "main_vs_1m.tex").read_text()
    assert "PoI vs ppo-rnn" in vs
    assert "p Holm pqn-rnn" in vs


def test_write_tables_needs_frames_or_the_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frames or the"):
        write_tables(None, tmp_path)


def test_game_summary_says_not_logged_for_the_reused_runs(
    tables: dict[str, pd.DataFrame],
) -> None:
    text = latex.game_summary_tex(tables["game"], "1m")
    line = next(ln for ln in text.splitlines() if ln.startswith("ppo &"))
    assert "not logged" in line


# --- numbers --------------------------------------------------------------------


def test_numbers_define_one_newcommand_per_reported_number(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
    tmp_path: Path,
) -> None:
    path = tmp_path / "numbers.tex"
    result = write_numbers(path, frames, tables=tables, complexity_root=complexity_root)
    lines = [
        ln for ln in path.read_text().splitlines() if ln.startswith(r"\newcommand")
    ]
    assert len(lines) == len(result) == len({n.name for n in result})
    for ln in lines:
        assert re.fullmatch(
            r"\\newcommand\{\\[A-Za-z]+\}\{[^{}]*(\{[^{}]*\})?[^{}]*\}", ln
        ), ln
    assert latex.defined_macros(path.read_text()) == {n.name for n in result}


def test_score_macros_expand_to_the_t1_cells(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    got = numbers(frames, tables, complexity_root)
    t1 = tables["main"].set_index(["benchmark", "method"])
    for (budget, method), row in list(t1.groupby(level=[0, 1]).first().iterrows())[:30]:
        key = macro("score", budget, method)
        assert got[key] == f"{row['mean']:.2f}", key
        assert got[macro("scoreSe", budget, method)] == f"{row['se']:.2f}"
        assert (
            got[macro("evalStoch", budget, method)] == f"{row['eval_stochastic']:.2f}"
        )
        if budget == "1m":
            assert got[macro("iqm", budget, method)] == f"{row['iqm']:.2f}"
            assert macro("median", budget, method) not in got or np.isfinite(
                row["median"]
            )
        else:
            assert macro("iqm", budget, method) not in got  # no IQM at 1B (3 seeds)
            assert got[macro("median", budget, method)] == f"{row['median']:.2f}"


def test_poi_and_holm_macros_name_the_control_and_exist_at_1m_only(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    got = numbers(frames, tables, complexity_root)
    row = tables["main"].query("benchmark == '1m' and method == 'sil'").iloc[0]
    assert got["poiOneMSilPpoRnn"] == f"{row['poi_ppo-rnn']:.2f}"
    p = row["p_holm_ppo-rnn"]
    assert got["pHolmOneMSilPpoRnn"] == ("<0.001" if p < 0.001 else f"{p:.3f}")
    assert not any(k.startswith(("poiOneB", "pHolmOneB")) for k in got)
    assert "poiOneMPpoPpo" not in got  # a control against itself is NaN


def test_ablation_macros_carry_the_ref_word(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    got = numbers(frames, tables, complexity_root)
    row = (
        tables["ablation"]
        .query(
            "benchmark == '1m' and method == 'smdp-skills' "
            "and config_ref == 'monte-carlo'"
        )
        .iloc[0]
    )
    assert got["iqmOneMSmdpSkillsMonteCarlo"] == f"{row['iqm']:.2f}"
    assert got["scoreOneMDynaQKappaZero"]
    assert got["scoreOneBSilSingleBuffer"]
    assert "iqmOneBSilSingleBuffer" not in got


def test_complexity_macros_include_leaves_fidelity_and_the_unreached_case(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    got = numbers(frames, tables, complexity_root)
    assert got["lNinetyOneMSil"] == "32"
    assert got["lEightyOneMDynaQKappaZeroDagger"] == "4.5"
    assert got["lNinetyFiveOneMSil"] == latex.NEVER  # inf: never reached
    assert got["fidSixteenOneMSil"] == "0.712"
    assert got["closedLoopOneMSil"] == "0.500"
    assert got["depthNinetyOneMSil"] == "5.000"
    assert got["spearmanScoreNinetyOneM"] == "-0.432"
    assert got["spearmanScoreNinetyOneMDagger"] == "0.100"


def test_a_missing_1m_complexity_csv_names_interpretation_task_6(
    frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="interpretation Task 6"):
        write_numbers(
            tmp_path / "n.tex", frames, tables=tables, complexity_root=tmp_path
        )
    assert not (tmp_path / "n.tex").exists()


def test_without_1b_runs_only_1m_macros_are_emitted(
    frames_1m: dict[str, pd.DataFrame], complexity_root: Path
) -> None:
    got = {
        n.name
        for n in collect_numbers(frames_1m, complexity_root=complexity_root, n_boot=N)
    }
    assert got
    assert not any("OneB" in k for k in got)
    assert any("OneM" in k for k in got)


def test_a_missing_1b_complexity_csv_is_skipped_when_1b_runs_exist(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    got = numbers(frames, tables, complexity_root)
    assert "scoreOneBSil" in got
    assert not any(k.startswith(("lNinetyOneB", "fidSixteenOneB")) for k in got)


def test_one_macro_with_two_values_is_an_error(
    frames: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    complexity_root: Path,
) -> None:
    broken = tables["main"].copy()
    rows = (broken["method"] == "ppo") & (broken["block"] == "skills")
    broken.loc[rows, "mean"] += 1.0  # the control repeats per block and must agree
    with pytest.raises(ValueError, match="two values"):
        collect_numbers(
            frames, tables={**tables, "main": broken}, complexity_root=complexity_root
        )


# --- report.tex and figures -----------------------------------------------------


def test_every_macro_used_in_report_tex_is_defined(tmp_path: Path) -> None:
    numbers_tex = tmp_path / "numbers.tex"
    numbers_tex.write_text("\\newcommand{\\scoreOneMPpo}{1.00}\n")
    report = tmp_path / "report.tex"
    report.write_text(r"We reach \scoreOneMPpo{} and \iqmOneMSil, in \textbf{bold}.\n")
    assert undefined_macros(report, numbers_tex) == ["iqmOneMSil"]
    numbers_tex.write_text(
        numbers_tex.read_text() + "\\newcommand{\\iqmOneMSil}{2.00}\n"
    )
    assert undefined_macros(report, numbers_tex) == []
    assert undefined_macros(report, tmp_path / "missing.tex") == [
        "iqmOneMSil",
        "scoreOneMPpo",
    ]


def test_the_real_report_uses_no_number_macro_yet() -> None:
    root = Path(__file__).resolve().parents[2] / "report" / "report.tex"
    assert not latex.USED.findall(root.read_text())  # vacuous today (R75, R86)


def test_copy_figures_takes_the_report_pdfs_only(tmp_path: Path) -> None:
    src, out = tmp_path / "results", tmp_path / "report" / "figures"
    src.mkdir()
    for name in (
        "aggregate.pdf",
        "aggregate_double.pdf",
        "aggregate_video.pdf",
        "aggregate.png",
    ):
        (src / name).write_bytes(b"%PDF")
    copied = copy_figures(src, out)
    assert sorted(p.name for p in copied) == ["aggregate.pdf", "aggregate_double.pdf"]
    assert sorted(p.name for p in out.iterdir()) == [
        "aggregate.pdf",
        "aggregate_double.pdf",
    ]
