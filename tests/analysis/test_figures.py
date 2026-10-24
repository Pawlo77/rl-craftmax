"""Tests of the core figures F1-F5 and F18-F20 on the comparison fixture.

Every figure writes a PDF and a PNG, is sized for one Interspeech column (report) or a
1920 x 1080 frame (video), takes its colours from `METHOD_COLORS` and its numbers from
the comparison tables; the tests read the drawn artists back (found by their ``gid``,
``<kind>/<budget or phase>/<block>/<method>/<ref>``) and compare them with the tables.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _compare_fixture import ABLATIONS
from _figures_fixture import ACHIEVEMENTS, figure_frames, tuning_frames
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.text import Text

from wut_murl_craftax.analysis import compare, figures
from wut_murl_craftax.analysis.compare import (
    CONTEXT_NOTE,
    ablation_table,
    hardware_table,
    literature_table,
    main_table,
    poi_matrix,
    run_stability,
    sensitivity_table,
    stability_table,
)
from wut_murl_craftax.analysis.figures import (
    FIGURES,
    NAMES,
    PENDING_1B,
    plot_aggregate,
    plot_curves_score,
    plot_hardware,
    plot_poi_matrix,
    plot_seed_spread,
    plot_sensitivity,
    plot_stability,
    plot_success_heatmap,
)
from wut_murl_craftax.analysis.hard import achievement_status
from wut_murl_craftax.evaluation.stability import config_scores
from wut_murl_craftax.evaluation.stats import curve_band, step_grid
from wut_murl_craftax.visualisation.charts import METHOD_COLORS, METHOD_COLORS_LIGHT

N = 200
"""Bootstrap resamples in tests (the protocol's 10,000 only slow them down)."""
THEMES = ("report", "video")
CONTROLS = ("ppo", "ppo-rnn", "pqn-rnn")
PDF, PNG = b"%PDF", b"\x89PNG\r\n\x1a\n"
LABELS = {
    "F1": ("env steps", "score (% of max)"),
    "F2": ("final score (% of max)",),
    "F3": ("versus (column)", "method (row)"),
    "F4": ("final score (% of max)",),
    "F5": ("tuned method",),
    "F18": ("config score r_c", "sensitivity S_h"),
    "F19": ("max drawdown", "share of runs"),
    "F20": ("steps per second", "peak memory"),
}
"""Substrings every figure's axis labels must contain, one per label."""


def _colors(theme: str) -> dict[str, str]:
    return METHOD_COLORS if theme == "video" else METHOD_COLORS_LIGHT


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """1M (10 seeds) and 1B (3 seeds) with achievements."""
    return figure_frames()


@pytest.fixture(scope="module")
def frames_1m() -> dict[str, pd.DataFrame]:
    """Only Craftax-1M: 1B is pending (R86)."""
    return figure_frames(("1m",))


@pytest.fixture(scope="module")
def tuning() -> dict[str, pd.DataFrame]:
    return tuning_frames()


@pytest.fixture(scope="module")
def literature(
    frames: dict[str, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> pd.DataFrame:
    """T2 with no published control values (the gate is not under test here)."""
    published = tmp_path_factory.mktemp("published") / "published.yaml"
    published.write_text("{}")
    return literature_table(
        frames,
        published=published,
        published_hash=lambda c, _b: f"{c}/tuned",
        n_boot=N,
    )


def draw(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    out_dir: Path,
    theme: str = "report",
) -> Figure:
    """Call the figure of ``fid`` with test-sized bootstraps and the shared tables."""
    plain: dict[str, Callable[..., Figure]] = {
        "F1": lambda: plot_curves_score(frames, out_dir, theme, n_boot=N),
        "F2": lambda: plot_aggregate(frames, out_dir, theme, n_boot=N),
        "F3": lambda: plot_poi_matrix(frames, out_dir, theme, n_boot=N),
        "F4": lambda: plot_seed_spread(frames, out_dir, theme, literature=literature),
        "F5": lambda: plot_success_heatmap(frames, out_dir, theme),
        "F18": lambda: plot_sensitivity(tuning, out_dir, theme),
        "F19": lambda: plot_stability(frames, out_dir, theme),
        "F20": lambda: plot_hardware(frames, out_dir, theme),
    }
    return plain[fid]()


def artists(fig: Figure, kind: str) -> list[Artist]:
    """Artists whose gid starts with ``kind + '/'``, in drawing order."""
    return fig.findobj(lambda a: (a.get_gid() or "").startswith(f"{kind}/"))


def one(fig: Figure, gid: str) -> Artist:
    found = fig.findobj(lambda a: a.get_gid() == gid)
    assert len(found) == 1, f"{gid}: {len(found)} artists"
    return found[0]


def texts(fig: Figure) -> list[str]:
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def axis_labels(fig: Figure) -> str:
    labels = []
    for ax in fig.axes:
        labels += [ax.get_xlabel(), ax.get_ylabel()]
    return " | ".join(labels)


def tick_labels(ax: object) -> list[str]:
    return [t.get_text() for t in ax.get_yticklabels()]  # type: ignore[attr-defined]


def with_gid(fig: Figure, kind: str, *parts: str) -> Artist:
    return one(fig, "/".join((kind, *parts)))


# --- every figure -------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", list(NAMES))
def test_each_figure_writes_a_pdf_and_a_png(
    fid: str,
    theme: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    fig = draw(fid, frames, tuning, literature, tmp_path, theme)
    assert isinstance(fig, Figure)
    stem = NAMES[fid] + ("_video" if theme == "video" else "")
    assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(PDF)
    assert (tmp_path / f"{stem}.png").read_bytes().startswith(PNG)
    assert FIGURES[fid] is getattr(figures, FIGURES[fid].__name__)
    for want in LABELS[fid]:
        assert want in axis_labels(fig), (fid, want, axis_labels(fig))


def test_the_two_themes_are_two_files_and_two_surfaces(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    light = plot_aggregate(frames, tmp_path, "report", n_boot=N)
    dark = plot_aggregate(frames, tmp_path, "video", n_boot=N)
    assert light.get_facecolor() != dark.get_facecolor()
    assert {p.name for p in tmp_path.iterdir()} == {
        "aggregate.pdf",
        "aggregate.png",
        "aggregate_video.pdf",
        "aggregate_video.png",
    }


def test_an_unknown_theme_is_refused(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="theme"):
        plot_aggregate(frames, tmp_path, "poster", n_boot=N)


@pytest.mark.parametrize("fid", list(NAMES))
def test_report_figures_are_one_column_wide_with_7pt_fonts(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    fig = draw(fid, frames, tuning, literature, tmp_path)
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
    fig.canvas.draw()
    sizes = {
        t.get_text(): t.get_fontsize()
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
    }
    assert sizes
    small = {k: v for k, v in sizes.items() if v < 7}
    assert not small, small


@pytest.mark.parametrize("fid", list(NAMES))
def test_video_figures_fill_a_1080p_frame(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    fig = draw(fid, frames, tuning, literature, tmp_path, "video")
    width, height = fig.get_size_inches() * fig.dpi
    assert (round(width), round(height)) == (1920, 1080)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", ["F1", "F2", "F3", "F4", "F5", "F19", "F20"])
def test_a_missing_1b_is_pending_not_an_empty_axis(
    fid: str,
    theme: str,
    frames_1m: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    fig = (
        draw(fid, frames_1m, tuning, pd.DataFrame(), tmp_path, theme)
        if (fid != "F4")
        else plot_seed_spread(frames_1m, tmp_path, theme, literature=_none_lit())
    )
    shown = texts(fig)
    if fid != "F3":  # F3 is a 1M figure by design: nothing is pending in it
        assert any(PENDING_1B in t for t in shown), shown
    for ax in fig.axes:
        if ax.get_visible() and ax.axison:
            assert ax.has_data() or ax.texts, (
                "an empty axis needs a label or must be off"
            )
            assert ax.get_title(loc="left") or ax.get_xlabel(), "an unlabelled axis"


def _none_lit() -> pd.DataFrame:
    return pd.DataFrame(columns=compare.LITERATURE_TABLE_COLUMNS)


@pytest.mark.parametrize("fid", ["F1", "F2", "F3", "F4", "F5", "F19", "F20"])
def test_no_run_data_gives_a_labelled_placeholder(
    fid: str, tmp_path: Path, tuning: dict[str, pd.DataFrame]
) -> None:
    empty = {
        "runs": figure_frames()["runs"].iloc[:0],
        "curves": figure_frames()["curves"].iloc[:0],
        "eval_episodes": figure_frames()["eval_episodes"].iloc[:0],
    }
    fig = (
        plot_seed_spread(empty, tmp_path, literature=_none_lit())
        if fid == "F4"
        else draw(fid, empty, tuning, pd.DataFrame(), tmp_path)
    )
    assert any("no data" in t for t in texts(fig))
    assert (tmp_path / f"{NAMES[fid]}.pdf").exists()


# --- F1 ----------------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
def test_f1_colours_and_styles_follow_the_contract(
    theme: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames, tmp_path, theme, n_boot=N)
    colors = _colors(theme)
    lines = artists(fig, "curve")
    assert lines
    for line in lines:
        *_, method, ref = (line.get_gid() or "").split("/")
        assert ref == "tuned"
        assert line.get_color() == colors[method]
        assert line.get_linestyle() == ("--" if method in CONTROLS else "-")
    grey = {
        one_line.get_color()
        for one_line in lines
        if one_line.get_gid().split("/")[3] in CONTROLS
    }
    assert len(grey) == 3


def test_f1_shows_the_two_blocks_and_the_controls_in_both(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames_1m, tmp_path, n_boot=N)
    gids = [a.get_gid().split("/") for a in artists(fig, "curve")]
    for control in CONTROLS:
        assert {b for _, _, b, m, _ in gids if m == control} == {"main", "skills"}
    assert {m for _, _, b, m, _ in gids if b == "skills"} - set(CONTROLS) == {
        "smdp-skills",
        "dyna-q",
    }
    assert "smdp-skills" not in {m for _, _, b, m, _ in gids if b == "main"}


def test_f1_is_the_shared_curve_band_over_seeds(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames_1m, tmp_path, n_boot=N, seed=3)
    runs, curves = frames_1m["runs"], frames_1m["curves"]
    mine = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]
    score = curves[curves["metric"] == "train/score_pct"]
    per_seed = [
        (g["env_steps"].to_numpy(), g["value"].to_numpy())
        for _, g in score[score["run_id"].isin(mine["run_id"])].groupby("run_id")
    ]
    want = curve_band(per_seed, step_grid(1e6), "mean", N, 3)
    line = with_gid(fig, "curve", "1m", "main", "sil", "tuned")
    x, y = line.get_data()
    np.testing.assert_allclose(x, want["env_steps"])
    np.testing.assert_allclose(y, want["estimate"])
    band = with_gid(fig, "band", "1m", "main", "sil", "tuned")
    got = np.vstack([p.vertices for p in band.get_paths()])
    assert got[:, 1].min() == pytest.approx(np.nanmin(want["lo"]))
    assert got[:, 1].max() == pytest.approx(np.nanmax(want["hi"]))


def test_f1_ablations_are_dotted_hollow_and_a_unit_of_their_own(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames_1m, tmp_path, n_boot=N, ablations=True)
    colors = _colors("report")
    for method, ref in ABLATIONS:
        line = with_gid(fig, "curve", "1m", _block(method), method, ref)
        assert line.get_linestyle() == ":"
        assert line.get_color() == colors[method]
        assert line.get_label() == f"{method} ({ref})"
        tuned = with_gid(fig, "curve", "1m", _block(method), method, "tuned")
        assert tuned.get_linestyle() == "-"  # never pooled into the method's curve
        assert not np.allclose(line.get_ydata(), tuned.get_ydata())


def _block(method: str) -> str:
    return "skills" if method in ("smdp-skills", "dyna-q") else "main"


def test_f1_1b_panels_are_marked_indicative(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames, tmp_path, n_boot=N)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
    assert any("1B" in t and "indicative" in t for t in titles)
    assert not any("1M" in t and "indicative" in t for t in titles)
    one(fig, "curve/1b/main/sil/tuned")


# --- F2 ----------------------------------------------------------------------------


def test_f2_reads_the_main_table_and_separates_the_blocks(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = main_table(frames, n_boot=N)
    fig = plot_aggregate(frames, tmp_path, n_boot=N, table=table)
    for row in table[table["benchmark"] == "1m"].itertuples():
        base = (row.benchmark, row.block, row.method, "tuned")
        iqm = with_gid(fig, "iqm", *base)
        assert iqm.get_xdata()[0] == pytest.approx(row.iqm)
        ci = with_gid(fig, "ci-iqm", *base)
        assert (ci.get_segments()[0][0][0], ci.get_segments()[0][1][0]) == (
            pytest.approx(row.iqm_lo),
            pytest.approx(row.iqm_hi),
        )
        mean = with_gid(fig, "mean", *base)
        assert mean.get_xdata()[0] == pytest.approx(row.mean)
    ax = with_gid(fig, "iqm", "1m", "main", "sil", "tuned").axes
    labels = tick_labels(ax)
    assert "main" in labels
    assert "uses hand-written skills" in labels
    assert labels.index("main") < labels.index("uses hand-written skills")
    after = labels[labels.index("uses hand-written skills") :]
    assert [c for c in CONTROLS if c in after] == list(CONTROLS)  # controls repeated
    assert "smdp-skills" not in labels[: labels.index("uses hand-written skills")]


def test_f2_1b_gives_median_and_mean_never_iqm(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = main_table(frames, n_boot=N)
    fig = plot_aggregate(frames, tmp_path, n_boot=N, table=table)
    row = table[(table["benchmark"] == "1b") & (table["method"] == "sil")].iloc[0]
    assert with_gid(fig, "median", "1b", "main", "sil", "tuned").get_xdata()[0] == (
        pytest.approx(row["median"])
    )
    ci = with_gid(fig, "ci-mean", "1b", "main", "sil", "tuned")
    segment = ci.get_segments()[0]
    assert (segment[0][0], segment[1][0]) == (
        pytest.approx(row["t_lo"]),
        pytest.approx(row["t_hi"]),
    )
    assert not [a for a in artists(fig, "iqm") if "/1b/" in a.get_gid()]
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
    assert any("1B" in t and "indicative" in t for t in titles)


def test_f2_ablations_are_dotted_rows_under_their_method(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = ablation_table(frames, n_boot=N)
    fig = plot_aggregate(
        frames, tmp_path, n_boot=N, ablation=table, table=main_table(frames, n_boot=N)
    )
    colors = _colors("report")
    for row in table[table["benchmark"] == "1m"].itertuples():
        base = (row.benchmark, row.block, row.method, row.config_ref)
        mean = with_gid(fig, "mean", *base)
        assert mean.get_xdata()[0] == pytest.approx(row.mean)
        assert mean.get_markerfacecolor() != colors[row.method]  # hollow
        assert with_gid(fig, "ci-mean", *base).get_linestyle()[0][1] is not None
        labels = tick_labels(mean.axes)
        assert f"{row.method} ({row.config_ref})" in labels
        assert labels.index(f"{row.method} ({row.config_ref})") > labels.index(
            row.method
        )
    tuned = with_gid(fig, "mean", "1m", "main", "sil", "tuned")
    ablated = with_gid(fig, "mean", "1m", "main", "sil", "single-buffer")
    assert tuned.get_xdata()[0] != ablated.get_xdata()[0]  # not pooled


# --- F3 ----------------------------------------------------------------------------


def test_f3_cells_are_the_poi_table_one_panel_per_block(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = poi_matrix(frames, n_boot=N)
    fig = plot_poi_matrix(frames, tmp_path, n_boot=N, table=table)
    for block, other in (("main", "smdp-skills"), ("skills", "sil")):
        image = with_gid(fig, "poi", "1m", block)
        ax = image.axes
        names = [t.get_text() for t in ax.get_yticklabels()]
        assert [t.get_text() for t in ax.get_xticklabels()] == names
        assert set(CONTROLS) <= set(names)
        assert other not in names  # no cross-block cell
        assert block in ax.get_title(loc="left")
        cells = np.asarray(image.get_array(), dtype=float)
        rows = table[(table["benchmark"] == "1m") & (table["block"] == block)]
        for r in rows.itertuples():
            assert cells[names.index(r.method), names.index(r.versus)] == (
                pytest.approx(r.poi)
            )
        assert np.isnan(np.diag(cells)).all()


def test_f3_marks_the_cells_whose_interval_excludes_one_half(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = poi_matrix(frames, n_boot=N)
    fig = plot_poi_matrix(frames, tmp_path, n_boot=N, table=table)
    bold = {
        t.get_text()
        for t in fig.findobj(Text)
        if t.get_gid() == "poi-cell/bold" and t.get_text()
    }
    rows = table[table["benchmark"] == "1m"]
    excluded = (rows["poi_lo"] > 0.5) | (rows["poi_hi"] < 0.5)
    assert excluded.any()
    assert (~excluded).any()
    assert {f"{p:.2f}" for p in rows.loc[excluded, "poi"]} <= {
        b.split("\n")[0] for b in bold
    }
    plain = {
        t.get_text().split("\n")[0]
        for t in fig.findobj(Text)
        if t.get_gid() == "poi-cell/plain"
    }
    assert {f"{p:.2f}" for p in rows.loc[~excluded, "poi"]} <= plain


def test_f3_needs_1m_so_a_1b_only_export_says_so(tmp_path: Path) -> None:
    fig = plot_poi_matrix(figure_frames(("1b",)), tmp_path, n_boot=N)
    assert any("Craftax-1M" in t and "pending" in t for t in texts(fig))


# --- F4 ----------------------------------------------------------------------------


def test_f4_dots_are_the_per_seed_final_scores(
    frames: dict[str, pd.DataFrame], literature: pd.DataFrame, tmp_path: Path
) -> None:
    fig = plot_seed_spread(frames, tmp_path, literature=literature)
    runs = frames["runs"]
    for budget in ("1m", "1b"):
        for method in ("ppo", "sil"):
            got = with_gid(fig, "seeds", budget, _block(method), method)
            want = runs[
                (runs["budget"] == budget)
                & (runs["method"] == method)
                & (runs["config_ref"] == "tuned")
            ]["final_score"]
            assert sorted(got.get_offsets()[:, 0]) == pytest.approx(sorted(want))


def test_f4_literature_values_are_marks_of_the_comparable_block(
    frames: dict[str, pd.DataFrame], literature: pd.DataFrame, tmp_path: Path
) -> None:
    fig = plot_seed_spread(frames, tmp_path, literature=literature)
    lit = literature[
        (literature["source"] == "literature")
        & (literature["score_kind"] == "training")
    ]
    marks = artists(fig, "lit")
    assert len(marks) == len(lit)
    for budget in ("1m", "1b"):
        rows = lit[lit["benchmark"] == budget]
        got = sorted(
            float(m.get_offsets()[0, 0])
            for m in marks
            if m.get_gid().split("/")[1] == budget
        )
        assert got == pytest.approx(sorted(rows["score_pct"]))
    same = {
        m.get_gid() for m in marks if m.get_facecolor().size and m.get_facecolor()[0][3]
    }
    assert same  # protocol-matching rows are filled, the others hollow
    assert len(same) < len(marks)
    assert not [t for t in texts(fig) if "[[" in t]


def test_f4_without_literature_rows_draws_only_our_dots(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_seed_spread(frames, tmp_path, literature=_none_lit())
    assert not artists(fig, "lit")
    assert artists(fig, "seeds")


# --- F5 ----------------------------------------------------------------------------


def test_f5_cells_are_the_final_success_rates_by_tier_and_floor(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_success_heatmap(frames_1m, tmp_path)
    eps = frames_1m["eval_episodes"]
    runs = frames_1m["runs"]
    tuned = runs[(runs["config_ref"] == "tuned") & (runs["budget"] == "1m")]
    status = achievement_status(eps[eps["run_id"].isin(tuned["run_id"])])
    for block in ("main", "skills"):
        image = with_gid(fig, "heat", "1m", block)
        names = [t.get_text() for t in image.axes.get_xticklabels()]
        assert set(CONTROLS) <= set(names)
        cells = np.asarray(image.get_array(), dtype=float)
        assert cells.shape == (len(ACHIEVEMENTS), len(names))
        for j, method in enumerate(names):
            mine = status[(status["method"] == method)].reset_index(drop=True)
            np.testing.assert_allclose(cells[:, j], mine["final_rate"])
    assert "smdp-skills" not in [
        t.get_text() for t in with_gid(fig, "heat", "1m", "main").axes.get_xticklabels()
    ]
    rows = [a.get_text() for a in fig.findobj(Text) if a.get_gid() == "heat/tier"]
    assert rows == ["basic", "intermediate", "advanced"]


# --- F18 ---------------------------------------------------------------------------


def test_f18_strip_and_bars_read_config_scores_and_t7(
    tuning: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = sensitivity_table(tuning)
    fig = plot_sensitivity(tuning, tmp_path, table=table)
    runs = tuning["runs"]
    for phase in ("tune-coarse", "tune-fine"):
        for method in ("sil", "smdp-skills"):
            mine = runs[(runs["method"] == method) & (runs["phase"] == phase)]
            dots = with_gid(fig, "rc", phase, method)
            assert sorted(dots.get_offsets()[:, 0]) == pytest.approx(
                sorted(config_scores(mine)["r_c"])
            )
            row = table[(table["phase"] == phase) & (table["method"] == method)].iloc[0]
            for rank, (name, s) in enumerate(
                ((row["top1"], row["top1_s"]), (row["top2"], row["top2_s"])), 1
            ):
                bar = with_gid(fig, "sens", phase, method, str(rank))
                assert bar.get_width() == pytest.approx(s)
                assert bar.get_label() == name
            floor = with_gid(fig, "noise", phase, method)
            assert floor.get_xdata()[0] == pytest.approx(row["noise_floor"])
    one(fig, "rc/tune-coarse/ppo")
    with pytest.raises(AssertionError):
        one(fig, "rc/tune-fine/ppo")


def test_f18_marks_the_hyperparameters_below_the_noise_floor(
    tuning: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = sensitivity_table(tuning)
    fig = plot_sensitivity(tuning, tmp_path, table=table)
    row = table[(table["phase"] == "tune-coarse") & (table["method"] == "sil")].iloc[0]
    assert row["top1_insensitive"] is False or not row["top1_insensitive"]
    assert row["top2_insensitive"]
    strong = with_gid(fig, "sens", "tune-coarse", "sil", "1")
    weak = with_gid(fig, "sens", "tune-coarse", "sil", "2")
    assert strong.get_facecolor()[3] == 1.0
    assert weak.get_facecolor()[3] < 1.0 or weak.get_hatch()


def test_f18_is_a_1m_figure_without_a_grid_it_says_so(tmp_path: Path) -> None:
    runs = tuning_frames()["runs"]
    fig = plot_sensitivity({"runs": runs[runs["phase"] == "tune-coarse"]}, tmp_path)
    assert any("tune-fine" in t and "no runs" in t for t in texts(fig))


# --- F19 ---------------------------------------------------------------------------


def test_f19_dots_are_the_per_run_drawdowns_and_bars_are_t3(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_stability(frames, tmp_path)
    per_run = run_stability(frames)
    t3 = stability_table(frames)
    for budget in ("1m", "1b"):
        for method in ("sil", "ppo"):
            mine = per_run[
                (per_run["benchmark"] == budget) & (per_run["method"] == method)
            ]
            dots = with_gid(fig, "dd", budget, _block(method), method)
            got = sorted(dots.get_offsets()[:, 0])
            assert got == pytest.approx(sorted(mine["max_drawdown"].dropna()))
            row = t3[
                (t3["benchmark"] == budget)
                & (t3["method"] == method)
                & (t3["block"] == _block(method))
            ].iloc[0]
            median = with_gid(fig, "dd-median", budget, _block(method), method)
            assert median.get_xdata()[0] == pytest.approx(row["max_drawdown_median"])
            drops = with_gid(fig, "drops", budget, _block(method), method)
            assert drops.get_width() == pytest.approx(
                row["runs_with_drops"] / row["seeds"]
            )
            collapse = with_gid(fig, "collapse", budget, _block(method), method)
            assert collapse.get_width() == pytest.approx(
                row["runs_with_collapse"] / row["seeds"]
            )
    assert (t3["runs_with_collapse"] > 0).any()  # the fixture has a collapsing run
    marked = with_gid(fig, "collapsed", "1m", "main", "sil")
    assert len(marked.get_offsets()) == 1  # sil seed 0 collapses


# --- F20 ---------------------------------------------------------------------------


def test_f20_reads_t4_medians_and_carries_the_context_note(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_hardware(frames, tmp_path)
    t4 = hardware_table(frames)
    row = t4[
        (t4["benchmark"] == "1m")
        & (t4["profile"] == "laptop")
        & (t4["method"] == "sil")
    ].iloc[0]
    sps = with_gid(fig, "hw", "laptop", "steady_sps", "sil")
    assert sps.get_offsets()[0, 0] == pytest.approx(row["steady_sps"])
    hours = with_gid(fig, "hw", "laptop", "wall_clock_s", "sil")
    assert hours.get_offsets()[0, 0] == pytest.approx(row["wall_clock_s"] / 3600)
    shown = " ".join(texts(fig))
    assert CONTEXT_NOTE in " ".join(shown.split())
    assert "n.m." in shown  # no GPU, energy or device-memory series on the laptop
    assert not [a for a in artists(fig, "hw") if a.get_gid().endswith("/1b")]


def test_f20_keeps_profiles_apart_and_has_no_1b_rows_to_draw(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs = frames["runs"].copy()
    other = runs[runs["method"].isin(["ppo", "sil"])].assign(profile="cuda")
    fig = plot_hardware({**frames, "runs": pd.concat([runs, other])}, tmp_path)
    one(fig, "hw/cuda/steady_sps/ppo")
    one(fig, "hw/cuda/steady_sps/sil")
    one(fig, "hw/laptop/steady_sps/dyna-q")
    with pytest.raises(AssertionError):
        one(fig, "hw/cuda/steady_sps/dyna-q")  # a profile only shows its own runs
    note = [t for t in texts(fig) if CONTEXT_NOTE in " ".join(t.split())]
    assert note
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
    assert any("cuda" in t for t in titles)
    assert any("laptop" in t for t in titles)
