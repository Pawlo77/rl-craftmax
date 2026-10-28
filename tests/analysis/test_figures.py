"""Tests of the core figures F1-F5 and F18-F20 on the comparison fixture.

Every figure writes a PNG (and no PDF), is sized for one Interspeech column (report)
or a 1920 x 1080 frame (video), takes its colours from `METHOD_COLORS` and its numbers
from the comparison tables; the tests read the drawn artists back (found by their
``gid``, ``<kind>/<budget or phase>/<block>/<method>/<ref>``) and compare them with the
tables.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _compare_fixture import ABLATIONS
from _figures_fixture import ACHIEVEMENTS, figure_frames, tuning_frames
from matplotlib.artist import Artist
from matplotlib.colors import to_hex
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
    plot_aggregate_poi,
    plot_curves_score,
    plot_hardware,
    plot_legend_methods,
    plot_poi_matrix,
    plot_seed_spread,
    plot_sensitivity,
    plot_stability,
    plot_success_heatmap,
)
from wut_murl_craftax.evaluation.stability import config_scores
from wut_murl_craftax.evaluation.stats import curve_band, step_grid
from wut_murl_craftax.visualisation.charts import (
    METHOD_COLORS,
    METHOD_COLORS_LIGHT,
    capture_captions,
)

N = 200
"""Bootstrap resamples in tests (the protocol's 10,000 only slow them down)."""
THEMES = ("report", "video")
CONTROLS = ("ppo", "ppo-rnn", "pqn-rnn")
PNG = b"\x89PNG\r\n\x1a\n"
LABELS = {
    "LEG": (),
    "F1": ("env steps", "score (% of max)"),
    "F2": ("final score (% of max)",),
    "F3": ("versus (column)", "method (row)"),
    "F4": ("final score (% of max)",),
    "F5": ("tuned method",),
    "F18": ("config score r_c", "sensitivity S_h"),
    "F19": ("max drawdown", "share of runs"),
    "F20": ("steps per second", "peak memory"),
    "F2F3": ("final score (% of max)", "PoI vs control"),
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
    width: str = "column",
) -> Figure:
    """Call the figure of ``fid`` with test-sized bootstraps and the shared tables."""
    w = {"width": width}
    plain: dict[str, Callable[..., Figure]] = {
        "F1": lambda: plot_curves_score(frames, out_dir, theme, n_boot=N, **w),
        "F2": lambda: plot_aggregate(frames, out_dir, theme, n_boot=N, **w),
        "F3": lambda: plot_poi_matrix(frames, out_dir, theme, n_boot=N, **w),
        "F4": lambda: plot_seed_spread(
            frames, out_dir, theme, literature=literature, **w
        ),
        "F5": lambda: plot_success_heatmap(frames, out_dir, theme, **w),
        "F18": lambda: plot_sensitivity(tuning, out_dir, theme, **w),
        "F19": lambda: plot_stability(frames, out_dir, theme, **w),
        "F20": lambda: plot_hardware(frames, out_dir, theme, **w),
        "F2F3": lambda: plot_aggregate_poi(frames, out_dir, theme, n_boot=N, **w),
        "LEG": lambda: plot_legend_methods(frames, out_dir, theme, **w),
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
def test_each_figure_writes_a_png_and_no_pdf(
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
    assert not (tmp_path / f"{stem}.pdf").exists()
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
        "aggregate.png",
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
@pytest.mark.parametrize("fid", ["F1", "F2", "F3", "F4", "F5", "F19", "F20", "F2F3"])
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


@pytest.mark.parametrize("fid", ["F1", "F2", "F3", "F4", "F5", "F19", "F20", "F2F3"])
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
    assert (tmp_path / f"{NAMES[fid]}.png").exists()


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


def _cell_texts(fig: Figure, image: Artist) -> dict[tuple[int, int], Text]:
    """The PoI text of each (row, column) cell of ``image``, found by its position."""
    return {
        (int(t.get_position()[1]), int(t.get_position()[0])): t
        for t in fig.findobj(Text)
        if (t.get_gid() or "").startswith("poi-cell/") and t.axes is image.axes
    }


def test_f3_bold_is_exactly_the_cells_whose_interval_excludes_one_half(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = poi_matrix(frames, n_boot=N)
    fig = plot_poi_matrix(frames, tmp_path, n_boot=N, table=table)
    seen = set()
    for block in ("main", "skills"):
        image = with_gid(fig, "poi", "1m", block)
        names = [t.get_text() for t in image.axes.get_yticklabels()]
        cells = _cell_texts(fig, image)
        rows = table[(table["benchmark"] == "1m") & (table["block"] == block)]
        assert len(cells) == len(rows)
        for r in rows.itertuples():
            text = cells[(names.index(r.method), names.index(r.versus))]
            excluded = r.poi_lo > 0.5 or r.poi_hi < 0.5
            assert text.get_text().split("\n")[0] == f"{r.poi:.2f}"
            assert text.get_gid() == ("poi-cell/bold" if excluded else "poi-cell/plain")
            assert (text.get_fontweight() == "bold") == excluded
            seen.add(excluded)
    assert seen == {True, False}


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


def _expected_rates(
    frames: dict[str, pd.DataFrame], budget: str = "1m"
) -> pd.DataFrame:
    """Method x achievement rates straight from the episodes (not via hard.py)."""
    eps, runs = frames["eval_episodes"], frames["runs"]
    ids = runs.loc[
        (runs["config_ref"] == "tuned") & (runs["budget"] == budget), "run_id"
    ]
    final = eps[eps["run_id"].isin(ids) & (eps["mode"] == "stochastic")]
    cols = [f"ach/{a}" for a in ACHIEVEMENTS]
    per_seed = final.groupby(["method", "run_id"])[cols].mean()
    return per_seed.groupby("method").mean().rename(columns=lambda c: c[4:])


def test_f5_rates_differ_so_a_misplaced_row_or_column_fails(
    frames_1m: dict[str, pd.DataFrame],
) -> None:
    rates = _expected_rates(frames_1m)
    assert rates.nunique().min() > 3  # every achievement differs across methods
    assert rates.T.nunique().min() > 2  # and every method across achievements


def test_f5_cells_are_the_final_success_rates_by_tier_and_floor(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_success_heatmap(frames_1m, tmp_path)
    rates = _expected_rates(frames_1m)
    for block in ("main", "skills"):
        image = with_gid(fig, "heat", "1m", block)
        names = [t.get_text() for t in image.axes.get_xticklabels()]
        assert set(CONTROLS) <= set(names)
        cells = np.asarray(image.get_array(), dtype=float)
        assert cells.shape == (len(ACHIEVEMENTS), len(names))
        for j, method in enumerate(names):
            for i, achievement in enumerate(ACHIEVEMENTS):  # row i is achievement i
                assert cells[i, j] == pytest.approx(rates.loc[method, achievement])
    assert "smdp-skills" not in [
        t.get_text() for t in with_gid(fig, "heat", "1m", "main").axes.get_xticklabels()
    ]
    rows = [a.get_text() for a in fig.findobj(Text) if a.get_gid() == "heat/tier"]
    assert rows == ["basic", "intermediate", "advanced"]


def test_f5_video_names_every_achievement_in_row_order(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_success_heatmap(frames_1m, tmp_path, "video")
    image = with_gid(fig, "heat", "1m", "main")
    assert [t.get_text() for t in image.axes.get_yticklabels()] == list(ACHIEVEMENTS)
    assert fig.findobj(lambda a: a.get_gid() == "heat/achievement")
    shown = texts(fig)
    assert "basic F0" in shown
    assert not [t for t in fig.findobj(Text) if t.get_gid() == "heat/tier"]


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
            assert row["top1"] in ("lr", "eps_top", "clip")
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
    ppo = table[(table["phase"] == "tune-coarse") & (table["method"] == "ppo")].iloc[0]
    assert not ppo["top1_insensitive"]
    assert ppo["top2_insensitive"]
    strong = with_gid(fig, "sens", "tune-coarse", "ppo", "1")
    weak = with_gid(fig, "sens", "tune-coarse", "ppo", "2")
    assert strong.get_facecolor()[3] == 1.0
    assert 0.4 <= weak.get_facecolor()[3] < 1.0  # faint but still readable
    assert weak.get_edgecolor()[3] == 1.0
    sil = with_gid(fig, "sens", "tune-coarse", "sil", "2")
    assert sil.get_facecolor()[3] == 1.0  # sil's second hyperparameter does matter


def test_f18_rows_and_tick_labels_follow_the_methods(
    tuning: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = sensitivity_table(tuning)
    fig = plot_sensitivity(tuning, tmp_path, table=table)
    order = {
        "tune-coarse": ["ppo", "sil", "smdp-skills"],
        "tune-fine": ["sil", "smdp-skills"],
    }
    for phase, methods in order.items():
        bar = with_gid(fig, "sens", phase, methods[0], "1")
        ax_s = bar.axes
        ax_rc = with_gid(fig, "rc", phase, methods[0]).axes
        assert [t.get_text() for t in ax_rc.get_yticklabels()] == methods
        want = []
        for m in methods:
            row = table[(table["phase"] == phase) & (table["method"] == m)].iloc[0]
            want += [row["top1"], row["top2"]]
        assert [t.get_text() for t in ax_s.get_yticklabels()] == want
        for k, m in enumerate(methods):
            for rank in (1, 2):
                rect = with_gid(fig, "sens", phase, m, str(rank))
                y = rect.get_y() + rect.get_height() / 2
                assert y == pytest.approx(2 * k + rank - 1)
                assert rect.get_label() == want[2 * k + rank - 1]
            dots = with_gid(fig, "rc", phase, m).get_offsets()[:, 1]
            assert np.all(np.abs(dots - (2 * k + 0.5)) < 0.5)
    smdp = table[(table["phase"] == "tune-coarse") & (table["method"] == "smdp-skills")]
    assert smdp.iloc[0]["top1"] == "eps_top"  # its own hyperparameter name


def test_sens_axes_refuses_a_method_without_a_t7_row(
    tuning: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = sensitivity_table(tuning)
    table = table[table["method"] != "sil"]
    ax_rc, ax_s = Figure().subplots(1, 2)
    with pytest.raises(ValueError, match="sil"):
        figures._sens_axes(
            ax_rc,
            ax_s,
            figures._theme("report"),
            "tune-coarse",
            [figures.Row(0, "sil", "method", "main", "sil", "tuned")],
            table,
            tuning["runs"],
        )


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


def test_f19_shares_divide_by_the_runs_that_have_a_curve(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs, curves = frames_1m["runs"], frames_1m["curves"]
    sil = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]
    gone = sil[sil["seed"] >= 5]["run_id"]  # half the runs logged no curve (R78)
    half = {**frames_1m, "curves": curves[~curves["run_id"].isin(gone)]}
    t3 = stability_table(half)
    row = t3[(t3["method"] == "sil") & (t3["block"] == "main")].iloc[0]
    assert (row["seeds"], row["curve_runs"]) == (10, 5)
    assert row["runs_with_collapse"] == 1  # sil seed 0 collapses
    fig = plot_stability(half, tmp_path)
    collapse = with_gid(fig, "collapse", "1m", "main", "sil")
    assert collapse.get_width() == pytest.approx(1 / 5)  # not 1 / 10
    drops = with_gid(fig, "drops", "1m", "main", "sil")
    assert drops.get_width() == pytest.approx(row["runs_with_drops"] / 5)
    assert len(with_gid(fig, "dd", "1m", "main", "sil").get_offsets()) <= 5


def test_f19_draws_no_share_bar_for_a_method_without_any_curve(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs, curves = frames_1m["runs"], frames_1m["curves"]
    sil = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]["run_id"]
    bare = {**frames_1m, "curves": curves[~curves["run_id"].isin(sil)]}
    fig = plot_stability(bare, tmp_path)
    assert not [a for a in artists(fig, "collapse") if a.get_gid().endswith("/sil")]


# --- report widths (R105) -------------------------------------------------------


@pytest.mark.parametrize("fid", list(NAMES))
def test_double_width_is_170_mm_with_its_own_files_and_7pt_fonts(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    fig = draw(fid, frames, tuning, literature, tmp_path, width="double")
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(170.0, abs=0.01)
    assert not (tmp_path / f"{NAMES[fid]}_double.pdf").exists()
    assert (tmp_path / f"{NAMES[fid]}_double.png").read_bytes().startswith(PNG)
    fig.canvas.draw()
    small = {
        t.get_text(): t.get_fontsize()
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7
    }
    assert not small, small


def test_the_width_option_is_refused_when_unknown(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="width"):
        plot_curves_score(frames, tmp_path, width="triple", n_boot=N)


def test_the_video_ignores_the_width_option(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames, tmp_path, "video", width="double", n_boot=N)
    assert tuple(round(v) for v in fig.get_size_inches() * fig.dpi) == (1920, 1080)
    assert (tmp_path / "curves_score_video.png").exists()


def test_f1_at_double_width_is_one_panel_per_budget_with_every_method(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_curves_score(frames, tmp_path, width="double", n_boot=N)
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.lines]
    assert len(panels) == 2  # 1M and 1B side by side, not four stacked panels
    xs = sorted(ax.get_position().x0 for ax in panels)
    assert xs[1] > xs[0]
    assert panels[0].get_position().y0 == pytest.approx(panels[1].get_position().y0)
    methods = {
        a.get_gid().split("/")[3]
        for a in artists(fig, "curve")
        if "/1m/" in a.get_gid()
    }
    assert methods == set(METHOD_COLORS_LIGHT)  # all nine, the controls once
    gids = [a.get_gid() for a in artists(fig, "curve")]
    assert len(gids) == len(set(gids))
    titles = [ax.get_title(loc="left") for ax in panels]
    assert any("1B" in t and "indicative" in t for t in titles)
    short = plot_curves_score(frames, tmp_path / "c", n_boot=N)
    assert (
        fig.get_size_inches()[1] < short.get_size_inches()[1]
    )  # 0.25 page, not 4 stacked


# --- F2 + F3 side by side -------------------------------------------------------


def test_f2f3_puts_the_poi_against_each_control_beside_the_intervals(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    poi = poi_matrix(frames, n_boot=N)
    t1 = main_table(frames, n_boot=N)
    fig = plot_aggregate_poi(frames, tmp_path, table=t1, poi=poi)
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
    iqm = with_gid(fig, "iqm", "1m", "main", "sil", "tuned")
    left = iqm.axes
    image = with_gid(fig, "poi", "1m", "controls")
    right = image.axes
    assert right.get_position().x0 > left.get_position().x1  # side by side
    assert right.get_position().y0 == pytest.approx(left.get_position().y0)
    assert [t.get_text() for t in right.get_xticklabels()] == list(CONTROLS)
    labels = [t.get_text() for t in left.get_yticklabels()]
    cells = {
        (int(t.get_position()[1]), int(t.get_position()[0])): t for t in right.texts
    }
    one_1m = poi[poi["benchmark"] == "1m"]
    n_cells = 0
    for r in one_1m.itertuples():
        if r.versus not in CONTROLS:
            continue
        # a method's row in its block; the controls repeat, so match by block order
        rows = [i for i, name in enumerate(labels) if name == r.method]
        block_start = labels.index(
            "main" if r.block == "main" else "uses hand-written skills"
        )
        row = next(i for i in rows if i > block_start)
        text = cells[(row, CONTROLS.index(r.versus))]
        assert text.get_text() == f"{r.poi:.2f}"
        excluded = r.poi_lo > 0.5 or r.poi_hi < 0.5
        assert text.get_gid() == ("poi-cell/bold" if excluded else "poi-cell/plain")
        n_cells += 1
    assert n_cells == len(cells)
    one(fig, "mean/1m/main/sil/tuned")


def test_f2f3_1b_rows_have_intervals_only(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_aggregate_poi(frames, tmp_path, n_boot=N)
    one(fig, "median/1b/main/sil/tuned")
    assert len(artists(fig, "poi")) == 1  # the 1M image only
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
    assert any("1B" in t and "indicative" in t for t in titles)


# --- every method artist uses METHOD_COLORS -------------------------------------

METHOD_KINDS = (
    "curve", "band", "iqm", "mean", "median", "ci-iqm", "ci-mean", "ci-median",
    "seeds", "rc", "sens", "dd", "drops", "collapse", "hw",
)  # fmt: skip
"""gid kinds drawn in a method's colour (the others are ink)."""


def _hexes(artist: Artist) -> set[str]:
    """Every colour an artist is drawn with, as hex without alpha."""
    out: set[str] = set()
    for getter in (
        "get_color",
        "get_facecolor",
        "get_edgecolor",
        "get_markerfacecolor",
    ):
        value = getattr(artist, getter, lambda: None)()
        if value is None:
            continue
        if isinstance(value, str):
            out.add(to_hex(value))
            continue
        arr = np.atleast_2d(np.asarray(value, dtype=float))
        out |= {to_hex(tuple(row[:3])) for row in arr if row.size >= 3}
    return out


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", list(NAMES))
def test_every_method_artist_is_drawn_in_a_method_colour(
    fid: str,
    theme: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    fig = draw(fid, frames, tuning, literature, tmp_path, theme)
    palette = {to_hex(c) for c in _colors(theme).values()}
    checked = 0
    for kind in METHOD_KINDS:
        for artist in artists(fig, kind):
            checked += 1
            assert _hexes(artist) & palette, (artist.get_gid(), _hexes(artist))
            parts = artist.get_gid().split("/")
            method = parts[2] if kind in ("rc", "sens") else parts[3]
            assert to_hex(_colors(theme)[method]) in _hexes(artist), artist.get_gid()
    assert checked or fid in ("F3", "F5", "LEG")


# --- the one method legend -----------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
def test_legend_methods_lists_every_method_with_its_colour_style_and_marker(
    theme: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_legend_methods(frames, tmp_path, theme)
    assert not fig.axes, "the legend is the whole figure"
    (legend,) = fig.legends
    labels = [t.get_text() for t in legend.get_texts()]
    methods = [m for m in labels if m in METHOD_COLORS]
    assert set(methods) == set(frames["runs"]["method"]) - {"ablation"}
    assert methods[:3] == ["ppo", "ppo-rnn", "pqn-rnn"][: len(methods[:3])]  # controls
    palette = _colors(theme)
    for handle, label in zip(legend.legend_handles, labels, strict=True):
        if label in CONTROLS:
            assert handle.get_linestyle() == "--"
        if label in METHOD_COLORS:
            assert to_hex(handle.get_color()) == to_hex(palette[label])
            assert handle.get_marker() not in ("", "None", None)


def test_legend_methods_adds_the_ablation_style_when_runs_have_ablations(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_legend_methods(frames, tmp_path)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert ("ablation" in labels) == bool(
        (frames["runs"]["config_ref"] != "tuned").any()
    )


def test_legend_methods_has_a_caption_and_the_three_variants(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with capture_captions() as rows:
        for theme, width in (
            ("report", "column"),
            ("report", "double"),
            ("video", "column"),
        ):
            plot_legend_methods(frames, tmp_path, theme, width=width)
    assert {p.name for p in tmp_path.iterdir()} == {
        "legend_methods.png",
        "legend_methods_double.png",
        "legend_methods_video.png",
    }
    assert [c.stem for c in rows] == ["legend_methods"]
    assert "colour" in rows[0].description
