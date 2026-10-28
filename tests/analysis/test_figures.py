"""Tests of the core figures F1-F5 and F18-F20 on the comparison fixture.

Every figure is one panel per PNG (a function writes a set of stems), is sized for one
Interspeech column (report) or a 1920 x 1080 frame (video), takes its colours from
`METHOD_COLORS` and its numbers from the comparison tables, and registers a caption;
the tests read the drawn artists back (found by their ``gid``,
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
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from wut_murl_craftax.analysis import compare, figures
from wut_murl_craftax.analysis.compare import (
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
    plot_aggregate,
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
from wut_murl_craftax.visualisation import charts
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
STEMS = {
    "LEG": ["legend_methods"],
    "F1": ["curves_score_main", "curves_score_skills"],
    "F2": ["aggregate"],
    "F3": ["poi_matrix_main", "poi_matrix_skills"],
    "F4": ["seed_spread_main", "seed_spread_skills", "seed_spread_published"],
    "F5": ["success_heatmap_main", "success_heatmap_skills"],
    "F18": [
        "config_scores_coarse", "sensitivity_coarse", "config_scores_fine",
        "sensitivity_fine",
    ],
    "F19": ["stability_drawdown", "stability_events"],
    "F20": [
        "hardware_sps", "hardware_wall_clock", "hardware_compile", "hardware_memory",
        "hardware_utilisation",
    ],
}  # fmt: skip
"""The files each figure id writes at 1M on the fixture (one panel per file)."""
LABELS = {
    "LEG": (),
    "F1": ("env steps", "score (% of max)"),
    "F2": ("final score (% of max)",),
    "F3": (),
    "F4": ("final score (% of max)",),
    "F5": (),
    "F18": ("config score r_c", "sensitivity S_h"),
    "F19": ("max drawdown", "share of runs"),
    "F20": ("steps per second", "peak memory"),
}
"""Substrings the axis labels of a figure set must contain."""


class Bundle:
    """The figures one call wrote (stem -> figure), searched as if they were one."""

    def __init__(
        self, figs: dict[str, Figure], captions: dict[str, str] | None = None
    ) -> None:
        self.figs = figs
        self.captions = captions or {}
        """Stem -> caption description registered while the figures were drawn."""

    @property
    def axes(self) -> list:
        return [a for f in self.figs.values() for a in f.axes]

    def findobj(self, match: Callable[[Artist], bool]) -> list[Artist]:
        return [a for f in self.figs.values() for a in f.findobj(match)]

    def __getitem__(self, stem: str) -> Figure:
        return self.figs[stem]


def _colors(theme: str) -> dict[str, str]:
    return METHOD_COLORS if theme == "video" else METHOD_COLORS_LIGHT


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Craftax-1M: 10 seeds, the ablations and achievements (one benchmark a call)."""
    return figure_frames(("1m",))


@pytest.fixture(scope="module")
def frames_1m(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return frames


@pytest.fixture(scope="module")
def frames_1b() -> dict[str, pd.DataFrame]:
    """Craftax-1B: 3 seeds (indicative: median and mean, no PoI)."""
    return figure_frames(("1b",))


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
) -> Bundle:
    """Call the figure of ``fid`` with test-sized bootstraps and the shared tables."""
    w = {"width": width}
    plain: dict[str, Callable[[], dict[str, Figure]]] = {
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
        "LEG": lambda: {
            "legend_methods": plot_legend_methods(frames, out_dir, theme, **w)
        },
    }
    return Bundle(plain[fid]())


def artists(fig: Bundle | Figure, kind: str) -> list[Artist]:
    """Artists whose gid starts with ``kind + '/'``, in drawing order."""
    return fig.findobj(lambda a: (a.get_gid() or "").startswith(f"{kind}/"))


def one(fig: Bundle | Figure, gid: str) -> Artist:
    found = fig.findobj(lambda a: a.get_gid() == gid)
    assert len(found) == 1, f"{gid}: {len(found)} artists"
    return found[0]


def texts(fig: Bundle | Figure) -> list[str]:
    for f in fig.figs.values() if isinstance(fig, Bundle) else [fig]:
        f.canvas.draw()
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def axis_labels(fig: Bundle | Figure) -> str:
    labels = []
    for ax in fig.axes:
        labels += [ax.get_xlabel(), ax.get_ylabel()]
    return " | ".join(labels)


def tick_labels(ax: object) -> list[str]:
    return [t.get_text() for t in ax.get_yticklabels()]  # type: ignore[attr-defined]


def with_gid(fig: Bundle | Figure, kind: str, *parts: str) -> Artist:
    return one(fig, "/".join((kind, *parts)))


def _block(method: str) -> str:
    return "skills" if method in ("smdp-skills", "dyna-q") else "main"


def _none_lit() -> pd.DataFrame:
    return pd.DataFrame(columns=compare.LITERATURE_TABLE_COLUMNS)


# --- every figure -------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", list(NAMES))
def test_each_figure_writes_its_files_as_png_with_captions(
    fid: str,
    theme: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    width = "double" if fid == "LEG" else "column"  # the legend spans both columns
    with capture_captions() as caps:
        figs = draw(fid, frames, tuning, literature, tmp_path, theme, width)
    assert sorted(figs.figs) == sorted(STEMS[fid])
    suffix = "_video" if theme == "video" else ""
    for stem in STEMS[fid]:
        assert not (tmp_path / f"{stem}{suffix}.pdf").exists()
        assert (tmp_path / f"{stem}{suffix}.png").read_bytes().startswith(PNG)
    assert sorted(c.stem for c in caps) == sorted(STEMS[fid])
    assert FIGURES[fid] is getattr(figures, FIGURES[fid].__name__)
    for want in LABELS[fid]:
        assert want in axis_labels(figs), (fid, want, axis_labels(figs))


def test_the_two_themes_are_two_files_and_two_surfaces(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    light = plot_aggregate(frames, tmp_path, "report", n_boot=N)["aggregate"]
    dark = plot_aggregate(frames, tmp_path, "video", n_boot=N)["aggregate"]
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


def test_a_figure_shows_one_benchmark_at_a_time(tmp_path: Path) -> None:
    both = figure_frames()
    with pytest.raises(ValueError, match="one benchmark"):
        plot_aggregate(both, tmp_path, n_boot=N)


@pytest.mark.parametrize("fid", list(NAMES))
def test_report_figures_are_one_column_wide_with_7pt_fonts(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    figs = draw(fid, frames, tuning, literature, tmp_path)
    for fig in figs.figs.values():
        assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
        fig.canvas.draw()
        sizes = {
            t.get_text(): t.get_fontsize()
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip()
        }
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
    for fig in draw(fid, frames, tuning, literature, tmp_path, "video").figs.values():
        width, height = fig.get_size_inches() * fig.dpi
        assert (round(width), round(height)) == (1920, 1080)


@pytest.mark.parametrize("fid", [f for f in NAMES if f != "LEG"])
def test_a_column_figure_is_not_drawn_at_the_double_width(
    fid: str,
    frames: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    literature: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Every core figure but the legend fits the column: one image, no 170 mm copy."""
    figs = draw(fid, frames, tuning, literature, tmp_path, width="double")
    assert not figs.figs
    assert not list(tmp_path.iterdir())


def test_a_listed_stem_is_one_170_mm_image_without_a_column_copy(
    frames: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(charts, "DOUBLE_STEMS", ("aggregate",))
    plot_aggregate(frames, tmp_path / "col", width="column", n_boot=N)
    assert not list((tmp_path / "col").glob("*.png"))
    fig = plot_aggregate(frames, tmp_path, width="double", n_boot=N)["aggregate"]
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(170.0, abs=0.01)
    assert (tmp_path / "aggregate.png").read_bytes().startswith(PNG)
    assert not (tmp_path / "aggregate_double.png").exists()


def test_the_width_option_is_refused_when_unknown(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="width"):
        plot_curves_score(frames, tmp_path, width="triple", n_boot=N)


def test_the_video_ignores_the_width_option(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = plot_curves_score(frames, tmp_path, "video", width="double", n_boot=N)
    for fig in figs.values():
        assert tuple(round(v) for v in fig.get_size_inches() * fig.dpi) == (1920, 1080)
    assert (tmp_path / "curves_score_main_video.png").exists()


@pytest.mark.parametrize("fid", ["F1", "F2", "F3", "F4", "F5", "F19", "F20"])
def test_no_run_data_gives_a_labelled_placeholder(
    fid: str, tmp_path: Path, tuning: dict[str, pd.DataFrame]
) -> None:
    full = figure_frames(("1m",))
    empty = {
        "runs": full["runs"].iloc[:0],
        "curves": full["curves"].iloc[:0],
        "eval_episodes": full["eval_episodes"].iloc[:0],
    }
    figs = (
        Bundle(plot_seed_spread(empty, tmp_path, literature=_none_lit()))
        if fid == "F4"
        else draw(fid, empty, tuning, pd.DataFrame(), tmp_path)
    )
    assert any("no data" in t for t in texts(figs))
    assert (tmp_path / f"{NAMES[fid]}.png").exists()


@pytest.mark.parametrize("fid", ["F1", "F2", "F4", "F5", "F19"])
def test_1b_figures_have_no_pending_or_indicative_text(
    fid: str,
    frames_1b: dict[str, pd.DataFrame],
    tuning: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """The 1B set is drawn from 1B alone and says nothing about the other budget."""
    if fid == "F4":
        figs = Bundle(plot_seed_spread(frames_1b, tmp_path, literature=_none_lit()))
    else:
        figs = draw(fid, frames_1b, tuning, pd.DataFrame(), tmp_path)
    assert figs.figs
    for t in texts(figs):
        assert not any(w in t for w in ("pending", "indicative", "1B", "1M")), t


# --- F1 ----------------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
def test_f1_colours_and_styles_follow_the_contract(
    theme: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = Bundle(plot_curves_score(frames, tmp_path, theme, n_boot=N))
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


def test_f1_has_one_file_per_block_with_the_controls_in_both_and_one_y_range(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = plot_curves_score(frames, tmp_path, n_boot=N)
    assert sorted(figs) == ["curves_score_main", "curves_score_skills"]
    methods = {}
    for stem, fig in figs.items():
        gids = [a.get_gid().split("/") for a in artists(fig, "curve")]
        methods[stem] = {m for _, _, _, m, _ in gids}
        assert {b for _, _, b, _, _ in gids} == {stem.removeprefix("curves_score_")}
    assert set(CONTROLS) <= methods["curves_score_main"]
    assert set(CONTROLS) <= methods["curves_score_skills"]
    assert methods["curves_score_skills"] - set(CONTROLS) == {"smdp-skills", "dyna-q"}
    assert "smdp-skills" not in methods["curves_score_main"]
    first, second = (f.axes[0] for f in figs.values())
    assert first.get_ylim() == second.get_ylim()
    assert first.get_position().bounds[2] == pytest.approx(
        second.get_position().bounds[2]
    )


def test_f1_is_the_shared_curve_band_over_seeds(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = Bundle(plot_curves_score(frames, tmp_path, n_boot=N, seed=3))
    runs, curves = frames["runs"], frames["curves"]
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
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = Bundle(plot_curves_score(frames, tmp_path, n_boot=N, ablations=True))
    colors = _colors("report")
    for method, ref in ABLATIONS:
        line = with_gid(fig, "curve", "1m", _block(method), method, ref)
        assert line.get_linestyle() == ":"
        assert line.get_color() == colors[method]
        assert line.get_label() == f"{method} ({ref})"
        tuned = with_gid(fig, "curve", "1m", _block(method), method, "tuned")
        assert tuned.get_linestyle() == "-"  # never pooled into the method's curve
        assert not np.allclose(line.get_ydata(), tuned.get_ydata())


# --- F2 ----------------------------------------------------------------------------


def test_f2_reads_the_main_table_and_separates_the_blocks(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = main_table(frames, n_boot=N)
    fig = Bundle(plot_aggregate(frames, tmp_path, n_boot=N, table=table))
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
    assert "skills" in labels
    assert labels.index("main") < labels.index("skills")
    after = labels[labels.index("skills") :]
    assert [c for c in CONTROLS if c in after] == list(CONTROLS)  # controls repeated
    assert "smdp-skills" not in labels[: labels.index("skills")]


def test_f2_1b_gives_median_and_mean_never_iqm(
    frames_1b: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = main_table(frames_1b, n_boot=N)
    fig = Bundle(plot_aggregate(frames_1b, tmp_path, n_boot=N, table=table))
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
    assert not artists(fig, "iqm")


def test_f2_ablations_are_dotted_rows_under_their_method(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = ablation_table(frames, n_boot=N)
    fig = Bundle(
        plot_aggregate(
            frames, tmp_path, n_boot=N, ablation=table,
            table=main_table(frames, n_boot=N),
        )
    )  # fmt: skip
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


def test_f3_cells_are_the_poi_table_one_square_file_per_block(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = poi_matrix(frames, n_boot=N)
    figs = plot_poi_matrix(frames, tmp_path, n_boot=N, table=table)
    for block, other in (("main", "smdp-skills"), ("skills", "sil")):
        fig = figs[f"poi_matrix_{block}"]
        w, h = fig.get_size_inches()
        assert h / w == pytest.approx(1.0)
        image = with_gid(fig, "poi", "1m", block)
        ax = image.axes
        names = [t.get_text() for t in ax.get_yticklabels()]
        assert [t.get_text() for t in ax.get_xticklabels()] == names
        assert set(CONTROLS) <= set(names)
        assert other not in names  # no cross-block cell
        cells = np.asarray(image.get_array(), dtype=float)
        rows = table[(table["benchmark"] == "1m") & (table["block"] == block)]
        for r in rows.itertuples():
            assert cells[names.index(r.method), names.index(r.versus)] == (
                pytest.approx(r.poi)
            )
        assert np.isnan(np.diag(cells)).all()


def _cell_texts(fig: Bundle | Figure, image: Artist) -> dict[tuple[int, int], Text]:
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
    fig = Bundle(plot_poi_matrix(frames, tmp_path, n_boot=N, table=table))
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
    figs = plot_poi_matrix(figure_frames(("1b",)), tmp_path, n_boot=N)
    assert any("10-seed 1M" in t for t in texts(Bundle(figs)))


# --- F4 ----------------------------------------------------------------------------


def test_f4_dots_are_the_per_seed_final_scores(
    frames: dict[str, pd.DataFrame], literature: pd.DataFrame, tmp_path: Path
) -> None:
    fig = Bundle(plot_seed_spread(frames, tmp_path, literature=literature))
    runs = frames["runs"]
    for method in ("ppo", "sil"):
        got = with_gid(fig, "seeds", "1m", _block(method), method)
        want = runs[(runs["method"] == method) & (runs["config_ref"] == "tuned")][
            "final_score"
        ]
        assert sorted(got.get_offsets()[:, 0]) == pytest.approx(sorted(want))


def test_f4_published_rows_are_their_own_file_on_the_same_x_range_and_inset(
    frames: dict[str, pd.DataFrame], literature: pd.DataFrame, tmp_path: Path
) -> None:
    figs = plot_seed_spread(frames, tmp_path, literature=literature)
    assert sorted(figs) == [
        "seed_spread_main", "seed_spread_published", "seed_spread_skills",
    ]  # fmt: skip
    lit = literature[
        (literature["source"] == "literature")
        & (literature["score_kind"] == "training")
        & (literature["benchmark"] == "1m")
    ]
    marks = artists(figs["seed_spread_published"], "lit")
    assert len(marks) == len(lit)
    got = sorted(float(m.get_offsets()[0, 0]) for m in marks)
    assert got == pytest.approx(sorted(lit["score_pct"]))
    same = [m for m in marks if m.get_facecolor().size and m.get_facecolor()[0][3]]
    assert same  # protocol-matching rows are filled, the others hollow
    assert len(same) < len(marks)
    axes = [f.axes[0] for f in figs.values()]
    assert len({ax.get_xlim() for ax in axes}) == 1
    assert len({round(ax.get_position().x0, 6) for ax in axes}) == 1
    assert len({round(ax.get_position().x1, 6) for ax in axes}) == 1


def test_f4_without_literature_rows_has_no_published_file(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = plot_seed_spread(frames, tmp_path, literature=_none_lit())
    assert "seed_spread_published" not in figs
    assert artists(Bundle(figs), "seeds")


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
    frames: dict[str, pd.DataFrame],
) -> None:
    rates = _expected_rates(frames)
    assert rates.nunique().min() > 3  # every achievement differs across methods
    assert rates.T.nunique().min() > 2  # and every method across achievements


def test_f5_cells_are_the_final_success_rates_one_file_per_block(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = plot_success_heatmap(frames, tmp_path)
    rates = _expected_rates(frames)
    for block in ("main", "skills"):
        fig = figs[f"success_heatmap_{block}"]
        image = with_gid(fig, "heat", "1m", block)
        names = [t.get_text() for t in image.axes.get_xticklabels()]
        assert set(CONTROLS) <= set(names)
        rows = [t.get_text() for t in image.axes.get_yticklabels()]
        assert rows == list(ACHIEVEMENTS)  # every fixture rate is far above 1%
        cells = np.asarray(image.get_array(), dtype=float)
        assert cells.shape == (len(ACHIEVEMENTS), len(names))
        for j, method in enumerate(names):
            for i, achievement in enumerate(ACHIEVEMENTS):  # row i is achievement i
                assert cells[i, j] == pytest.approx(rates.loc[method, achievement])
    main = [
        t.get_text() for t in figs["success_heatmap_main"].axes[0].get_xticklabels()
    ]
    assert "smdp-skills" not in main
    sizes = {tuple(f.get_size_inches()) for f in figs.values()}
    assert len(sizes) == 1  # the two files tile
    a, b = (f.axes[0].get_position().bounds for f in figs.values())
    assert a[1] == pytest.approx(b[1])
    assert a[3] == pytest.approx(b[3])


def test_f5_folds_the_achievements_nobody_reaches_into_the_caption(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    eps = frames["eval_episodes"].copy()
    eps["ach/enter_sewers"] = False  # nobody reaches it
    with capture_captions() as caps:
        figs = plot_success_heatmap({**frames, "eval_episodes": eps}, tmp_path)
    rows = [
        t.get_text() for t in figs["success_heatmap_main"].axes[0].get_yticklabels()
    ]
    assert "enter_sewers" not in rows
    assert all("enter_sewers" in c.description for c in caps)


# --- F18 ---------------------------------------------------------------------------


def test_f18_strip_and_bars_read_config_scores_and_t7(
    tuning: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = sensitivity_table(tuning)
    fig = Bundle(plot_sensitivity(tuning, tmp_path, table=table))
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
    fig = Bundle(plot_sensitivity(tuning, tmp_path, table=table))
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
    fig = Bundle(plot_sensitivity(tuning, tmp_path, table=table))
    order = {
        "tune-coarse": ["ppo", "sil", "smdp-skills"],
        "tune-fine": ["sil", "smdp-skills"],
    }
    for phase, methods in order.items():
        bar = with_gid(fig, "sens", phase, methods[0], "1")
        ax_s = bar.axes
        ax_rc = with_gid(fig, "rc", phase, methods[0]).axes
        assert ax_rc is not ax_s  # one panel per file
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
            assert np.all(np.abs(dots - k) < 0.5)
    smdp = table[(table["phase"] == "tune-coarse") & (table["method"] == "smdp-skills")]
    assert smdp.iloc[0]["top1"] == "eps_top"  # its own hyperparameter name


def test_the_bars_panel_refuses_a_method_without_a_t7_row(
    tuning: dict[str, pd.DataFrame],
) -> None:
    table = sensitivity_table(tuning)
    table = table[table["method"] != "sil"]
    ax = Figure().subplots()
    with pytest.raises(ValueError, match="sil"):
        figures._bars_panel(
            ax,
            figures._theme("report"),
            "tune-coarse",
            [figures.Row(0, "sil", "method", "main", "sil", "tuned")],
            table,
        )


def test_f18_without_a_grid_draws_only_the_phase_it_has(tmp_path: Path) -> None:
    runs = tuning_frames()["runs"]
    figs = plot_sensitivity({"runs": runs[runs["phase"] == "tune-coarse"]}, tmp_path)
    assert sorted(figs) == ["config_scores_coarse", "sensitivity_coarse"]


# --- F19 ---------------------------------------------------------------------------


def test_f19_dots_are_the_per_run_drawdowns_and_bars_are_t3(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = plot_stability(frames, tmp_path)
    fig = Bundle(figs)
    per_run = run_stability(frames)
    t3 = stability_table(frames)
    for method in ("sil", "ppo"):
        mine = per_run[(per_run["benchmark"] == "1m") & (per_run["method"] == method)]
        dots = with_gid(fig, "dd", "1m", _block(method), method)
        got = sorted(dots.get_offsets()[:, 0])
        assert got == pytest.approx(sorted(mine["max_drawdown"].dropna()))
        row = t3[
            (t3["benchmark"] == "1m")
            & (t3["method"] == method)
            & (t3["block"] == _block(method))
        ].iloc[0]
        median = with_gid(fig, "dd-median", "1m", _block(method), method)
        assert median.get_xdata()[0] == pytest.approx(row["max_drawdown_median"])
        drops = with_gid(fig, "drops", "1m", _block(method), method)
        assert drops.get_width() == pytest.approx(row["runs_with_drops"] / row["seeds"])
        collapse = with_gid(fig, "collapse", "1m", _block(method), method)
        assert collapse.get_width() == pytest.approx(
            row["runs_with_collapse"] / row["seeds"]
        )
    assert (t3["runs_with_collapse"] > 0).any()  # the fixture has a collapsing run
    marked = with_gid(fig, "collapsed", "1m", "main", "sil")
    assert len(marked.get_offsets()) == 1  # sil seed 0 collapses


def test_f19_files_have_one_size_and_one_axes_box(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    a, b = plot_stability(frames, tmp_path).values()
    assert tuple(a.get_size_inches()) == tuple(b.get_size_inches())
    assert a.axes[0].get_position().bounds == b.axes[0].get_position().bounds


def test_f19_shares_divide_by_the_runs_that_have_a_curve(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs, curves = frames["runs"], frames["curves"]
    sil = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]
    gone = sil[sil["seed"] >= 5]["run_id"]  # half the runs logged no curve (R78)
    half = {**frames, "curves": curves[~curves["run_id"].isin(gone)]}
    t3 = stability_table(half)
    row = t3[(t3["method"] == "sil") & (t3["block"] == "main")].iloc[0]
    assert (row["seeds"], row["curve_runs"]) == (10, 5)
    assert row["runs_with_collapse"] == 1  # sil seed 0 collapses
    fig = Bundle(plot_stability(half, tmp_path))
    collapse = with_gid(fig, "collapse", "1m", "main", "sil")
    assert collapse.get_width() == pytest.approx(1 / 5)  # not 1 / 10
    drops = with_gid(fig, "drops", "1m", "main", "sil")
    assert drops.get_width() == pytest.approx(row["runs_with_drops"] / 5)
    assert len(with_gid(fig, "dd", "1m", "main", "sil").get_offsets()) <= 5


def test_f19_draws_no_share_bar_for_a_method_without_any_curve(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs, curves = frames["runs"], frames["curves"]
    sil = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]["run_id"]
    bare = {**frames, "curves": curves[~curves["run_id"].isin(sil)]}
    fig = Bundle(plot_stability(bare, tmp_path))
    assert not [a for a in artists(fig, "collapse") if a.get_gid().endswith("/sil")]


# --- F20 ---------------------------------------------------------------------------


def test_f20_reads_t4_medians_one_file_per_metric_and_names_the_machine(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with capture_captions() as caps:
        figs = plot_hardware(frames, tmp_path)
    fig = Bundle(figs)
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
    assert "hardware_efficiency" not in figs  # no energy series: not drawn, said below
    text = {c.stem: c.description for c in caps}
    assert "Apple Silicon laptop" in text["hardware_sps"]
    assert "steps per joule" in text["hardware_sps"]
    assert compare.CONTEXT_NOTE in " ".join(text["hardware_sps"].split())
    assert not [a for a in artists(fig, "hw") if a.get_gid().endswith("/1b")]
    assert not [t for t in texts(fig) if "n.m." in t]


def test_f20_keeps_profiles_apart_in_files_of_their_own(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs = frames["runs"].copy()
    other = runs[runs["method"].isin(["ppo", "sil"])].assign(profile="cuda")
    figs = plot_hardware({**frames, "runs": pd.concat([runs, other])}, tmp_path)
    assert {"hardware_sps_cuda", "hardware_sps_laptop"} <= set(figs)
    cuda, laptop = (
        Bundle({"c": figs["hardware_sps_cuda"]}),
        Bundle({"l": figs["hardware_sps_laptop"]}),
    )
    one(cuda, "hw/cuda/steady_sps/ppo")
    one(cuda, "hw/cuda/steady_sps/sil")
    one(laptop, "hw/laptop/steady_sps/dyna-q")
    with pytest.raises(AssertionError):
        one(cuda, "hw/cuda/steady_sps/dyna-q")  # a profile only shows its own runs
    assert not artists(cuda, "hw/laptop")


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


def test_legend_methods_has_a_caption_one_report_image_and_the_video(
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
        "legend_methods_video.png",
    }
    png = (tmp_path / "legend_methods.png").read_bytes()
    assert int.from_bytes(png[16:20], "big") == pytest.approx(170 / 25.4 * 300, abs=1)
    assert [c.stem for c in rows] == ["legend_methods"]
    assert rows[0].width_mm == 170.0
    assert "colour" in rows[0].description
