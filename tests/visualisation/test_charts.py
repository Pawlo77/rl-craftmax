"""Tests of the shared chart palette: method colours in both themes (contracts.md 6)."""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.visualisation import charts

CONTROLS = ("ppo", "ppo-rnn", "pqn-rnn")


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two ``#rrggbb`` colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("palette", ["METHOD_COLORS", "METHOD_COLORS_LIGHT"])
def test_every_registry_method_has_a_colour(palette: str) -> None:
    assert set(getattr(charts, palette)) == set(METHODS)


@pytest.mark.parametrize(
    ("palette", "surface"),
    [("METHOD_COLORS", "SURFACE"), ("METHOD_COLORS_LIGHT", "SURFACE_LIGHT")],
)
def test_colours_have_3_to_1_contrast_with_their_surface(
    palette: str, surface: str
) -> None:
    for method, color in getattr(charts, palette).items():
        assert contrast(color, getattr(charts, surface)) >= 3.0, method


@pytest.mark.parametrize("palette", ["METHOD_COLORS", "METHOD_COLORS_LIGHT"])
def test_the_controls_are_three_distinct_greys_and_the_rest_are_not_grey(
    palette: str,
) -> None:
    colors = getattr(charts, palette)
    greys = [colors[c] for c in CONTROLS]
    assert len(set(greys)) == 3
    for hex_color in greys:
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        assert max(r, g, b) - min(r, g, b) <= 12
    for method, hex_color in colors.items():
        if method not in CONTROLS:
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
            assert max(r, g, b) - min(r, g, b) > 40, method
    assert len(set(colors.values())) == len(colors)


def test_the_light_theme_has_its_own_surface_ink_muted_and_grid() -> None:
    assert charts.SURFACE_LIGHT == "#ffffff"
    assert contrast(charts.INK_LIGHT, charts.SURFACE_LIGHT) >= 7
    assert contrast(charts.MUTED_LIGHT, charts.SURFACE_LIGHT) >= 4.5
    assert charts.GRID_LIGHT != charts.SURFACE_LIGHT
    assert charts.SURFACE == "#1a1a19"  # the dark constants stay as they were


@pytest.mark.parametrize(
    ("seq", "div"),
    [("SEQUENTIAL_LIGHT", "DIVERGING_LIGHT"), ("SEQUENTIAL_DARK", "DIVERGING_DARK")],
)
def test_heat_map_ramps_never_reuse_a_method_colour(seq: str, div: str) -> None:
    methods = {
        c.lower()
        for c in {*charts.METHOD_COLORS.values(), *charts.METHOD_COLORS_LIGHT.values()}
    }
    for name in (seq, div):
        ramp = getattr(charts, name)
        assert not {c.lower() for c in ramp} & methods
        assert len(set(ramp)) == len(ramp)
    assert len(getattr(charts, div)) == 3


# --- the figure theme ---------------------------------------------------------------


def test_the_figure_is_drawn_under_its_own_rc_values(tmp_path: Path) -> None:
    th = charts.theme("video")
    fig = charts.themed_figure(th, 0)
    fig.add_subplot().plot([0, 1])
    seen: list[float] = []
    plain = Figure.savefig

    def spy(self: Figure, *args: Any, **kwargs: Any) -> None:
        seen.append(mpl.rcParams["xtick.labelsize"])  # lazily read while drawing
        plain(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "savefig", spy)
        charts.save_png(fig, th, tmp_path, "x")
    assert seen == [th.font]
    assert mpl.rcParams["xtick.labelsize"] != th.font  # nothing leaked


def test_a_note_wider_than_the_canvas_is_wrapped(tmp_path: Path) -> None:
    th = charts.theme("report")
    fig = charts.themed_figure(th, 60)
    fig.add_subplot().plot([0, 1])
    fig.supxlabel("word " * 60, x=0.01, ha="left", fontsize=th.font)
    charts.settle(fig)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    (note,) = fig.texts
    assert note.get_window_extent(renderer).width <= fig.get_size_inches()[0] * fig.dpi
    assert "\n" in note.get_text()


def test_a_long_panel_title_is_wrapped_to_its_axes(tmp_path: Path) -> None:
    th = charts.theme("report", "double")
    fig = charts.themed_figure(th, 60)
    left, right = fig.subplots(1, 2)
    charts.set_title(left, th, "a very long title " * 6, width=400)
    charts.set_title(right, th, "short")
    charts.settle(fig)
    title = charts._left_title(left)
    assert title is not None
    assert title.get_window_extent(fig.canvas.get_renderer()).x1 <= (  # type: ignore[attr-defined]
        right.get_tightbbox(fig.canvas.get_renderer()).x0  # type: ignore[attr-defined]
    )
    empty = charts._left_title(fig.add_subplot(3, 1, 3))
    assert empty is not None
    assert empty.get_text() == ""


# --- captions and per-benchmark folders -------------------------------------------


def _saved_figure(tmp_path: Path, name: str, **caption: str) -> Path:
    th = charts.report_theme()
    fig = charts.themed_figure(th, 30)
    charts.style_axes(fig.add_subplot(), th).plot([0, 1], [0, 1])
    charts.save_png(fig, th, tmp_path, name, **caption)
    return tmp_path / f"{name}.png"


def test_a_saved_figure_registers_its_caption_under_the_stem(tmp_path: Path) -> None:
    with charts.capture_captions() as rows:
        _saved_figure(tmp_path, "toy", title="Toy", description="A line from 0 to 1.")
        _saved_figure(tmp_path, "toy", title="Toy", description="The last one wins.")
        _saved_figure(tmp_path, "plain")  # no description: nothing registered
    assert rows == [charts.Caption("toy", "Toy", "The last one wins.")]


def test_a_caption_registered_outside_a_block_is_dropped() -> None:
    charts.register_caption("x", "X", "Nobody collects this.")  # no error


@pytest.mark.parametrize("bad", ["", "  ", "two\nlines", "a | b"])
def test_a_caption_text_is_one_cell(bad: str) -> None:
    with pytest.raises(ValueError, match="caption of 'x'"):
        charts.register_caption("x", "X", bad)
    with pytest.raises(ValueError, match="caption of 'x'"):
        charts.register_caption("x", bad, "fine")


def test_captions_csv_keeps_the_rows_of_the_other_figure_ids(tmp_path: Path) -> None:
    rows = [charts.Caption("a", "A", "first"), charts.Caption("b", "B", "second")]
    charts.write_captions(tmp_path, "F1", rows)
    charts.write_captions(tmp_path, "F2", [charts.Caption("c", "C", "third")])
    path = charts.write_captions(tmp_path, "F1", [charts.Caption("a", "A", "new")])
    lines = path.read_text().splitlines()
    assert lines[0] == "stem,id,title,description"
    assert sorted(lines[1:]) == ["a,F1,A,new", "c,F2,C,third"]


def test_the_figure_folder_of_a_benchmark_is_a_subfolder() -> None:
    assert charts.figure_dir(Path("figures"), "1m") == Path("figures/1m")


# --- size classes: widths, aspect ratios, tiling insets ------------------------------


@pytest.mark.parametrize(
    ("width", "mm"), [("column", 80.0), ("double", 170.0), ("tile", 39.0)]
)
def test_report_widths_are_exact_and_the_tile_is_half_a_column(
    width: str, mm: float
) -> None:
    th = charts.theme("report", width)
    assert th.width * 25.4 == pytest.approx(mm)
    assert charts.TILE_MM == (charts.COLUMN_MM - 2) / 2
    assert th.suffix == {"column": "", "double": "_double", "tile": "_tile"}[width]


@pytest.mark.parametrize(
    ("shape", "ratio"),
    [(charts.Shape.CURVE, 0.62), (charts.Shape.DOTS, 0.45), (charts.Shape.SQUARE, 1.0)],
)
def test_a_fixed_aspect_shape_gives_height_over_width(
    shape: charts.Shape, ratio: float
) -> None:
    th = charts.theme("report")
    fig = charts.themed_figure(th, shape=shape)
    w, h = fig.get_size_inches()
    assert h / w == pytest.approx(ratio)
    assert fig.shape == shape  # type: ignore[attr-defined]


def test_a_rows_shape_grows_by_the_theme_pitch_per_row() -> None:
    th = charts.theme("report")
    two = charts.shape_height_mm(th, charts.Shape.ROWS, 2)
    ten = charts.shape_height_mm(th, charts.Shape.ROWS, 10)
    assert ten - two == pytest.approx(8 * th.pitch_mm)
    with pytest.raises(ValueError, match="rows"):
        charts.shape_height_mm(th, charts.Shape.ROWS)
    with pytest.raises(ValueError, match="height_mm or a shape"):
        charts.themed_figure(th)


def test_the_video_keeps_its_frame_whatever_the_shape() -> None:
    fig = charts.themed_figure(charts.VIDEO, shape=charts.Shape.CURVE)
    assert tuple(fig.get_size_inches() * fig.dpi) == (1920, 1080)


def test_tiled_panels_with_one_inset_have_identical_axes_boxes() -> None:
    th = charts.theme("report")
    inset = max(
        charts.label_inset_mm(th, names) for names in (["ppo"], ["count-bonus-pqn-rnn"])
    )
    boxes = []
    for _ in range(2):
        fig = charts.themed_figure(th, shape=charts.Shape.DOTS)
        ax = charts.add_panel(fig, left_mm=inset)
        boxes.append(tuple(ax.get_position().bounds))
    assert boxes[0] == boxes[1]
    left, _, width, _ = boxes[0]
    assert left * 80.0 == pytest.approx(inset)
    assert (left + width) * 80.0 == pytest.approx(78.0)
