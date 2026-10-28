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
