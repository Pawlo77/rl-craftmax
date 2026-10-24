"""Tests of the shared chart palette: method colours in both themes (contracts.md 6)."""

import pytest

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
