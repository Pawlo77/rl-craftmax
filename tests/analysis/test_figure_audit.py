"""The audit of `_figure_audit` bites: planted defects are found, a clean figure is not.

Without these the style test could pass because a detector silently stopped working.
"""

from collections.abc import Callable
from pathlib import Path

from _figure_audit import Saved, audit_saves
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _fig() -> Figure:
    fig = Figure(figsize=(3, 2), dpi=100)
    FigureCanvasAgg(fig)
    return fig


def _clean(fig: Figure) -> None:
    ax = fig.add_axes((0.25, 0.3, 0.65, 0.6))
    ax.plot([0, 1], [0, 1], label="a line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _cropped(fig: Figure) -> None:
    _clean(fig)
    fig.text(-0.1, 0.5, "far left of the canvas")


def _legend_over_data(fig: Figure) -> None:
    _clean(fig)
    fig.axes[0].legend(loc="center")


def _overlapping(fig: Figure) -> None:
    _clean(fig)
    fig.text(0.5, 0.5, "first text")
    fig.text(0.5, 0.5, "second text")


def _tilted_ylabel(fig: Figure) -> None:
    _clean(fig)
    fig.axes[0].set_ylabel("y", rotation=0)


def _save(tmp_path: Path, build: Callable[[Figure], None]) -> Saved:
    fig = _fig()
    build(fig)
    with audit_saves() as saved:
        fig.savefig(tmp_path / "f.png")
    (record,) = saved
    return record


def test_a_clean_figure_has_nothing_to_report(tmp_path: Path) -> None:
    f = _save(tmp_path, _clean)
    assert not f.cut
    assert not f.covered
    assert not f.overlaps
    assert not f.wrapped_titles
    assert not f.lone_yticks
    assert f.ylabel_rotations == [90.0]


def test_a_text_off_the_canvas_is_found(tmp_path: Path) -> None:
    f = _save(tmp_path, _cropped)
    assert any("far left" in c for c in f.cut)


def test_a_legend_over_the_data_is_found(tmp_path: Path) -> None:
    f = _save(tmp_path, _legend_over_data)
    assert f.covered


def test_two_overlapping_texts_are_found(tmp_path: Path) -> None:
    f = _save(tmp_path, _overlapping)
    assert f.overlaps


def test_a_y_label_that_is_not_vertical_is_found(tmp_path: Path) -> None:
    f = _save(tmp_path, _tilted_ylabel)
    assert f.ylabel_rotations == [0.0]
