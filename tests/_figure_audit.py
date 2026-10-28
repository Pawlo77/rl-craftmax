"""Audit of the figures a test saves, as written (after the layout engine ran).

`audit_saves` wraps ``Figure.savefig`` and ``RendererAgg.draw_text`` for a ``with``
block and records per file, never raising: text off the canvas, legends over an axes,
overlapping texts (ticks included), layout warnings, wrapped titles, method legends,
the number of data axes (colour bars excepted), a lone y tick label naming a quantity,
the text kinds (only ticks, axis labels, legends, a title and numbers; no figure-level
text or sentence, no Craftax, 1B, pending, indicative), the declared size class and
pixel size, and the captions registered inside the block (`charts.capture_captions`).
A test decides what each record means.
"""

import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from matplotlib.axes import Axes
from matplotlib.axis import Axis, Tick
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

from wut_murl_craftax.visualisation import charts

TOL_PX = 1.0
"""Pixels a text may poke out of the canvas, or a legend into an axes (AA)."""


@dataclass
class Saved:
    """What one ``savefig`` call wrote and what was wrong with it."""

    path: Path
    cut: list[str] = field(default_factory=list)
    """Texts (and their extent in pixels) that leave the canvas."""
    covered: list[str] = field(default_factory=list)
    """Legends that overlap an axes window."""
    overlaps: list[str] = field(default_factory=list)
    """Pairs of texts whose extents intersect."""
    wrapped_titles: list[str] = field(default_factory=list)
    """Titles (axes or figure) that run over more than one line."""
    method_legends: list[list[str]] = field(default_factory=list)
    """Legends that list two or more method names, as their entries."""
    data_axes: int = 0
    """Visible axes that draw data (colour bars and text-only axes not counted)."""
    lone_yticks: list[str] = field(default_factory=list)
    """Axes whose only y tick label is a word (a category naming the quantity)."""
    figure_texts: list[str] = field(default_factory=list)
    """Texts attached to the figure rather than an axes: notes, suptitles."""
    sentences: list[str] = field(default_factory=list)
    """Texts inside an axes that are not numbers (annotations, placeholders)."""
    forbidden: list[str] = field(default_factory=list)
    """Drawn texts that say Craftax, 1B, 1M, pending or indicative."""
    shape: str | None = None
    """The size class the figure declared (`charts.Shape`), None if it declared none."""
    size_mm: tuple[float, float] = (0.0, 0.0)
    """Declared width and height of the canvas in mm."""
    pixels: tuple[int, int] = (0, 0)
    """Width and height in pixels of the PNG as written (its header)."""
    expected_px: tuple[int, int] = (0, 0)
    """Pixels the declared size and dpi give."""


class Saves(list[Saved]):
    """The `Saved` files of a block, the layout warnings and captions it raised."""

    def __init__(self) -> None:
        super().__init__()
        self.warnings: list[str] = []
        self.captions: list[charts.Caption] = []


def _overlap(a: Any, b: Any) -> bool:
    """True if two bboxes share more than `TOL_PX` in both directions."""
    wide = min(a.x1, b.x1) - max(a.x0, b.x0)
    high = min(a.y1, b.y1) - max(a.y0, b.y0)
    return wide > TOL_PX and high > TOL_PX


def _cut(drawn: list[tuple[str, Any]], size: tuple[int, int]) -> list[str]:
    width, height = size
    out = []
    for text, box in drawn:
        x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
        if x0 < -TOL_PX or y0 < -TOL_PX or x1 > width + TOL_PX or y1 > height + TOL_PX:
            out.append(
                f"{text!r} at ({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}) in {size}"
            )
    return out


def _overlapping(drawn: list[tuple[str, Any, Any]]) -> list[str]:
    straight = [
        (t, b)
        for t, b, m in drawn
        if abs(m.get_rotation()) % 90 < 1e-6 or abs(m.get_rotation()) % 90 > 90 - 1e-6
    ]
    out = []
    for i, (ta, ba) in enumerate(straight):
        out += [
            f"{ta!r} over {tb!r}" for tb, bb in straight[i + 1 :] if _overlap(ba, bb)
        ]
    return out


def _covered(fig: Figure, drawn: list[tuple[str, Any, Any]]) -> list[str]:
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    legends = fig.findobj(Legend)
    out = []
    for legend in legends:
        if not legend.get_visible():
            continue
        box = legend.get_window_extent(renderer)
        label = "|".join(t.get_text() for t in legend.get_texts())[:60]
        out += [
            f"legend [{label}] over axes {ax.get_title(loc='left')!r}"
            for ax in fig.axes
            if ax.get_visible() and _overlap(box, ax.get_window_extent(renderer))
        ]
        own = {id(t) for t in (*legend.get_texts(), legend.get_title())}
        for text, other, mtext in drawn:
            if id(mtext) not in own and _overlap(box, other):
                out.append(f"legend [{label}] over the text {text!r}")
    return out


NUMBER = re.compile(r"^[\s\d.,:%+\-\u2212\u2013<>=\[\]()x/]*$")
"""A data value printed in a cell or beside a mark: digits and their punctuation."""
FORBIDDEN = re.compile(r"Craftax|\b1B\b|\b1M\b|pending|indicative", re.IGNORECASE)
"""What an image must not say (the caption does, in the notes)."""
NUMERIC_TICK = re.compile(r"^[\s\d.,%+\-\u2212<>kMB]*$")


def _axes_of(fig: Figure) -> list[Axes]:
    return [a for a in fig.findobj(Axes) if a.get_visible()]


def _is_colorbar(ax: Axes) -> bool:
    return hasattr(ax, "_colorbar")  # set by matplotlib on a colour bar's axes


def _wrapped_titles(fig: Figure) -> list[str]:
    out = []
    for ax in _axes_of(fig):
        out += [
            t for loc in ("left", "center", "right") if "\n" in (t := ax.get_title(loc))
        ]
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and "\n" in sup.get_text():
        out.append(sup.get_text())
    return out


def _method_legends(fig: Figure) -> list[list[str]]:
    names = set(charts.METHOD_COLORS_LIGHT)
    out = []
    for leg in fig.findobj(Legend):
        labels = [t.get_text() for t in leg.get_texts()]
        if sum(label.split(" (")[0] in names for label in labels) >= 2:
            out.append(labels)
    return out


def _data_axes(fig: Figure) -> list[Axes]:
    return [a for a in _axes_of(fig) if a.has_data() and not _is_colorbar(a)]


def _lone_yticks(axes: list[Axes]) -> list[str]:
    out = []
    for ax in axes:
        if not ax.yaxis.get_visible():
            continue
        labels = [
            t.get_text()
            for t in ax.get_yticklabels()
            if t.get_visible() and t.get_text().strip()
        ]
        if len(labels) == 1 and not NUMERIC_TICK.match(labels[0]):
            out.append(labels[0])
    return out


def _kinds(fig: Figure) -> dict[int, str]:
    """Id of each Text the figure owns -> its kind (tick, axis label, legend, title,
    annotation or figure text)."""
    kinds: dict[int, str] = {}
    for tick in fig.findobj(Tick):
        kinds[id(tick.label1)] = kinds[id(tick.label2)] = "tick"
    for axis in fig.findobj(Axis):
        kinds[id(axis.label)] = "axis label"
        kinds[id(axis.offsetText)] = "tick"
    for leg in fig.findobj(Legend):
        for t in [*leg.get_texts(), leg.get_title()]:
            kinds[id(t)] = "legend"
    for ax in fig.findobj(Axes):
        for name in ("title", "_left_title", "_right_title"):  # no public accessor
            kinds[id(getattr(ax, name))] = "title"
        for t in ax.texts:
            kinds[id(t)] = "annotation"
    for t in fig.findobj(Text):
        if t in fig.texts or t in [
            getattr(fig, n, None) for n in ("_suptitle", "_supxlabel", "_supylabel")
        ]:
            kinds[id(t)] = "figure"
    return kinds


def _text_problems(
    fig: Figure, drawn: list[tuple[str, Any, Any]]
) -> tuple[list[str], list[str], list[str]]:
    """Figure-level texts, sentences inside axes, forbidden words among ``drawn``."""
    kinds = _kinds(fig)
    figure_texts, sentences, forbidden = [], [], []
    for text, _box, mtext in drawn:
        kind = kinds.get(id(mtext), "other")
        if kind == "figure":
            figure_texts.append(text)
        elif kind in ("annotation", "other") and not NUMBER.match(text):
            sentences.append(text)
        if kind != "tick" and FORBIDDEN.search(text):  # '1M' is a step tick
            forbidden.append(text)
    return figure_texts, sentences, forbidden


@contextmanager
def audit_saves() -> Iterator[Saves]:
    """Collect a `Saved` per ``Figure.savefig`` call inside the block."""
    saved = Saves()
    drawn: list[tuple[str, Any, Any]] = []
    plain_save, plain_text = Figure.savefig, RendererAgg.draw_text

    def draw_text(
        self: RendererAgg, gc: Any, x: Any, y: Any, s: str, *a: Any, **k: Any
    ) -> Any:
        mtext = k.get("mtext", a[3] if len(a) > 3 else None)
        if mtext is not None and s.strip():
            drawn.append((
                s,
                Text.get_window_extent(mtext, self),
                mtext,
            ))  # not the arrow
        return plain_text(self, gc, x, y, s, *a, **k)

    def savefig(self: Figure, fname: Any, *a: Any, **k: Any) -> Any:
        drawn.clear()
        result = plain_save(self, fname, *a, **k)
        record = Saved(Path(str(fname)))
        renderer = self.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
        size = (int(renderer.width), int(renderer.height))
        record.cut = _cut([(t, b) for t, b, _ in drawn], size)
        record.covered = _covered(self, list(drawn))
        record.overlaps = _overlapping(list(drawn))
        record.wrapped_titles = _wrapped_titles(self)
        record.method_legends = _method_legends(self)
        data = _data_axes(self)
        record.data_axes = len(data)
        record.lone_yticks = _lone_yticks(data)
        record.figure_texts, record.sentences, record.forbidden = _text_problems(
            self, list(drawn)
        )
        w_in, h_in = self.get_size_inches()
        record.shape = getattr(self, "shape", None)
        record.size_mm = (float(w_in * 25.4), float(h_in * 25.4))
        record.expected_px = (round(w_in * self.dpi), round(h_in * self.dpi))
        head = Path(str(fname)).read_bytes()[16:24]  # PNG IHDR: width, height
        record.pixels = (
            int.from_bytes(head[:4], "big"),
            int.from_bytes(head[4:], "big"),
        )
        saved.append(record)
        return result

    with (
        pytest.MonkeyPatch.context() as mp,
        warnings.catch_warnings(record=True) as caught,
        charts.capture_captions() as captions,
    ):
        warnings.simplefilter("always")
        mp.setattr(Figure, "savefig", savefig)
        mp.setattr(RendererAgg, "draw_text", draw_text)
        yield saved
    saved.captions = captions
    saved.warnings = [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
        and ("layout" in str(w.message) or "Layout" in str(w.message))
    ]
