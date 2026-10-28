"""Audit of the figures a test saves: text cut off the canvas, legends over data.

`audit_saves` wraps ``Figure.savefig`` and ``RendererAgg.draw_text`` for a ``with``
block. Every saved figure is checked in the state it was written in (the layout engine
has run): each text the renderer drew (titles, ticks, legends, notes, annotations) must
lie inside the canvas, and no legend may overlap the window of any axes, the area its
data is drawn in. Problems are collected per file, never raised, so a test decides.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend

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


@contextmanager
def audit_saves() -> Iterator[list[Saved]]:
    """Collect a `Saved` per ``Figure.savefig`` call inside the block."""
    saved: list[Saved] = []
    drawn: list[tuple[str, Any, Any]] = []
    plain_save, plain_text = Figure.savefig, RendererAgg.draw_text

    def draw_text(
        self: RendererAgg, gc: Any, x: Any, y: Any, s: str, *a: Any, **k: Any
    ) -> Any:
        mtext = k.get("mtext", a[3] if len(a) > 3 else None)
        if mtext is not None and s.strip():
            drawn.append((s, mtext.get_window_extent(self), mtext))
        return plain_text(self, gc, x, y, s, *a, **k)

    def savefig(self: Figure, fname: Any, *a: Any, **k: Any) -> Any:
        drawn.clear()
        result = plain_save(self, fname, *a, **k)
        record = Saved(Path(str(fname)))
        renderer = self.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
        size = (int(renderer.width), int(renderer.height))
        record.cut = _cut([(t, b) for t, b, _ in drawn], size)
        record.covered = _covered(self, list(drawn))
        saved.append(record)
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "savefig", savefig)
        mp.setattr(RendererAgg, "draw_text", draw_text)
        yield saved
