"""Tests for the achievement timeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from _figure_audit import audit_saves
from _learning_run import needs_ffmpeg, tiny_run
from PIL import Image

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.charts import VIDEO
from wut_murl_craftax.visualisation.timeline import first_unlock_order, render_timeline


def test_rows_sorted_by_first_unlock_never_last() -> None:
    first = np.array([5, -1, 1, 5, -1, 3])
    assert first_unlock_order(first).tolist() == [2, 5, 0, 3, 1, 4]


@needs_ffmpeg
def test_timeline_frames_and_thumbnail(tmp_path: Path) -> None:
    out = render_timeline(
        tiny_run(), tmp_path / "t.mkv", seconds=1, fps=6, hold=0.5, size=(480, 300)
    )
    assert ffmpeg.count_frames(out) == 9
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (480, 300)
    assert out.with_suffix(".png").exists()


@needs_ffmpeg
def test_timeline_png_is_not_cropped_and_has_the_shared_look(tmp_path: Path) -> None:
    with audit_saves() as saved:
        render_timeline(
            tiny_run(), tmp_path / "t.mkv", seconds=1, fps=2, hold=0, size=(1920, 1080)
        )
    (png,) = saved
    assert png.path.suffix == ".png"
    assert not png.cut, png.cut
    assert not png.covered, png.covered
    assert not png.figure_texts, png.figure_texts
    assert not png.sentences, png.sentences  # no title or counter on the image
    assert not png.forbidden, png.forbidden
    assert png.data_axes == 1
    pixel = np.asarray(Image.open(png.path).convert("RGB"))[2, 2]
    assert tuple(pixel) == tuple(int(VIDEO.surface[i : i + 2], 16) for i in (1, 3, 5))


@needs_ffmpeg
def test_the_timeline_writes_its_caption_beside_the_files(tmp_path: Path) -> None:
    render_timeline(
        tiny_run(), tmp_path / "timeline.mkv", seconds=1, fps=2, hold=0, size=(480, 300)
    )
    rows = pd.read_csv(tmp_path / "captions.csv")
    assert rows["stem"].tolist() == ["timeline"]
    assert rows["id"].tolist() == ["TL"]
    assert "first unlock" in rows["description"][0]
    assert "\n" not in rows["description"][0]
