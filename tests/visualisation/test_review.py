"""Tests for the review CSV, the static review page and the review assets."""

import json
from pathlib import Path

from _gameplay_run import classic_master, needs_ffmpeg
from PIL import Image

from wut_murl_craftax.visualisation import review

ROW = {
    "id": "c001",
    "keep": "",
    "score": "12.5",
    "type": "fight",
    "caption": 'Fight, 3 enemies, "close"',
    "rollout": "classic_heur_1",
    "game": "classic",
    "policy": "heuristic",
    "label": "scripted agent",
    "start": "5",
    "end": "40",
    "peak": "20",
    "seconds": "3.6",
    "speed": "1",
    "events": "fight:x; death:y",
}


def test_csv_round_trip_and_kept(tmp_path: Path) -> None:
    rows = [
        ROW,
        {**ROW, "id": "c002", "keep": "keep"},
        {**ROW, "id": "c003", "keep": "drop"},
    ]
    path = tmp_path / "review.csv"
    review.write_rows(rows, path)
    back = review.read_rows(path)
    assert back == rows
    assert list(back[0]) == list(review.COLUMNS)
    assert [r["id"] for r in review.kept(back)] == ["c002"]


def test_page_is_the_template_with_one_data_block(tmp_path: Path) -> None:
    page = review.write_page([ROW, {**ROW, "id": "c002"}], tmp_path, "candidates.csv")
    html = page.read_text()
    template = review.TEMPLATE.read_text()
    assert html.count("<tr") == template.count("<tr")  # rows are built in the browser
    data = html.split('<script id="data" type="application/json">')[1].split(
        "</script>"
    )[0]
    payload = json.loads(data)
    assert payload["columns"] == list(review.COLUMNS)
    assert [r["id"] for r in payload["rows"]] == ["c001", "c002"]


@needs_ffmpeg
def test_assets_and_contact_sheet(tmp_path: Path) -> None:
    rows = [ROW, {**ROW, "id": "c002", "start": "0", "end": "10", "peak": "10"}]
    review.build_assets(rows, classic_master().parent, tmp_path)
    strip = Image.open(tmp_path / "strips" / "c001.png")
    assert strip.size == (4 * 288 + 3 * 4, 288)
    assert Image.open(tmp_path / "thumbs" / "c002.png").size == (288, 288)
    assert (tmp_path / "previews" / "c001.mp4").stat().st_size > 0
    sheet = review.contact_sheet(rows, tmp_path)
    assert sheet.size == (6 * 360, 470)
