"""Review: keep/drop CSV (source of truth), previews, contact sheet, one static page.

``python -m ...review outputs/visualisation/review/candidates.csv`` writes ``strips/``,
``thumbs/``, ``previews/``, ``contact_sheet.png`` and ``review.html`` next to the CSV;
page's "Download review.csv" saves the reviewer's choices for the cut step.
"""

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.text import font

COLUMNS = (
    "id",
    "keep",
    "score",
    "type",
    "caption",
    "rollout",
    "game",
    "policy",
    "label",
    "start",
    "end",
    "peak",
    "seconds",
    "speed",
    "events",
)
"""Review CSV columns; add new ones at the end only (every step reads them by name)."""
TEMPLATE = Path(__file__).with_name("review_template.html")
"""The one static page; ``__DATA__`` is replaced by the candidates as JSON."""
DEFAULT_MASTERS = Path("outputs/visualisation/masters")
"""Masters folder named by rollout stem."""

type Row = dict[str, str]


def read_rows(path: Path) -> list[Row]:
    """Rows of a candidates or review CSV."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: Iterable[Mapping[str, str]], path: Path) -> Path:
    """Write rows with ``COLUMNS`` (extra keys are an error, missing ones empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def kept(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Rows the reviewer marked ``keep``, in file (score) order."""
    return [r for r in rows if r.get("keep") == "keep"]


def build_assets(
    rows: Sequence[Mapping[str, str]], masters_dir: Path, out_dir: Path
) -> None:
    """Per candidate: 4-frame strip, peak thumbnail, 2x preview MP4 (exact frames)."""
    for sub in ("strips", "thumbs", "previews"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    x2 = "scale=iw*2:ih*2:flags=neighbor"
    for r in rows:
        m, i = masters_dir / f"{r['rollout']}.mkv", r["id"]
        start, end, peak = int(r["start"]), int(r["end"]), int(r["peak"])
        picks = "+".join(f"eq(n,{n})" for n in (start, (start + peak) // 2, peak, end))
        ffmpeg.run([
            "-i",
            m,
            "-vf",
            f"select='{picks}',{x2},tile=4x1:padding=4:color=white",
            "-frames:v",
            "1",
            out_dir / "strips" / f"{i}.png",
        ])
        ffmpeg.run([
            "-i",
            m,
            "-vf",
            f"select='eq(n,{peak})',{x2}",
            "-frames:v",
            "1",
            out_dir / "thumbs" / f"{i}.png",
        ])
        trim = f"trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS"
        ffmpeg.run([
            "-i",
            m,
            "-vf",
            f"{trim},{x2},format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            out_dir / "previews" / f"{i}.mp4",
        ])


def contact_sheet(
    rows: Sequence[Mapping[str, str]], review_dir: Path, cols: int = 6
) -> Image.Image:
    """Peak thumbnails in a grid; id, score, type, caption and time range under each."""
    cw, ch = 360, 470
    sheet = Image.new(
        "RGB", (cw * cols, ch * ((len(rows) + cols - 1) // cols)), "white"
    )
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * cw, (i // cols) * ch
        im = Image.open(review_dir / "thumbs" / f"{r['id']}.png")
        im.thumbnail((cw - 20, ch - 70), Image.Resampling.NEAREST)
        sheet.paste(im, (x + (cw - im.width) // 2, y + 8))
        d.text(
            (x + 10, y + ch - 58),
            f"{r['id']}  score {r['score']}  {r['type']}",
            fill="black",
            font=font(15),
        )
        d.text(
            (x + 10, y + ch - 38),
            r["caption"][:44],
            fill="#333",
            font=font(13, bold=False),
        )
        span = f"{r['rollout']}  {int(r['start']) / 10:.1f}-{int(r['end']) / 10:.1f}s"
        d.text((x + 10, y + ch - 20), span, fill="#666", font=font(13, bold=False))
    return sheet


def write_page(rows: Sequence[Mapping[str, str]], out_dir: Path, source: str) -> Path:
    """Copy the static template with the candidates as JSON; the browser adds rows."""
    data = json.dumps({
        "source": source,
        "columns": COLUMNS,
        "rows": [dict(r) for r in rows],
    })
    page = out_dir / "review.html"
    page.write_text(
        TEMPLATE.read_text().replace("__DATA__", data.replace("</", "<\\/"))
    )
    return page


def main(argv: list[str] | None = None) -> None:
    """Build the review material for a candidates CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--masters-dir", type=Path, default=DEFAULT_MASTERS)
    a = p.parse_args(argv)
    rows, out = read_rows(a.csv), a.csv.parent
    build_assets(rows, a.masters_dir, out)
    contact_sheet(rows, out).save(out / "contact_sheet.png", optimize=True)
    page = write_page(rows, out, a.csv.name)
    print(f"{page} and {out / 'contact_sheet.png'} for {len(rows)} candidates")
    print(f"serve with: python -m http.server 8765 --bind 127.0.0.1 --directory {out}")


if __name__ == "__main__":
    main()
