"""Read book pages from a PDF by PRINTED page number (self-contained helper).

usage:
  uv run --no-project --with pypdfium2 --with pillow python -I pages.py text   <pdf> <first> <last> [--offset 13]
  uv run --no-project --with pypdfium2 --with pillow python -I pages.py render <pdf> <first> <last> --out <dir> [--dpi 130] [--offset 13]

The PDF index = printed page + offset (13 for the Sutton & Barto draft; the front matter
uses roman numerals). `text` prints the text layer (equations come out scrambled: take
equations and numbers from `render`, then read the PNGs). `render` writes page-<printed>.png.
"""

import argparse
import re
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "render"])
    ap.add_argument("pdf")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--offset", type=int, default=13)
    ap.add_argument("--out", default="pages")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()
    doc = pdfium.PdfDocument(a.pdf)
    out = Path(a.out)
    if a.mode == "render":
        out.mkdir(parents=True, exist_ok=True)
    for printed in range(a.first, a.last + 1):
        i = printed + a.offset
        if not 0 <= i < len(doc):
            continue
        if a.mode == "text":
            t = doc[i].get_textpage().get_text_range()
            print(f"=== printed {printed} (index {i})\n" + re.sub(r"\s+", " ", t))
        else:
            img = doc[i].render(scale=a.dpi / 72).to_pil()
            path = out / f"page-{printed:03d}.png"
            img.save(path)
            print(path)


main()
