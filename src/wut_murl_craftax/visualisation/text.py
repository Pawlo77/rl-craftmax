"""Pillow text overlays (captions, banners, cards, labels): ffmpeg lacks drawtext."""

import logging
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

FONT_DIRS = (
    Path("/System/Library/Fonts/Supplemental"),
    Path("/usr/share/fonts/truetype/msttcorefonts"),
)
"""Where Arial is looked up (macOS, Debian/Ubuntu msttcorefonts)."""
BACKGROUND = (18, 20, 24)
"""Dark grey of cards, padding and labels (``0x121418``)."""

type Size = tuple[int, int]
"""Width, height in pixels."""
type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


@cache
def font(px: int, bold: bool = True) -> Font:
    """Arial (Bold) at ``px``; Pillow's built-in font when Arial is not installed."""
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    for d in FONT_DIRS:
        for candidate in (d / name, d / name.replace(" ", "_")):
            if candidate.exists():
                return ImageFont.truetype(str(candidate), px)
    logger.warning("Arial not found in %s; using Pillow's default font", FONT_DIRS)
    return ImageFont.load_default(px)


def pretty(name: str) -> str:
    """``MAKE_STONE_PICKAXE`` -> ``Make Stone Pickaxe``; other text unchanged."""
    return name.replace("_", " ").title() if name.isupper() else name


def wrap(text: str, f: Font, width: int) -> list[str]:
    """Greedy word wrap of ``text`` to lines at most ``width`` pixels wide."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and f.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return [*lines, cur]


def caption_panel(
    size: Size,
    box: tuple[int, int, int],
    title: str,
    subtitle: str,
    title_px: int = 40,
    sub_px: int = 24,
) -> Image.Image:
    """Transparent frame, dark box at ``box = (x, y, width)``: title over subtitle."""
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y, w = box
    tf, sf = font(title_px), font(sub_px, bold=False)
    head, sub = wrap(pretty(title), tf, w), wrap(subtitle, sf, w)
    th, sh = int(title_px * 1.3), int(sub_px * 1.25)
    d.rounded_rectangle(
        (x - 16, y - 20, x + w + 16, y + th * len(head) + sh * len(sub) + 34),
        18,
        fill=(0, 0, 0, 170),
    )
    for i, line in enumerate(head):
        d.text((x, y + th * i), line, font=tf, fill="white")
    for i, line in enumerate(sub):
        d.text(
            (x, y + th * len(head) + 14 + sh * i), line, font=sf, fill=(210, 210, 210)
        )
    return im


def banner(
    size: Size, center: tuple[int, int], head: str, body: str, px: int = 84
) -> Image.Image:
    """Transparent frame: yellow headline over a white line, boxed around ``center``."""
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fb, fs = font(px), font(px * 5 // 8)
    cx, cy = center
    half = (max(fb.getlength(head), fs.getlength(body)) + 80) / 2
    d.rounded_rectangle(
        (cx - half, cy - px, cx + half, cy + px), 24, fill=(0, 0, 0, 190)
    )
    d.text((cx, cy - px * 3 // 8), head, font=fb, fill=(255, 214, 0), anchor="mm")
    d.text((cx, cy + px // 2), pretty(body), font=fs, fill="white", anchor="mm")
    return im


def title_card(size: Size, title: str, subtitle: str = "") -> Image.Image:
    """Opaque card: wrapped title in the middle, grey subtitle below."""
    w, h = size
    im = Image.new("RGB", size, BACKGROUND)
    d = ImageDraw.Draw(im)
    f = font(max(24, min(w, h) // 11))
    step = int(f.size * 1.15) if isinstance(f, ImageFont.FreeTypeFont) else 30
    lines = wrap(title, f, w - w // 10)
    y0 = h // 2 - step * len(lines) // 2
    for i, line in enumerate(lines):
        d.text((w // 2, y0 + step * i), line, font=f, fill="white", anchor="mm")
    if subtitle:
        d.text(
            (w // 2, y0 + step * len(lines) + step // 3),
            subtitle,
            font=font(max(14, step * 2 // 5), bold=False),
            fill=(180, 180, 180),
            anchor="mm",
        )
    return im


def label_bar(text: str, width: int, height: int = 40) -> Image.Image:
    """Opaque one-line label for grid cells."""
    im = Image.new("RGB", (width, height), BACKGROUND)
    ImageDraw.Draw(im).text(
        (10, height // 2),
        text,
        font=font(int(height * 0.55)),
        fill="white",
        anchor="lm",
    )
    return im


def text_strip(text: str, width: int, height: int = 34) -> Image.Image:
    """Black caption strip stacked under GIFs."""
    im = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    f = font(int(height * 0.6))
    line = pretty(text)
    while line and f.getlength(line) > width - 16:
        line = line[:-1]
    ImageDraw.Draw(im).text((8, height // 2), line, font=f, fill="white", anchor="lm")
    return im
