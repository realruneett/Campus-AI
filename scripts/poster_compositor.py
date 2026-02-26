#!/usr/bin/env python3
"""
poster_compositor.py
====================
Smart Poster Compositor — Text Placement Engine

Renders PIL typography on SDXL-generated artwork with three placement modes:

    "auto"    Scans the image for the quietest region (fewest edges) and
              places text there automatically.
    manual    Pass text_position="top" | "center" | "bottom" to pin the text
              block to a fixed zone — useful when you have already reviewed
              the artwork and know where the clean space is.
    "none"    Returns the artwork untouched (useful for debugging raw art).

A feathered dark scrim is applied only under the text block when scrim=True.
Set scrim=False for bright or vivid artworks where a dark overlay would ruin
the visual — text rendering already includes drop shadows and strokes for
standalone legibility.

Styles:
    modern   Centered Montserrat, accent rules, info pill.
    bold     Left-aligned heavy display, side accent bars, right-aligned organiser.
    elegant  Centered Playfair Display, fine horizontal rules.
"""

from __future__ import annotations

import os
import textwrap
from typing import Literal

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FONTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")
)

_FONT_URLS: dict[str, str] = {
    "Montserrat-Regular":
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf",
    "Montserrat-Medium":
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Medium.ttf",
    "Montserrat-Bold":
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
    "Montserrat-ExtraBold":
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf",
    "PlayfairDisplay-Bold":
        "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "PlayfairDisplay-Regular":
        "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay-Italic%5Bwght%5D.ttf",
}

Style    = Literal["modern", "bold", "elegant"]
Position = Literal["auto", "top", "center", "bottom", "none"]

# Vertical centre of the text block as a fraction of image height
_POSITION_RATIOS: dict[str, float] = {
    "top":    0.14,   # tight to the very top — above most subjects
    "center": 0.50,
    "bottom": 0.80,
}

# Scrim intensity per style — bold needs more coverage to hide busy artwork
_SCRIM_INTENSITY: dict[str, float] = {
    "bold":    0.90,
    "modern":  0.78,
    "elegant": 0.75,
}


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def ensure_fonts() -> str:
    """Download fonts to the assets directory if they are not already cached."""
    os.makedirs(_FONTS_DIR, exist_ok=True)
    for name, url in _FONT_URLS.items():
        dest = os.path.join(_FONTS_DIR, f"{name}.ttf")
        if os.path.exists(dest):
            continue
        print(f"  Downloading font: {name} ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(r.content)
        except Exception as exc:
            print(f"  Warning — could not download {name}: {exc}")
    return _FONTS_DIR


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a font by logical name and point size, with graceful fallback."""
    path = os.path.join(_FONTS_DIR, f"{name}.ttf")
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    for fallback in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text measurement
# ---------------------------------------------------------------------------

def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


def _wrap_title(title: str, style: Style) -> tuple[list[str], int]:
    """Return (wrapped lines, font size) for the title based on length and style."""
    length = len(title)
    if style == "bold":
        size  = 70 if length < 15 else 56 if length < 25 else 44
        width = 14 if size > 56 else 18
    elif style == "elegant":
        size  = 56 if length < 20 else 44 if length < 30 else 36
        width = 18 if size > 44 else 22
    else:  # modern
        size  = 64 if length < 20 else 50 if length < 30 else 40
        width = 20 if size > 50 else 24

    display = title if style == "elegant" else title.upper()
    return textwrap.wrap(display, width=width), size


# ---------------------------------------------------------------------------
# Quiet-zone detection  (used only when text_position="auto")
# ---------------------------------------------------------------------------

def _score_bands(image: Image.Image, n: int = 5) -> list[tuple[int, int, int, float]]:
    """Score horizontal bands by edge density. Returns list sorted quietest-first.

    The top 15 % of the image is always excluded — that space is reserved
    for organiser branding and top chrome elements.
    """
    w, h       = image.size
    edges      = np.array(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    band_h     = h // n
    top_margin = int(h * 0.15)
    bands: list[tuple[int, int, int, float]] = []

    for i in range(n):
        y0 = i * band_h
        y1 = min((i + 1) * band_h, h)
        if y1 <= top_margin:
            score = 9999.0
        elif y0 < top_margin:
            score = float(np.mean(edges[top_margin:y1, :]))
        else:
            score = float(np.mean(edges[y0:y1, :]))
        bands.append((i, y0, y1, score))

    bands.sort(key=lambda b: b[3])
    return bands


def _find_text_region(
    image: Image.Image,
    block_height: int,
    n_bands: int = 5,
) -> tuple[int, int, int, str]:
    """Return (y_center, y_top, y_bottom, hint) for the quietest usable region."""
    w, h  = image.size
    bands = _score_bands(image, n_bands)
    _, y0, y1, _ = bands[0]

    if (y1 - y0) < block_height:
        expand = (block_height - (y1 - y0)) // 2
        y0 = max(0, y0 - expand)
        y1 = min(h, y1 + expand)

    y_center = (y0 + y1) // 2
    rel      = y_center / h
    hint     = "top" if rel < 0.33 else "bottom" if rel > 0.66 else "center"
    return y_center, y0, y1, hint


# ---------------------------------------------------------------------------
# Localized dark scrim  (feathered, only under the text block)
# ---------------------------------------------------------------------------

def _apply_scrim(
    image:     Image.Image,
    y_top:     int,
    y_bottom:  int,
    intensity: float = 0.78,
) -> Image.Image:
    """Burn a soft dark gradient over *image* between y_top and y_bottom ONLY.

    60-pixel feathered edges ensure the scrim blends invisibly into the
    surrounding artwork. Nothing outside the text region is darkened.
    """
    w, h        = image.size
    scrim       = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw        = ImageDraw.Draw(scrim)
    feather     = 60
    fade_top    = max(0, y_top    - feather)
    fade_bottom = min(h, y_bottom + feather)

    for y in range(fade_top, fade_bottom):
        if y < y_top:
            t = (y - fade_top) / max(1, y_top - fade_top)
        elif y > y_bottom:
            t = 1.0 - (y - y_bottom) / max(1, fade_bottom - y_bottom)
        else:
            t = 1.0
        alpha = min(int(200 * t * intensity), 215)
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

    base = image.convert("RGBA")
    return Image.alpha_composite(base, scrim).convert("RGB")


# ---------------------------------------------------------------------------
# Text-rendering primitives
# ---------------------------------------------------------------------------

def _shadowed(
    draw:          ImageDraw.ImageDraw,
    xy:            tuple[int, int],
    text:          str,
    font:          ImageFont.FreeTypeFont,
    fill:          str = "#FFFFFF",
    shadow_color:  str = "#000000",
    shadow_offset: int = 4,
    anchor:        str = "lt",
) -> None:
    """Render text with a layered drop shadow and thin stroke for legibility."""
    x, y = xy
    draw.text((x + shadow_offset,     y + shadow_offset),     text, font=font, fill=(0, 0, 0, 220), anchor=anchor)
    draw.text((x + shadow_offset * 2, y + shadow_offset * 2), text, font=font, fill=(0, 0, 0, 100), anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, stroke_width=2, stroke_fill=shadow_color, anchor=anchor)


def _pill(
    draw:    ImageDraw.ImageDraw,
    xy:      tuple[int, int],
    text:    str,
    font:    ImageFont.FreeTypeFont,
    fill:    str             = "#FFFFFF",
    bg:      tuple[int, ...] = (0, 0, 0, 160),
    padding: int             = 12,
    anchor:  str             = "lt",
) -> None:
    """Render text on a semi-transparent rounded-rectangle background."""
    bb = font.getbbox(text, anchor=anchor)
    x, y = xy
    draw.rounded_rectangle(
        [
            (x + bb[0] - padding, y + bb[1] - padding),
            (x + bb[2] + padding, y + bb[3] + padding),
        ],
        radius=8,
        fill=bg,
    )
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Block height estimator
# ---------------------------------------------------------------------------

def _estimate_block_height(
    title: str, subtitle: str, date: str, venue: str, organizer: str, style: Style,
) -> int:
    lines, size = _wrap_title(title, style)
    font_title  = load_font(
        "PlayfairDisplay-Bold" if style == "elegant" else "Montserrat-ExtraBold", size
    )
    total = sum(_text_size(ln, font_title)[1] + 12 for ln in lines) + 24
    if subtitle:  total += 34
    if date:      total += 28
    if venue:     total += 28
    if organizer: total += 36
    return total + 40


# ---------------------------------------------------------------------------
# Layout: MODERN
# ---------------------------------------------------------------------------

def _layout_modern(
    draw: ImageDraw.ImageDraw, w: int, h: int,
    title: str, subtitle: str, date: str, venue: str,
    organizer: str, accent: str, start_y: int,
) -> None:
    """Centered layout with accent bars top and bottom."""
    cx  = w // 2
    gap = 16

    draw.rectangle([(0, 0), (w, 5)], fill=accent)
    if organizer:
        font_org = load_font("Montserrat-Medium", 22)
        _shadowed(draw, (cx, 28), organizer.upper(), font_org, anchor="mt")

    cursor = start_y
    lines, size = _wrap_title(title, "modern")
    font_title  = load_font("Montserrat-ExtraBold", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, font_title, anchor="mt")
        cursor += _text_size(line, font_title)[1] + 12

    cursor += 6
    draw.rectangle([(cx - 90, cursor), (cx + 90, cursor + 3)], fill=accent)
    cursor += 3 + gap

    if subtitle:
        font_sub = load_font("PlayfairDisplay-Regular", 26)
        _shadowed(draw, (cx, cursor), subtitle, font_sub, fill=accent, anchor="mt")
        cursor += _text_size(subtitle, font_sub)[1] + gap

    parts: list[str] = []
    if date:  parts.append(f"📅  {date}")
    if venue: parts.append(f"📍  {venue}")
    if parts:
        font_info = load_font("Montserrat-Regular", 18)
        _pill(draw, (cx, cursor), "   •   ".join(parts), font_info,
              bg=(0, 0, 0, 170), anchor="mt")

    draw.rectangle([(0, h - 5), (w, h)], fill=accent)


# ---------------------------------------------------------------------------
# Layout: BOLD
# ---------------------------------------------------------------------------

def _layout_bold(
    draw: ImageDraw.ImageDraw, w: int, h: int,
    title: str, subtitle: str, date: str, venue: str,
    organizer: str, accent: str, start_y: int,
) -> None:
    """Left-aligned heavy display. Organiser pill pinned top-right."""
    LEFT = 50
    gap  = 18

    draw.rectangle([(0, 0),     (6, h)], fill=accent)
    draw.rectangle([(w - 6, 0), (w, h)], fill=accent)

    # Organiser — top-right so it never clashes with left-aligned title
    if organizer:
        font_org = load_font("Montserrat-Bold", 18)
        _pill(draw, (w - LEFT, 28), organizer.upper(), font_org,
              fill=accent, bg=(0, 0, 0, 200), padding=10, anchor="rt")

    cursor = start_y
    lines, size = _wrap_title(title, "bold")
    font_title  = load_font("Montserrat-ExtraBold", size)
    for line in lines:
        _shadowed(draw, (LEFT, cursor), line, font_title, shadow_offset=5)
        cursor += _text_size(line, font_title)[1] + 8
    cursor += gap

    if subtitle:
        font_sub = load_font("Montserrat-Bold", 24)
        _shadowed(draw, (LEFT, cursor), subtitle.upper(), font_sub, fill=accent)
        cursor += _text_size(subtitle.upper(), font_sub)[1] + gap

    font_info = load_font("Montserrat-Regular", 20)
    if date:
        _shadowed(draw, (LEFT, cursor), f"📅  {date}", font_info, fill="#DDDDDD")
        cursor += _text_size(f"📅  {date}", font_info)[1] + 10
    if venue:
        _shadowed(draw, (LEFT, cursor), f"📍  {venue}", font_info, fill="#DDDDDD")


# ---------------------------------------------------------------------------
# Layout: ELEGANT
# ---------------------------------------------------------------------------

def _layout_elegant(
    draw: ImageDraw.ImageDraw, w: int, h: int,
    title: str, subtitle: str, date: str, venue: str,
    organizer: str, accent: str, start_y: int,
) -> None:
    """Centered serif layout with fine horizontal rules."""
    cx     = w // 2
    rule_w = 160
    gap    = 18

    draw.rectangle([(cx - rule_w, 46), (cx + rule_w, 48)], fill=accent)
    if organizer:
        font_org = load_font("Montserrat-Medium", 20)
        _shadowed(draw, (cx, 62), organizer, font_org, anchor="mt")
    draw.rectangle([(cx - rule_w, 94), (cx + rule_w, 96)], fill=accent)

    cursor = start_y
    lines, size = _wrap_title(title, "elegant")
    font_title  = load_font("PlayfairDisplay-Bold", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, font_title,
                  shadow_color="#1A1A1A", shadow_offset=3, anchor="mt")
        cursor += _text_size(line, font_title)[1] + 14

    cursor += 8
    draw.rectangle([(cx - 60, cursor), (cx + 60, cursor + 1)], fill=accent)
    cursor += 1 + gap

    if subtitle:
        font_sub = load_font("PlayfairDisplay-Regular", 26)
        _shadowed(draw, (cx, cursor), subtitle, font_sub, fill=accent, anchor="mt")
        cursor += _text_size(subtitle, font_sub)[1] + gap

    font_info = load_font("Montserrat-Regular", 17)
    if date:
        _shadowed(draw, (cx, cursor), date.upper(), font_info,
                  fill="#E8E8E8", anchor="mt")
        cursor += _text_size(date.upper(), font_info)[1] + 8
    if venue:
        _pill(draw, (cx, cursor), venue, font_info,
              fill="#FFFFFF", bg=(0, 0, 0, 150), padding=10, anchor="mt")

    draw.rectangle([(cx - rule_w, h - 48), (cx + rule_w, h - 46)], fill=accent)
    draw.rectangle([(cx - rule_w, h - 36), (cx + rule_w, h - 34)], fill=accent)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LAYOUTS = {
    "modern":  _layout_modern,
    "bold":    _layout_bold,
    "elegant": _layout_elegant,
}


def composite_poster(
    artwork:       Image.Image,
    title:         str,
    subtitle:      str      = "",
    date:          str      = "",
    venue:         str      = "",
    organizer:     str      = "",
    accent_color:  str      = "#FFD700",
    style:         Style    = "modern",
    text_position: Position = "auto",
    scrim:         bool     = True,
) -> Image.Image:
    """Composite event text onto an SDXL artwork image.

    Args:
        artwork:       Raw SDXL-generated PIL Image.
        title:         Primary event name (required).
        subtitle:      Short tagline or theme (optional).
        date:          Human-readable date string (optional).
        venue:         Location or venue name (optional).
        organizer:     Host shown at the top of the poster (optional).
        accent_color:  Hex colour for decorative elements and rules.
        style:         "modern" | "bold" | "elegant"
        text_position: "auto"   — detect quietest region automatically.
                       "top"    — pin text block near the top (y=14%).
                       "center" — pin to vertical centre (y=50%).
                       "bottom" — pin to bottom area (y=80%).
                       "none"   — return artwork unchanged.
        scrim:         True  — feathered dark gradient under text block only.
                               Intensity is style-aware: bold=0.90, others lower.
                       False — no scrim; rely on shadow/stroke for legibility.
                               Use for bright, vivid, or light-bg artworks.

    Returns:
        Composited PIL Image (RGB).
    """
    if text_position == "none":
        return artwork.copy().convert("RGB")

    ensure_fonts()

    img = artwork.copy().convert("RGB")
    w, h = img.size

    block_h = _estimate_block_height(title, subtitle, date, venue, organizer, style)

    if text_position in _POSITION_RATIOS:
        y_center = int(h * _POSITION_RATIOS[text_position])
    else:
        y_center, _, _, _ = _find_text_region(img, block_h)

    pad = 44
    if scrim:
        scrim_top       = max(0, y_center - block_h // 2 - pad)
        scrim_bottom    = min(h, y_center + block_h // 2 + pad)
        scrim_intensity = _SCRIM_INTENSITY.get(style, 0.78)
        img = _apply_scrim(img, scrim_top, scrim_bottom, intensity=scrim_intensity)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    text_start_y = max(pad, y_center - block_h // 2)
    _LAYOUTS.get(style, _layout_modern)(
        draw, w, h, title, subtitle, date, venue, organizer, accent_color, text_start_y
    )

    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result.convert("RGB")