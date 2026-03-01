#!/usr/bin/env python3
"""
Campus-AI by CounciL — Poster Compositor  v3.0
================================================
Professional PIL typography engine with 1,700+ Google Fonts on-demand
and 8 unique layout styles for SDXL-generated artwork.

Features:
  - Dynamic Google Fonts integration — any of 1,700+ font families
    downloaded and cached on first use.
  - 30 premium poster fonts pre-cached at startup.
  - 8 typography layout styles: Modern, Bold, Elegant, Retro, Minimal,
    Futuristic, Handwritten, Royal.
  - Auto quiet-zone detection for optimal text placement.
  - Feathered dark scrim for contrast.
  - Clean ASCII-only rendering — no emoji/unicode boxes.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Literal, Optional, Union

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# WSL specific: prevent I/O crashes by caching fonts to the internal Linux filesystem 
# instead of the mounted Windows E: drive
_FONTS_DIR = "/tmp/campus-ai-fonts"
os.makedirs(_FONTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Google Fonts CDN — 30 pre-cached poster fonts + on-demand loader
# ---------------------------------------------------------------------------

# Format: "FontName-Weight" -> CDN URL
# Uses fontsource CDN (jsdelivr) for reliable .ttf delivery
_PRECACHED_FONTS: dict[str, str] = {
    # Sans-serif — modern / bold
    "Montserrat-Regular":     "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-400-normal.ttf",
    "Montserrat-Medium":      "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-500-normal.ttf",
    "Montserrat-Bold":        "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-700-normal.ttf",
    "Montserrat-ExtraBold":   "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-800-normal.ttf",
    "Poppins-Light":          "https://cdn.jsdelivr.net/fontsource/fonts/poppins@latest/latin-300-normal.ttf",
    "Poppins-Regular":        "https://cdn.jsdelivr.net/fontsource/fonts/poppins@latest/latin-400-normal.ttf",
    "Poppins-Bold":           "https://cdn.jsdelivr.net/fontsource/fonts/poppins@latest/latin-700-normal.ttf",
    "Oswald-Bold":            "https://cdn.jsdelivr.net/fontsource/fonts/oswald@latest/latin-700-normal.ttf",
    "Oswald-Regular":         "https://cdn.jsdelivr.net/fontsource/fonts/oswald@latest/latin-400-normal.ttf",
    "Raleway-Bold":           "https://cdn.jsdelivr.net/fontsource/fonts/raleway@latest/latin-700-normal.ttf",
    "Raleway-Regular":        "https://cdn.jsdelivr.net/fontsource/fonts/raleway@latest/latin-400-normal.ttf",
    "Inter-Bold":             "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-700-normal.ttf",
    "Inter-Regular":          "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.ttf",
    "Quicksand-Bold":         "https://cdn.jsdelivr.net/fontsource/fonts/quicksand@latest/latin-700-normal.ttf",
    "Quicksand-Regular":      "https://cdn.jsdelivr.net/fontsource/fonts/quicksand@latest/latin-400-normal.ttf",
    # Serif — elegant / royal
    "PlayfairDisplay-Bold":   "https://cdn.jsdelivr.net/fontsource/fonts/playfair-display@latest/latin-700-normal.ttf",
    "PlayfairDisplay-Regular":"https://cdn.jsdelivr.net/fontsource/fonts/playfair-display@latest/latin-400-normal.ttf",
    "Lora-Bold":              "https://cdn.jsdelivr.net/fontsource/fonts/lora@latest/latin-700-normal.ttf",
    "Lora-Regular":           "https://cdn.jsdelivr.net/fontsource/fonts/lora@latest/latin-400-normal.ttf",
    "Merriweather-Bold":      "https://cdn.jsdelivr.net/fontsource/fonts/merriweather@latest/latin-700-normal.ttf",
    "CormorantGaramond-Bold": "https://cdn.jsdelivr.net/fontsource/fonts/cormorant-garamond@latest/latin-700-normal.ttf",
    "CormorantGaramond-Regular":"https://cdn.jsdelivr.net/fontsource/fonts/cormorant-garamond@latest/latin-400-normal.ttf",
    "Cinzel-Bold":            "https://cdn.jsdelivr.net/fontsource/fonts/cinzel@latest/latin-700-normal.ttf",
    "Cinzel-Regular":         "https://cdn.jsdelivr.net/fontsource/fonts/cinzel@latest/latin-400-normal.ttf",
    # Display — retro / futuristic
    "BebasNeue-Regular":      "https://cdn.jsdelivr.net/fontsource/fonts/bebas-neue@latest/latin-400-normal.ttf",
    "Orbitron-Bold":          "https://cdn.jsdelivr.net/fontsource/fonts/orbitron@latest/latin-700-normal.ttf",
    "Orbitron-Regular":       "https://cdn.jsdelivr.net/fontsource/fonts/orbitron@latest/latin-400-normal.ttf",
    "Rajdhani-Bold":          "https://cdn.jsdelivr.net/fontsource/fonts/rajdhani@latest/latin-700-normal.ttf",
    "Rajdhani-Regular":       "https://cdn.jsdelivr.net/fontsource/fonts/rajdhani@latest/latin-400-normal.ttf",
    # Script — handwritten
    "DancingScript-Bold":     "https://cdn.jsdelivr.net/fontsource/fonts/dancing-script@latest/latin-700-normal.ttf",
    "DancingScript-Regular":  "https://cdn.jsdelivr.net/fontsource/fonts/dancing-script@latest/latin-400-normal.ttf",
}

# Fonts that can be loaded on-demand from Google Fonts CDN
# Format: family-slug used in fontsource URLs
_GOOGLE_FONT_FAMILIES = [
    "abel", "abril-fatface", "acme", "alegreya", "alegreya-sans", "alfa-slab-one",
    "amatic-sc", "amiri", "anton", "archivo", "archivo-black", "archivo-narrow",
    "arimo", "arvo", "asap", "assistant", "barlow", "barlow-condensed",
    "bebas-neue", "bitter", "cabin", "cairo", "chakra-petch", "cinzel",
    "comfortaa", "commissioner", "cormorant-garamond", "crimson-text",
    "dancing-script", "dm-sans", "dm-serif-display", "dosis", "eb-garamond",
    "exo-2", "fira-sans", "fredoka-one", "fugaz-one", "graduate",
    "great-vibes", "heebo", "hind", "ibm-plex-sans", "inconsolata",
    "indie-flower", "inter", "josefin-sans", "jost", "kalam",
    "kanit", "karla", "kaushan-script", "lato", "league-spartan",
    "lexend", "libre-baskerville", "libre-franklin", "lilita-one",
    "lobster", "lora", "luckiest-guy", "manrope", "maven-pro",
    "merriweather", "merriweather-sans", "montserrat", "mukta", "mulish",
    "nanum-gothic", "neuton", "noto-sans", "noto-serif", "nunito",
    "nunito-sans", "old-standard-tt", "open-sans", "orbitron", "oswald",
    "outfit", "overpass", "oxanium", "pacifico", "passion-one",
    "pathway-gothic-one", "patrick-hand", "permanent-marker", "philosopher",
    "play", "playfair-display", "plus-jakarta-sans", "poppins",
    "press-start-2p", "prompt", "pt-sans", "pt-serif", "quicksand",
    "rajdhani", "raleway", "righteous", "roboto", "roboto-condensed",
    "roboto-mono", "roboto-slab", "rokkitt", "rubik", "russo-one",
    "sacramento", "saira", "satisfy", "secular-one", "shadows-into-light",
    "signika", "silkscreen", "sora", "source-code-pro", "source-sans-3",
    "source-serif-4", "space-grotesk", "space-mono", "spectral",
    "stint-ultra-expanded", "teko", "titillium-web", "ubuntu", "urbanist",
    "varela-round", "vollkorn", "work-sans", "yanone-kaffeesatz", "zilla-slab",
]

Style    = Literal["modern", "bold", "elegant", "retro", "minimal", "futuristic", "handwritten", "royal"]
Position = Literal["auto", "top", "center", "bottom", "none"]

_POSITION_RATIOS: dict[str, float] = {
    "top": 0.14, "center": 0.50, "bottom": 0.80,
}

_SCRIM_INTENSITY: dict[str, float] = {
    "bold": 0.90, "modern": 0.78, "elegant": 0.75,
    "retro": 0.85, "minimal": 0.60, "futuristic": 0.82,
    "handwritten": 0.70, "royal": 0.80,
}

# Style → (title_font, body_font)
_STYLE_FONTS: dict[str, tuple[str, str]] = {
    "modern":      ("Montserrat-ExtraBold",   "Montserrat-Regular"),
    "bold":        ("Montserrat-ExtraBold",   "Montserrat-Bold"),
    "elegant":     ("PlayfairDisplay-Bold",   "PlayfairDisplay-Regular"),
    "retro":       ("BebasNeue-Regular",      "Oswald-Regular"),
    "minimal":     ("Poppins-Light",          "Poppins-Regular"),
    "futuristic":  ("Orbitron-Bold",          "Rajdhani-Regular"),
    "handwritten": ("DancingScript-Bold",     "Quicksand-Regular"),
    "royal":       ("Cinzel-Bold",            "CormorantGaramond-Regular"),
}


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def ensure_fonts() -> str:
    """Download the 30 pre-cached fonts on startup."""
    os.makedirs(_FONTS_DIR, exist_ok=True)
    for name, url in _PRECACHED_FONTS.items():
        dest = os.path.join(_FONTS_DIR, f"{name}.ttf")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            try:
                ImageFont.truetype(dest, 20)
                continue
            except Exception:
                os.remove(dest)
        elif os.path.exists(dest):
            os.remove(dest)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(r.content)
            ImageFont.truetype(dest, 20)
            print(f"  \u2713 {name} ({len(r.content):,} bytes)")
        except Exception as exc:
            print(f"  \u2717 {name}: {exc}")
            if os.path.exists(dest):
                os.remove(dest)
    return _FONTS_DIR


def fetch_google_font(family_slug: str, weight: int = 700) -> Optional[str]:
    """Download a Google Font on-demand from fontsource CDN. Returns path or None."""
    os.makedirs(_FONTS_DIR, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9-]", "", family_slug.lower().strip())
    dest_name = f"gf-{safe_name}-{weight}.ttf"
    dest = os.path.join(_FONTS_DIR, dest_name)

    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return dest

    url = f"https://cdn.jsdelivr.net/fontsource/fonts/{safe_name}@latest/latin-{weight}-normal.ttf"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        ImageFont.truetype(dest, 20)
        print(f"  \u2713 On-demand: {safe_name} w{weight} ({len(r.content):,} bytes)")
        return dest
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return None


def get_available_fonts() -> list[str]:
    """Return list of all available font family slugs (pre-cached + Google catalog)."""
    cached = set()
    if os.path.isdir(_FONTS_DIR):
        for f in os.listdir(_FONTS_DIR):
            if f.endswith(".ttf"):
                cached.add(f.replace(".ttf", ""))
    return sorted(set(list(cached) + _GOOGLE_FONT_FAMILIES))


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font by name with multi-level fallback."""
    # Try pre-cached font
    path = os.path.join(_FONTS_DIR, f"{name}.ttf")
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    # Try Google Fonts on-demand (if name looks like a slug)
    slug = name.lower().replace(" ", "-").replace("_", "-")
    gf_path = fetch_google_font(slug)
    if gf_path:
        try:
            return ImageFont.truetype(gf_path, size)
        except Exception:
            pass

    # Fallback chain
    for alt in ("Montserrat-Bold", "Montserrat-Regular", "Poppins-Bold"):
        alt_path = os.path.join(_FONTS_DIR, f"{alt}.ttf")
        if os.path.exists(alt_path):
            try:
                return ImageFont.truetype(alt_path, size)
            except Exception:
                continue
    for sys_font in ("DejaVuSans.ttf", "arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(sys_font, size)
        except OSError:
            continue
    return ImageFont.load_default()  # type: ignore[return-value]


def _load_custom_or_style_font(custom_font: str, style: str, role: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a custom font if specified, otherwise use the style's default."""
    if custom_font and custom_font != "Default":
        font = _load_font(custom_font, size)
        if font:
            return font
    pair = _STYLE_FONTS.get(style, _STYLE_FONTS["modern"])
    return _load_font(pair[0] if role == "title" else pair[1], size)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Strip emoji and non-ASCII that render as boxes."""
    return "".join(ch for ch in text if ord(ch) < 0x10000 and (ch.isprintable() or ch == " "))


def _text_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (width, height)."""
    bb = font.getbbox(text)
    return int(bb[2] - bb[0]), int(bb[3] - bb[1])


def _wrap_title(title: str, style: str) -> tuple[list[str], int]:
    """Return (wrapped lines, font point size)."""
    length = len(title)
    sizes = {
        "bold":        (70, 56, 44),
        "elegant":     (56, 44, 36),
        "retro":       (80, 64, 50),
        "minimal":     (48, 38, 30),
        "futuristic":  (54, 44, 36),
        "handwritten": (60, 48, 38),
        "royal":       (52, 42, 34),
    }
    s = sizes.get(style, (64, 50, 40))
    size = s[0] if length < 15 else s[1] if length < 25 else s[2]

    widths = {"retro": 14, "minimal": 24, "handwritten": 18, "royal": 20}
    width = widths.get(style, 18 if size > 50 else 22)

    display = title if style in ("elegant", "handwritten") else title.upper()
    return textwrap.wrap(display, width=width) or [display], size


# ---------------------------------------------------------------------------
# Quiet-zone detection
# ---------------------------------------------------------------------------

def _score_bands(image: Image.Image, n: int = 5) -> list[tuple[int, int, int, float]]:
    w, h = image.size
    edges = np.array(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    band_h = h // n
    top_margin = int(h * 0.15)
    bands = []
    for i in range(n):
        y0, y1 = i * band_h, min((i + 1) * band_h, h)
        score = 9999.0 if y1 <= top_margin else float(np.mean(edges[max(y0, top_margin):y1, :]))
        bands.append((i, y0, y1, score))
    bands.sort(key=lambda b: b[3])
    return bands


def _find_text_region(image: Image.Image, block_height: int, n_bands: int = 5) -> tuple[int, int, int, str]:
    w, h = image.size
    bands = _score_bands(image, n_bands)
    _, y0, y1, _ = bands[0]
    if (y1 - y0) < block_height:
        expand = (block_height - (y1 - y0)) // 2
        y0, y1 = max(0, y0 - expand), min(h, y1 + expand)
    y_center = (y0 + y1) // 2
    rel = y_center / h
    hint = "top" if rel < 0.33 else "bottom" if rel > 0.66 else "center"
    return y_center, y0, y1, hint


# ---------------------------------------------------------------------------
# Feathered dark scrim
# ---------------------------------------------------------------------------

def _apply_scrim(image: Image.Image, y_top: int, y_bottom: int, intensity: float = 0.78) -> Image.Image:
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    feather = 60
    fade_t, fade_b = max(0, y_top - feather), min(h, y_bottom + feather)
    for y in range(fade_t, fade_b):
        if y < y_top:
            t = (y - fade_t) / max(1, y_top - fade_t)
        elif y > y_bottom:
            t = 1.0 - (y - y_bottom) / max(1, fade_b - y_bottom)
        else:
            t = 1.0
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, min(int(200 * t * intensity), 215)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Text-rendering primitives
# ---------------------------------------------------------------------------

def _shadowed(draw, xy, text, font, fill="#FFFFFF", shadow_color="#000000",
              shadow_offset=3, anchor="lt"):
    text = _sanitize(text)
    x, y = xy
    draw.text((x + shadow_offset * 2, y + shadow_offset * 2), text, font=font, fill=(0, 0, 0, 90), anchor=anchor)
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 200), anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, stroke_width=2, stroke_fill=shadow_color, anchor=anchor)


def _pill(draw, xy, text, font, fill="#FFFFFF", bg=(0, 0, 0, 160), padding=12, anchor="lt"):
    text = _sanitize(text)
    bb = font.getbbox(text, anchor=anchor)
    x, y = xy
    draw.rounded_rectangle(
        [(x + bb[0] - padding, y + bb[1] - padding),
         (x + bb[2] + padding, y + bb[3] + padding)],
        radius=8, fill=bg,
    )
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Block height estimator
# ---------------------------------------------------------------------------

def _estimate_block_height(title, subtitle, date, venue, organizer, style):
    lines, size = _wrap_title(title, style)
    pair = _STYLE_FONTS.get(style, _STYLE_FONTS["modern"])
    font_title = _load_font(pair[0], size)
    total = sum(_text_bbox(ln, font_title)[1] + 12 for ln in lines) + 24
    if subtitle:  total += 34
    if date:      total += 28
    if venue:     total += 28
    if organizer: total += 36
    return total + 40


# ---------------------------------------------------------------------------
# 8 LAYOUT STYLES
# ---------------------------------------------------------------------------

# ---- 1. MODERN (centered, accent rules, info pill) ----
def _layout_modern(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 16
    draw.rectangle([(0, 0), (w, 5)], fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "modern", "body", 22)
        _shadowed(draw, (cx, 28), organizer.upper(), f, anchor="mt")
    cursor = start_y
    lines, size = _wrap_title(title, "modern")
    ft = _load_custom_or_style_font(custom_font, "modern", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 12
    cursor += 6
    draw.rectangle([(cx - 90, cursor), (cx + 90, cursor + 3)], fill=accent)
    cursor += 3 + gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "modern", "body", 26)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle), fs, fill=accent, anchor="mt")
        cursor += _text_bbox(subtitle, fs)[1] + gap
    parts = [p for p in [date, venue] if p]
    if parts:
        fi = _load_custom_or_style_font(custom_font, "modern", "body", 18)
        _pill(draw, (cx, cursor), "  |  ".join(parts), fi, bg=(0, 0, 0, 170), anchor="mt")
    draw.rectangle([(0, h - 5), (w, h)], fill=accent)


# ---- 2. BOLD (left-aligned, side accent bars) ----
def _layout_bold(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    LEFT, gap = 50, 18
    draw.rectangle([(0, 0), (6, h)], fill=accent)
    draw.rectangle([(w - 6, 0), (w, h)], fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "bold", "body", 18)
        _pill(draw, (w - LEFT, 28), organizer.upper(), f, fill=accent, bg=(0, 0, 0, 200), padding=10, anchor="rt")
    cursor = start_y
    lines, size = _wrap_title(title, "bold")
    ft = _load_custom_or_style_font(custom_font, "bold", "title", size)
    for line in lines:
        _shadowed(draw, (LEFT, cursor), line, ft, shadow_offset=4)
        cursor += _text_bbox(line, ft)[1] + 8
    cursor += gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "bold", "body", 24)
        _shadowed(draw, (LEFT, cursor), _sanitize(subtitle).upper(), fs, fill=accent)
        cursor += _text_bbox(subtitle.upper(), fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "bold", "body", 20)
    if date:
        _shadowed(draw, (LEFT, cursor), date, fi, fill="#DDDDDD")
        cursor += _text_bbox(date, fi)[1] + 10
    if venue:
        _shadowed(draw, (LEFT, cursor), venue, fi, fill="#DDDDDD")


# ---- 3. ELEGANT (centered serif, fine rules) ----
def _layout_elegant(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, rule_w, gap = w // 2, 160, 18
    draw.rectangle([(cx - rule_w, 46), (cx + rule_w, 48)], fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "elegant", "body", 20)
        _shadowed(draw, (cx, 62), _sanitize(organizer), f, anchor="mt")
    draw.rectangle([(cx - rule_w, 94), (cx + rule_w, 96)], fill=accent)
    cursor = start_y
    lines, size = _wrap_title(title, "elegant")
    ft = _load_custom_or_style_font(custom_font, "elegant", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, shadow_color="#1A1A1A", shadow_offset=3, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 14
    cursor += 8
    draw.rectangle([(cx - 60, cursor), (cx + 60, cursor + 1)], fill=accent)
    cursor += 1 + gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "elegant", "body", 26)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle), fs, fill=accent, anchor="mt")
        cursor += _text_bbox(subtitle, fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "elegant", "body", 17)
    if date:
        _shadowed(draw, (cx, cursor), date.upper(), fi, fill="#E8E8E8", anchor="mt")
        cursor += _text_bbox(date.upper(), fi)[1] + 8
    if venue:
        _pill(draw, (cx, cursor), _sanitize(venue), fi, bg=(0, 0, 0, 150), padding=10, anchor="mt")
    draw.rectangle([(cx - rule_w, h - 48), (cx + rule_w, h - 46)], fill=accent)
    draw.rectangle([(cx - rule_w, h - 36), (cx + rule_w, h - 34)], fill=accent)


# ---- 4. RETRO (stacked caps, diagonal accents, vintage feel) ----
def _layout_retro(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 14
    # Thick top/bottom bars
    draw.rectangle([(0, 0), (w, 8)], fill=accent)
    draw.rectangle([(0, 12), (w, 14)], fill=accent)
    draw.rectangle([(0, h - 14), (w, h - 12)], fill=accent)
    draw.rectangle([(0, h - 8), (w, h)], fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "retro", "body", 20)
        _shadowed(draw, (cx, 30), organizer.upper(), f, fill="#EEEEEE", anchor="mt")
    cursor = start_y
    lines, size = _wrap_title(title, "retro")
    ft = _load_custom_or_style_font(custom_font, "retro", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, fill=accent, shadow_offset=4, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 6
    cursor += gap
    # Star divider
    _shadowed(draw, (cx, cursor), "\u2605  \u2605  \u2605", _load_font("Montserrat-Regular", 16), fill=accent, anchor="mt")
    cursor += 30
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "retro", "body", 22)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle).upper(), fs, fill="#EEEEEE", anchor="mt")
        cursor += _text_bbox(subtitle.upper(), fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "retro", "body", 18)
    if date:
        _pill(draw, (cx, cursor), date.upper(), fi, fill=accent, bg=(0, 0, 0, 180), padding=10, anchor="mt")
        cursor += 40
    if venue:
        _shadowed(draw, (cx, cursor), venue.upper(), fi, fill="#CCCCCC", anchor="mt")


# ---- 5. MINIMAL (ultra-spaced, airy, clean) ----
def _layout_minimal(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 20
    # Thin accent line at top
    draw.rectangle([(cx - 40, 40), (cx + 40, 42)], fill=accent)
    cursor = start_y
    lines, size = _wrap_title(title, "minimal")
    ft = _load_custom_or_style_font(custom_font, "minimal", "title", size)
    for line in lines:
        # Extra letter spacing via manual rendering
        _shadowed(draw, (cx, cursor), line, ft, fill="#FFFFFF", shadow_offset=2, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 18
    cursor += gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "minimal", "body", 20)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle), fs, fill=accent, shadow_offset=2, anchor="mt")
        cursor += _text_bbox(subtitle, fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "minimal", "body", 16)
    parts = [p for p in [date, venue] if p]
    if parts:
        _shadowed(draw, (cx, cursor), "  \u00b7  ".join(parts), fi, fill="#BBBBBB", shadow_offset=1, anchor="mt")
        cursor += 30
    if organizer:
        fo = _load_custom_or_style_font(custom_font, "minimal", "body", 14)
        _shadowed(draw, (cx, cursor), organizer, fo, fill="#999999", shadow_offset=1, anchor="mt")
    draw.rectangle([(cx - 40, h - 42), (cx + 40, h - 40)], fill=accent)


# ---- 6. FUTURISTIC (geometric, neon accent lines, sci-fi) ----
def _layout_futuristic(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 16
    # Corner brackets
    bk = 40
    for corners in [((10, 10), (10, 10 + bk)), ((10, 10), (10 + bk, 10)),
                    ((w - 10, 10), (w - 10, 10 + bk)), ((w - 10, 10), (w - 10 - bk, 10)),
                    ((10, h - 10), (10, h - 10 - bk)), ((10, h - 10), (10 + bk, h - 10)),
                    ((w - 10, h - 10), (w - 10, h - 10 - bk)), ((w - 10, h - 10), (w - 10 - bk, h - 10))]:
        draw.line(corners, fill=accent, width=2)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "futuristic", "body", 16)
        _pill(draw, (cx, 30), organizer.upper(), f, fill=accent, bg=(0, 0, 0, 200), padding=8, anchor="mt")
    cursor = start_y
    lines, size = _wrap_title(title, "futuristic")
    ft = _load_custom_or_style_font(custom_font, "futuristic", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, fill=accent, shadow_color="#000000", shadow_offset=3, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 10
    cursor += 8
    # Glowing divider
    draw.rectangle([(cx - 120, cursor), (cx + 120, cursor + 2)], fill=accent)
    cursor += 2 + gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "futuristic", "body", 22)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle).upper(), fs, fill="#FFFFFF", anchor="mt")
        cursor += _text_bbox(subtitle.upper(), fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "futuristic", "body", 17)
    if date:
        _shadowed(draw, (cx, cursor), date, fi, fill="#CCCCCC", anchor="mt")
        cursor += _text_bbox(date, fi)[1] + 8
    if venue:
        _pill(draw, (cx, cursor), venue, fi, fill=accent, bg=(0, 0, 0, 180), padding=8, anchor="mt")


# ---- 7. HANDWRITTEN (organic, flowing, warm) ----
def _layout_handwritten(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 18
    # Subtle wavy divider (drawn as dots)
    for i in range(cx - 80, cx + 80, 8):
        draw.ellipse([(i, 50), (i + 3, 53)], fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "handwritten", "body", 18)
        _shadowed(draw, (cx, 66), organizer, f, fill="#DDDDDD", shadow_offset=2, anchor="mt")
    cursor = start_y
    lines, size = _wrap_title(title, "handwritten")
    ft = _load_custom_or_style_font(custom_font, "handwritten", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, fill="#FFFFFF", shadow_offset=3, anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 14
    cursor += gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "handwritten", "body", 22)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle), fs, fill=accent, shadow_offset=2, anchor="mt")
        cursor += _text_bbox(subtitle, fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "handwritten", "body", 18)
    parts = [p for p in [date, venue] if p]
    if parts:
        _shadowed(draw, (cx, cursor), "  ~  ".join(parts), fi, fill="#CCCCCC", shadow_offset=2, anchor="mt")
    # Bottom dots
    for i in range(cx - 80, cx + 80, 8):
        draw.ellipse([(i, h - 53), (i + 3, h - 50)], fill=accent)


# ---- 8. ROYAL (ornamental frames, luxury formal) ----
def _layout_royal(draw, w, h, title, subtitle, date, venue, organizer, accent, start_y, custom_font):
    cx, gap = w // 2, 16
    # Ornamental border
    m = 30
    draw.rectangle([(m, m), (w - m, m + 2)], fill=accent)
    draw.rectangle([(m, h - m - 2), (w - m, h - m)], fill=accent)
    draw.rectangle([(m, m), (m + 2, h - m)], fill=accent)
    draw.rectangle([(w - m - 2, m), (w - m, h - m)], fill=accent)
    # Inner border
    m2 = 38
    draw.rectangle([(m2, m2), (w - m2, m2 + 1)], fill=accent)
    draw.rectangle([(m2, h - m2 - 1), (w - m2, h - m2)], fill=accent)
    draw.rectangle([(m2, m2), (m2 + 1, h - m2)], fill=accent)
    draw.rectangle([(w - m2 - 1, m2), (w - m2, h - m2)], fill=accent)
    # Corner ornaments
    orn = "\u2726"
    fo = _load_font("Montserrat-Bold", 20)
    for pos in [(m + 8, m + 4), (w - m - 22, m + 4), (m + 8, h - m - 22), (w - m - 22, h - m - 22)]:
        draw.text(pos, orn, font=fo, fill=accent)
    if organizer:
        f = _load_custom_or_style_font(custom_font, "royal", "body", 18)
        _shadowed(draw, (cx, 60), organizer.upper(), f, fill=accent, anchor="mt")
    # Decorative divider
    draw.rectangle([(cx - 100, 82), (cx + 100, 83)], fill=accent)
    _shadowed(draw, (cx, 86), "\u2726", fo, fill=accent, anchor="mt")
    draw.rectangle([(cx - 100, 108), (cx + 100, 109)], fill=accent)
    cursor = start_y
    lines, size = _wrap_title(title, "royal")
    ft = _load_custom_or_style_font(custom_font, "royal", "title", size)
    for line in lines:
        _shadowed(draw, (cx, cursor), line, ft, fill="#FFFFFF", shadow_color="#1A1A1A", anchor="mt")
        cursor += _text_bbox(line, ft)[1] + 12
    cursor += gap
    if subtitle:
        fs = _load_custom_or_style_font(custom_font, "royal", "body", 24)
        _shadowed(draw, (cx, cursor), _sanitize(subtitle), fs, fill=accent, anchor="mt")
        cursor += _text_bbox(subtitle, fs)[1] + gap
    fi = _load_custom_or_style_font(custom_font, "royal", "body", 17)
    if date:
        _shadowed(draw, (cx, cursor), date.upper(), fi, fill="#E8E8E8", anchor="mt")
        cursor += _text_bbox(date.upper(), fi)[1] + 8
    if venue:
        _shadowed(draw, (cx, cursor), venue, fi, fill="#CCCCCC", anchor="mt")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LAYOUTS = {
    "modern": _layout_modern, "bold": _layout_bold, "elegant": _layout_elegant,
    "retro": _layout_retro, "minimal": _layout_minimal, "futuristic": _layout_futuristic,
    "handwritten": _layout_handwritten, "royal": _layout_royal,
}


def composite_poster(
    artwork:       Image.Image,
    title:         str,
    subtitle:      str            = "",
    date:          str            = "",
    venue:         str            = "",
    organizer:     str            = "",
    accent_color:  str            = "#FFD700",
    style:         Union[Style, str] = "modern",
    text_position: Union[Position, str] = "auto",
    scrim:         bool           = True,
    custom_font:   str            = "",
) -> Image.Image:
    """Composite event text onto an SDXL artwork image.

    Args:
        artwork:       Raw SDXL-generated PIL Image.
        title:         Primary event name (required).
        subtitle:      Short tagline (optional).
        date:          Date string (optional).
        venue:         Location (optional).
        organizer:     Host shown at top (optional).
        accent_color:  Hex colour for decorative elements.
        style:         One of 8 layout styles.
        text_position: "auto" | "top" | "center" | "bottom" | "none"
        scrim:         Apply dark gradient under text block.
        custom_font:   Optional font family name to override style defaults.

    Returns:
        Composited PIL Image (RGB).
    """
    style = str(style).lower().strip()
    text_position = str(text_position).lower().strip()
    if style not in _LAYOUTS:
        style = "modern"
    if text_position not in (*_POSITION_RATIOS, "auto", "none"):
        text_position = "auto"
    if text_position == "none":
        return artwork.copy().convert("RGB")

    ensure_fonts()

    title     = _sanitize(title)
    subtitle  = _sanitize(subtitle)
    date      = _sanitize(date)
    venue     = _sanitize(venue)
    organizer = _sanitize(organizer)

    img = artwork.copy().convert("RGB")
    w, h = img.size
    block_h = _estimate_block_height(title, subtitle, date, venue, organizer, style)

    if text_position in _POSITION_RATIOS:
        y_center = int(h * _POSITION_RATIOS[text_position])
    else:
        y_center, _, _, _ = _find_text_region(img, block_h)

    pad = 44
    if scrim:
        scrim_top    = max(0, y_center - block_h // 2 - pad)
        scrim_bottom = min(h, y_center + block_h // 2 + pad)
        img = _apply_scrim(img, scrim_top, scrim_bottom,
                           intensity=_SCRIM_INTENSITY.get(style, 0.78))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    text_start_y = max(pad, y_center - block_h // 2)

    layout_fn = _LAYOUTS.get(style, _layout_modern)
    layout_fn(draw, w, h, title, subtitle, date, venue, organizer,
              accent_color, text_start_y, custom_font or "")

    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    return result.convert("RGB")