#!/usr/bin/env python3
"""
Campus-AI by CounciL — Production Gradio Application  (v3.0)
=============================================================
Premium multi-tab poster generation platform.

Tabs:
  1. Text → Poster        (Two-Stage: SDXL artwork + PIL typography)
  2. Reference Image       (IP-Adapter + LoRA)
  3. Image Transform       (Img2Img pipeline)
  4. Inpaint / Edit        (Mask-based regeneration)
  5. HD Upscale            (Real-ESRGAN 4x)
"""

import os
import time
import logging
from typing import Optional

import torch
import gradio as gr

# WSL specific: prevent I/O crashes by forcing Gradio to use a local temp directory
os.makedirs("tmp/gradio", exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("tmp/gradio")

# HF Spaces ZeroGPU decorator (works even if package isn't installed)
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _FakeSpaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(fn):
                return fn
            return decorator
    spaces = _FakeSpaces()

from pipelines import get_pipeline_manager, flush_vram
from prompt_engine import (
    build_text2img_prompt,
    build_img2img_prompt,
    build_inpaint_prompt,
    STYLE_MAP,
    EVENT_TYPE_HINTS,
    NEGATIVE_PROMPT,
)
from poster_compositor import composite_poster, ensure_fonts, get_available_fonts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dropdowns
# ─────────────────────────────────────────────────────────────────────────────
EVENT_TYPES = list(EVENT_TYPE_HINTS.keys())
STYLES = list(STYLE_MAP.keys())

# Pre-cached poster fonts for dropdown (display-friendly names)
FONT_CHOICES = [
    "Default",
    "Montserrat-ExtraBold", "Montserrat-Bold", "Montserrat-Regular",
    "Poppins-Bold", "Poppins-Light", "Poppins-Regular",
    "PlayfairDisplay-Bold", "PlayfairDisplay-Regular",
    "Oswald-Bold", "Oswald-Regular",
    "Raleway-Bold", "Raleway-Regular",
    "Inter-Bold", "Inter-Regular",
    "BebasNeue-Regular",
    "Orbitron-Bold", "Orbitron-Regular",
    "Rajdhani-Bold", "Rajdhani-Regular",
    "DancingScript-Bold", "DancingScript-Regular",
    "Cinzel-Bold", "Cinzel-Regular",
    "CormorantGaramond-Bold", "CormorantGaramond-Regular",
    "Lora-Bold", "Lora-Regular",
    "Merriweather-Bold",
    "Quicksand-Bold", "Quicksand-Regular",
]

TYPOGRAPHY_STYLES = ["Modern", "Bold", "Elegant", "Retro", "Minimal", "Futuristic", "Handwritten", "Royal"]

RESOLUTION_PRESETS = {
    "Square (1024×1024)": (1024, 1024),
    "Portrait (768×1152)": (768, 1152),
    "Portrait Tall (768×1344)": (768, 1344),
    "Landscape (1152×768)": (1152, 768),
    "Landscape Wide (1344×768)": (1344, 768),
    "Instagram Story (768×1365)": (768, 1365),
    "A4 Poster (768×1086)": (768, 1086),
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@spaces.GPU(duration=90)
def generate_text2img(
    event_description: str,
    event_type: str,
    style: str,
    resolution: str,
    num_variants: int,
    num_steps: int,
    guidance_scale: float,
    lora_strength: float,
    enable_upscale: bool,
    seed: int,
    poster_title: str,
    poster_subtitle: str,
    poster_date: str,
    poster_venue: str,
    poster_organizer: str,
    poster_text_style: str,
    poster_text_position: str,
    poster_scrim: bool,
    poster_accent: str,
    poster_font: str,
):
    """Tab 1: Two-stage poster generation — SDXL artwork + PIL typography."""
    if not event_description.strip():
        raise gr.Error("Please enter an event description!")

    manager = get_pipeline_manager()
    pipe = manager.get_text2img()

    prompt = build_text2img_prompt(event_description, event_type, style)
    logger.info(f"[Text2Img] Artwork prompt: {prompt[:120]}...")

    width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))

    if seed == -1:
        seed = int(time.time()) % (2**32)

    if manager.is_lora_loaded:
        pipe.fuse_lora(lora_scale=lora_strength)

    images = []
    raw_artworks = []  # Store raw SDXL artworks for live typography updates
    generator = torch.Generator("cpu").manual_seed(seed)
    start = time.time()

    for i in range(num_variants):
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        artwork = result.images[0]
        raw_artworks.append(artwork.copy())

        # Stage 2: PIL typography compositor
        if poster_title.strip():
            try:
                img = composite_poster(
                    artwork=artwork,
                    title=poster_title.strip(),
                    subtitle=poster_subtitle.strip(),
                    date=poster_date.strip(),
                    venue=poster_venue.strip(),
                    organizer=poster_organizer.strip(),
                    accent_color=poster_accent or "#FFD700",
                    style=poster_text_style.lower(),
                    text_position=poster_text_position.lower(),
                    scrim=poster_scrim,
                    custom_font=poster_font or "",
                )
            except Exception as e:
                logger.warning(f"Compositor error: {e}. Returning raw artwork.")
                img = artwork
        else:
            img = artwork

        if enable_upscale:
            img = manager.upscale_image(img, scale=2)

        images.append(img)

    elapsed = time.time() - start

    if manager.is_lora_loaded:
        pipe.unfuse_lora()

    info = (
        f"**Generated {num_variants} poster(s) in {elapsed:.1f}s** | "
        f"Seed: {seed} | {width}×{height} | Steps: {num_steps}\n\n"
        f"**Prompt:**\n{prompt}"
    )

    return images, info, raw_artworks


def update_typography(
    stored_artworks,
    poster_title: str,
    poster_subtitle: str,
    poster_date: str,
    poster_venue: str,
    poster_organizer: str,
    poster_text_style: str,
    poster_text_position: str,
    poster_scrim: bool,
    poster_accent: str,
    poster_font: str,
):
    """Re-apply typography on stored raw artworks without regenerating SDXL."""
    if not stored_artworks:
        raise gr.Error("No artwork stored — generate a poster first!")
    if not poster_title.strip():
        raise gr.Error("Please enter a poster title!")

    start = time.time()
    images = []
    for artwork in stored_artworks:
        try:
            img = composite_poster(
                artwork=artwork,
                title=poster_title.strip(),
                subtitle=poster_subtitle.strip(),
                date=poster_date.strip(),
                venue=poster_venue.strip(),
                organizer=poster_organizer.strip(),
                accent_color=poster_accent or "#FFD700",
                style=poster_text_style.lower(),
                text_position=poster_text_position.lower(),
                scrim=poster_scrim,
                custom_font=poster_font or "",
            )
        except Exception as e:
            logger.warning(f"Typography update error: {e}")
            img = artwork
        images.append(img)

    elapsed = time.time() - start
    info = (
        f"**Typography updated in {elapsed:.1f}s** | "
        f"{len(images)} poster(s) | "
        f"Style: {poster_text_style} | Position: {poster_text_position}"
    )
    return images, info


@spaces.GPU(duration=90)
def generate_with_reference(
    event_description: str,
    reference_image,
    style: str,
    style_strength: float,
    resolution: str,
    num_steps: int,
    guidance_scale: float,
    enable_upscale: bool,
    seed: int,
):
    """Tab 2: Reference image + text → poster (IP-Adapter)."""
    if reference_image is None:
        raise gr.Error("Please upload a reference image!")
    if not event_description.strip():
        raise gr.Error("Please enter an event description!")

    from PIL import Image

    manager = get_pipeline_manager()
    pipe = manager.get_text2img()
    pipe = manager.load_ip_adapter(pipe)
    manager.set_ip_adapter_scale(pipe, scale=style_strength)

    prompt = build_text2img_prompt(event_description, "Other", style)
    width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))

    if seed == -1:
        seed = int(time.time()) % (2**32)

    generator = torch.Generator("cpu").manual_seed(seed)
    start = time.time()

    # Prepare reference image
    ref_img = Image.fromarray(reference_image).convert("RGB").resize((224, 224))

    result = pipe(
        prompt=prompt,
        ip_adapter_image=ref_img,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    img = result.images[0]
    elapsed = time.time() - start

    if enable_upscale:
        img = manager.upscale_image(img, scale=2)

    info = (
        f"**Generated in {elapsed:.1f}s** | Seed: {seed} | "
        f"Style strength: {style_strength}\n\n"
        f"**Prompt:**\n{prompt}"
    )

    return img, info


@spaces.GPU(duration=90)
def generate_img2img(
    input_image,
    transform_description: str,
    style: str,
    denoising_strength: float,
    num_steps: int,
    guidance_scale: float,
    enable_upscale: bool,
    seed: int,
):
    """Tab 3: Image-to-image transformation."""
    if input_image is None:
        raise gr.Error("Please upload an image to transform!")

    from PIL import Image

    manager = get_pipeline_manager()
    pipe = manager.get_img2img()

    prompt = build_img2img_prompt(transform_description, style)

    if seed == -1:
        seed = int(time.time()) % (2**32)

    generator = torch.Generator("cpu").manual_seed(seed)
    init_image = Image.fromarray(input_image).convert("RGB").resize((1024, 1024))

    start = time.time()
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=denoising_strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    img = result.images[0]
    elapsed = time.time() - start

    if enable_upscale:
        img = manager.upscale_image(img, scale=2)

    info = (
        f"**Transformed in {elapsed:.1f}s** | Seed: {seed} | "
        f"Denoise: {denoising_strength}\n\n"
        f"**Prompt:**\n{prompt}"
    )

    return img, info


@spaces.GPU(duration=90)
def generate_inpaint(
    input_data: dict,
    fill_description: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Tab 4: Inpainting – regenerate masked region."""
    if input_data is None:
        raise gr.Error("Please upload an image and draw a mask!")

    from PIL import Image
    import numpy as np

    manager = get_pipeline_manager()
    pipe = manager.get_inpaint()

    prompt = build_inpaint_prompt(fill_description)

    if seed == -1:
        seed = int(time.time()) % (2**32)

    generator = torch.Generator("cpu").manual_seed(seed)

    # Extract image and mask from ImageEditor output
    source_image = Image.fromarray(input_data["background"]).convert("RGB").resize((1024, 1024))

    # Build mask from composite layers
    if "layers" in input_data and len(input_data["layers"]) > 0:
        mask_layer = input_data["layers"][0]
        mask = Image.fromarray(mask_layer).convert("L").resize((1024, 1024))
        # Binarize mask
        mask = mask.point(lambda x: 255 if x > 10 else 0)
    else:
        raise gr.Error("Please draw on the image to create a mask!")

    start = time.time()
    result = pipe(
        prompt=prompt,
        image=source_image,
        mask_image=mask,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    img = result.images[0]
    elapsed = time.time() - start

    info = (
        f"**Inpainted in {elapsed:.1f}s** | Seed: {seed}\n\n"
        f"**Prompt:**\n{prompt}"
    )

    return img, info


def upscale_only(input_image, scale_factor: int):
    """Tab 5: Standalone HD upscaling."""
    if input_image is None:
        raise gr.Error("Please upload an image to upscale!")

    from PIL import Image

    manager = get_pipeline_manager()
    img = Image.fromarray(input_image).convert("RGB")

    original_size = f"{img.width}×{img.height}"

    start = time.time()
    result = manager.upscale_image(img, scale=scale_factor)
    elapsed = time.time() - start

    new_size = f"{result.width}×{result.height}"
    info = f"**Upscaled in {elapsed:.1f}s** | {original_size} → {new_size}"

    return result, info


def edit_poster(
    input_image,
    poster_title: str,
    poster_subtitle: str,
    poster_date: str,
    poster_venue: str,
    poster_organizer: str,
    poster_text_style: str,
    poster_text_position: str,
    poster_scrim: bool,
    poster_accent: str,
    poster_font: str,
):
    """Tab 6: Edit poster — apply/re-apply typography overlay on any artwork."""
    if input_image is None:
        raise gr.Error("Please upload an artwork image!")
    if not poster_title.strip():
        raise gr.Error("Please enter a poster title!")

    from PIL import Image

    img = Image.fromarray(input_image).convert("RGB")
    start = time.time()

    try:
        result = composite_poster(
            artwork=img,
            title=poster_title.strip(),
            subtitle=poster_subtitle.strip(),
            date=poster_date.strip(),
            venue=poster_venue.strip(),
            organizer=poster_organizer.strip(),
            accent_color=poster_accent or "#FFD700",
            style=poster_text_style.lower(),
            text_position=poster_text_position.lower(),
            scrim=poster_scrim,
            custom_font=poster_font or "",
        )
    except Exception as e:
        logger.warning(f"Compositor error in editor: {e}")
        raise gr.Error(f"Compositor error: {e}")

    elapsed = time.time() - start
    info = (
        f"**Typography applied in {elapsed:.1f}s** | "
        f"Size: {result.width}×{result.height} | "
        f"Style: {poster_text_style} | Position: {poster_text_position}"
    )

    return result, info


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

css = """
/* ═══════════════════════════════════════════════════════════════
   Campus-AI by CounciL — Premium Dark Gold Theme
   ═══════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');

/* ── Root Variables ─────────────────────────────────────────── */
:root {
    --gold: #D4AF37;
    --gold-light: #F5D87A;
    --gold-dark: #B8960C;
    --bg-primary: #0A0A0F;
    --bg-secondary: #12121A;
    --bg-card: #1A1A25;
    --bg-elevated: #22222F;
    --border-subtle: rgba(212, 175, 55, 0.12);
    --border-gold: rgba(212, 175, 55, 0.25);
    --text-primary: #F0EDE6;
    --text-secondary: #9B9AA3;
    --text-muted: #6B6A73;
    --accent-purple: #764ba2;
    --accent-blue: #667eea;
    --shadow-gold: 0 0 30px rgba(212, 175, 55, 0.06);
    --radius: 12px;
    --radius-lg: 16px;
}

/* ── Global Container ───────────────────────────────────────── */
.gradio-container {
    max-width: 1440px !important;
    margin: auto;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
}

.dark .gradio-container {
    background: var(--bg-primary) !important;
}

/* ── Hero Header ────────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 2.5em 1em 1.8em;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 50% 0%, rgba(212, 175, 55, 0.06) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.2em;
    font-weight: 900;
    color: #D4AF37;
    text-shadow: 0 0 40px rgba(212, 175, 55, 0.3), 0 0 80px rgba(212, 175, 55, 0.1);
    margin-bottom: 0.15em;
    letter-spacing: 0.02em;
    line-height: 1.1;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}

.hero-byline {
    font-family: 'Inter', sans-serif;
    font-size: 0.85em;
    font-weight: 400;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--gold-dark);
    margin-bottom: 0.8em;
    opacity: 0.7;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    color: var(--text-secondary);
    font-size: 1.05em;
    font-weight: 300;
    line-height: 1.6;
    max-width: 700px;
    margin: 0 auto;
    letter-spacing: 0.01em;
}

.hero-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.2em auto 0;
}

/* ── Tab Navigation ─────────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    padding: 0.4em 0.5em !important;
    gap: 4px !important;
}

.tabs > .tab-nav > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92em !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65em 1.2em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: transparent !important;
    letter-spacing: 0.01em;
}

.tabs > .tab-nav > button:hover {
    color: var(--text-primary) !important;
    background: rgba(212, 175, 55, 0.06) !important;
}

.tabs > .tab-nav > button.selected {
    color: var(--gold) !important;
    background: rgba(212, 175, 55, 0.1) !important;
    box-shadow: 0 1px 0 0 var(--gold) !important;
    font-weight: 600 !important;
}

/* ── Tab Content Panels ─────────────────────────────────────── */
.tabitem {
    background: var(--bg-secondary) !important;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    padding: 1.5em !important;
}

/* ── Tab Description Banners ────────────────────────────────── */
.tab-description {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.04) 0%, rgba(102, 126, 234, 0.04) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1em 1.4em;
    margin-bottom: 1.2em;
    color: var(--text-secondary);
    font-size: 0.9em;
    line-height: 1.5;
}

/* ── Input Controls ─────────────────────────────────────────── */
.gradio-textbox textarea,
.gradio-textbox input,
.gradio-dropdown select,
.gradio-number input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.gradio-textbox textarea:focus,
.gradio-textbox input:focus,
.gradio-dropdown select:focus {
    border-color: var(--gold-dark) !important;
    box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.1) !important;
    outline: none !important;
}

label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
    letter-spacing: 0.02em;
}

/* ── Accordion Panels ───────────────────────────────────────── */
.gradio-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

.gradio-accordion > .label-wrap {
    background: var(--bg-elevated) !important;
    padding: 0.8em 1.2em !important;
}

/* ── Generate Buttons ───────────────────────────────────────── */
.generate-btn {
    background: linear-gradient(135deg, var(--gold-dark) 0%, var(--gold) 50%, var(--gold-light) 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    color: #0A0A0F !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius) !important;
    padding: 0.9em 2em !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2) !important;
}

.generate-btn:hover {
    background-position: right center !important;
    box-shadow: 0 6px 25px rgba(212, 175, 55, 0.35) !important;
    transform: translateY(-1px) !important;
}

.generate-btn:active {
    transform: translateY(0px) !important;
}

/* ── Gallery ────────────────────────────────────────────────── */
.gradio-gallery {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden;
}

.gradio-gallery .gallery-item {
    border-radius: 8px !important;
    border: 1px solid var(--border-subtle) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.gradio-gallery .gallery-item:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-gold) !important;
}

/* ── Output Info Blocks ─────────────────────────────────────── */
.prose {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9em !important;
    line-height: 1.6 !important;
}

.prose strong {
    color: var(--gold-light) !important;
}

/* ── Sliders ────────────────────────────────────────────────── */
input[type="range"] {
    accent-color: var(--gold) !important;
}

/* ── Checkbox ───────────────────────────────────────────────── */
input[type="checkbox"]:checked {
    accent-color: var(--gold) !important;
}

/* ── Color Picker ───────────────────────────────────────────── */
.gradio-colorpicker input {
    border-radius: var(--radius) !important;
}

/* ── Examples Section ───────────────────────────────────────── */
.examples-table {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* ── Footer ─────────────────────────────────────────────────── */
.footer-container {
    text-align: center;
    padding: 2em 1em 1.5em;
    margin-top: 1.5em;
    border-top: 1px solid var(--border-subtle);
    position: relative;
}

.footer-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dark), transparent);
}

.footer-brand {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.15em;
    font-weight: 700;
    color: #D4AF37;
    display: inline-block;
    margin-bottom: 0.4em;
}

.footer-details {
    color: var(--text-muted);
    font-size: 0.78em;
    font-weight: 300;
    line-height: 1.8;
    letter-spacing: 0.02em;
}

.footer-tech {
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.4em;
}

.footer-badge {
    display: inline-block;
    background: rgba(212, 175, 55, 0.08);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-size: 0.72em;
    padding: 0.2em 0.6em;
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border-gold);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gold-dark); }

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero-title { font-size: 2em; }
    .hero-subtitle { font-size: 0.9em; }
    .tabs > .tab-nav > button { font-size: 0.82em !important; padding: 0.5em 0.8em !important; }
}
"""

EXAMPLES = [
    ["IIT Indore Techfest 2026 — Robotics & AI Championships", "Technical Fest", "Tech-Futuristic"],
    ["Diwali Mela 2026 — Spark of Joy", "Diwali Celebration", "Festival of Lights"],
    ["Inter-College Basketball Championship", "Sports Tournament", "Vibrant and Energetic"],
    ["Photography Club Portfolio Night", "Club Recruitment", "Dark Premium"],
    ["ML/AI Workshop Series — From Zero to GPT", "Workshop / Seminar", "Gradient Modern"],
    ["Classical Kathak Dance Night", "Cultural Event", "Traditional Indian"],
    ["Holi Hai! Campus Color Run", "Holi Festival", "Artistic and Creative"],
    ["Navratri Garba Night 2026", "Navratri / Garba", "Bollywood Glamour"],
    ["End-of-Year Farewell Party", "Freshers / Farewell", "Watercolor Dreamscape"],
    ["Independence Day Cultural Program", "Independence Day / Republic Day", "Traditional Indian"],
    ["24-Hour Hackathon — Code for Good", "Hackathon", "Neon Glow"],
    ["Annual Fashion Show — Threads of Time", "Fashion Show", "Elegant and Professional"],
]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Campus-AI by CounciL") as demo:

        # ── Hero Header ──────────────────────────────────────────
        gr.HTML(
            '<div class="hero-container">'
            '<div class="hero-title">Campus-AI</div>'
            '<div class="hero-byline">by CounciL</div>'
            '<div class="hero-subtitle">'
            'AI-powered event poster generation — describe your event, '
            'choose a style, and get pixel-perfect posters with clean typography in seconds.'
            '</div>'
            '<div class="hero-divider"></div>'
            '</div>'
        )

        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════
            # TAB 1: Text → Poster
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  Text → Poster", id="text2img"):
                gr.HTML(
                    '<div class="tab-description">'
                    '<strong>Two-Stage Pipeline</strong> — SDXL generates pure visual artwork, '
                    'then PIL compositor overlays pixel-perfect typography. '
                    'Fill in the event details and poster text below.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_event = gr.Textbox(
                            label="📝 Describe Your Event",
                            placeholder="e.g., 'Annual tech fest with AI and robotics competitions at IIT Indore, March 2026'",
                            lines=3,
                        )
                        t2i_type = gr.Dropdown(
                            EVENT_TYPES, value="Technical Fest",
                            label="🏷️ Event Type",
                        )
                        t2i_style = gr.Dropdown(
                            STYLES, value="Vibrant and Energetic",
                            label="🎨 Visual Style",
                        )
                        t2i_resolution = gr.Dropdown(
                            list(RESOLUTION_PRESETS.keys()),
                            value="Portrait (768×1152)",
                            label="📐 Resolution",
                        )
                        t2i_variants = gr.Slider(
                            1, 4, value=1, step=1,
                            label="🔢 Number of Variants",
                        )

                        with gr.Accordion("✦ Typography Overlay", open=True):
                            gr.HTML('<div style="color: var(--text-muted); font-size: 0.82em; margin-bottom: 0.8em;">These fields are rendered as real text on top — not generated by AI.</div>')
                            t2i_title = gr.Textbox(
                                label="Poster Title",
                                placeholder="e.g., TechFest 2026",
                                value="",
                            )
                            t2i_subtitle = gr.Textbox(
                                label="Subtitle / Tagline",
                                placeholder="e.g., Innovation Beyond Limits",
                                value="",
                            )
                            with gr.Row():
                                t2i_date = gr.Textbox(
                                    label="Date",
                                    placeholder="March 15-17, 2026",
                                    value="",
                                )
                                t2i_venue = gr.Textbox(
                                    label="Venue",
                                    placeholder="IIT Indore, Main Auditorium",
                                    value="",
                                )
                            t2i_organizer = gr.Textbox(
                                label="Organizer",
                                placeholder="e.g., Student Technical Council",
                                value="",
                            )
                            with gr.Row():
                                t2i_text_style = gr.Dropdown(
                                    TYPOGRAPHY_STYLES,
                                    value="Modern",
                                    label="Typography Style",
                                )
                                t2i_text_pos = gr.Dropdown(
                                    ["Auto", "Top", "Center", "Bottom"],
                                    value="Auto",
                                    label="Text Position",
                                )
                            t2i_font = gr.Dropdown(
                                FONT_CHOICES,
                                value="Default",
                                label="Font Family (30 pre-cached + 1,700 on-demand)",
                            )
                            with gr.Row():
                                t2i_accent = gr.ColorPicker(
                                    label="Accent Color",
                                    value="#FFD700",
                                )
                                t2i_scrim = gr.Checkbox(
                                    label="Dark Scrim (contrast behind text)",
                                    value=True,
                                )

                        with gr.Accordion("⚙  Advanced Settings", open=False):
                            t2i_steps = gr.Slider(10, 50, value=28, step=1, label="Inference Steps")
                            t2i_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                            t2i_lora = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="LoRA Strength")
                            t2i_upscale = gr.Checkbox(label="HD Upscale (2×)", value=False)
                            t2i_seed = gr.Number(value=-1, label="Seed (-1 = random)")

                        t2i_btn = gr.Button("Generate Poster", variant="primary", size="lg", elem_classes=["generate-btn"])
                        t2i_update_btn = gr.Button("Update Typography", variant="secondary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        t2i_gallery = gr.Gallery(
                            label="Generated Posters", columns=2,
                            height=650, object_fit="contain",
                        )
                        t2i_info = gr.Markdown(label="Generation Info")
                        # Hidden state to store raw artworks for live typography updates
                        t2i_raw_state = gr.State(value=[])

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[t2i_event, t2i_type, t2i_style],
                    label="💡 Try These Examples",
                )

                t2i_btn.click(
                    fn=generate_text2img,
                    inputs=[
                        t2i_event, t2i_type, t2i_style, t2i_resolution,
                        t2i_variants, t2i_steps, t2i_cfg, t2i_lora,
                        t2i_upscale, t2i_seed,
                        t2i_title, t2i_subtitle, t2i_date, t2i_venue,
                        t2i_organizer, t2i_text_style, t2i_text_pos,
                        t2i_scrim, t2i_accent, t2i_font,
                    ],
                    outputs=[t2i_gallery, t2i_info, t2i_raw_state],
                )

                t2i_update_btn.click(
                    fn=update_typography,
                    inputs=[
                        t2i_raw_state,
                        t2i_title, t2i_subtitle, t2i_date, t2i_venue,
                        t2i_organizer, t2i_text_style, t2i_text_pos,
                        t2i_scrim, t2i_accent, t2i_font,
                    ],
                    outputs=[t2i_gallery, t2i_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 2: Reference Image
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  Reference Image", id="reference"):
                gr.HTML(
                    '<div class="tab-description">'
                    'Upload a poster you like — the AI extracts its <strong>visual style</strong> '
                    'and blends it with your event description using IP-Adapter.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_image = gr.Image(
                            label="📎 Upload Reference Poster",
                            type="numpy", height=300,
                        )
                        ref_event = gr.Textbox(
                            label="📝 Describe Your Event",
                            placeholder="e.g., 'Annual cultural night with dance performances'",
                            lines=2,
                        )
                        ref_style = gr.Dropdown(
                            STYLES, value="Vibrant and Energetic",
                            label="🎨 Base Style",
                        )
                        ref_strength = gr.Slider(
                            0.0, 1.0, value=0.6, step=0.05,
                            label="🎚️ Reference Influence (0=ignore, 1=copy)",
                        )
                        ref_resolution = gr.Dropdown(
                            list(RESOLUTION_PRESETS.keys()),
                            value="Portrait (768×1152)",
                            label="📐 Resolution",
                        )

                        with gr.Accordion("⚙️ Advanced", open=False):
                            ref_steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
                            ref_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance")
                            ref_upscale = gr.Checkbox(label="🔍 HD Upscale (2x)", value=False)
                            ref_seed = gr.Number(value=-1, label="Seed")

                        ref_btn = gr.Button("Generate with Reference", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        ref_output = gr.Image(label="Generated Poster", type="pil", height=600)
                        ref_info = gr.Markdown()

                ref_btn.click(
                    fn=generate_with_reference,
                    inputs=[
                        ref_event, ref_image, ref_style, ref_strength,
                        ref_resolution, ref_steps, ref_cfg, ref_upscale, ref_seed,
                    ],
                    outputs=[ref_output, ref_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 3: Image Transform
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  Image Transform", id="img2img"):
                gr.HTML(
                    '<div class="tab-description">'
                    'Upload an existing poster → describe how you want it <strong>transformed</strong>. '
                    'Lower denoising = subtle changes, higher = dramatic restyle.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        i2i_image = gr.Image(
                            label="📎 Upload Poster to Transform",
                            type="numpy", height=300,
                        )
                        i2i_desc = gr.Textbox(
                            label="🔄 Describe the Transformation",
                            placeholder="e.g., 'Make it neon-themed with darker background and glow effects'",
                            lines=2,
                        )
                        i2i_style = gr.Dropdown(
                            STYLES, value="Tech-Futuristic",
                            label="🎨 Target Style",
                        )
                        i2i_denoise = gr.Slider(
                            0.1, 1.0, value=0.65, step=0.05,
                            label="🎚️ Transformation Strength (0.1=subtle, 1.0=complete restyle)",
                        )

                        with gr.Accordion("⚙️ Advanced", open=False):
                            i2i_steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
                            i2i_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance")
                            i2i_upscale = gr.Checkbox(label="🔍 HD Upscale (2x)", value=False)
                            i2i_seed = gr.Number(value=-1, label="Seed")

                        i2i_btn = gr.Button("Transform Poster", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        i2i_output = gr.Image(label="Transformed Poster", type="pil", height=600)
                        i2i_info = gr.Markdown()

                i2i_btn.click(
                    fn=generate_img2img,
                    inputs=[
                        i2i_image, i2i_desc, i2i_style, i2i_denoise,
                        i2i_steps, i2i_cfg, i2i_upscale, i2i_seed,
                    ],
                    outputs=[i2i_output, i2i_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 4: Inpainting
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  Inpaint / Edit", id="inpaint"):
                gr.HTML(
                    '<div class="tab-description">'
                    'Upload a poster → <strong>draw over the area</strong> you want to change → '
                    'describe what should replace it. The rest stays intact.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_editor = gr.ImageEditor(
                            label="🖌️ Draw Mask on Poster",
                            type="numpy",
                            height=400,
                            brush=gr.Brush(
                                default_size=30,
                                colors=["#FFFFFF"],
                                color_mode="fixed",
                            ),
                            eraser=gr.Eraser(default_size=20),
                            layers=True,
                        )
                        inp_desc = gr.Textbox(
                            label="📝 What Should Fill the Masked Area?",
                            placeholder="e.g., 'A golden trophy with confetti'",
                            lines=2,
                        )

                        with gr.Accordion("⚙️ Advanced", open=False):
                            inp_steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
                            inp_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance")
                            inp_seed = gr.Number(value=-1, label="Seed")

                        inp_btn = gr.Button("Inpaint Region", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        inp_output = gr.Image(label="Inpainted Poster", type="pil", height=600)
                        inp_info = gr.Markdown()

                inp_btn.click(
                    fn=generate_inpaint,
                    inputs=[inp_editor, inp_desc, inp_steps, inp_cfg, inp_seed],
                    outputs=[inp_output, inp_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 5: HD Upscale
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  HD Upscale", id="upscale"):
                gr.HTML(
                    '<div class="tab-description">'
                    'Upload any image → get a <strong>4× upscaled</strong> HD version using Real-ESRGAN. '
                    'Perfect for making generated posters print-ready.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        up_image = gr.Image(
                            label="📎 Upload Image",
                            type="numpy", height=300,
                        )
                        up_scale = gr.Radio(
                            [2, 4], value=4, label="🔍 Scale Factor",
                        )
                        up_btn = gr.Button("Upscale", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        up_output = gr.Image(label="Upscaled Image", type="pil", height=600)
                        up_info = gr.Markdown()

                up_btn.click(
                    fn=upscale_only,
                    inputs=[up_image, up_scale],
                    outputs=[up_output, up_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 6: Poster Editor
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✦  Edit Poster", id="edit_poster"):
                gr.HTML(
                    '<div class="tab-description">'
                    'Upload any artwork or generated poster → <strong>apply or re-apply typography</strong> '
                    'with full control over title, subtitle, date, venue, style, and position. '
                    'No AI generation — instant compositor overlay.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ed_image = gr.Image(
                            label="Upload Artwork / Poster",
                            type="numpy", height=300,
                        )

                        with gr.Accordion("✦ Typography Settings", open=True):
                            ed_title = gr.Textbox(
                                label="Poster Title",
                                placeholder="e.g., TechFest 2026",
                            )
                            ed_subtitle = gr.Textbox(
                                label="Subtitle / Tagline",
                                placeholder="e.g., Innovation Beyond Limits",
                            )
                            with gr.Row():
                                ed_date = gr.Textbox(
                                    label="Date",
                                    placeholder="March 15-17, 2026",
                                )
                                ed_venue = gr.Textbox(
                                    label="Venue",
                                    placeholder="IIT Indore, Main Auditorium",
                                )
                            ed_organizer = gr.Textbox(
                                label="Organizer",
                                placeholder="e.g., Student Technical Council",
                            )
                            with gr.Row():
                                ed_text_style = gr.Dropdown(
                                    TYPOGRAPHY_STYLES,
                                    value="Modern",
                                    label="Typography Style",
                                )
                                ed_text_pos = gr.Dropdown(
                                    ["Auto", "Top", "Center", "Bottom"],
                                    value="Auto",
                                    label="Text Position",
                                )
                            ed_font = gr.Dropdown(
                                FONT_CHOICES,
                                value="Default",
                                label="Font Family",
                            )
                            with gr.Row():
                                ed_accent = gr.ColorPicker(
                                    label="Accent Color",
                                    value="#FFD700",
                                )
                                ed_scrim = gr.Checkbox(
                                    label="Dark Scrim (contrast behind text)",
                                    value=True,
                                )

                        ed_btn = gr.Button("Apply Typography", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        ed_output = gr.Image(label="Edited Poster", type="pil", height=650)
                        ed_info = gr.Markdown()

                ed_btn.click(
                    fn=edit_poster,
                    inputs=[
                        ed_image, ed_title, ed_subtitle, ed_date, ed_venue,
                        ed_organizer, ed_text_style, ed_text_pos,
                        ed_scrim, ed_accent, ed_font,
                    ],
                    outputs=[ed_output, ed_info],
                )

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML(
            '<div class="footer-container">'
            '<div class="footer-brand">Campus-AI</div><br>'
            '<div class="footer-details">'
            'Fine-tuned on 55,000+ event poster images across 55 subcategories<br>'
            '<div class="footer-tech">'
            '<span class="footer-badge">SDXL 1.0 · 2.6B</span>'
            '<span class="footer-badge">LoRA</span>'
            '<span class="footer-badge">IP-Adapter</span>'
            '<span class="footer-badge">Real-ESRGAN</span>'
            '<span class="footer-badge">Groq Llama 3.3 70B</span>'
            '<span class="footer-badge">PIL Compositor</span>'
            '</div>'
            '<div style="margin-top: 0.6em; color: var(--text-muted); font-size: 0.85em;">'
            'Crafted by <strong style="color: var(--gold-dark);">CounciL</strong> for the Indian campus community'
            '</div>'
            '</div>'
            '</div>'
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, css=css, theme=gr.themes.Base())
