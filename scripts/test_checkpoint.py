#!/usr/bin/env python3
"""
test_checkpoint.py
==================
Two-Stage Poster Generation Pipeline — SDXL Artwork + PIL Typography

Stage 1  Generate pure visual artwork with SDXL + Campus AI LoRA.
         Prompts describe ONLY visual atmosphere — zero text references.
         guidance_scale=7.5 ensures the negative prompt suppresses all
         hallucinated text/watermarks from the diffusion output.

Stage 2  PIL Compositor overlays pixel-perfect typography on the raw artwork.

Usage:
    python test_checkpoint.py

Outputs in output/test_generations/:
    <slug>_artwork.png   — raw SDXL output, no text
    <slug>_poster.png    — final composited poster

Per-poster controls:
    text_position  "top" | "center" | "bottom" | "auto"
                   Set based on where the artwork has clean negative space.
    scrim          True  for dark/busy artworks — adds contrast under text.
                   False for vivid/bright artworks — keep colours untouched.
"""

from __future__ import annotations

import os
import sys

import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poster_compositor import composite_poster, ensure_fonts


# ---------------------------------------------------------------------------
# Shared negative prompt
# ---------------------------------------------------------------------------
# Explicitly blocks ALL forms of text/typography from the raw artwork.
# garbled_text and illegible_text added specifically to kill LoRA artefacts
# like BOMIELLOOOKD / OULSTECS seen in previous generations.

_NEG = (
    "text, words, letters, typography, fonts, captions, labels, watermark, "
    "signature, logo, banner, title, heading, writing, written text, "
    "illegible text, garbled text, gibberish text, distorted words, "
    "random letters, fake words, blurry, low quality, deformed, ugly, "
    "disfigured, oversaturated, bad anatomy, cropped, out of frame"
)


# ---------------------------------------------------------------------------
# Poster definitions
# ---------------------------------------------------------------------------

POSTERS: list[tuple[str, str, dict]] = [

    # ── Freshers Party ──────────────────────────────────────────────────────
    (
        "freshers_party",

        "campus_ai_poster  Vibrant freshers welcome party background.  "
        "Confetti explosion in electric blue and neon purple raining from above.  "
        "Disco ball casting prismatic reflections across a dark concert stage.  "
        "Bokeh light circles in hot pink and cyan filling the frame.  "
        "Bollywood dance-floor energy with glitter dust in a single spotlight beam.  "
        "Shallow depth of field, cinematic wide-angle composition.  "
        "No text, no signs, no banners anywhere in the scene.",

        dict(
            title          = "Freshers Bash 2026",
            subtitle       = "Welcome to the Jungle, First Years!",
            date           = "August 22, 2026  •  6 PM Onwards",
            venue          = "Open Air Theatre, DTU",
            organizer      = "Student Council 2026–27",
            accent_color   = "#E040FB",
            style          = "bold",
            text_position  = "bottom",
            scrim          = True,
        ),
    ),

    # ── Navratri Garba ──────────────────────────────────────────────────────
    (
        "navratri_garba",

        "campus_ai_poster  Stunning Navratri Garba night celebration background.  "
        "Swirling dandiya sticks and ghagra choli silhouettes mid-spin viewed from above.  "
        "Warm saffron, deep crimson, and gold falling flower petals.  "
        "Intricate mirror-work embroidery and marigold garland borders framing the scene.  "
        "Glowing earthen diyas reflecting off a polished stone floor.  "
        "Rich festive atmosphere, painterly detail, vibrant colour contrast.  "
        "No text, no signs, no labels anywhere in the scene.",

        dict(
            title          = "Garba Raas Night",
            subtitle       = "Nine Nights of Dandiya & Dance",
            date           = "October 2–10, 2026",
            venue          = "College Ground, SVNIT Surat",
            organizer      = "Gujarat Cultural Committee",
            accent_color   = "#FF6F00",
            style          = "elegant",
            text_position  = "bottom",
            scrim          = False,
        ),
    ),

    # ── Coding Hackathon ────────────────────────────────────────────────────
    (
        "coding_hackathon",

        "campus_ai_poster  Dark futuristic hackathon coding environment background.  "
        "Multiple holographic screens floating in 3-D space with scrolling green "
        "terminal animations and binary rain patterns.  "
        "Glowing cyan circuit-board traces on a deep black background.  "
        "Keyboard and laptop silhouettes lit from below by a cool blue glow.  "
        "High-contrast, ultra-sharp, cyberpunk aesthetic.  "
        "No text, no readable characters, no words anywhere in the scene.",

        dict(
            title          = "Code-a-thon 4.0",
            subtitle       = "36 Hours.  No Sleep.  Pure Code.",
            date           = "January 18–19, 2026",
            venue          = "CS Lab 301, IIT Bombay",
            organizer      = "WnCC & DevClub",
            accent_color   = "#00E676",
            style          = "bold",
            text_position  = "bottom",
            scrim          = True,
        ),
    ),

    # ── Blood Donation Camp ─────────────────────────────────────────────────
    (
        "blood_donation",

        "campus_ai_poster  Warm heartfelt blood donation awareness background.  "
        "A large red blood drop with a heartbeat ECG line running through its center.  "
        "Clean white and soft crimson minimalist medical composition.  "
        "Two open hands gently cupping the drop from below.  "
        "Gentle radial light bloom.  Compassionate, hopeful healthcare aesthetic.  "
        "No text, no words, no labels in the scene.",

        dict(
            title          = "Donate Blood, Save Lives",
            subtitle       = "NSS Blood Donation Camp",
            date           = "March 5, 2026  •  9 AM – 4 PM",
            venue          = "Health Centre, NIT Trichy",
            organizer      = "NSS Unit & Red Cross Society",
            accent_color   = "#D32F2F",
            style          = "modern",
            text_position  = "bottom",
            scrim          = False,
        ),
    ),

    # ── Farewell ────────────────────────────────────────────────────────────
    (
        "farewell",

        "campus_ai_poster  Sentimental farewell celebration background.  "
        "Golden fairy lights strung across a twilight campus courtyard.  "
        "Graduation caps thrown upward against a warm amber-peach sunset sky.  "
        "Bokeh spheres in champagne gold and soft peach.  "
        "Petals falling slowly through the air from above.  "
        "Nostalgic, bittersweet, and celebratory mood.  Warm film-grain texture.  "
        "No text, no banners, no words in the scene.",

        dict(
            title          = "Alvida — Farewell 2026",
            subtitle       = "For the Batch That Made It Legendary",
            date           = "May 15, 2026  •  5 PM",
            venue          = "Main Auditorium, NSUT",
            organizer      = "Third Year Organizing Committee",
            accent_color   = "#FFD54F",
            style          = "elegant",
            text_position  = "bottom",
            scrim          = False,
        ),
    ),

    # ── Annual Cultural Fest ─────────────────────────────────────────────────
    (
        "annual_fest",

        "campus_ai_poster  Epic grand annual college cultural fest background.  "
        "Massive paint-splash explosion in rainbow neon colours filling the entire frame.  "
        "Fireworks bursting above a packed outdoor main stage.  "
        "Laser beams sweeping over a roaring silhouette crowd.  "
        "Smoke machines and confetti cannons firing simultaneously.  "
        "Maximum energy, blockbuster festival scale, ultra-vivid colour grading.  "
        "Absolutely no text, no stage signs, no banners, no readable characters.",

        dict(
            title          = "MOKSHA 2026",
            subtitle       = "The Biggest College Fest in India",
            date           = "February 14–16, 2026",
            venue          = "NSUT Main Campus, Dwarka",
            organizer      = "Moksha Organizing Committee",
            accent_color   = "#FF1744",
            style          = "bold",
            text_position  = "bottom",
            scrim          = True,
        ),
    ),

    # ── Robotics Competition ─────────────────────────────────────────────────
    (
        "robotics_competition",

        "campus_ai_poster  Futuristic robotics competition arena background.  "
        "A sleek industrial robot arm mid-motion under dramatic blue-white spotlights.  "
        "Metallic gears, pistons, and carbon-fibre surface textures.  "
        "Electric sparks flying off welded joints.  Dark smoke and industrial haze.  "
        "High-contrast dramatic lighting, mechanical precision aesthetic.  "
        "No text, no labels, no signage anywhere in the scene.",

        dict(
            title          = "RoboWars 2026",
            subtitle       = "Build It.  Break It.  Win It.",
            date           = "March 22, 2026",
            venue          = "Innovation Hub, BITS Pilani",
            organizer      = "Robotics & Automation Society",
            accent_color   = "#40C4FF",
            style          = "modern",
            text_position  = "bottom",
            scrim          = True,
        ),
    ),

    # ── Standup Comedy Night ─────────────────────────────────────────────────
    (
        "standup_comedy",

        "campus_ai_poster  Moody open-mic comedy night stage background.  "
        "Single golden spotlight cone hitting a lone microphone stand centre stage.  "
        "Deep maroon velvet curtains framing the wings on both sides.  "
        "Brick wall texture visible at the back — classic comedy club look.  "
        "Warm amber footlights and a faint laughing crowd silhouette at the bottom.  "
        "Intimate, atmospheric, slightly gritty feel.  "
        "No text, no words, no chalk board writing, no signs anywhere.",

        dict(
            title          = "Laugh Riot 2026",
            subtitle       = "Open Mic Comedy Night",
            date           = "April 5, 2026  •  7 PM",
            venue          = "Black Box Theatre, Miranda House",
            organizer      = "The Comedy Collective",
            accent_color   = "#FFAB40",
            style          = "modern",
            text_position  = "top",      # mic + spotlight fill center/bottom
            scrim          = True,
        ),
    ),

    # ── Diwali Celebration ───────────────────────────────────────────────────
    (
        "diwali",

        "campus_ai_poster  Magical Diwali festival night background.  "
        "Hundreds of glowing earthen diyas arranged in concentric circles on dark stone.  "
        "Fireworks bursting in gold, silver, and emerald green overhead.  "
        "Intricate rangoli patterns in vibrant pink, blue, and orange surrounding the diyas.  "
        "Warm golden bokeh light spheres floating throughout.  "
        "Festive, divine, deeply traditional Indian atmosphere.  "
        "No text, no words, no labels anywhere in the scene.",

        dict(
            title          = "Diwali Utsav 2026",
            subtitle       = "Festival of Lights on Campus",
            date           = "October 20, 2026  •  6 PM",
            venue          = "Central Lawn, IIT Delhi",
            organizer      = "Cultural Committee & NSS",
            accent_color   = "#FFD700",
            style          = "elegant",
            text_position  = "top",      # rangoli / diyas fill bottom beautifully
            scrim          = False,
        ),
    ),

]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _load_pipeline(base_id: str, lora_dir: str, lora_file: str):
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

    print("  Loading SDXL base model ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_id,
        torch_dtype     = torch.float16,
        variant         = "fp16",
        use_safetensors = True,
    ).to("cuda")

    # DPM++ 2M Karras — sharper outputs, better prompt adherence than DDPM
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas = True,
    )

    lora_path = os.path.join(lora_dir, lora_file)
    if os.path.exists(lora_path):
        pipe.load_lora_weights(lora_dir, weight_name=lora_file, adapter_name="campus_poster")
        pipe.set_adapters(["campus_poster"], adapter_weights=[1.0])
        print(f"  LoRA loaded  →  {lora_path}")
    else:
        print(f"  WARNING: LoRA not found at {lora_path}  —  using base SDXL only")

    return pipe


def generate_posters() -> None:
    out_dir   = Path("output/test_generations")
    lora_dir  = "models/sdxl/checkpoints/campus_ai_poster_sdxl_phase3"
    lora_file = "campus_ai_poster_sdxl_phase3.safetensors"
    base_id   = "stabilityai/stable-diffusion-xl-base-1.0"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CAMPUS AI  —  TWO-STAGE POSTER PIPELINE")
    print("=" * 60)

    print("\n[Stage 0]  Downloading / verifying fonts ...")
    ensure_fonts()

    print("\n[Stage 1]  Loading SDXL + Campus AI LoRA ...")
    pipe = _load_pipeline(base_id, lora_dir, lora_file)

    print(f"\n[Stage 2]  Generating {len(POSTERS)} posters ...\n")

    for slug, artwork_prompt, text_cfg in POSTERS:
        label = slug.upper().replace("_", " ")
        print(f"  🎨  {label}")

        artwork = pipe(
            artwork_prompt,
            negative_prompt     = _NEG,
            num_inference_steps = 35,   # +5 steps for cleaner detail
            guidance_scale      = 7.5,  # stronger negative adherence — kills hallucinated text
        ).images[0]

        artwork_path = out_dir / f"{slug}_artwork.png"
        artwork.save(artwork_path)
        print(f"       artwork  →  {artwork_path}")

        final = composite_poster(artwork, **text_cfg)
        poster_path = out_dir / f"{slug}_poster.png"
        final.save(poster_path)
        print(f"       poster   →  {poster_path}\n")

    del pipe
    torch.cuda.empty_cache()

    print("=" * 60)
    print(f"  ✅  Done.  All outputs in  {out_dir}/")
    print("       *_artwork.png  →  raw SDXL art, no text")
    print("       *_poster.png   →  final composited poster")
    print("=" * 60)


if __name__ == "__main__":
    generate_posters()