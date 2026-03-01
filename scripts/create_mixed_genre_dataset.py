#!/usr/bin/env python3
"""
Campus-AI by CounciL — Phase 4 Mixed-Genre Dataset Creator  v2
================================================================
Generates a massive, EQUALLY DISTRIBUTED cross-genre tuning dataset.

Key features:
  - Balanced output: every subcategory contributes the SAME number of
    samples, preventing dominant categories from drowning smaller ones.
  - Auto mixes-per-image: automatically calculated so that smaller
    categories get MORE mixes and larger ones get fewer, resulting
    in equal output volume.
  - Caption dropout: randomly drops caption segments (subtitle, date,
    venue, style description) to teach the model unconditional and
    partially-conditioned generation.
  - 55 subcategory style profiles with rich visual descriptions.

Output:  data/tuning-2/

Usage:
    python scripts/create_mixed_genre_dataset.py --source data/train --output data/tuning-2 --target-per-cat 3000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 55 subcategory style profiles
# ---------------------------------------------------------------------------

SUBCATEGORY_STYLES: dict[str, str] = {
    # ── college_events ──
    "alumni_reunion": (
        "nostalgic warm-toned gathering atmosphere, vintage photo collage borders, "
        "golden hour campus silhouettes, university crest motifs, old book texture overlays, "
        "warm sepia #D2691E to cream #FFFDD0 gradient"
    ),
    "annual_fest": (
        "explosive celebration energy, stadium spotlight beams, confetti particle burst, "
        "massive crowd silhouette panorama, concert stage volumetric fog, "
        "electric magenta #FF00FF to deep violet #4B0082 neon gradients"
    ),
    "clubs_recruitment": (
        "modern recruitment poster aesthetic, bold geometric grid sections, "
        "speech bubble and megaphone motifs, energetic diagonal compositions, "
        "clean navy #001F3F with bright coral #FF6B6B accents"
    ),
    "farewell": (
        "soft emotional twilight atmosphere, lantern glow bokeh, "
        "silhouette figures walking toward sunset horizon, scattered rose petals, "
        "gentle lavender #E6E6FA to warm peach #FFDAB9 dreamy gradient"
    ),
    "freshers": (
        "vibrant welcome energy, colorful bunting flag rows, "
        "open gate and pathway leading forward, fresh green foliage, "
        "bright teal #20B2AA to sunshine yellow #FFD700 uplifting palette"
    ),
    "graduation": (
        "ceremonial grandeur, mortarboard caps in mid-throw freeze, "
        "rolled diploma with ribbon and wax seal, ornate podium spotlight, "
        "regal navy #000080 and gold #FFD700 formal palette"
    ),
    # ── cultural_fest ──
    "art_exhibition": (
        "gallery white-wall minimalism with dramatic spot lighting, "
        "paintbrush stroke texture overlays, gilt ornate frames, "
        "neutral gallery cream #F5F5DC with bold primary accent splashes"
    ),
    "dance": (
        "frozen dynamic dance pose mid-movement, flowing fabric trails, "
        "stage spotlight rim lighting on silhouette, motion blur energy streaks, "
        "deep burgundy #800020 to gold #FFD700 dramatic palette"
    ),
    "drama_theatre": (
        "theatrical stage with red velvet curtain drapes, footlight row glow, "
        "dramatic chiaroscuro spotlight, comedy-tragedy mask silhouettes, "
        "rich crimson #DC143C to jet black #0A0A0A stage palette"
    ),
    "fashion_show": (
        "runway catwalk perspective with model silhouettes, "
        "camera flash burst effects, elegant fabric drape textures, "
        "glossy luxe black #111111 with rose gold #B76E79 accents"
    ),
    "literary": (
        "open book with pages mid-flutter, quill and ink motifs, "
        "library shelf bokeh background, vintage paper texture overlay, "
        "warm brown #8B4513 to ivory #FFFFF0 scholarly palette"
    ),
    "music": (
        "concert stage volumetric laser beams, guitar and microphone silhouettes, "
        "audio waveform visualization patterns, smoke machine haze, "
        "electric blue #00BFFF to purple #8B00FF rock palette"
    ),
    "standup_comedy": (
        "brick wall background with single spotlight circle, "
        "microphone stand center stage, warm intimate club lighting, "
        "warm amber #FFBF00 spotlight on dark charcoal #36454F"
    ),
    # ── entertainment ──
    "food_fest": (
        "vibrant food stall banner aesthetic, steam and aroma curl wisps, "
        "colorful spice powder patterns, rustic wooden table texture, "
        "warm saffron #FF9933 to chili red #C41E3A appetite palette"
    ),
    "gaming": (
        "pixel art collage borders, controller and joystick motifs, "
        "retro arcade cabinet glow, pixelated particle effects, "
        "neon green #39FF14 to electric purple #BF00FF gamer palette"
    ),
    "movie_night": (
        "cinema projector beam cutting through darkness, film strip borders, "
        "popcorn bucket and clapperboard motifs, theater seat silhouettes, "
        "warm projector amber #FFB347 on cinema black #1A1A1A"
    ),
    # ── festivals ──
    "christmas": (
        "pine wreath borders with red ribbon bows, snowflake particle overlay, "
        "candy cane stripe accents, warm fireplace glow, "
        "classic red #CC0000 and hunter green #355E3B with gold #FFD700"
    ),
    "diwali": (
        "rows of lit diyas with flickering flames, intricate rangoli mandala patterns, "
        "marigold garland borders, sparkler trail long-exposure effects, "
        "rich saffron #FF9933 to deep maroon #800000 with gold #FFD700"
    ),
    "durga_puja": (
        "ornate pandal archway with fabric drapes, dhol drum motifs, "
        "sindoor red accents, lotus flower patterns, "
        "vermillion #E34234 to gold #FFD700 on cream #FFFDD0"
    ),
    "eid": (
        "crescent moon and star motifs, mosque silhouette against twilight sky, "
        "intricate geometric arabesque patterns, lantern glow, "
        "emerald green #046307 to gold #FFD700 on deep blue #00008B"
    ),
    "ganesh_chaturthi": (
        "modak and laddu offering motifs, lotus throne, "
        "marigold and hibiscus garland borders, traditional bell silhouettes, "
        "saffron #FF9933 to vermillion #E34234 with gold accents"
    ),
    "holi": (
        "explosive color powder burst mid-air freeze, rainbow paint splatter, "
        "water balloon splash effects, joyful crowd silhouettes, "
        "full rainbow spectrum palette from magenta #FF00FF to cyan #00FFFF"
    ),
    "independence_republic": (
        "tricolor saffron-white-green gradient flow, Ashoka chakra motif, "
        "flag fabric ripple texture, soldier silhouette salute, "
        "saffron #FF9933 to white #FFFFFF to green #138808"
    ),
    "navratri_garba": (
        "dandiya stick cross patterns, embroidered chaniya choli fabric textures, "
        "colorful mirror work reflections, circular garba dance formation, "
        "vibrant magenta #FF00FF to orange #FF6600 to yellow #FFD700"
    ),
    "new_year": (
        "midnight clock striking twelve, champagne glass clink freeze, "
        "golden firework burst patterns, streamers and confetti rain, "
        "midnight blue #191970 to champagne gold #F7E7CE"
    ),
    "onam": (
        "pookalam floral carpet circular patterns, vallam kali boat silhouettes, "
        "banana leaf decoration, traditional lamp nilavilakku, "
        "golden yellow #FFBF00 to deep green #006400 Kerala palette"
    ),
    "pongal_sankranti": (
        "traditional kolam patterns, sugarcane bundle motifs, "
        "overflowing pot with sunrise backdrop, kite festival sky, "
        "warm turmeric yellow #FFD300 to terracotta #E2725B"
    ),
    # ── social ──
    "awareness": (
        "megaphone and speech bubble motifs, raised fist solidarity symbol, "
        "world globe cradled in open palms, awareness ribbon loop, "
        "strong blue #0066CC to white #FFFFFF clean impact palette"
    ),
    "blood_donation": (
        "blood drop motifs, heartbeat pulse line, "
        "medical cross symbol, helping hands reaching together, "
        "life red #FF0000 to warm coral #FF6B6B on clean white #FFFFFF"
    ),
    "charity": (
        "warm hands reaching together toward central heart symbol, "
        "tree of life with spreading branches and green leaves, "
        "dove of peace in flight, community circle silhouettes, "
        "warm sunrise orange #FF8C00 to hopeful green #228B22"
    ),
    "environment": (
        "lush green canopy foliage borders, earth globe with leaf crown, "
        "recycling arrow cycle motifs, clean water droplet reflections, "
        "deep emerald #006400 to sky blue #87CEEB eco palette"
    ),
    # ── sports ──
    "athletics": (
        "running track vanishing point perspective, sprinter frozen mid-stride, "
        "stadium floodlight beams, stopwatch and medal motifs, "
        "fierce red #DC143C to olympic gold #FFD700"
    ),
    "badminton_tennis": (
        "shuttlecock and racket silhouette, net grid pattern divider, "
        "court lines perspective, action freeze mid-serve, "
        "court green #3CB371 to white #FFFFFF with yellow #FFD700"
    ),
    "basketball": (
        "basketball court hardwood texture, slam dunk silhouette freeze, "
        "hoop and backboard motifs, scoreboard glow, "
        "orange #FF8C00 to black #111111 with white #FFFFFF court lines"
    ),
    "cricket": (
        "cricket pitch green stripes, bat and ball silhouette, "
        "stadium floodlight night-match atmosphere, wicket stumps motif, "
        "rich green #006400 to white #FFFFFF cricket palette"
    ),
    "esports": (
        "gaming monitor setup glow, keyboard RGB lighting streaks, "
        "virtual arena digital landscape, pixel particle effects, "
        "neon cyan #00FFFF to deep purple #4B0082 RGB palette"
    ),
    "football": (
        "football pitch bird's eye perspective, goal net pattern overlay, "
        "dynamic kick silhouette, stadium crowd wave, "
        "pitch green #228B22 to floodlight white with black #000000 accents"
    ),
    "kabaddi_kho": (
        "traditional Indian wrestling arena, mud texture ground, "
        "dynamic tackle silhouette, tribal pattern borders, "
        "earthy brown #8B4513 to saffron #FF9933 traditional palette"
    ),
    "yoga_fitness": (
        "serene sunrise silhouette pose, lotus flower motifs, "
        "smooth mountain landscape backdrop, floating leaf particles, "
        "calming lavender #B57EDC to soft mint #98FFB3 wellness palette"
    ),
    # ── styles ──
    "3d_futuristic": (
        "sleek 3D geometric floating shapes with metallic reflections, "
        "holographic iridescent material surfaces, depth-of-field parallax layers, "
        "chrome #C0C0C0 to electric cyan #00CED1 futuristic palette"
    ),
    "dark_theme": (
        "deep matte black background with subtle noise grain texture, "
        "single concentrated spotlight pool of warm light, thin gold #FFD700 accent lines, "
        "luxurious dark atmosphere, charcoal #1A1A1A to jet #0A0A0A"
    ),
    "gradient": (
        "smooth multi-step color gradient flowing diagonally across frame, "
        "glass morphism frosted panels floating at varied depths, "
        "harmonious color transitions from warm to cool tones"
    ),
    "illustration": (
        "hand-drawn illustration style with visible brush texture, "
        "bold outline strokes, flat color fills with subtle shading, "
        "artistic palette with muted earth tones and bright pop accents"
    ),
    "minimalist": (
        "vast generous negative space breathing room, single bold design element, "
        "hairline geometric rules as visual anchors, surgical precision layout, "
        "muted ivory #FFFFF0 or warm charcoal #36454F with one accent stripe"
    ),
    "neon_glow": (
        "neon tube lighting outlines glowing against dark background, "
        "electric glow halos with light bleed and bloom effects, "
        "synthwave grid horizon vanishing point, "
        "hot pink #FF1493 to electric blue #00BFFF neon palette"
    ),
    "retro_vintage": (
        "aged paper and cardboard texture with worn edges, "
        "halftone dot pattern overlays, bold slab serif typography silhouettes, "
        "faded postcard color palette of mustard #FFDB58 to burnt sienna #E97451"
    ),
    "typography": (
        "abstract decorative typographic layout composition shapes (no readable text), "
        "letterform-inspired geometric patterns at various scales, "
        "ink splatter and brush stroke textures, monochrome black #000000 on cream #FFFDD0"
    ),
    "watercolor": (
        "soft watercolor paint wash backgrounds with organic bleeding edges, "
        "wet-on-wet diffusion effects, visible paper grain texture, "
        "pastel palette of soft pink #FFB6C1 and dusty sage #B2BEB5"
    ),
    # ── tech_fest ──
    "ai_ml": (
        "neural network node-and-edge visualization, data matrix rain backdrop, "
        "brain silhouette with circuit pattern overlay, GPU chip motifs, "
        "deep blue #001F3F to electric cyan #00CED1 AI palette"
    ),
    "coding_competition": (
        "code syntax highlighting color blocks, terminal green-on-black aesthetic, "
        "bracket and semicolon decorative patterns, binary cascade, "
        "terminal green #00FF41 to dark screen #0D1117"
    ),
    "cybersecurity": (
        "digital lock and shield motifs, binary code rain matrix, "
        "warning triangle alert elements, fingerprint scan overlay, "
        "dark black #0A0A0A to warning red #FF0000 to electric green #00FF00"
    ),
    "hackathon": (
        "laptop glow in dark room atmosphere, coffee cup steam wisps, "
        "code window overlapping panels, late-night skyline, "
        "midnight blue #191970 to bright orange #FF6600 hacker palette"
    ),
    "robotics": (
        "mechanical gear and cog motifs, robotic arm silhouette, "
        "metallic reflective surfaces, blueprint grid background, "
        "industrial silver #C0C0C0 to electric blue #4169E1"
    ),
    "web_app_dev": (
        "browser window frame mockup composition, responsive device outlines, "
        "colorful app icon grid pattern, code bracket decorations, "
        "modern indigo #4B0082 to fresh teal #20B2AA dev palette"
    ),
    # ── workshops ──
    "business": (
        "corporate presentation screen glow, handshake silhouette motifs, "
        "ascending chart graph pattern, briefcase and tie elements, "
        "professional slate #708090 to gold #FFD700 executive palette"
    ),
    "coding": (
        "IDE editor syntax color blocks stacked, version control branch visualization, "
        "developer desk setup with multiple screens, "
        "dark editor #1E1E1E to syntax green #A8FF60"
    ),
    "conference": (
        "large auditorium perspective with stage spotlights, "
        "microphone and podium silhouette, row of chairs, "
        "deep navy #001F3F to warm spotlight amber #FFB347"
    ),
    "design": (
        "color wheel and swatch palette grids, Bezier curve tool motifs, "
        "layered artboard composition, gradient mesh patterns, "
        "creative purple #800080 to fresh coral #FF6B6B"
    ),
    "placement": (
        "professional handshake silhouette, resume document stack, "
        "corporate building skyline, ascending career path steps, "
        "confident navy #000080 to success green #228B22"
    ),
    "seminar": (
        "lecture hall podium with focused spotlight beam, "
        "raised hand audience silhouettes, whiteboard calculations, "
        "academic blue #4169E1 to warm cream #FFFDD0"
    ),
    "soft_skills": (
        "team collaboration circle of silhouettes, speech bubble clouds, "
        "handshake and thumbs-up motifs, growth plant metaphor, "
        "friendly orange #FF8C00 to calm teal #20B2AA people palette"
    ),
}

# Design quality boosters
DESIGN_BOOSTERS = [
    "masterpiece, best quality, ultra-detailed, 8k resolution",
    "professional color grading, sharp focus, cinematic composition",
    "volumetric lighting, ray tracing, photorealistic rendering",
    "award-winning visual design, studio quality",
    "dramatic atmosphere, museum-quality artwork",
    "editorial quality composition, magazine cover worthy",
    "pixel-perfect detail, razor-sharp clarity",
    "expert-level visual hierarchy with clear focal point",
    "premium design with intentional negative space for overlay elements",
    "glossy high-production-value aesthetic with polished finishing",
    "rich layered depth with foreground, midground, background separation",
    "balanced color harmony with professionally curated palette",
    "smooth gradients with no banding, print-quality color transitions",
    "luxurious premium feel with attention to micro-detail textures",
]

NO_TEXT_SUFFIX = (
    "CRITICAL: absolutely no text, no words, no letters, no writing, "
    "no typography, no signs, no labels, no captions anywhere in the entire scene. "
    "All surfaces must be smooth, clean, and completely free of character-like shapes. "
    "Replace any area where text might appear with abstract visual patterns instead."
)

# Caption dropout strategies (helps model learn partial conditioning)
DROPOUT_MODES = [
    "full",              # Keep full mixed caption
    "full",              # Double weight for full captions
    "no_style_b",        # Drop second style blend
    "no_boosters",       # Drop quality boosters
    "minimal",           # Ultra-short caption (subject + trigger only)
    "subject_only",      # Just the subject, no style info
    "empty",             # Empty caption (unconditional training)
    "style_only",        # Only style descriptors, no subject
]


def apply_caption_dropout(
    trigger: str,
    subject: str,
    style_a_desc: str,
    style_b_desc: str,
    boosters: str,
    style_a_name: str,
    style_b_name: str,
) -> str:
    """Apply caption dropout strategy for training robustness."""
    mode = random.choice(DROPOUT_MODES)

    if mode == "empty":
        # 12.5% chance: fully empty caption → unconditional generation
        return ""

    if mode == "subject_only":
        # Just trigger + subject
        return f"{trigger}  {subject}. {NO_TEXT_SUFFIX}"

    if mode == "minimal":
        # Trigger + short subject
        short = subject.split(".")[0].strip()
        return f"{trigger}  {short}. {NO_TEXT_SUFFIX}"

    if mode == "style_only":
        # No subject, only styles
        return (
            f"{trigger}  Professional event poster background artwork. "
            f"{style_a_desc}. {NO_TEXT_SUFFIX}"
        )

    if mode == "no_style_b":
        # Drop the second genre blend
        return (
            f"{trigger}  {subject}. "
            f"Visual style: {style_a_name.replace('_', ' ')} aesthetic: {style_a_desc}. "
            f"{boosters}. {NO_TEXT_SUFFIX}"
        )

    if mode == "no_boosters":
        # Drop quality boosters
        return (
            f"{trigger}  {subject}. "
            f"Blending {style_a_name.replace('_', ' ')} and "
            f"{style_b_name.replace('_', ' ')}: {style_a_desc}. "
            f"Interleaved with {style_b_desc}. {NO_TEXT_SUFFIX}"
        )

    # "full" mode — complete caption
    return (
        f"{trigger}  {subject}. "
        f"Visual style blending {style_a_name.replace('_', ' ')} and "
        f"{style_b_name.replace('_', ' ')} aesthetics: {style_a_desc}. "
        f"Interleaved with {style_b_desc}. "
        f"{boosters}. {NO_TEXT_SUFFIX}"
    )


def find_captioned_images(source_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Find image-caption pairs grouped by subcategory (recursive)."""
    categories: dict[str, list[tuple[Path, Path]]] = {}

    for parent in sorted(source_dir.iterdir()):
        if not parent.is_dir():
            continue
        for sub in sorted(parent.iterdir()):
            if not sub.is_dir():
                continue
            cat_name = sub.name
            pairs = []
            for f in sub.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    txt = f.with_suffix(".txt")
                    if txt.exists():
                        pairs.append((f, txt))
            if pairs:
                categories[cat_name] = pairs

    return categories


def create_mixed_dataset(
    source_dir: Path,
    output_dir: Path,
    target_per_cat: int = 3000,
    seed: int = 42,
) -> None:
    """Create the Phase 4 mixed-genre dataset with EQUAL distribution.

    Every subcategory will produce exactly `target_per_cat` output samples,
    regardless of how many source images it has. Smaller categories get
    more mixes per image; larger ones fewer.
    """
    random.seed(seed)

    print("=" * 66)
    print("  Campus-AI by CounciL — Phase 4 Mixed-Genre Dataset  v2")
    print("  Equal Distribution + Caption Dropout")
    print("=" * 66)

    categories = find_captioned_images(source_dir)
    if not categories:
        print("ERROR: No captioned images found!")
        return

    total_source = sum(len(v) for v in categories.values())
    cat_names = sorted(categories.keys())
    n_cats = len(cat_names)

    print(f"\n  Source:  {source_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Subcategories found:  {n_cats}")
    print(f"  Total source images:  {total_source:,}")
    print(f"  Target per category:  {target_per_cat:,}")
    print(f"  Expected output:      ~{n_cats * target_per_cat:,} samples")
    print()

    # Show per-category plan
    print(f"  {'Category':<30s}  {'Source':>7s}  {'Mixes/img':>9s}  {'Output':>7s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*7}")

    cat_plans: list[tuple[str, list[tuple[Path, Path]], int]] = []
    for cat_name in cat_names:
        pairs = categories[cat_name]
        n_source = len(pairs)
        # Auto-calculate mixes per image to hit target
        mixes = max(1, target_per_cat // n_source)
        actual_output = n_source * mixes
        print(f"  {cat_name:<30s}  {n_source:>7,}  {mixes:>9}  {actual_output:>7,}")
        cat_plans.append((cat_name, pairs, mixes))

    total_planned = sum(len(p) * m for _, p, m in cat_plans)
    print(f"\n  Total planned output: {total_planned:,} samples")
    print(f"  Caption dropout: ~12.5% empty, ~12.5% minimal, ~75% full/partial\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_created = 0
    mix_id = 0

    for cat_name, pairs, mixes_per_img in cat_plans:
        other_cats = [c for c in cat_names if c != cat_name]
        print(f"  Processing {cat_name} ({len(pairs):,} images x {mixes_per_img} mixes)...")

        for img_path, txt_path in pairs:
            original_caption = txt_path.read_text(encoding="utf-8", errors="ignore").strip()

            # Extract subject from original caption
            cap_parts = original_caption.split(".")
            subject = ". ".join(cap_parts[:2]).strip()
            if not subject or len(subject) < 20:
                subject = "a stunning professional event poster background artwork"

            # Pick random partner subcategories for each mix
            partners = random.choices(other_cats, k=mixes_per_img)

            for partner in partners:
                style_a_desc = SUBCATEGORY_STYLES.get(cat_name, "professional design")
                style_b_desc = SUBCATEGORY_STYLES.get(partner, "modern visual elements")
                boosters = ", ".join(random.sample(DESIGN_BOOSTERS, 3))

                # Apply caption dropout
                caption = apply_caption_dropout(
                    trigger="campusai poster",
                    subject=subject,
                    style_a_desc=style_a_desc,
                    style_b_desc=style_b_desc,
                    boosters=boosters,
                    style_a_name=cat_name,
                    style_b_name=partner,
                )

                ext = img_path.suffix
                out_img = output_dir / f"mix_{mix_id:07d}{ext}"
                out_txt = output_dir / f"mix_{mix_id:07d}.txt"

                try:
                    shutil.copy2(img_path, out_img)
                    out_txt.write_text(caption, encoding="utf-8")
                    total_created += 1
                    mix_id += 1
                except Exception as e:
                    print(f"    Warning: {e}")

        print(f"    Done. Running total: {total_created:,}")

    print(f"\n{'=' * 66}")
    print(f"  PHASE 4 DATASET COMPLETE!")
    print(f"  Total samples:      {total_created:,}")
    print(f"  Total files:        {total_created * 2:,} (image + caption)")
    print(f"  Output directory:   {output_dir}")
    print(f"  Categories:         {n_cats} (equally distributed)")
    print(f"  Caption dropout:    enabled (empty/minimal/partial/full)")
    print(f"\n  Next: update training config and start Phase 4 fine-tuning")
    print(f"{'=' * 66}")


def main():
    parser = argparse.ArgumentParser(
        description="Campus-AI Phase 4 — Mixed-Genre Dataset (Equal Distribution)"
    )
    parser.add_argument(
        "--source", type=str, default="data/train",
        help="Source directory with subcategorized captioned images"
    )
    parser.add_argument(
        "--output", type=str, default="data/tuning-2",
        help="Output directory (default: data/tuning-2)"
    )
    parser.add_argument(
        "--target-per-cat", type=int, default=3000,
        help="Target samples per subcategory for equal distribution (default: 3000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"Error: source directory '{source}' does not exist.")
        return

    create_mixed_dataset(source, output, args.target_per_cat, args.seed)


if __name__ == "__main__":
    main()
