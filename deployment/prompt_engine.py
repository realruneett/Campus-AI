#!/usr/bin/env python3
"""
CampusGen AI – Prompt Engine
Uses Groq Llama 3.3 70B to transform simple event descriptions
into detailed, high-quality image generation prompts.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
TRIGGER_WORD = "campus_ai_poster"

# ─────────────────────────────────────────────────────────────────────────────
# System Prompts (per mode)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_TEXT2IMG = f"""You are a world-class poster design expert specializing in Indian college event posters. Given an event description, generate a detailed, cinematic image generation prompt.

Your prompt MUST include:
1. Composition & layout (center-stage focal point, text hierarchy areas, decorative borders)
2. Color palette (specific hex-inspired descriptions, gradients, mood)
3. Typography style (bold sans-serif, elegant serif, handwritten, neon glow)
4. Background elements (abstract patterns, venue imagery, thematic textures)
5. Lighting & atmosphere (dramatic spotlights, warm glow, neon reflections)
6. Cultural/thematic motifs appropriate to the event

RULES:
- ALWAYS start with "{TRIGGER_WORD}"
- Keep under 200 words
- Be extremely specific about visual details
- For Indian events, include culturally authentic motifs (rangoli, diyas, mehendi, etc.)
- Describe the poster as a finished design, not a scene
- Output ONLY the prompt, nothing else"""

SYSTEM_IMG2IMG = f"""You are a poster restyling expert. Given a description of how the user wants to transform an existing poster, generate a detailed prompt describing the desired output.

Focus on:
1. The new visual style to apply
2. Color palette changes
3. Typography modifications
4. Atmosphere and mood shifts
5. Elements to preserve vs. change

RULES:
- ALWAYS start with "{TRIGGER_WORD}"
- Keep under 150 words
- Describe the desired RESULT, not the process
- Output ONLY the prompt"""

SYSTEM_INPAINT = f"""You are a poster editing expert. Given a description of what region the user wants to regenerate on a poster, generate a prompt describing what should fill that region.

Focus on:
1. What visual elements should appear in the masked area
2. Style consistency with the surrounding poster
3. Color and lighting continuity

RULES:
- ALWAYS start with "{TRIGGER_WORD}"
- Keep under 100 words
- Be specific about what fills the masked area
- Output ONLY the prompt"""

# ─────────────────────────────────────────────────────────────────────────────
# Style Descriptions
# ─────────────────────────────────────────────────────────────────────────────
STYLE_MAP = {
    "Vibrant and Energetic": (
        "vibrant energetic colors, electric gradients from magenta to cyan, "
        "dynamic diagonal composition, bold sans-serif typography, "
        "particle effects and light streaks"
    ),
    "Elegant and Professional": (
        "elegant professional design, deep navy and gold color scheme, "
        "clean serif typography, subtle gradient backgrounds, "
        "refined geometric accents"
    ),
    "Modern Minimalist": (
        "modern minimalist design, generous white space, "
        "monochromatic palette with single accent color, "
        "thin geometric lines, clean sans-serif typography"
    ),
    "Traditional Indian": (
        "traditional Indian design, warm gold saffron and deep red palette, "
        "ornate mandala borders, rangoli-inspired patterns, "
        "decorative Devanagari-style typography, paisley motifs"
    ),
    "Tech-Futuristic": (
        "futuristic cyberpunk tech design, dark background with neon glow, "
        "holographic elements, circuit board patterns, "
        "glitch text effects, electric blue and purple neon"
    ),
    "Artistic and Creative": (
        "artistic watercolor splash design, fluid organic shapes, "
        "hand-painted texture, eclectic mixed typography, "
        "ink splatter accents, warm earthy tones"
    ),
    "Neon Glow": (
        "neon glow poster design, deep black background, "
        "vivid neon tubes in pink cyan and yellow, "
        "reflective surfaces, urban night atmosphere, glow typography"
    ),
    "Retro Vintage": (
        "retro vintage poster design, distressed paper texture, "
        "muted warm color palette, bold block letters, "
        "halftone dot patterns, 70s inspired graphics"
    ),
    "Dark Premium": (
        "dark premium poster design, matte black with metallic gold accents, "
        "luxury typography, subtle emboss effects, "
        "dramatic lighting, high contrast minimal elements"
    ),
    "Gradient Modern": (
        "modern gradient poster, smooth multi-color gradient backgrounds, "
        "floating 3D geometric shapes, soft shadows, "
        "rounded sans-serif typography, glass morphism effects"
    ),
}

EVENT_TYPE_HINTS = {
    "Technical Fest": "coding symbols, circuit patterns, robotic elements, binary code, tech logos",
    "Cultural Event": "stage lights, dance silhouettes, musical instruments, spotlights, curtains",
    "Sports Tournament": "dynamic action poses, sports equipment, stadium lights, motion blur, trophy",
    "Workshop / Seminar": "whiteboard, notebooks, professional setting, light bulb icons, knowledge symbols",
    "College Fest": "college campus backdrop, festive decorations, diverse crowd silhouettes, confetti",
    "Diwali Celebration": "diyas, rangoli, fireworks, marigold garlands, Lord Ganesha motifs, sparklers",
    "Holi Festival": "color powder splashes, water balloons, vibrant rainbow, pichkari, crowd celebration",
    "Navratri / Garba": "dandiya sticks, ghagra choli silhouettes, Durga motifs, festive lights",
    "Ganesh Chaturthi": "Lord Ganesha, modak, marigold, mandap, festive procession elements",
    "Eid Celebration": "crescent moon and star, mosque silhouette, lanterns, arabesque patterns",
    "Christmas / New Year": "Christmas tree, snowflakes, countdown clock, fireworks, candy canes",
    "Club Recruitment": "diverse student silhouettes, creative tools, speech bubbles, join-us energy",
    "Academic Event": "graduation cap, books, podium, academic shields, scholarly elements",
    "Freshers / Farewell": "welcome banner, photo frames, nostalgic elements, stage performance",
    "Blood Donation": "red cross, heart, blood drop, helping hands, medical symbols",
    "Music Concert": "guitar, microphone, soundwaves, stage spotlights, crowd silhouettes",
    "Food Festival": "food illustrations, chef hat, spice bowls, colorful plates, steam",
    "Marathon / Fitness": "running silhouettes, finish line, stopwatch, sneakers, energy",
    "Other": "professional event design, modern layout, eye-catching visual elements",
}


def _call_groq(system_prompt: str, user_message: str) -> Optional[str]:
    """Make a Groq API call and return the response text."""
    if not GROQ_API_KEY:
        return None

    try:
        import requests

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.8,
                "max_tokens": 350,
                "top_p": 0.9,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return None


def _ensure_trigger(prompt: str) -> str:
    """Ensure the trigger word is at the start of the prompt."""
    if not prompt.lower().startswith(TRIGGER_WORD):
        prompt = f"{TRIGGER_WORD} {prompt}"
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_text2img_prompt(
    event_description: str,
    event_type: str = "Other",
    style: str = "Vibrant and Energetic",
) -> str:
    """Build a rich prompt for text-to-poster generation."""

    style_desc = STYLE_MAP.get(style, STYLE_MAP["Vibrant and Energetic"])
    event_hints = EVENT_TYPE_HINTS.get(event_type, EVENT_TYPE_HINTS["Other"])

    user_msg = (
        f"Create an image generation prompt for this event poster:\n"
        f"Event: {event_description}\n"
        f"Type: {event_type}\n"
        f"Style: {style}\n"
        f"Style hints: {style_desc}\n"
        f"Thematic elements: {event_hints}\n"
    )

    result = _call_groq(SYSTEM_TEXT2IMG, user_msg)

    if result:
        return _ensure_trigger(result)

    # Fallback without LLM
    return _ensure_trigger(
        f"A professional {event_type.lower()} event poster for {event_description}. "
        f"{style_desc}. {event_hints}. "
        f"High quality typography, well-organized layout, eye-catching design."
    )


def build_img2img_prompt(
    transform_description: str,
    style: str = "Vibrant and Energetic",
) -> str:
    """Build a prompt for img2img poster transformation."""

    style_desc = STYLE_MAP.get(style, STYLE_MAP["Vibrant and Energetic"])

    user_msg = (
        f"Transform this poster with the following changes:\n"
        f"Changes: {transform_description}\n"
        f"New style: {style}\n"
        f"Style hints: {style_desc}\n"
    )

    result = _call_groq(SYSTEM_IMG2IMG, user_msg)

    if result:
        return _ensure_trigger(result)

    return _ensure_trigger(
        f"A transformed poster: {transform_description}. "
        f"{style_desc}. Professional quality, cohesive design."
    )


def build_inpaint_prompt(
    fill_description: str,
) -> str:
    """Build a prompt for inpainting a region of a poster."""

    user_msg = f"Fill the masked region with: {fill_description}"

    result = _call_groq(SYSTEM_INPAINT, user_msg)

    if result:
        return _ensure_trigger(result)

    return _ensure_trigger(
        f"{fill_description}. Seamless blending, consistent style."
    )
