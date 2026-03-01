#!/usr/bin/env python3
"""
Campus-AI — Prompt Engine  (v3.0 – Production)
===============================================
Transforms simple event descriptions into rich, SDXL-optimized artwork prompts.

Architecture:
  ┌─────────────────────────────┐
  │  User Input (event desc)    │
  └──────────┬──────────────────┘
             ▼
  ┌──────────────────────────────────────────────────┐
  │  1. Style lookup  →  STYLE_MAP[style]            │
  │  2. Event hints   →  EVENT_TYPE_HINTS[type]      │
  │  3. Composition   →  COMPOSITION_STRATEGIES[res]  │
  │  4. Lighting      →  LIGHTING_PRESETS[mood]       │
  │  5. Color harmony →  COLOR_HARMONIES              │
  │  6. Quality       →  QUALITY_SUFFIX               │
  ├──────────────────────────────────────────────────┤
  │  Groq Llama 3.3 70B assembles final prompt       │
  │  (or intelligent fallback if API unavailable)    │
  ├──────────────────────────────────────────────────┤
  │  _append_no_text()  → forces text suppression    │
  │  _append_quality()  → max visual fidelity        │
  │  NEGATIVE_PROMPT    → blocks 50+ text tokens     │
  └──────────────────────────────────────────────────┘

Key design:
  • Every generated prompt is artwork-only — ZERO text references.
  • NEGATIVE_PROMPT aggressively blocks garbled text from SDXL.
  • QUALITY_SUFFIX + QUALITY_BOOSTERS guarantee premium output.
  • Composition strategies adapt to portrait/landscape/square.
  • Lighting presets add cinematic atmosphere per event mood.
  • Color harmony suggestions ensure visually pleasing palettes.
"""

import os
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
TRIGGER_WORD = "campus_ai_poster"


# ═════════════════════════════════════════════════════════════════════════════
# NEGATIVE PROMPT — blocks ALL text/glyphs + quality defects from SDXL
# ═════════════════════════════════════════════════════════════════════════════
NEGATIVE_PROMPT = (
    # ─── Text suppression (80+ tokens, ultra-aggressive) ───
    "text, words, letters, numbers, digits, numerals, alphabet, characters, "
    "typography, typeface, fonts, serif, sans-serif, handwriting, cursive, "
    "captions, labels, watermark, signature, logo, emblem, insignia, stamp, "
    "banner, placard, signboard, billboard, marquee, nameplate, plaque, "
    "title, heading, subtitle, subheading, headline, caption, annotation, "
    "writing, written, inscriptions, engravings, calligraphy, monogram, "
    "illegible text, garbled text, gibberish, distorted words, random letters, "
    "fake words, misspelled, scrambled, corrupted characters, "
    "any readable content, any writing, any lettering, any glyphs, "
    "text overlay, text block, text banner, text area, word art, "
    "blurred text, faded text, backwards text, upside down text, "
    "symbols, punctuation marks, mathematical symbols, equations, "
    "printed text, typed text, stamped text, embossed text, "
    "graffiti, scribbles, doodle letters, scrawled words, "
    # ─── Quality defects ───
    "blurry, low quality, lowres, jpeg artifacts, compression artifacts, "
    "deformed, ugly, disfigured, oversaturated, underexposed, overexposed, "
    "bad anatomy, cropped, out of frame, duplicate, clone, "
    "poorly drawn, extra limbs, missing limbs, malformed, mutilated, "
    "fused fingers, too many fingers, long neck, error, "
    "worst quality, low quality, normal quality, amateur, unprofessional"
)


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY SYSTEMS — appended to every prompt for maximum visual fidelity
# ═════════════════════════════════════════════════════════════════════════════
QUALITY_SUFFIX = (
    "masterpiece, best quality, ultra-detailed, 8k resolution, "
    "professional color grading, sharp focus, cinematic composition, "
    "volumetric lighting, ray tracing, photorealistic rendering, "
    "award-winning visual design, studio quality"
)

# Additional quality boosters randomly selected and appended for variation
QUALITY_BOOSTERS = [
    "trending on ArtStation, breathtaking detail",
    "unreal engine quality, octane render aesthetic",
    "professional photography quality, DSLR sharpness",
    "hyper-realistic material textures, subsurface scattering",
    "dramatic atmosphere, museum-quality artwork",
    "editorial quality composition, magazine cover worthy",
    "pixel-perfect detail, razor-sharp clarity",
    "cinematic color science, ACES color pipeline",
]


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSITION STRATEGIES — adapt to output resolution aspect ratio
# ═════════════════════════════════════════════════════════════════════════════
COMPOSITION_STRATEGIES = {
    "portrait": (
        "vertical portrait composition with strong top-to-bottom visual flow, "
        "generous negative space in upper third for overlay elements, "
        "focal point anchored in lower center, visual weight distributed bottom-heavy, "
        "leading lines drawing eye downward, layered foreground-midground-background depth"
    ),
    "landscape": (
        "panoramic horizontal composition with wide establishing visual sweep, "
        "rule-of-thirds focal placement slightly left of center, "
        "strong horizon line with gradient sky, lateral visual flow left to right, "
        "flanking decorative elements creating natural frame"
    ),
    "square": (
        "balanced centered radial composition with strong focal nucleus, "
        "concentric visual layers radiating outward from center, "
        "symmetrical decorative borders, four-corner accent elements, "
        "diagonal energy lines creating dynamic X-pattern within square frame"
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# LIGHTING PRESETS — cinematic atmosphere per event mood
# ═════════════════════════════════════════════════════════════════════════════
LIGHTING_PRESETS = {
    "dramatic": (
        "dramatic chiaroscuro lighting with deep shadows and brilliant highlights, "
        "single hard key light from upper-left casting long shadows, "
        "volumetric God rays cutting through atmospheric haze"
    ),
    "warm": (
        "warm golden-hour lighting with soft amber tones permeating the scene, "
        "diffused sunlight creating gentle shadow gradients, "
        "candlelight warmth radiating from center, cozy intimate atmosphere"
    ),
    "cool": (
        "cool blue-toned ambient lighting with steel-gray accents, "
        "moonlit atmosphere with soft silver rim lighting on edges, "
        "subtle cyan fill light from below, mysterious nocturnal mood"
    ),
    "neon": (
        "vivid neon emission lighting in hot pink electric cyan and acid green, "
        "hard colored shadows from multiple neon sources, "
        "wet surface reflections doubling every color, chromatic light bleeding"
    ),
    "festive": (
        "warm festive multi-source lighting from dozens of small warm light sources, "
        "fairy light bokeh scattered throughout, diya flame flicker warmth, "
        "sparkler trail light painting effects, golden party atmosphere"
    ),
    "ethereal": (
        "soft ethereal dreamlike lighting with gentle pastel color washes, "
        "backlighting creating angelic rim glow on silhouettes, "
        "diffused light through translucent fabric, otherworldly divine radiance"
    ),
    "studio": (
        "professional three-point studio lighting setup, "
        "clean key light with soft fill and subtle rim separation, "
        "controlled gradient backdrop, precise highlight placement"
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# COLOR HARMONIES — ensure visually pleasing palettes
# ═════════════════════════════════════════════════════════════════════════════
COLOR_HARMONIES = {
    "fire": "rich warm palette of deep crimson #DC143C, molten gold #FFD700, burnt orange #CC5500, and amber #FFBF00",
    "ocean": "cool palette of deep navy #001F3F, teal #008080, aquamarine #7FFFD4, and seafoam #98FFE0",
    "sunset": "gradient palette from molten tangerine #FF6347 through coral pink #FF7F7F to deep magenta #C71585 to violet #7B2D8E",
    "forest": "natural palette of deep emerald #046307, sage green #87AE73, warm cedar #8B4513, and golden sunbeam #FFD700",
    "royal": "luxurious palette of deep purple #4B0082, regal gold #FFD700, midnight blue #191970, and ruby red #E0115F",
    "pastel": "soft dreamy palette of blush pink #FFB6C1, lavender #E6E6FA, mint #98FF98, and buttercup #FFFACD",
    "monochrome": "sophisticated monochromatic palette with deep charcoal #333333, silver #C0C0C0, platinum #E5E5E5, and pure white highlights",
    "indian_festive": "traditional Indian palette of saffron #FF9933, deep red #B22222, gold #FFD700, emerald #50C878, and royal blue #4169E1",
    "cyberpunk": "electric cyberpunk palette of neon magenta #FF00FF, electric cyan #00FFFF, acid green #ADFF2F, and void black #0A0A0A",
    "earth": "organic earth-tone palette of terracotta #E2725B, ochre #CC7722, olive #808000, and warm cream #FFFDD0",
}

# Map event types to their ideal lighting and color presets
EVENT_MOOD_MAP = {
    "Technical Fest": ("neon", "cyberpunk"),
    "Cultural Event": ("warm", "indian_festive"),
    "Sports Tournament": ("dramatic", "fire"),
    "Workshop / Seminar": ("studio", "monochrome"),
    "College Fest": ("festive", "sunset"),
    "Diwali Celebration": ("festive", "indian_festive"),
    "Holi Festival": ("dramatic", "pastel"),
    "Navratri / Garba": ("festive", "indian_festive"),
    "Ganesh Chaturthi": ("warm", "indian_festive"),
    "Eid Celebration": ("ethereal", "royal"),
    "Christmas / New Year": ("festive", "fire"),
    "Club Recruitment": ("neon", "sunset"),
    "Academic Event": ("studio", "royal"),
    "Freshers / Farewell": ("festive", "pastel"),
    "Blood Donation": ("studio", "fire"),
    "Music Concert": ("neon", "cyberpunk"),
    "Food Festival": ("warm", "earth"),
    "Marathon / Fitness": ("dramatic", "sunset"),
    "Hackathon": ("neon", "cyberpunk"),
    "Film / Theatre": ("dramatic", "royal"),
    "Fashion Show": ("studio", "pastel"),
    "Debate / MUN": ("studio", "royal"),
    "Independence Day / Republic Day": ("dramatic", "indian_festive"),
    "Art Exhibition": ("ethereal", "pastel"),
    "Science Fair": ("cool", "ocean"),
    "Charity / NGO Drive": ("warm", "earth"),
    "Prom Night": ("neon", "sunset"),
    "Convocation": ("warm", "royal"),
    "Other": ("dramatic", "sunset"),
}


# ═════════════════════════════════════════════════════════════════════════════
# TEXTURE LAYERS — add depth and materiality
# ═════════════════════════════════════════════════════════════════════════════
TEXTURE_LAYERS = [
    "subtle film grain overlay adding organic analog warmth",
    "fine noise texture creating depth and richness",
    "gentle paper texture with soft fiber details",
    "subtle metallic shimmer on decorative elements",
    "delicate silk fabric texture with light-catching folds",
    "soft bokeh particle dust floating in light beams",
    "fine geometric mesh pattern at very low opacity",
    "gentle watercolor paper tooth texture underneath colors",
]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — LLM instructions for artwork-only generation
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_TEXT2IMG = f"""You are a world-class visual artwork designer specializing in Indian college event poster BACKGROUNDS. You generate detailed, cinematic image generation prompts for VISUAL BACKGROUND ARTWORK ONLY.

Your prompt MUST describe ALL of these in RICH DETAIL:

1. ATMOSPHERE & MOOD — gradients, cinematic lighting, ambient glow, volumetric fog, bokeh, lens flares, mood-specific atmosphere
2. COLOR PALETTE — specific hex colors and their relationships, complementary/analogous schemes, gradient directions and transitions
3. DECORATIVE ELEMENTS — ornamental patterns, geometric shapes, mandalas, borders, floral motifs, abstract design elements
4. LIGHTING — type (dramatic/warm/cool/neon), direction, intensity, shadows, rim lighting, God rays, reflections, caustics
5. TEXTURES & MATERIALS — silk, velvet, marble, metallic gold foil, holographic shimmer, frosted glass, embossed patterns, fabric weaves
6. DEPTH & COMPOSITION — foreground/midground/background layers, bokeh, depth-of-field, leading lines, rule of thirds, negative space for overlay
7. CULTURAL MOTIFS — For Indian events: diyas, rangoli, marigold garlands, mehendi patterns, lotus motifs, peacock feathers, temple silhouettes, dandiya sticks, traditional fabric patterns
8. FINE DETAILS — particle effects, dust motes in light beams, sparkle highlights, subtle gradients within gradients, micro-texture variations

CRITICAL RULES:
- ALWAYS start your prompt with "{TRIGGER_WORD}"
- THIS IS THE MOST IMPORTANT RULE: NEVER EVER mention text, typography, fonts, titles, headings, letters, words, writing, captions, labels, signs, or any readable content in your prompt
- NEVER describe anything that could appear as text, glyphs, or character-like shapes in the generated image
- The image must be PURE VISUAL ARTWORK with ABSOLUTELY ZERO text-like elements anywhere
- Where a poster would normally have text (center, corners, banners), describe abstract visual patterns, gradients, decorative motifs, or clean empty space instead
- NEVER reference any specific words, event names, dates, or any content that could be rendered as text
- ALWAYS end with exactly: "CRITICAL: absolutely no text, no words, no letters, no writing, no typography, no signs, no labels, no captions, no readable content anywhere in the entire scene. All surfaces must be smooth, clean, and completely free of any character-like shapes."
- Use 200-280 words — be EXTREMELY specific about every visual detail
- Use cinematic language: "shallow depth of field", "anamorphic bokeh", "volumetric caustics"
- For Indian cultural events: include at LEAST 4 authentic cultural visual motifs with specific descriptive detail
- Include at least one mention of material texture (e.g., "brushed gold foil", "crushed velvet")
- Specify at least TWO specific hex colors or named colors
- Output ONLY the prompt — no preamble, no quotation marks, no explanation"""

SYSTEM_IMG2IMG = f"""You are a visual artwork restyling expert. Given a description of how to transform an existing visual artwork, generate a detailed prompt describing the desired output ARTWORK with maximum visual specificity.

Focus on:
1. The new visual style, color palette, mood, and atmosphere
2. Lighting changes — type, direction, color temperature
3. Texture and material modifications — surface quality, finish, grain
4. Elements to preserve vs. dramatically change
5. Overall compositional shifts and depth modifications

CRITICAL RULES:
- ALWAYS start with "{TRIGGER_WORD}"
- NEVER mention text, typography, fonts, or any written content
- Describe ONLY visual artwork transformations
- ALWAYS end with: "No text, no words, no letters anywhere in the scene."
- Use 150-200 words, be extremely specific
- Output ONLY the prompt"""

SYSTEM_INPAINT = f"""You are a visual artwork editing expert. Given a description of what region to regenerate, generate a prompt describing the VISUAL ELEMENTS to fill that region with maximum detail.

Focus on:
1. Specific visual elements for the masked area
2. Color and lighting continuity with surrounding artwork
3. Texture and pattern matching
4. Seamless blending at boundaries

CRITICAL RULES:
- ALWAYS start with "{TRIGGER_WORD}"
- NEVER mention text, typography, fonts, or any written content
- Describe ONLY visual elements
- ALWAYS end with: "No text, no words, no letters."
- Use 100-130 words, very specific
- Output ONLY the prompt"""


# ═════════════════════════════════════════════════════════════════════════════
# STYLE MAP — 20 ultra-rich visual-only styles (ZERO text references)
# ═════════════════════════════════════════════════════════════════════════════
STYLE_MAP = {
    "Vibrant and Energetic": (
        "vibrant energetic explosion of colors, electric gradients sweeping from hot magenta #FF00FF "
        "to electric cyan #00FFFF to deep violet #7B2D8E, dynamic diagonal composition with "
        "motion blur streaks converging at focal center, particle effects raining like confetti, "
        "volumetric light beams cutting through atmospheric haze, prismatic rainbow light refractions, "
        "bokeh circles in neon pink and electric blue, high-energy pulse-wave visual ripples "
        "radiating outward, metallic glitter dust suspended in mid-air"
    ),
    "Elegant and Professional": (
        "elegant professional visual design, deep navy #001F3F to midnight black gradient, "
        "brushed gold accent lines and thin geometric frames, subtle vignette darkening at edges, "
        "frosted glass panel overlays with soft gaussian blur, refined diamond-pattern lattice, "
        "warm amber spotlight from above casting precise shadows, silk-smooth color transitions, "
        "polished marble texture accents with golden veining, understated luxury atmosphere, "
        "clean balanced composition with surgical precision"
    ),
    "Modern Minimalist": (
        "ultra-modern minimalist composition, vast generous negative space in muted ivory #FFFFF0 "
        "or warm charcoal #36454F, single bold accent color stripe cutting through the frame, "
        "hairline geometric rules and perfect circles as visual anchors, "
        "soft precise drop shadows on floating geometric cards, monochromatic tonal gradients, "
        "matte surface finishes, Bauhaus-inspired geometric purity, "
        "intentional asymmetric balance, whitespace as a design element itself"
    ),
    "Traditional Indian": (
        "rich traditional Indian festive visual, warm gold-saffron #FF9933 to deep crimson #B22222 "
        "gradient background, ornate mandala borders with intricate geometric precision, "
        "rangoli-inspired floral patterns radiating from center in vibrant powder colors, "
        "decorative paisley and mehndi henna motifs framing the composition, "
        "marigold garland arches draping across the top edge, "
        "glowing earthen diyas casting warm amber firelight, "
        "mirror-work kutch embroidery sparkle accents, temple gopuram silhouette at base, "
        "traditional bandhani tie-dye fabric texture patches"
    ),
    "Tech-Futuristic": (
        "futuristic cyberpunk tech atmosphere, deep space black #0A0A0A background with electric neon glow, "
        "holographic grid planes receding into perspective vanishing point at infinity, "
        "circuit board trace patterns etched in glowing cyan #00FFFF light, "
        "floating translucent HUD-style geometric panel outlines, "
        "data stream particle trails in electric blue #4169E1 and violet #8A2BE2, "
        "hard-edge geometric shapes with chromatic aberration at edges, "
        "pulsing reactor-core energy glow at focal center, carbon fiber woven texture, "
        "scanning laser line sweeping across the frame"
    ),
    "Artistic and Creative": (
        "expressive artistic watercolor splash artwork, fluid organic paint splatters "
        "in warm ochre #CC7722, amber #FFBF00, and deep teal #008080, "
        "hand-painted impasto texture with visible thick brushstrokes, "
        "ink splatter accents blooming outward from center in every direction, "
        "mixed-media collage aesthetic with torn kraft paper edges and layered textures, "
        "warm earthy terracotta and forest green tones, "
        "painterly depth with dripping paint rivulets running down, "
        "charcoal graphite sketch elements underneath watercolor washes, "
        "palette knife texture marks in highlights"
    ),
    "Watercolor Dreamscape": (
        "fluid organic watercolor dreamscape, soft pastel washes of lavender #E6E6FA, mint green #98FF98, "
        "and baby pink #FFB6C1 blending seamlessly, translucent wet-on-wet watercolor technique, "
        "dreamy ethereal atmosphere with soft glowing edges, "
        "delicate ink line art accents bleeding into the paint, "
        "paper grain texture visible beneath the washes, gentle ambient lighting, "
        "floating pigment dust motes, calm and peaceful visual tone"
    ),
    "Neon Glow": (
        "intense neon glow atmosphere, pitch-black void #000000 background, "
        "vivid neon tubes bending into geometric shapes in hot pink #FF69B4 cyan #00FFFF "
        "and electric yellow #FFFF00, wet asphalt reflections perfectly doubling every neon glow, "
        "volumetric fog catching colored beams into visible light shafts, "
        "urban night noir atmosphere with rain-slicked surfaces, "
        "glowing emission halos around every light source with soft falloff, "
        "chromatic light bleeding, cross-shaped lens flare streaks, "
        "distant city skyline silhouette with scattered window lights"
    ),
    "Retro Vintage": (
        "retro vintage visual design from the 1970s, warm distressed paper texture with age spots "
        "and coffee-stain rings, muted color palette of burnt sienna #A0522D mustard yellow #E1AD01 "
        "and olive green #556B2F, halftone Ben-Day dot patterns creating visual depth gradients, "
        "sun-faded film grain overlay at 15% opacity, rounded geometric shapes, "
        "vintage sunburst rays radiating from center point, retro color-shift gradients, "
        "analog print registration offset effects in CMYK colors, "
        "worn edge vignette darkening like aged photograph"
    ),
    "Dark Premium": (
        "ultra-dark premium visual, matte velvety black #0D0D0D base with metallic gold #FFD700 foil accents, "
        "subtle embossed geometric patterns catching dramatic side-lighting on raised surfaces, "
        "extreme high contrast with pure black shadows and sharp brilliant highlights, "
        "luxury material textures — brushed titanium #8B8680, woven carbon fiber, polished obsidian #0B0B0B, "
        "minimal elements floating in vast dark negative space, "
        "single warm golden spotlight creating dramatic chiaroscuro from above, "
        "gold leaf flake particles drifting in spotlight beam"
    ),
    "Gradient Modern": (
        "modern gradient visual artwork, buttery smooth multi-color gradient sweeping "
        "from coral pink #FF7F7F through lavender #E6E6FA to electric indigo #4B0082, "
        "floating 3D geometric shapes with realistic contact shadows, "
        "glass morphism frosted panels with 40% translucent overlays, "
        "subtle noise grain texture adding analog depth, rounded organic blob shapes, "
        "iridescent rainbow shimmer on curved surfaces catching virtual light, "
        "dreamlike ethereal atmosphere with soft gaussian emission glow, "
        "thin wireframe sphere accent elements"
    ),
    "Bollywood Glamour": (
        "Bollywood glamour spectacle, rich saturated jewel tones of ruby #E0115F emerald #50C878 "
        "and sapphire #0F52BA, ornate gold filigree borders inspired by Mughal jharokha architecture, "
        "dramatic stage spotlight beams cutting through theatrical dry-ice haze, "
        "scattered pink rose petals frozen mid-air in slow-motion, shimmering sequin bokeh, "
        "Art Deco meets Indian royalty aesthetic, heavy velvet curtain drapes with gold tassels, "
        "mirror mosaic reflections creating kaleidoscope patterns, "
        "luxury palace interior ambiance with marble pillars and arches"
    ),
    "Festival of Lights": (
        "magical festival of lights atmosphere, hundreds of floating paper lanterns "
        "glowing warm amber #FFBF00 against a deep indigo #4B0082 twilight sky gradient, "
        "bokeh circles of gold orange and warm yellow scattered throughout the frame, "
        "strings of fairy lights creating warm constellation patterns across the composition, "
        "soft candlelight diya warmth pooling at the base, firefly particles drifting lazily, "
        "warm-to-cool color temperature gradient from golden base to cool blue sky, "
        "dreamy long-exposure light trails creating flowing ribbons of light, "
        "floating lotus flowers on reflective water surface"
    ),
    "Street Art Urban": (
        "vibrant street art urban aesthetic, raw concrete wall texture as canvas with visible aggregate, "
        "bold spray-paint gradients in electric saturated colors, stencil art geometric patterns "
        "layered with dripping thick paint streaks and splatter effects, "
        "torn wheat-paste layers creating archaeological depth and history, "
        "gritty urban texture with exposed brick industrial pipe and rusted metal details, "
        "neon accent highlights against raw concrete gray #808080, "
        "skateboard culture and hip-hop visual elements, chain-link fence overlay texture"
    ),
    "Cosmic Space": (
        "cosmic deep space visual, swirling nebula clouds in violet #8A2BE2 magenta #FF00FF "
        "and teal #008080 with internal luminosity, scattered star field with varying brightness "
        "and color temperature from cool blue to warm yellow, "
        "planetary ring arcs catching distant starlight in iridescent bands, "
        "cosmic dust lanes creating depth and grand-scale perspective, "
        "aurora borealis curtains of emerald green and violet light, "
        "gravitational lens light-bending distortion near center focal point, "
        "vast infinite scale with tiny detailed celestial formations"
    ),
    "Tropical Paradise": (
        "lush tropical paradise atmosphere, vibrant palm frond silhouettes framing top corners, "
        "sunset gradient from molten orange #FF6347 through coral pink #FF7F7F to deep purple #800080, "
        "exotic frangipani and hibiscus flowers scattered decoratively across composition, "
        "ocean wave crests catching golden hour directional light creating sparkle, "
        "tropical bird-of-paradise plant accents with vivid orange and blue, "
        "coconut husk textures, warm humid atmosphere with diffused golden sunlight, "
        "palm shadow stripe patterns on sand-tone background"
    ),
    "Sacred Geometry": (
        "intricate sacred geometry visual, perfectly mathematical Flower of Life pattern "
        "rendered in luminous gold #FFD700 lines on deep indigo #1A0533 background, "
        "Metatron's Cube with glowing vertices pulsing with energy, "
        "Fibonacci spiral curves flowing through the composition, "
        "hexagonal honeycomb patterns with iridescent fill, "
        "mandala-like concentric ring structures with fractal detail at every scale, "
        "subtle energy flow lines connecting geometric nodes, "
        "crystalline lattice structures catching and refracting light"
    ),
    "Waterfront Evening": (
        "serene waterfront evening scene, mirror-still lake reflecting deep twilight sky, "
        "silhouetted treeline along horizon, last traces of sunset in peach #FFDAB9 and rose, "
        "string lights reflected as wavy golden snakes on water surface, "
        "atmospheric mist rolling across water creating depth layers, "
        "wooden dock planks as textured foreground element, "
        "firefly dots of warm light scattered in the dusk air, "
        "star emergence in deep blue #191970 upper sky gradient"
    ),
    "Pop Art": (
        "bold Pop Art visual inspired by Roy Lichtenstein and Andy Warhol, "
        "stark primary color blocks in red #FF0000 blue #0000FF and yellow #FFFF00, "
        "heavy black outlines around every element, Ben-Day halftone dot patterns filling shapes, "
        "comic-style starburst explosion shapes as decorative accents, "
        "screen-print color registration offsets, flat unmodulated color fills, "
        "action lines and speed streaks radiating from center, "
        "high contrast graphic design with zero subtlety — maximum visual punch"
    ),
    "Zen Japanese": (
        "serene Japanese Zen aesthetic, minimalist wabi-sabi composition, "
        "soft ink wash sumi-e gradients from deep black to paper white, "
        "cherry blossom sakura petals falling in gentle slow-motion, "
        "enso circle brushstroke as central meditative element, "
        "bamboo grove silhouettes in gentle fog, koi fish ripple pattern on still water, "
        "traditional wave pattern seigaiha in subtle background, "
        "muted earth tones with single accent of red #C41E3A torii gate, "
        "asymmetric ikebana balance, generous contemplative negative space"
    ),
    "Psychedelic 60s": (
        "vivid psychedelic 1960s visual, swirling liquid lava-lamp color patterns "
        "flowing between hot pink #FF69B4 electric orange #FF4500 acid green #7FFF00 "
        "and deep purple #800080, kaleidoscopic mirror symmetry patterns, "
        "op-art optical illusion concentric patterns creating depth, "
        "paisley swirl motifs enlarged to fill entire composition regions, "
        "tie-dye spiral color blending effects, mushroom dome shapes as decorative elements, "
        "peace symbol as subtle background watermark pattern, "
        "wavy undulating horizon lines, rainbow prism light spectrum bands"
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# EVENT TYPE HINTS — 30 event types with ultra-rich visual descriptions
# ═════════════════════════════════════════════════════════════════════════════
EVENT_TYPE_HINTS = {
    "Technical Fest": (
        "glowing circuit board trace patterns radiating from center, robotic arm silhouettes "
        "in dynamic poses, holographic projection display pyramids, "
        "binary-code-inspired decorative data stream cascades (purely visual pattern), "
        "quadcopter drone silhouettes hovering in formation, "
        "3D-printed geometric lattice sculptures, laser grid projections in perspective, "
        "quantum computing orbital visualization spheres, "
        "PCB green solder mask texture as background element"
    ),
    "Cultural Event": (
        "grand stage with dramatic red velvet curtains and golden cord tiebacks, "
        "classical Bharatanatyam dance mudra hand silhouettes in graceful poses, "
        "traditional Indian musical instruments — ornate sitar veena tabla mridangam, "
        "elaborate rangoli mandala floor patterns in vibrant powder colors, "
        "stage spotlights with volumetric atmospheric haze beams, "
        "silk fabric drapes cascading in festive jewel-tone colors, "
        "flower garland toran borders with marigold and jasmine"
    ),
    "Sports Tournament": (
        "dynamic action-frozen athlete silhouettes mid-leap with explosive energy, "
        "detailed sports equipment — basketball with leather texture, cricket bat grain, "
        "football hexagonal panels, stadium floodlights creating dramatic God ray beams, "
        "motion blur speed lines implying velocity, gleaming trophy under spotlight, "
        "stadium crowd silhouette creating atmosphere in background, "
        "athletic starting-block texture, running track lane line patterns, "
        "championship medal with ribbon flutter"
    ),
    "Workshop / Seminar": (
        "soft professional diffused lighting on abstract knowledge visualization, "
        "glowing light bulb filament icon as central innovation motif, "
        "interconnected neural network nodes with pulsing energy pathways, "
        "laboratory Erlenmeyer flask and precision gear mechanism motifs, "
        "isometric blueprint grid background with faint construction lines, "
        "innovation energy spark particles bursting from ideas, "
        "abstract flowchart connections representing systematic thinking"
    ),
    "College Fest": (
        "aerial view of festive campus quad decorated with colorful triangular bunting pennants, "
        "confetti explosion frozen in mid-air — multi-colored paper squares and circles, "
        "carnival Ferris wheel rides silhouetted against sunset sky, "
        "diverse crowd energy represented by abstract color splash waves, "
        "balloon arches in college colors and streamer cascades, "
        "festival stage with dramatic lighting rig silhouette, "
        "food stall warm lighting and fabric canopy outlines"
    ),
    "Diwali Celebration": (
        "hundreds of glowing earthen diyas arranged in beautiful flower-of-life patterns on marble, "
        "spectacular aerial firework bursts painting the night sky in gold emerald and ruby, "
        "intricate rangoli designs in vibrant gulal powder — magenta saffron cyan white, "
        "marigold genda phool and jasmine mogra flower garlands draping across frame, "
        "ornate brass deep jyoti oil lamp centerpiece with seven flames, "
        "shimmering sparkler hand trails creating light-painting arcs, "
        "warm golden candlelight ambiance suffusing the entire scene, "
        "Lord Ganesha silhouette motif with divine golden aura halo, "
        "floating rose petals on brass urli water bowl"
    ),
    "Holi Festival": (
        "explosive clouds of vibrant color powder — magenta turquoise saffron violet electric green — "
        "frozen mid-burst against brilliant blue sky creating rainbow nebula, "
        "pichkari water gun splashes catching light in prismatic streams, "
        "rainbow color powder streaks painting the air in every direction, "
        "gulal color handprints on white surfaces creating abstract art, "
        "water droplets catching full-spectrum prismatic sunlight, "
        "joyful impossible energy color explosion filling every corner of frame"
    ),
    "Navratri / Garba": (
        "ornate dandiya raas sticks crossed mid-dance from artistic overhead angle, "
        "ghagra choli chaniya fabric patterns with intricate kutchi mirror-work embroidery, "
        "Goddess Durga fierce silhouette with divine golden radiant aura, "
        "festive chaniya choli fabric textures in vibrant navrang nine-color spectrum, "
        "garba circle formation viewed from dramatic above angle, "
        "decorative toran door hangings and fresh flower garlands, "
        "nine-pointed Navratri star motif, warm diya oil-lamp glow, "
        "ghungroo ankle bell details catching light"
    ),
    "Ganesh Chaturthi": (
        "majestic Lord Ganesha idol silhouette with divine golden glow emanating outward, "
        "modak sweet offerings arranged on ornate brass thali plate, "
        "mountain of marigold flowers in orange and yellow heaped around base, "
        "festive pandal mandap with colorful fabric-draped canopy and banana trunk pillars, "
        "incense dhoop smoke wisps catching golden directional light beautifully, "
        "coconut and banana leaf ceremonial arrangements, "
        "visarjan water reflection of idol silhouette"
    ),
    "Eid Celebration": (
        "elegant crescent moon hilaal and star motif rendered in golden filigree metalwork, "
        "ornate mosque dome and minaret silhouettes against starlit indigo sky, "
        "hanging Moroccan fanoos lanterns casting intricate geometric shadow patterns, "
        "intricate arabesque and Islamic geometric zellige tile patterns, "
        "soft moonlight illumination with silver-blue color cast, "
        "dates and tasbih prayer beads still-life arrangement, "
        "open Quran stand silhouette with divine light radiating"
    ),
    "Christmas / New Year": (
        "sparkling Christmas tree silhouette with baubles ornaments and golden star topper, "
        "falling snowflakes in unique crystalline patterns against dark sky, "
        "midnight countdown clock face with ornate hands approaching twelve, "
        "spectacular New Year firework display over illuminated city skyline silhouette, "
        "candy cane stripes holly berries and pine cone decorations, "
        "warm fireplace hearth glow ambiance, gift box ribbons and satin bows, "
        "champagne glass with rising golden bubbles"
    ),
    "Club Recruitment": (
        "diverse student silhouettes in collaborative creative active poses, "
        "artistic tools scattered decoratively — paintbrush camera guitar code symbols, "
        "dynamic upward-reaching energy visual with rising gradient, "
        "interconnected community network nodes glowing with warm connection lines, "
        "colorful empty speech bubble shapes as decorative elements only, "
        "welcoming open-door motif with warm inviting light spilling out, "
        "joining-hands unity circle silhouette"
    ),
    "Academic Event": (
        "stately graduation mortarboard cap floating with golden tassel catching light, "
        "stacked vintage leather-bound books with gilt page edges reflecting warmth, "
        "wooden podium under single dramatic focused spotlight beam, "
        "academic institution shield crest decorative heraldry motif, "
        "celestial armillary globe and compass rose decorative elements, "
        "scholarly quill feather and crystal inkwell accents, "
        "laurel wreath victory border in burnished gold"
    ),
    "Freshers / Farewell": (
        "nostalgic Polaroid instant photo frames floating on soft dreamy background, "
        "welcome arch draped with colorful fabric fairy lights and fresh flowers, "
        "confetti and metallic streamer cascade frozen in time, "
        "graduation mortarboard toss silhouettes against sunset sky, "
        "memory scrapbook collage aesthetic with overlapping faded images, "
        "warm vintage film color grading with soft amber tint, "
        "emotional sunset hues blending from warm to cool"
    ),
    "Blood Donation": (
        "bold red cross medical symbol rendered in sleek clean modern style, "
        "anatomical heart illustration with gentle pulse-wave emanating outward, "
        "blood drop motif with internal DNA helix swirl pattern, "
        "diverse helping hands reaching toward central warm light, "
        "clean clinical white and deep crimson red color scheme, "
        "lifeline ECG pulse wave as decorative flowing border element, "
        "stethoscope coil as frame accent"
    ),
    "Music Concert": (
        "electric guitar fretboard silhouette with stage light flares blooming at headstock, "
        "professional condenser studio microphone under single dramatic spot, "
        "radiating soundwave concentric circles as visual ripple pattern, "
        "stage fog machine dense haze with colored laser beams cutting through, "
        "concert crowd silhouette ocean with thousands of raised hands and phone lights, "
        "Marshall speaker stack tower walls, vinyl record groove spiral pattern, "
        "equalizer frequency bars visualization pulsing with energy"
    ),
    "Food Festival": (
        "artistic top-down flat-lay food photography arrangement with vibrant spices in brass bowls, "
        "chef toque hat and crossed wooden spoon silhouettes, "
        "aromatic steam wisps rising artfully from hot dishes, "
        "colorful spice market aesthetic — turmeric yellow chili red saffron orange cardamom green, "
        "rustic hand-hewn wooden board texture as base, fresh coriander herb garnishes, "
        "warm kitchen tungsten lighting ambiance, "
        "Indian thali round plate with multiple colorful dishes arranged in circle"
    ),
    "Marathon / Fitness": (
        "dynamic running silhouettes captured mid-stride with explosive motion energy trails, "
        "finish line ribbon flutter in breeze, precision chronograph stopwatch face, "
        "athletic shoe sole print trail pattern receding into distance, "
        "morning golden sunrise over athletic track field with dramatic long shadows, "
        "energy burst particle effects radiating around peak-performance athletes, "
        "medal and olive laurel wreath victory iconography, "
        "heartbeat pulse line overlaid as dynamic compositional element"
    ),
    "Hackathon": (
        "glowing laptop screen blue-light reflected on focused face silhouettes in dark room, "
        "floating code bracket angle-bracket and algorithm flowchart decorative elements, "
        "coffee cup steam wisps intertwining with coiled USB-C cable shapes, "
        "digital matrix data cascade as visual particle rain in green, "
        "wall-clock hands showing midnight representing non-stop coding marathon, "
        "collaborative whiteboard with abstract wireframe sketch diagrams, "
        "energy drink can tower stacked in pyramid, "
        "GitHub contribution graph heat-map pattern as subtle background texture"
    ),
    "Film / Theatre": (
        "dramatic theatre Fresnel spotlight cutting through darkness onto empty wooden stage, "
        "35mm film reel and director's clapperboard silhouettes, vintage cinema projector beam, "
        "heavy red velvet curtains with golden rope tassels drawn to sides, "
        "comedy and tragedy theatrical mask motifs in gold, "
        "popcorn bucket and unwinding film strip decorative border, "
        "directorial chair under single key light with dramatic shadow"
    ),
    "Fashion Show": (
        "sleek runway perspective lines vanishing into dramatic contrasty backlighting, "
        "elegant fabric swatch samples floating mid-air — silk charmeuse chiffon crushed velvet, "
        "fashion dress-form mannequin silhouettes in dramatic angular poses, "
        "sewing needle trailing golden thread in flowing spiral motif, "
        "scattered rose petals trailing along glossy catwalk surface, "
        "glamorous full-length mirror reflections with soft bokeh focus"
    ),
    "Debate / MUN": (
        "stately wooden podium under ornate parliamentary crystal chandelier lighting, "
        "gavel striking sound block as powerful central motif with impact ripple, "
        "world globe with warm-lit highlighted nations and latitude lines, "
        "classical Corinthian column architecture framing the composition symmetrically, "
        "balance scales of justice decorative element in brushed brass, "
        "diplomatic olive branch and ribbon border motifs, "
        "circular amphitheatre seating silhouette in background"
    ),
    "Independence Day / Republic Day": (
        "Indian tricolor national flag saffron white and green fabric billowing in wind with folds, "
        "Ashoka Chakra 24-spoke wheel spinning with golden metallic spokes, "
        "India Gate and Red Fort monument silhouettes against golden hour sky, "
        "soaring eagle silhouette against dramatic sunrise rays, "
        "patriotic ribbon streamers in tricolor flowing across composition, "
        "soldier salute silhouette with dramatic rim-lit backlighting, "
        "marigold flower arrangements in tri-color saffron white green patterns"
    ),
    "Art Exhibition": (
        "elegant white-cube gallery space with dramatic track lighting on empty walls, "
        "ornate gold picture frames at various angles creating depth, "
        "abstract paint palette with rich impasto color dollops and bristle brushes, "
        "sculptor's chisel and marble dust particles catching spotlight, "
        "canvas easel silhouette under gallery spot beam, "
        "installation art suspended geometric mobiles, "
        "gallery opening wine glass reflections creating elegant bokeh"
    ),
    "Science Fair": (
        "laboratory flask bubbling with luminous colored liquid and vapor wisps, "
        "DNA double helix structure rotating with bioluminescent glow, "
        "atom orbital model with electron trail paths in electric blue, "
        "microscope and telescope instrument silhouettes, "
        "periodic table element tile pattern as subtle background, "
        "Tesla coil electrical discharge arcs in purple, "
        "galaxy-brain knowledge explosion visual emanating from scientific apparatus"
    ),
    "Charity / NGO Drive": (
        "diverse warm hands reaching together toward central glowing heart symbol, "
        "world globe cradled gently in caring open palms, "
        "warm sunrise hope gradient from dark earth to golden sky, "
        "tree of life growing from base with spreading branches and green leaves, "
        "ribbon awareness loop in campaign colors, "
        "community circle of diverse people silhouettes holding hands, "
        "dove of peace in flight with olive branch against warm sky"
    ),
    "Prom Night": (
        "glamorous ballroom with crystal chandelier casting thousands of rainbow light spots, "
        "disco mirror ball refractions creating dancing light patterns on walls and floor, "
        "elegant corsage flower arrangement detail, "
        "starry night sky visible through grand ballroom windows, "
        "champagne gold #C5B358 and rose pink #FF007F color scheme, "
        "flowing evening gown fabric silhouettes in slow-motion twirl, "
        "fairy light canopy overhead creating warm star ceiling"
    ),
    "Convocation": (
        "ceremonial academic procession with flowing gown fabric and hood colors, "
        "degree scroll tied with ribbon and wax seal, "
        "podium microphone under institutional spotlight, "
        "mortarboard graduation caps frozen mid-throw celebration, "
        "university crest shield with laurel branch border in gold, "
        "proud family silhouettes in audience, "
        "confetti and serpentine streamer celebration burst, "
        "warm golden afternoon light through gothic arch windows"
    ),
    "Other": (
        "professional event atmosphere with modern geometric visual design elements, "
        "eye-catching radial gradient background from warm center to cool edges, "
        "floating decorative geometric shapes with soft shadows at various depths, "
        "dramatic centered spotlight with volumetric atmospheric haze, "
        "subtle vignette framing drawing focus inward, "
        "clean generous negative space zones for visual breathing room, "
        "metallic accent lines creating structural visual framework"
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# GROQ API CALLER — SDK + raw HTTP dual fallback
# ═════════════════════════════════════════════════════════════════════════════

def _call_groq(system_prompt: str, user_message: str) -> Optional[str]:
    """Make a Groq API call — tries SDK first, then raw HTTP fallback."""
    if not GROQ_API_KEY:
        return None

    # Try Groq SDK first
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.85,
            max_tokens=450,
            top_p=0.92,
        )
        return chat.choices[0].message.content.strip()

    except ImportError:
        logger.info("Groq SDK not installed. Using raw HTTP fallback...")
        return _call_groq_raw(system_prompt, user_message)
    except Exception as e:
        logger.warning(f"Groq SDK error: {e}. Trying raw HTTP fallback...")
        return _call_groq_raw(system_prompt, user_message)


def _call_groq_raw(system_prompt: str, user_message: str) -> Optional[str]:
    """Fallback: call Groq via raw HTTP if the SDK is unavailable."""
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
                "temperature": 0.85,
                "max_tokens": 450,
                "top_p": 0.92,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Groq raw API error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT ASSEMBLY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_trigger(prompt: str) -> str:
    """Ensure the trigger word is at the start of the prompt."""
    if not prompt.lower().startswith(TRIGGER_WORD):
        prompt = f"{TRIGGER_WORD}  {prompt}"
    return prompt


def _append_quality(prompt: str) -> str:
    """Append quality suffix + a random quality booster for variation."""
    booster = random.choice(QUALITY_BOOSTERS)
    return f"{prompt.rstrip('. ')}. {QUALITY_SUFFIX}. {booster}"


def _append_no_text(prompt: str) -> str:
    """Ensure the prompt ends with an aggressive no-text instruction."""
    no_text_tag = (
        "CRITICAL: absolutely no text, no words, no letters, no writing, "
        "no typography, no signs, no labels, no captions, no readable content "
        "anywhere in the entire scene. All surfaces must be smooth, clean, "
        "and completely free of any character-like shapes or glyph-like patterns. "
        "Replace any area where text might appear with abstract visual patterns, "
        "gradients, or decorative motifs instead."
    )
    if "absolutely no text" not in prompt.lower():
        prompt = f"{prompt.rstrip('. ')}. {no_text_tag}"
    return prompt


def _get_composition(resolution: str = "portrait") -> str:
    """Return composition strategy based on aspect ratio."""
    if resolution in COMPOSITION_STRATEGIES:
        return COMPOSITION_STRATEGIES[resolution]
    return COMPOSITION_STRATEGIES["portrait"]


def _get_mood_context(event_type: str) -> tuple[str, str]:
    """Return (lighting_preset, color_harmony) for an event type."""
    mood = EVENT_MOOD_MAP.get(event_type, ("dramatic", "sunset"))
    lighting = LIGHTING_PRESETS.get(mood[0], LIGHTING_PRESETS["dramatic"])
    colors = COLOR_HARMONIES.get(mood[1], COLOR_HARMONIES["sunset"])
    return lighting, colors


def _get_texture() -> str:
    """Return a random texture layer for visual depth."""
    return random.choice(TEXTURE_LAYERS)


def _detect_aspect(resolution_name: str) -> str:
    """Detect aspect ratio category from resolution preset name."""
    r = resolution_name.lower()
    if "portrait" in r or "story" in r or "a4" in r or "tall" in r:
        return "portrait"
    elif "landscape" in r or "wide" in r:
        return "landscape"
    return "square"


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def build_text2img_prompt(
    event_description: str,
    event_type: str = "Other",
    style: str = "Vibrant and Energetic",
    resolution: str = "Portrait (768×1152)",
) -> str:
    """Build a rich artwork-only prompt for SDXL poster background generation.

    Combines: Style + Event Hints + Lighting + Color Harmony + Composition + Texture + Quality.
    """
    style_desc = STYLE_MAP.get(style, STYLE_MAP["Vibrant and Energetic"])
    event_hints = EVENT_TYPE_HINTS.get(event_type, EVENT_TYPE_HINTS["Other"])
    lighting, colors = _get_mood_context(event_type)
    aspect = _detect_aspect(resolution)
    composition = _get_composition(aspect)
    texture = _get_texture()

    user_msg = (
        f"Create a visual artwork background prompt for this event:\n"
        f"Event: {event_description}\n"
        f"Type: {event_type}\n"
        f"Visual style: {style}\n"
        f"Style details: {style_desc}\n"
        f"Thematic visual elements: {event_hints}\n"
        f"Lighting atmosphere: {lighting}\n"
        f"Color palette: {colors}\n"
        f"Composition strategy: {composition}\n"
        f"Texture layer: {texture}\n"
        f"\nIMPORTANT: Describe ONLY the visual artwork background. "
        f"Do NOT mention any text, words, titles, names, dates, numbers, "
        f"typography, or readable content of any kind. "
        f"The artwork must have clean negative space suitable for text overlay later.\n"
    )

    result = _call_groq(SYSTEM_TEXT2IMG, user_msg)

    if result:
        prompt = _ensure_trigger(result)
        prompt = _append_no_text(prompt)
        return _append_quality(prompt)

    # ── Intelligent fallback without LLM ──────────────────────────────
    prompt = _ensure_trigger(
        f"A stunning visual artwork background for a {event_type.lower()} event "
        f"themed around {event_description}. "
        f"{style_desc}. "
        f"Decorative visual elements: {event_hints}. "
        f"{lighting}. Color palette: {colors}. "
        f"{composition}. "
        f"{texture}. "
        f"Beautiful composition with dramatic cinematic atmosphere and rich layered depth."
    )
    prompt = _append_no_text(prompt)
    return _append_quality(prompt)


def build_img2img_prompt(
    transform_description: str,
    style: str = "Vibrant and Energetic",
) -> str:
    """Build a prompt for img2img poster transformation."""
    style_desc = STYLE_MAP.get(style, STYLE_MAP["Vibrant and Energetic"])

    user_msg = (
        f"Transform this visual artwork with the following changes:\n"
        f"Changes: {transform_description}\n"
        f"New visual style: {style}\n"
        f"Style details: {style_desc}\n"
        f"\nIMPORTANT: Describe ONLY the visual transformation. "
        f"Do NOT mention any text, words, or typography.\n"
    )

    result = _call_groq(SYSTEM_IMG2IMG, user_msg)

    if result:
        prompt = _ensure_trigger(result)
        prompt = _append_no_text(prompt)
        return _append_quality(prompt)

    prompt = _ensure_trigger(
        f"A transformed visual artwork: {transform_description}. "
        f"{style_desc}. Professional quality, cohesive artistic design, "
        f"seamless style transformation with maintained composition integrity."
    )
    prompt = _append_no_text(prompt)
    return _append_quality(prompt)


def build_inpaint_prompt(
    fill_description: str,
) -> str:
    """Build a prompt for inpainting a region of a poster artwork."""
    user_msg = (
        f"Fill the masked region of this artwork with: {fill_description}\n"
        f"\nIMPORTANT: Describe ONLY visual elements. No text, words, or writing.\n"
    )

    result = _call_groq(SYSTEM_INPAINT, user_msg)

    if result:
        prompt = _ensure_trigger(result)
        prompt = _append_no_text(prompt)
        return _append_quality(prompt)

    prompt = _ensure_trigger(
        f"{fill_description}. Seamless blending with surrounding artwork, "
        f"consistent lighting direction and color palette, "
        f"matching texture and material quality at boundaries."
    )
    prompt = _append_no_text(prompt)
    return _append_quality(prompt)
