#!/usr/bin/env python3
"""
CampusGen AI – Full-Feature Gradio Application
Multi-tab poster generation platform for Hugging Face Spaces.

Tabs:
  1. Text → Poster        (Flux + LoRA + Groq LLM)
  2. Reference Image       (IP-Adapter + LoRA)
  3. Image Transform       (Img2Img pipeline)
  4. Inpainting / Edit     (Mask-based regeneration)
  5. HD Upscale            (Real-ESRGAN 4x)
"""

import os
import time
import logging
from typing import Optional

import torch
import gradio as gr

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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dropdowns
# ─────────────────────────────────────────────────────────────────────────────
EVENT_TYPES = list(EVENT_TYPE_HINTS.keys())
STYLES = list(STYLE_MAP.keys())

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
):
    """Tab 1: Text-to-Poster generation."""
    if not event_description.strip():
        raise gr.Error("Please enter an event description!")

    manager = get_pipeline_manager()
    pipe = manager.get_text2img()

    # Build prompt via Groq LLM
    prompt = build_text2img_prompt(event_description, event_type, style)
    logger.info(f"[Text2Img] Prompt: {prompt[:120]}...")

    # Resolution
    width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))

    # Seed
    if seed == -1:
        seed = int(time.time()) % (2**32)

    # LoRA strength
    if manager.is_lora_loaded:
        pipe.fuse_lora(lora_scale=lora_strength)

    # Generate variants
    images = []
    generator = torch.Generator("cpu").manual_seed(seed)
    start = time.time()

    for i in range(num_variants):
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        img = result.images[0]

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


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}
.title-text {
    text-align: center;
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
    letter-spacing: -0.02em;
}
.subtitle-text {
    text-align: center;
    color: #888;
    font-size: 1.15em;
    margin-bottom: 1.5em;
    font-weight: 300;
}
.tab-nav button {
    font-size: 1.05em !important;
    font-weight: 600 !important;
}
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
}
.footer-text {
    text-align: center;
    color: #999;
    font-size: 0.9em;
    margin-top: 1em;
    padding: 1em;
    border-top: 1px solid #333;
}
"""

EXAMPLES = [
    ["IIT Indore Techfest 2026 — Robotics & AI Championships", "Technical Fest", "Tech-Futuristic"],
    ["Diwali Mela 2026 — Spark of Joy", "Diwali Celebration", "Traditional Indian"],
    ["Inter-College Basketball Championship", "Sports Tournament", "Vibrant and Energetic"],
    ["Photography Club Portfolio Night", "Club Recruitment", "Dark Premium"],
    ["ML/AI Workshop Series — From Zero to GPT", "Workshop / Seminar", "Gradient Modern"],
    ["Classical Kathak Dance Night", "Cultural Event", "Elegant and Professional"],
    ["Holi Hai! Campus Color Run", "Holi Festival", "Artistic and Creative"],
    ["Navratri Garba Night 2026", "Navratri / Garba", "Traditional Indian"],
    ["End-of-Year Farewell Party", "Freshers / Farewell", "Neon Glow"],
    ["Blood Donation Camp — Save Lives", "Blood Donation", "Modern Minimalist"],
]


def build_app() -> gr.Blocks:
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="CampusGen AI") as demo:

        # ── Header ───────────────────────────────────────────────────
        gr.HTML(
            '<div class="title-text">🎨 CampusGen AI</div>'
            '<div class="subtitle-text">'
            "Generate stunning event posters in seconds — "
            "Text · Reference Image · Transform · Inpaint · Upscale"
            "</div>"
        )

        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════
            # TAB 1: Text → Poster
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("✍️ Text → Poster", id="text2img"):
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

                        with gr.Accordion("⚙️ Advanced", open=False):
                            t2i_steps = gr.Slider(10, 50, value=28, step=1, label="Inference Steps")
                            t2i_cfg = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="Guidance Scale")
                            t2i_lora = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="LoRA Strength")
                            t2i_upscale = gr.Checkbox(label="🔍 HD Upscale (2x)", value=False)
                            t2i_seed = gr.Number(value=-1, label="Seed (-1 = random)")

                        t2i_btn = gr.Button("🚀 Generate Poster", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        t2i_gallery = gr.Gallery(
                            label="Generated Posters", columns=2,
                            height=600, object_fit="contain",
                        )
                        t2i_info = gr.Markdown(label="Generation Info")

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
                    ],
                    outputs=[t2i_gallery, t2i_info],
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 2: Reference Image
            # ═══════════════════════════════════════════════════════════
            with gr.Tab("🖼️ Reference Image", id="reference"):
                gr.Markdown(
                    "Upload a poster you like → the AI will extract its **visual style** "
                    "and blend it with your event description using IP-Adapter."
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

                        ref_btn = gr.Button("🚀 Generate with Reference", variant="primary", size="lg", elem_classes=["generate-btn"])

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
            with gr.Tab("🔄 Image Transform", id="img2img"):
                gr.Markdown(
                    "Upload an existing poster → describe how you want it **transformed**. "
                    "Lower denoising = subtle changes, higher = dramatic restyle."
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

                        i2i_btn = gr.Button("🔄 Transform Poster", variant="primary", size="lg", elem_classes=["generate-btn"])

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
            with gr.Tab("🖌️ Inpaint / Edit", id="inpaint"):
                gr.Markdown(
                    "Upload a poster → **draw over the area** you want to change → "
                    "describe what should replace it. The rest of the poster stays intact."
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

                        inp_btn = gr.Button("🖌️ Inpaint Region", variant="primary", size="lg", elem_classes=["generate-btn"])

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
            with gr.Tab("🔍 HD Upscale", id="upscale"):
                gr.Markdown(
                    "Upload any image → get a **4x upscaled** HD version using Real-ESRGAN. "
                    "Great for making generated posters print-ready."
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
                        up_btn = gr.Button("🔍 Upscale", variant="primary", size="lg", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        up_output = gr.Image(label="Upscaled Image", type="pil", height=600)
                        up_info = gr.Markdown()

                up_btn.click(
                    fn=upscale_only,
                    inputs=[up_image, up_scale],
                    outputs=[up_output, up_info],
                )

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML(
            '<div class="footer-text">'
            "<strong>CampusGen AI</strong> — "
            "Fine-tuned on 71,000+ event poster images across 57 subcategories | "
            "Flux.1-dev + LoRA + IP-Adapter + Real-ESRGAN | "
            "Groq Llama 3.3 70B for smart prompts<br>"
            "Built with ❤️ for the Indian campus community"
            "</div>"
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
