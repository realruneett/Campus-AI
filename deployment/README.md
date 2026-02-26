---
title: CampusGen AI - Event Poster Generator
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: true
license: mit
hardware: zero-a10g
---

# 🎨 CampusGen AI – Universal Event Poster Generator

Generate professional event posters for **any occasion** in 10–15 seconds.

## Features

- **5 Generation Modes**: Text→Poster, Reference Image (IP-Adapter), Image Transform, Inpainting, HD Upscale
- **AI-Powered**: Flux.1-dev fine-tuned on 55,000+ diverse poster images via LoRA
- **55 Categories**: Tech fests, cultural events, festivals (Diwali, Holi, Navratri), sports, workshops, and more
- **Smart Prompts**: Groq Llama 3.3 70B understands your event semantics and generates optimal prompts
- **10 Visual Styles**: Vibrant, Elegant, Minimalist, Traditional Indian, Tech-Futuristic, Neon Glow, and more
- **HD Upscaling**: Real-ESRGAN 4x for print-ready posters
- **Batch Generation**: Generate up to 4 variants at once
- **Zero Cost**: Free deployment via ZeroGPU

## How to Use

### Tab 1: Text → Poster

1. Describe your event (e.g., "IIT Indore Techfest 2026 — Robotics & AI Championships")
2. Select event type and visual style
3. Click **Generate Poster**

### Tab 2: Reference Image

1. Upload a poster you like as a reference
2. Describe your event
3. Adjust style influence slider
4. Click **Generate with Reference**

### Tab 3: Image Transform

1. Upload an existing poster
2. Describe the transformation (e.g., "Make it neon-themed")
3. Adjust transformation strength
4. Click **Transform Poster**

### Tab 4: Inpaint / Edit

1. Upload a poster
2. Draw over the area you want to change
3. Describe what should fill it
4. Click **Inpaint Region**

### Tab 5: HD Upscale

1. Upload any image
2. Select 2x or 4x scale
3. Click **Upscale**

## Technical Details

| Component | Details |
|-----------|---------|
| Base Model | Flux.1-dev (12B params) |
| Fine-tuning | LoRA (rank 32, bf16) |
| Dataset | 55,000+ curated event posters, 55 categories |
| LLM | Llama 3.3 70B via Groq |
| IP-Adapter | Reference image style extraction |
| Upscaler | Real-ESRGAN 4x |
| Hardware | ZeroGPU (shared A100) |

## Pipeline (GPU-Accelerated)

```text
Scraping (CPU) → Quality Filter (GPU) → Captioning (GPU) → Split → Train LoRA (GPU) → Deploy
```

## Author

Built with ❤️ by M Runeet Kumar
