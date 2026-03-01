# Campus-AI by CounciL — Universal Event Poster Generator

> AI-powered event poster generation for any occasion in 10–15 seconds.

## Overview

Campus-AI generates professional event posters using:

- **Stable Diffusion XL 1.0** fine-tuned on 55,000+ diverse poster images via LoRA
- **Llama 3.3 70B** (Groq) for natural language event understanding
- **6 Generation Modes**: Text→Poster, Reference Image, Img2Img, Inpainting, HD Upscale, Edit Poster
- **Two-Stage Pipeline**: SDXL artwork generation + PIL typography compositor
- **Prompt Engine v3.0**: 20 visual styles, 30 event types, 7 lighting presets, 10 color harmonies
- **Poster Compositor v3.0**: 8 typography styles, 30+ premium fonts, 1,700+ Google Fonts on-demand
- **GPU-accelerated pipeline** from data processing to training
- **Zero cost** deployment on Hugging Face Spaces (ZeroGPU)

## Architecture

```text
User Input → Groq LLM (prompt engine v3.0) → SDXL 1.0 + LoRA → PIL Compositor → HD Upscale → Poster
                                                   ↑
                                      IP-Adapter (reference style)
                                      Img2Img (transform)
                                      Inpainting (edit regions)
                                      Edit Poster (re-apply typography)
```

| Component | Details |
|-----------|---------E
| Base Model | Stable Diffusion XL 1.0 (2.6B params) |
| Fine-tuning | Tri-Phase LoRA rank 32, bf16, 55K+ images |
| Curriculum | Phase 1 (Layout/1e-4) → Phase 2 (Perfection/2e-5) → Phase 3 (Style/5e-6) |
| Dataset | 55,000+ curated event posters, 55 categories |
| LLM | Llama 3.3 70B via Groq (free tier) |
| Prompt Engine | v3.0 — 20 styles, 30 events, 7 lighting, 10 color harmonies |
| Typography | Compositor v3.0 — 8 layout styles, 30+ fonts, 1,700+ on-demand, caption dropout |
| Upscaler | Real-ESRGAN 4x |
| Deployment | HF Spaces with ZeroGPU |

## Categories (55 themes)

| Group | Subcategories |
|-------|--------------E
| Tech Fest | Hackathons, AI/ML, robotics, coding competitions, cyber security |
| Cultural Event | Dance, music, drama, art exhibitions, poetry |
| College Events | Annual days, freshers, farewell, alumni meets |
| Sports | Cricket, football, basketball, athletics, chess |
| Festivals | Diwali, Holi, Navratri, Ganesh Chaturthi, Eid, Christmas |
| Workshops | Seminars, webinars, training sessions, conferences |
| Social | Blood donation, charity, environmental drives |
| Entertainment | DJ nights, concerts, standup comedy, movie screenings |

## Project Structure

```text
campus-ai/
├── configs/
│   └── config.yaml                  # Master configuration
├── scripts/
│   ├── pinterest_scraper.py         # Image scraper (CPU, network-bound)
│   ├── quality_filter.py            # GPU-accelerated quality filtering
│   ├── caption_generator.py         # Florence-2 captioning (GPU)
│   ├── split_dataset.py             # Dataset splitting (1000/200/100)
│   ├── test_checkpoint.py           # LoRA inference testing
│   ├── create_training_config.py    # ai-toolkit config generator
│   └── create_mixed_genre_dataset.py # Phase 4 cross-genre dataset
├── deployment/
│   ├── app.py                       # 6-tab Gradio application (v3.0)
│   ├── pipelines.py                 # Pipeline manager (SDXL/IP-Adapter/ESRGAN)
│   ├── prompt_engine.py             # Groq LLM prompt engine v3.0
│   ├── poster_compositor.py         # PIL typography compositor
│   ├── requirements.txt             # HF Space dependencies
│   └── README.md                    # HF Space card
├── assets/
│   └── fonts/                       # 30+ pre-cached fonts (Google Fonts CDN)
├── data/
│   ├── raw/                         # Scraped images (~1900/theme)
│   ├── processed/                   # GPU-filtered images (~1300/theme)
│   ├── final/                       # Captioned dataset (GPU)
│   ├── train/                       # 1000 images/theme
│   ├── val/                         # 200 images/theme
│   ├── test/                        # 100 images/theme
│   └── tuning-2/                    # Phase 4 mixed-genre dataset
├── models/                          # Trained LoRA checkpoints
├── outputs/                         # Generated outputs
├── docs/
│   ├── README.md                    # This file
│   ├── SETUP.md                     # Setup guide
│   ├── PIPELINE.md                  # Execution pipeline
│   ├── NOVELTY.md                   # Novelty & unique value proposition
│   ├── CAMPUS-AI-PROJECT-BRIEF.md   # Comprehensive project brief
│   └── architecture.html            # Visual architecture diagram
└── requirements.txt                 # Local dependencies
```

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate   # (or `source venv/bin/activate` on Linux/WSL)
pip install -r requirements.txt

# 2. Data Pipeline
python scripts/pinterest_scraper.py      # 🖥️ CPU  — Scrape posters (overnight)
python scripts/quality_filter.py         # 🎮 GPU  — Filter quality (~5 min)
python scripts/caption_generator.py      # 🎮 GPU  — Generate captions (overnight)
python scripts/split_dataset.py          # 🖥️ CPU  — Split 1000/200/100

# 3. Training
python scripts/create_training_config.py # 🖥️ CPU  — Generate ai-toolkit config
cd ai-toolkit && python run.py ../configs/train_sdxl_lora.yaml  # 🎮 GPU  — Phase 1 (Layout)
cd ai-toolkit && python run.py ../configs/train_sdxl_lora_phase2.yaml # 🎮 GPU — Phase 2 (Perfection)
cd ai-toolkit && python run.py ../configs/train_sdxl_lora_phase3.yaml # 🎮 GPU — Phase 3 (Style)
cd ai-toolkit && python run.py ../configs/train_sdxl_lora_phase4.yaml # 🎮 GPU — Phase 4 (Mixed-Genre)

# 4. Local Deployment
python deployment/app.py                 # 🖥️ CPU / 🎮 GPU — Launch 6-tab Gradio UI

# 5. Future Cloud Deployment (Hugging Face Spaces)
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/campus-ai-poster-sdxl models/sdxl/checkpoints/campus_ai_poster_sdxl/ .
# Push deployment/ files to HF Space
```

See [SETUP.md](SETUP.md) for detailed instructions. See [PIPELINE.md](PIPELINE.md) for step-by-step execution guide.

## Hardware

- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM) — used for quality filtering, captioning, training
- **CPU**: Intel Ultra 9 275HX — used for scraping, splitting
- **RAM**: 32GB
- **Training time**: ~10 hours (Phase 1 Layout + Phase 2 Perfection + Phase 3 Style)

## Author

**CounciL** — Campus-AI by CounciL

## License

MIT
