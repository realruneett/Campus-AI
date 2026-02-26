# CampusGen AI – Universal Event Poster Generator

> AI-powered event poster generation for any occasion in 10–15 seconds.

## Overview

CampusGen AI generates professional event posters using:

- **Stable Diffusion XL 1.0** fine-tuned on 55,000+ diverse poster images via LoRA
- **Llama 3.3 70B** (Groq) for natural language event understanding
- **5 Generation Modes**: Text→Poster, Reference Image, Img2Img, Inpainting, HD Upscale
- **GPU-accelerated pipeline** from data processing to training
- **Zero cost** deployment on Hugging Face Spaces (ZeroGPU)

## Architecture

```text
User Input → Groq LLM (prompt engineering) → SDXL 1.0 + LoRA → HD Upscale → Poster
                                                  ↑
                                     IP-Adapter (reference style)
                                     Img2Img (transform)
                                     Inpainting (edit regions)
```

| Component | Details |
|-----------|---------|
| Base Model | Stable Diffusion XL 1.0 (2.6B params) |
| Fine-tuning | Dual-Phase LoRA rank 32, bf16, 55K+ images |
| Curriculum | Phase 1 (Layout/1e-4) → Phase 2 (Perfection/2e-5) |
| Dataset | 55,000+ curated event posters, 55 categories |
| LLM | Llama 3.3 70B via Groq (free tier) |
| Upscaler | Real-ESRGAN 4x |
| Deployment | HF Spaces with ZeroGPU |

## Categories (55 themes)

| Group | Subcategories |
|-------|--------------|
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
│   └── create_training_config.py    # ai-toolkit config generator
├── deployment/
│   ├── app.py                       # 5-tab Gradio application
│   ├── pipelines.py                 # Pipeline manager (SDXL/IP-Adapter/ESRGAN)
│   ├── prompt_engine.py             # Groq LLM prompt engineering
│   ├── requirements.txt             # HF Space dependencies
│   └── README.md                    # HF Space card
├── data/
│   ├── raw/                         # Scraped images (~1900/theme)
│   ├── processed/                   # GPU-filtered images (~1300/theme)
│   ├── final/                       # Captioned dataset (GPU)
│   ├── train/                       # 1000 images/theme
│   ├── val/                         # 200 images/theme
│   └── test/                        # 100 images/theme
├── models/                          # Trained LoRA checkpoints
├── outputs/                         # Generated outputs
├── docs/
│   ├── README.md                    # This file
│   ├── SETUP.md                     # Setup guide
│   └── PIPELINE.md                  # Execution pipeline
└── requirements.txt                 # Local dependencies
```

## Quick Start

```bash
# 1. Setup
conda create -n campus-ai python=3.11
conda activate campus-ai
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

# 4. Deploy
huggingface-cli upload YOUR_USERNAME/campus-ai-poster-sdxl models/sdxl/lora/ .  # Upload LoRA
# Push deployment/ files to HF Space
```

See [SETUP.md](SETUP.md) for detailed instructions. See [PIPELINE.md](PIPELINE.md) for step-by-step execution guide.

## Hardware

- **GPU**: NVIDIA RTX 5070 Ti (12GB VRAM) — used for quality filtering, captioning, training
- **CPU**: Intel Ultra 9 275HX (24 cores) — used for scraping, splitting
- **RAM**: 32GB
- **Training time**: ~7.5 hours (Phase 1 Layout + Phase 2 Perfection)

## Author

**M Runeet Kumar** – Ashta/Indore, MP, India

## License

MIT
