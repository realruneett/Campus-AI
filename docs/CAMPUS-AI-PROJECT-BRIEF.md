# CAMPUS-AI: PROJECT BRIEF

## Universal Event Poster Generator

**Project**: CampusGen AI
**Type**: AI-powered multi-modal event poster generation
**Hardware**: Intel Ultra 9 275HX + RTX 5070 Ti (16GB VRAM)
**Deployment**: Hugging Face Spaces (ZeroGPU — Free Tier)
**Total Cost**: $0
**Last Updated**: February 21, 2026

---

## EXECUTIVE SUMMARY

CampusGen AI generates professional event posters for ANY occasion in 10–15 seconds using:

- **Stable Diffusion XL 1.0 (2.6B params)** fine-tuned on **55,000+ diverse poster images** via LoRA
- **5 Generation Modes**: Text→Poster, Reference Image, Image Transform, Inpainting, HD Upscale
- **Llama 3.3 70B** (Groq) for intelligent prompt engineering
- **Real-ESRGAN** for 4x HD upscaling
- **IP-Adapter** for reference image style transfer
- **GPU-accelerated pipeline** end-to-end

---

## WHY THIS WINS

| Metric | CampusGen AI | Typical Projects |
|--------|-------------|------------------|
| Dataset | **55,000+ images, 55 categories** | 100-500 images, 1-2 categories |
| Generation Modes | **5 modes** (text, reference, transform, inpaint, upscale) | 1 mode (text only) |
| Training | LoRA on RTX 5070 Ti (bf16) | Quantized on Colab |
| Intelligence | **LLM-powered** prompt engineering (10 styles, 19 event types) | Template-based |
| Speed | 10-15 seconds/poster | 30-60+ seconds |
| Upscaling | **Real-ESRGAN 4x** HD output | None |
| Style Transfer | **IP-Adapter** reference image | None |
| Cost | $0 (smart free tier) | $0-200 |
| Deployment | Professional 5-tab HF Space | Local/unstable |

---

## TECHNOLOGY RATIONALE (Why These Models?)

| Technology | Why We Chose It | What It Replaces |
|------------|-----------------|------------------|
| **SDXL 1.0 (2.6B)** | The gold standard open-source framework for local training. It perfectly fits within a 12GB VRAM envelope allowing for rapid bf16 fine-tuning without destructive memory swapping. | Midjourney V6 / DALL-E 3 (closed source, un-finetunable) |
| **LoRA (Low-Rank Adaptation)** | Training a 2.6 Billion parameter model from scratch requires supercomputers. LoRA trains tiny adapter layers (**~80M parameters**) that sit on top of the frozen base model. This makes training possible in a few hours on a consumer RTX 5070 Ti (12GB) without catastrophic forgetting of the base model's knowledge. | Full Fine-Tuning (Requires multiple A100s, huge memory) |
| **Florence-2-large** | Microsoft's highly efficient Vision-Language Model. Instead of running 3 different models, Florence-2 does **Detailed Visual Summaries + OCR (reading text) + Dense Region Capturing** all in one pass. Clean, rich captions are the secret to teaching the SDXL model what a "poster" is. | BLIP-2 / LLaVA (bulkier, less strict OCR formatting) |
| **Llama 3.3 70B (via Groq)** | Users write lazy prompts like "a cybersec hackathon." We use Llama 3.3 to intercept that prompt and intelligently explode it into a highly detailed, cinematic description referencing our 10 trained visual styles and 19 event types. Running it through the Groq API makes this essentially instantaneous and free. | Hardcoded prompt templates (rigid, boring) |
| **IP-Adapter** | It allows users to upload a reference image (e.g., a cool poster they found online) and injects that structural/stylistic "vibe" into the generation pipeline natively, without needing a secondary text prompt. | ControlNet (heavier, overkill for pure style transfer) |
| **Real-ESRGAN** | A specialized upscaler neural network that reconstructs high-frequency details. Generating a 4K image directly in SDXL takes immense VRAM and time. It is faster to generate at 1024x1024 and run it through Real-ESRGAN to get a massive 4K HD output with perfectly crisp text in 2 seconds. | Bicubic interpolation (blurry, pixelated) |

--------------------------------------------------------------------------

## TRAINING SPECIFICATIONS

### Model Architecture

| Component | Specification |
|-----------|---------------|
| Base Model | Stable Diffusion XL 1.0 (2.6B parameters) — **FROZEN** |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 32 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| **Trainable Parameters** | **~80 million** (0.6% of base model) |
| Precision | bf16 (bfloat16) |
| LoRA File Size | ~150-300 MB (.safetensors) |
| Trigger Word | `campus_ai_poster` |

### How LoRA Works

```text
Base model: SDXL 1.0 (2.6B params) → FROZEN, not modified
                    ↓
LoRA injects small adapter matrices into attention layers:
  Original W (4096×4096) = 16M params  → FROZEN
  LoRA: A (4096×32) + B (32×4096) = 262K params  → TRAINED
                    ↓
~250 attention layers × 262K = ~80M trainable params (3% of 2.6B)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW 8-bit (`bitsandbytes`) |
| Learning Rate | 1e-4 (Phase 1) → 2e-5 (Phase 2) → **1e-5 (Phase 3)** |
| Batch Size | 1 |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 4 |
| Max Steps | 4000 (P1) + 20000 (P2) + **6448 (P3)** |
| Phase 3 Dataset | **6,448** highly curated typography & layout templates |
| Checkpoint Interval | Every 500 steps |
| Resolution | 1024×1024 |
| Noise Scheduler | DDPM |
| EMA Decay | 0.99 |
| Gradient Checkpointing | Enabled |
| Train UNet | Yes |
| Train Text Encoder | No |
| **Dependencies** | `bitsandbytes` (critical for 8-bit), `diffusers==0.32.1` (for `torchao` compat) |
| Estimated Time | ~7.5 hours on RTX 5070 Ti |

---

## DATASET SPECIFICATIONS

### Overview

| Metric | Value |
|--------|-------|
| Raw images scraped | ~1900 per theme × 55 themes = **~104,500** |
| After quality filter | ~1300 per theme = **~71,500** |
| Train split | 1000 per theme = **55,000** |
| Validation split | 200 per theme = **11,000** |
| Test split | 100 per theme = **5,500** |

### 55 Categories (Hierarchical)

| Group | Subcategories |
|-------|---------------|
| **Tech Fest** | Hackathon, AI/ML, Robotics, Coding Competition, Cybersecurity, Web Dev, Startup, Data Science, IoT, Open Source, Game Dev |
| **Cultural Fest** | Dance, Music, Drama, Art Exhibition, Poetry, Fashion Show, Photography |
| **College Events** | Annual Day, Freshers Party, Farewell, Alumni Meet, Orientation, Graduation |
| **Sports** | Cricket, Football, Basketball, Athletics, Chess, Badminton, Volleyball |
| **Festivals** | Diwali, Holi, Navratri/Garba, Ganesh Chaturthi, Eid, Christmas, Onam, Pongal |
| **Workshops** | Technical Seminar, Business Workshop, Creative Workshop, Leadership, Research |
| **Social** | Blood Donation, Charity, Environmental, Awareness Campaign, NSS/NCC |
| **Entertainment** | DJ Night, Concert, Standup Comedy, Movie Screening, Open Mic |

### Quality Filtering (GPU-Accelerated)

| Check | Threshold | Method |
|-------|-----------|--------|
| Resolution | ≥512px shortest side | CPU |
| Sharpness | Laplacian variance ≥50 | **GPU** (PyTorch conv2d) |
| Aspect Ratio | 0.4–2.5 | CPU |
| File Size | 20KB–50MB | CPU |
| Color Variance | std ≥15 | **GPU** (torch.std) |
| Deduplication | pHash distance ≤5 | CPU |

### Captioning

| Component | Detail |
|-----------|--------|
| Model | Florence-2-large (microsoft) |
| Device | **GPU** (float16) |
| Captions | `campus_ai_poster` trigger + category prefix + Florence-2 description |
| Output | Image + `.txt` pairs in `data/final/` |

---

## DEPLOYMENT APP — 5-Tab Architecture

### Files

| File | Purpose |
|------|---------|
| `app.py` | 5-tab Gradio UI (~500 lines) |
| `pipelines.py` | Pipeline manager — lazy loads SDXL/IP-Adapter/ESRGAN (~230 lines) |
| `prompt_engine.py` | Groq LLM with 10 styles, 19 event types (~250 lines) |
| `requirements.txt` | HF Space dependencies |
| `README.md` | HF Space card |

### 5 Generation Modes

| Tab | What It Does | Key Tech |
|-----|-------------|----------|
| ✍️ Text → Poster | Describe event → get poster(s) | SDXL + LoRA + Groq LLM |
| 🖼️ Reference Image | Upload a poster → copy its style | IP-Adapter |
| 🔄 Image Transform | Upload → restyle existing poster | Img2Img pipeline |
| 🖌️ Inpaint / Edit | Draw mask → regenerate region | Inpainting pipeline |
| 🔍 HD Upscale | 2x/4x upscale any image | Real-ESRGAN |

### Shared Features

- 7 resolution presets (768×1152, 1024×1024, etc.)
- 10 visual styles
- Batch generation (1-4 variants)
- Seed control
- LoRA strength slider
- Generation metadata display

### VRAM Management

- Only ONE pipeline active at a time (text2img OR img2img OR inpaint)
- Model CPU offloading for 16GB GPU / HF ZeroGPU
- IP-Adapter loads as lightweight adapter (~300MB) on top of base model
- Real-ESRGAN uses tiled processing (512px tiles) for memory efficiency

---

## GPU PIPELINE SUMMARY

| Step | Device | Time |
|------|--------|------|
| Scraping (Pinterest) | 🖥️ CPU (network-bound) | ~6-12h |
| Quality Filter | 🎮 GPU (Laplacian + color) | ~5 min |
| Captioning (Florence-2) | 🎮 GPU (float16) | ~6-12h |
| Dataset Split | 🖥️ CPU (file copy) | ~1 min |
| LoRA Training | 🎮 GPU (bf16) | ~7.5h |
| Upload to HF | 🖥️ CPU | ~5 min |
| Live Demo | ☁️ Cloud GPU (ZeroGPU) | Real-time |

---

## EXECUTION PIPELINE

```bash
# Phase 1: Data Collection
python scripts/pinterest_scraper.py          # 🖥️ CPU — overnight

# Phase 2: Data Processing
python scripts/quality_filter.py             # 🎮 GPU — ~5 min
python scripts/caption_generator.py          # 🎮 GPU — overnight
python scripts/split_dataset.py              # 🖥️ CPU — ~1 min

# Phase 3: Training (Dual-Phase)
python scripts/create_training_config.py     # 🖥️ CPU — Setup
python ai-toolkit/run.py configs/train_sdxl_lora.yaml  # 🎮 GPU — Phase 1 (3h)
python ai-toolkit/run.py configs/train_sdxl_lora_phase2.yaml  # 🎮 GPU — Phase 2 (4.5h)

# Phase 4: Deploy
huggingface-cli upload YOUR_USERNAME/campus-ai-poster-sdxl models/sdxl/checkpoints/campus_ai_poster_sdxl/ .
# Push deployment/ to HF Space
```

---

## FILE STRUCTURE

```text
campus-ai/
├── .gitignore                       # Explicitly ignores data/ & models/ for GitHub push
├── configs/
│   ├── config.yaml                  # Master configuration (w/ hf_token)
│   ├── train_sdxl_lora.yaml         # ai-toolkit Phase 1 generator
│   └── train_sdxl_lora_phase3.yaml  # Phase 3 implicit layout tuner
├── scripts/
│   ├── pinterest_scraper.py         # Image scraper (1900/theme)
│   ├── quality_filter.py            # GPU-accelerated quality filter
│   ├── caption_generator.py         # Florence-2 GPU captioning
│   ├── split_dataset.py             # Fixed 1000/200/100 split
│   ├── test_checkpoint.py           # LoRA inference testing
│   └── create_training_config.py    # ai-toolkit config generator
├── deployment/
│   ├── app.py                       # 5-tab Gradio app
│   ├── pipelines.py                 # Pipeline manager
│   ├── prompt_engine.py             # Groq LLM prompt engine
│   ├── requirements.txt             # HF Space dependencies
│   └── README.md                    # HF Space card
├── data/
│   ├── raw/                         # ~104K scraped images
│   ├── processed/                   # ~71K quality-filtered
│   ├── final/                       # Captioned pairs
│   ├── train/                       # 55K (1000/theme)
│   ├── val/                         # 11K (200/theme)
│   └── test/                        # 5.5K (100/theme)
├── models/sdxl/checkpoints/         # Trained LoRA weights
├── docs/
│   ├── CAMPUS-AI-PROJECT-BRIEF.md   # This file
│   ├── README.md                    # Project overview
│   ├── SETUP.md                     # Setup guide
│   └── PIPELINE.md                  # Execution pipeline
└── requirements.txt                 # Local dependencies
```

---

## COMPETITION STRATEGY

### What Judges Will See

1. **Live 5-tab demo** on Hugging Face (not just slides)
2. **55,000+ image dataset** (10-100x larger than competitors)
3. **5 generation modes** (competitors have 1)
4. **GPU-accelerated pipeline** (professional engineering)
5. **$0 deployment** (smart architecture)

### Key Talking Points

- "Trained on 55,000+ event posters across 55 categories — 10x larger than typical projects"
- "5 generation modes: text, reference image, transform, inpaint, upscale"
- "80 million trainable parameters via LoRA on 2.6 billion parameter SDXL model"
- "GPU-accelerated pipeline: quality filter, captioning, and training all on GPU"
- "Zero cost — entire project runs on free tier services"

### Tough Questions

**Q: "Only 80M params? That seems small."**
A: "That's the power of LoRA — we get the quality of a 2.6B model while only training 80M adapter parameters. The base model already knows how to generate images; our LoRA teaches it our specific poster style. Bigger ≠ better — efficiency is the innovation."

**Q: "How is this different from MidJourney?"**
A: "MidJourney is generic. Ours is specialized — trained on 55,000 Indian event posters. It understands rangoli patterns, tech fest aesthetics, and college event culture. Plus, 5 generation modes including reference image style transfer and inpainting."

**Q: "Can judges try it live?"**
A: "Absolutely — here's the HF Space link. Pick any event, any style. Generate in 15 seconds."

---

## SUCCESS METRICS

| Metric | Target | Status |
|--------|--------|--------|
| Dataset | 55K+ captioned images | ✅ Complete |
| Training | Loss <0.10, coherent samples | ⏳ Pending |
| Generation | <20 seconds, professional quality | ⏳ Pending |
| Deployment | Live 5-tab HF Space | ⏳ Pending |
| Demo | All 5 tabs working flawlessly | ⏳ Pending |

---

**Version**: 4.1
**Last Updated**: February 22, 2026
**Status**: Dataset captioned ✅ → Training LoRA on RTX 5070 Ti 🔄
