# Campus-AI by CounciL — Full Execution Pipeline

> Step-by-step guide from raw data to live hackathon demo.

---

## Phase 1: Data Collection (Raw Ingestion) 🖥️ CPU

⏱️ **Runtime Strategy:** ~6-12 hours across distributed local instances (Run Async)
⚙️ **Hardware Requirement:** standard CPU, high bandwidth connection, 500GB+ NVMe SSD recommended.

```bash
cd e:\campus-ai
python scripts/pinterest_scraper.py
```

- Downloads **1900 images per theme** across 55 categories
- Saves to `data/raw/` with hierarchical folders (`tech_fest/hackathon/`, etc.)
- **Global Deduplication:** Uses a custom `GlobalImageDeduplicator` employing Perceptual Hashing (PHash) and a high-performance SQLite caching layer (`data/phash_cache.db`). Scans ~130,000+ existing images instantly to ensure zero duplicates across the entire corpus.
- Skips already-downloaded images safely — safe to restart

---

## Phase 1.5: Tuning Dataset Collection 🕸️ CPU

⏱️ ~1-2 hours (Targeted run)

```bash
cd e:\campus-ai
python scripts/pinterest_tuning_scraper.py
```

- **Strict Enforcement Engine:** Uses a heavily modified Selenium scraper that recursively scrolls and cycles through search queries until it achieves strictly **100 unique images** per 55 specific subcategories.
- **Data Isolation:** Saves uniquely to `data/tuning/<category>/<subcategory>/`.
- **Absolute Uniqueness:** Pipes newly scraped images through the identical `GlobalImageDeduplicator` cache, guaranteeing these 5,500 tuning images have absolutely zero overlap with the 100k+ images in the main `data/raw`, `data/train`, or `data/val` datasets.

---

## Phase 2: Data Processing & Quality Assurance

### 2a. Quality Filter 🎮 GPU (~5 min)

⚙️ **Algorithm:** Offloads Canny Edge / Laplacian Variance calculations to CUDA to rapidly sweep 130k+ images for optimal sharpness and color contrast.

```bash
python scripts/quality_filter.py
```

Removes blurry, low-res, duplicate images → saves to `data/processed/`

### 2b. Caption Generation 🎮 GPU (~6-12 hours)

⚙️ **Model Architecture:** Microsoft `Florence-2-large` via HuggingFace `transformers`.
⚙️ **Hardware Target:** RTX 4070 Ti / 5070 Ti (Float16 precision, ~12GB VRAM allocation).

```bash
python scripts/caption_generator.py
```

- Transforms pixel data into rich spatial text (e.g., "Bold sans-serif typography on the top left, neon cyber-punk background, dates on bottom right"). Saves `.txt` pairs to `data/final/`. These pairs are critical for SDXL cross-attention during LoRA tuning.

### 2c. Dataset Split 🖥️ CPU (~1 min)

⚙️ **Logic:** Deterministic pseudo-random seed to guarantee identical splits across team machines.

```bash
python scripts/split_dataset.py
```

Splits into **1000 train / 200 val / 100 test** per theme → `data/train/`, `data/val/`, `data/test/`

---

## Phase 3: Fine-Tune LoRA 🎮 GPU (~10 hours total)

**Core Training Engine:** `ai-toolkit` featuring LoRA+ optimization. Employs a tri-phase curriculum to circumvent catastrophic forgetting while molding the SDXL 1.0 architecture.

### 3a. Phase 1: Layout Pass (~3 hours)

- **Objective:** Teaches the model the macro-composition, layout, and lighting of the 55 event categories.
- **Data Source:** Exclusively uses `data/train/` (to preserve validation sets for Phase 2).

```bash
# 1. Generate optimal JSON layout training config
python scripts/create_training_config.py

# 2. Train Layout Pass (Learning Rate: 1e-4)
python ai-toolkit/run.py configs/train_sdxl_lora.yaml
```

Output: `models/sdxl/checkpoints/campus_ai_poster_sdxl/campus_ai_poster_sdxl.safetensors`

### 3b. Phase 2: Perfection Pass (~4.5 hours)

- **Objective:** Bakes in micro-details, sharp Indian cultural textures (e.g., diwali lamps, specific fonts), and perfect aesthetic adherence.
- **Mechanics:** Resumes gracefully from the Phase 1 `.safetensors` weights. Drops learning rate sequentially (2e-5) while utilizing the full 100% data blend (`train`, `val`, `test`).

```bash
# Train Perfection Pass (Internal Checkpoint Resume)
python ai-toolkit/run.py configs/train_sdxl_lora_phase2.yaml
```

Output: Overwrites the `.safetensors` with the high-fidelity weights.

### 3c. Phase 3: Style & Typography Pass (~2.5 hours)

- **Objective:** Final polish for 8 unique typography layout styles and precise color-matching, training the model to leave clean negative space for compositor overlay.
- **Mechanics:** Drops learning rate to (5e-6). Uses a small, highly curated dataset of typography-heavy posters.

```bash
# Train Style Pass
python ai-toolkit/run.py configs/train_sdxl_lora_phase3.yaml
```

Output: Final `campus_ai_poster_sdxl_phase3.safetensors` ready for deployment.

---

## Phase 4: Cross-Genre Dataset & Tuning (Optional Experimental)

**Objective.** To teach the model to blend styles across categories (e.g. tech fest layouts with cultural lighting).

```bash
python scripts/create_mixed_genre_dataset.py --source data/train --output data/tuning-2 --target-per-cat 3000
```

Produces ~165,000 blended caption samples for experimental Phase 4 training.

```bash
# Train Phase 4 Mixed-Genre
python ai-toolkit/run.py configs/train_sdxl_lora_phase4.yaml
```

Output: Final `campus_ai_poster_sdxl_phase4.safetensors` ready for deployment.

---

## Phase 5: Local Demo & Testing 🖥️ CPU / 🎮 GPU

Before deploying to the cloud, spin up the 6-tab Gradio UI locally using your RTX 5070 Ti.

```bash
python deployment/app.py
```

Open `http://localhost:7860` to verify all 6 tabs (Text→Poster, Reference Image, Image Transform, Inpaint/Edit, HD Upscale, Edit Poster) and LoRA integration are working correctly.

---

## Phase 6: Future Cloud Deployment (Hugging Face)

In the future, when ready to push to production:

### 5a. Install & Login

```bash
pip install huggingface-hub[cli]
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### 5b. Upload LoRA Weights

```bash
huggingface-cli upload YOUR_USERNAME/campus-ai-poster-sdxl models/sdxl/checkpoints/campus_ai_poster_sdxl/ .
```

### 5c. Create & Deploy HF Space

```bash
cd deployment
git init
huggingface-cli repo create campus-ai-poster-generator --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/campus-ai-poster-generator
git add app.py pipelines.py prompt_engine.py poster_compositor.py requirements.txt README.md
git commit -m "Deploy Campus-AI by CounciL"
git push space main
```

### 5d. Add Secrets (on HF website)

Go to **Space Settings → Variables and Secrets** and add:

| Secret Name    | Value                |
|---------------|----------------------|
| `HF_USERNAME`  | your HF username     |
| `GROQ_API_KEY` | your Groq API key    |

---

## Phase 6: Test Cloud Live ☁️ Cloud GPU

Open `https://huggingface.co/spaces/YOUR_USERNAME/campus-ai-poster-generator` and test all 6 tabs.

---

## HF Free vs Pro

| Feature | Free | Pro ($9/mo) |
|---------|------|-------------|
| ZeroGPU (shared A100) | ✅ Low priority | ✅ High priority |
| Private Spaces | ❌ | ✅ |
| Persistent Storage | ❌ | ✅ |
| Cold start | Slower | Faster |

**Verdict: Free tier works for a hackathon demo.** Upgrade to Pro only if the queue is too slow during judging.

---

## Quick Reference

```
pinterest_scraper.py  →  data/raw/          (1900 images/theme)
pinterest_tuning_scraper.py → data/tuning/  (Strictly 100 entirely unique images/theme)
image_deduplicator.py →  data/phash_cache.db (O(1) lookups via SQLite PHash)
quality_filter.py     →  data/processed/    (~1300 quality-passed/theme)
caption_generator.py  →  data/final/        (image + caption pairs)
split_dataset.py      →  data/train/val/test/ (1000/200/100)
create_training_config.py → configs/train_sdxl_lora.yaml
ai-toolkit/run.py     →  configs/train_sdxl_lora.yaml (Phase 1 Layout)
ai-toolkit/run.py     →  configs/train_sdxl_lora_phase2.yaml (Phase 2 Detail)
test_checkpoint.py    →  poster_compositor.py (SDXL Art + PIL Typography)
deployment/app.py     →  Local 6-Tab Gradio Server (Primary Test)
deployment/prompt_engine.py → v3.0 (20 styles, 30 events, 7 lighting, 10 color harmonies)
deployment/poster_compositor.py → PIL Typography (auto zone detection, 3 layouts)
huggingface-cli upload → HF Space             (Future Production)
```

---

*Campus-AI by CounciL — Built for the Indian campus community*
