# Campus-AI by CounciL – Setup Guide

## Prerequisites

- **OS**: Windows 10/11 or Ubuntu 22.04+
- **Python**: 3.11+
- **GPU**: NVIDIA GPU with 12GB VRAM (RTX 5070 Ti used for development)
- **CUDA**: 12.1+ with matching drivers
- **Disk**: 100GB+ free space
- **Chrome**: Latest version (for Pinterest scraping)

## 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On WSL/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 2. Configuration

Edit `configs/config.yaml`:

```yaml
project:
  creator: "YOUR_NAME"      # ← Change this

deployment:
  hf_username: "YOUR_HF_USERNAME"   # ← Change this
```

### API Keys

| Service      | Where to Get                   | Config Key                          |
| ------------ | ------------------------------ | ----------------------------------- |
| Kaggle       | kaggle.com/settings            | `api_keys.kaggle`                   |
| Unsplash     | unsplash.com/developers        | `api_keys.unsplash`                 |
| Pexels       | pexels.com/api                 | `api_keys.pexels`                   |
| Groq         | console.groq.com               | Environment: `GROQ_API_KEY`         |
| HuggingFace  | huggingface.co/settings/tokens | CLI: `huggingface-cli login`        |

## 3. Data Pipeline

### Step 1: Scrape Images 🖥️ CPU (~6-12 hours)

```bash
python scripts/pinterest_scraper.py
# Or scrape a single category:
python scripts/pinterest_scraper.py
# Or scrape a single category:
python scripts/pinterest_scraper.py --category tech_fest
# Or targeted top-up for specific counts:
python scripts/pinterest_scraper.py --category workshops/coding --target 2800
```

**Output**: `data/raw/{category}/{subcategory}/` with ~1900 images per theme

### Step 2: Quality Filter 🎮 GPU (~5 min)

```bash
python scripts/quality_filter.py
```

Uses GPU-accelerated sharpness detection (Laplacian via PyTorch CUDA) and color analysis. Auto-detects GPU, falls back to CPU.

**Output**: `data/processed/{category}/` with ~1300+ high-quality images per theme

### Step 3: Caption Generation 🎮 GPU (~6-12 hours)

```bash
python scripts/caption_generator.py
```

Florence-2 runs in float16 on GPU. Includes `campus_ai_poster` trigger word and category-aware prefixes.

**Output**: `data/final/{category}/` with image + `.txt` caption pairs + `metadata.json`

### Step 4: Dataset Split 🖥️ CPU (~1 min)

```bash
python scripts/split_dataset.py
```

Fixed counts: **1000 train / 200 val / 100 test** per theme.

**Output**: `data/train/`, `data/val/`, `data/test/`

## 4. Training 🎮 GPU (~15 hours total)

### Install ai-toolkit

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -e .
cd ..
```

### Phase 1: Layout Pass (~3 hours)

Generates the initial configuration and trains block-in composition.

```bash
python scripts/create_training_config.py
# Outputs: configs/train_sdxl_lora.yaml

cd ai-toolkit
set HF_TOKEN=your_token_here
python run.py ../configs/train_sdxl_lora.yaml
cd ..
```

### Phase 2: Perfection Pass (~4.5 hours)

Uses the static `configs/train_sdxl_lora_phase2.yaml` (0.1 dropout, 2e-5 LR) to refine micro-details across the entire dataset (train/val/test).

```bash
cd ai-toolkit
set HF_TOKEN=your_token_here
python run.py ../configs/train_sdxl_lora_phase2.yaml
cd ..
```

### Phase 3: Style & Typography Pass (~2.5 hours)

Locks in the 8 typography layout styles and precise negative space using `train_sdxl_lora_phase3.yaml` (5e-6 LR) on the exact 6,448-image tuning dataset.

```bash
cd ai-toolkit
python run.py ../configs/train_sdxl_lora_phase3.yaml
cd ..
```

### Phase 4: Mixed-Genre Fine-Tuning (~3.5+ hours)

Generates 148,800 paired samples balancing 62 categories evenly with 8 modes of caption dropout, then gently trains at `2e-6` LR to blend styles seamlessly.

```bash
# 1. Generate the massive mixed-genre dataset
python scripts/create_mixed_genre_dataset.py --source data/train --output data/tuning-2 --target-per-cat 3000

# 2. Run the tuning pass
cd ai-toolkit
python run.py ../configs/train_sdxl_lora_phase4.yaml
cd ..
```

### Monitor

```bash
# In a separate terminal
nvidia-smi -l 30

# TensorBoard
tensorboard --logdir logs/tensorboard
```

### Test Checkpoints

```bash
python scripts/test_checkpoint.py
```

## 5. Local Testing 🖥️ CPU / 🎮 GPU

To test the application on your own machine using the trained LoRA:

```bash
python deployment/app.py
```

Then open `http://127.0.0.1:7860` in your web browser to access the 6-tab Gradio UI.

## 6. Deploy to Cloud (Future) ☁️ Cloud

Once local testing is complete, you can deploy to Hugging Face Spaces.

### Upload LoRA to Hugging Face

```bash
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/campus-ai-poster-sdxl models/sdxl/checkpoints/campus_ai_poster_sdxl/ .
```

### Create & Deploy HF Space

```bash
cd deployment
git init
huggingface-cli repo create campus-ai-poster-generator --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/campus-ai-poster-generator
git add app.py pipelines.py prompt_engine.py poster_compositor.py requirements.txt README.md
git commit -m "Deploy Campus-AI by CounciL"
git push space main
```

### Configure Secrets

In Space Settings → Variables and Secrets:

| Secret Name    | Value                |
| -------------- | -------------------- |
| `HF_USERNAME`  | your HF username     |
| `GROQ_API_KEY` | your Groq API key    |

## GPU Usage Summary

| Step               | Device            | Time                    |
| ------------------ | ----------------- | ----------------------- |
| Scraping           | 🖥️ CPU            | ~6-12h (network-bound)  |
| Quality Filter     | 🎮 GPU            | ~5 min                  |
| Captioning         | 🎮 GPU            | ~6-12h                  |
| Split              | 🖥️ CPU            | ~1 min                  |
| Training (Phase 1) | 🎮 GPU            | ~3h                     |
| Training (Phase 2) | 🎮 GPU            | ~4.5h                   |
| Training (Phase 3) | 🎮 GPU            | ~2.5h                   |
| Training (Phase 4) | 🎮 GPU            | ~3.5h                   |
| Upload             | 🖥️ CPU            | ~5 min                  |
| Live Demo          | ☁️ Cloud GPU      | HF ZeroGPU              |

## Troubleshooting

| Issue                    | Solution                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------- |
| CUDA OOM during training | Set `batch_size: 1` and `gradient_accumulation_steps: 4` in config                 |
| Pinterest blocking       | Increase sleep time, use VPN, or try alt sources                                   |
| Blurry outputs           | Increase `num_inference_steps` to 40                                               |
| Slow cold start on HF    | Send Space link 24h before demo to warm it up                                      |
| Groq rate limit          | Create multiple accounts, rotate API keys                                          |
| GPU not detected         | Verify CUDA install: `python -c "import torch; print(torch.cuda.is_available())"`  |
