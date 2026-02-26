#!/usr/bin/env python3

"""
Caption Generator — Florence-2 Native (transformers >= 4.56)

Multi-task captioning: MORE_DETAILED_CAPTION + OCR + DENSE_REGION_CAPTION

SETUP (run ONCE):
    pip install "transformers==4.57.3" tokenizers --upgrade
    rm -rf ~/.cache/huggingface/modules/transformers_modules/

Outputs:
    data/{split}/{category}/image.txt
    data/{split}/metadata.json
"""

import os
import re
import sys
import json
import logging
import argparse
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import transformers
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/caption_generator.log"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "ducviet00/Florence-2-large-hf"

TASKS    = ["<MORE_DETAILED_CAPTION>", "<OCR>", "<DENSE_REGION_CAPTION>"]
TASK_KEY = {
    "<MORE_DETAILED_CAPTION>": "visual",
    "<OCR>":                   "ocr",
    "<DENSE_REGION_CAPTION>":  "regions",
}
CATEGORY_LABELS = {
    "tech_fest":      "A technology fest event poster",
    "cultural_fest":  "A cultural festival event poster",
    "college_events": "A college event poster",
    "sports":         "A sports tournament event poster",
    "festivals":      "A festival celebration event poster",
    "workshops":      "A workshop or seminar event poster",
    "social":         "A social awareness event poster",
    "entertainment":  "An entertainment event poster",
    "styles":         "A stylized event poster",
    "general":        "An event poster",
    "diwali":         "A Diwali celebration event poster",
    "holi":           "A Holi festival event poster",
    "navratri":       "A Navratri festival event poster",
    "eid":            "An Eid celebration event poster",
    "ganesh":         "A Ganesh Chaturthi event poster",
}

# ─────────────────────────────────────────────────────────────────────────────
# Cache guard
# ─────────────────────────────────────────────────────────────────────────────
def _check_stale_cache():
    stale = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if stale.exists():
        logger.warning(
            f"Stale remote-code cache at {stale} — "
            "run: rm -rf ~/.cache/huggingface/modules/transformers_modules/"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Florence-2 Captioner
# Direct-class loading — bypasses auto_map, no Auto* classes used
# ─────────────────────────────────────────────────────────────────────────────
class Florence2Captioner:
    """Multi-task Florence-2 captioner using native transformers classes."""

    def __init__(self, device: str = "auto"):
        from transformers import Florence2ForConditionalGeneration, Florence2Processor
        from transformers.models.bart import BartTokenizerFast
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.dtype  = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info(f"transformers : {transformers.__version__}")
        logger.info(f"torch        : {torch.__version__}")
        logger.info(f"device/dtype : {self.device} / {self.dtype}")
        logger.info(f"Loading {MODEL_ID} ...")

        # Direct tokenizer load — bypasses AutoTokenizer & auto_map
        tokenizer = BartTokenizerFast.from_pretrained(MODEL_ID)

        # Patch image_token if missing (required by Florence2Processor.__init__)
        if not hasattr(tokenizer, "image_token") or tokenizer.image_token is None:
            tok_vocab   = tokenizer.get_vocab()
            image_token = next(
                (t for t in ["<image>", "</s>", "<unk>"] if t in tok_vocab), None
            )
            if image_token is None:
                tokenizer.add_tokens(["<image>"], special_tokens=True)
                image_token = "<image>"
            tokenizer.image_token    = image_token
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            logger.info(f"Patched image_token='{image_token}' (id={tokenizer.image_token_id})")

        # Direct image processor load — bypasses AutoImageProcessor & auto_map
        image_processor = CLIPImageProcessor.from_pretrained(MODEL_ID)

        # Assemble processor from components (bypasses from_pretrained's AutoTokenizer call)
        self.processor = Florence2Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

        # Direct model load — bypasses AutoModel & auto_map in config.json
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            ignore_mismatched_sizes=False,
        ).to(self.device)
        self.model.eval()
        logger.info("Florence-2 loaded successfully.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _safe_to_device(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
            elif v.is_floating_point():
                out[k] = v.to(device=self.device, dtype=self.dtype)
            else:
                out[k] = v.to(device=self.device)
        return out

    def _run_task(self, image: Image.Image, task: str) -> str:
        """Run one Florence-2 task; returns clean decoded string."""
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = self._safe_to_device(inputs)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
            )

        # Decode directly — post_process_generation raises
        # "Unsupported parse task: pure_text/description_with_bboxes"
        # in transformers 4.57.3 due to processor_config task-type mismatch.
        # Direct decoding gives identical text for all tasks we use.
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Strip task prompt tokens if they leaked through decode
        for tok in TASKS:
            text = text.replace(tok, "").strip()

        # DENSE_REGION_CAPTION contains <loc_NNN> coordinate tokens;
        # strip them to keep only the human-readable region labels
        if task == "<DENSE_REGION_CAPTION>":
            text = re.sub(r"<loc_\d+>", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ,")

        return text

    # ── public API ────────────────────────────────────────────────────────────

    def caption(self, image: Image.Image) -> dict:
        """Run all tasks; returns {visual, ocr, regions}."""
        if image.width < 16 or image.height < 16:
            raise ValueError(f"Image too small: {image.size}")
        results = {}
        for task in TASKS:
            key = TASK_KEY[task]
            try:
                results[key] = self._run_task(image, task)
            except Exception as e:
                logger.warning(f"Task {task} failed: {e}\n{traceback.format_exc()}")
                results[key] = ""
        return results

    def build_caption(self, task_results: dict, category: str) -> str:
        """Merge multi-task results into one Flux fine-tuning caption."""
        parent = category.split("/")[0] if "/" in category else category
        prefix = CATEGORY_LABELS.get(category, CATEGORY_LABELS.get(parent, "An event poster"))

        visual  = task_results.get("visual", "").strip()
        ocr     = task_results.get("ocr", "").strip()
        regions = task_results.get("regions", "").strip()

        parts = [f"campus_ai_poster {prefix}."]
        if visual:
            parts.append(visual)
        if ocr:
            ocr_clean = " | ".join(dict.fromkeys(
                t.strip() for t in ocr.replace("\n", " | ").split(" | ") if t.strip()
            ))
            parts.append(f"[Text on poster: {ocr_clean}]")
        if regions:
            r = regions[:400].rsplit(".", 1)[0] + "." if len(regions) > 400 else regions
            parts.append(f"[Design elements: {r}]")

        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path) -> set:
    return set(json.loads(path.read_text())) if path.exists() else set()

def save_checkpoint(path: Path, done: set):
    path.write_text(json.dumps(sorted(done)))

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def run_captioning(config: dict, splits: list):
    _check_stale_cache()

    data_paths = config.get("paths", {}).get("data", {})
    if not data_paths:
        logger.error("Missing 'paths.data' in config.yaml")
        sys.exit(1)

    try:
        captioner = Florence2Captioner()
    except Exception:
        logger.error(f"Could not load Florence-2:\n{traceback.format_exc()}")
        sys.exit(1)

    for split in splits:
        if split not in data_paths:
            logger.warning(f"'{split}' not in config paths. Skipping.")
            continue
        split_dir = Path(data_paths[split])
        if not split_dir.exists():
            logger.warning(f"Dir not found: {split_dir}. Skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  Split: {split.upper()} ({split_dir})")
        logger.info(f"{'='*60}")

        ckpt_path = split_dir / ".caption_checkpoint.json"
        done = load_checkpoint(ckpt_path)
        logger.info(f"Checkpoint: {len(done)} already captioned.")

        all_imgs = []
        for root, _, files in os.walk(split_dir):
            rp = Path(root)
            for fname in sorted(files):
                fp = rp / fname
                if fp.suffix.lower() in IMAGE_EXTS:
                    cat = str(rp.relative_to(split_dir)).replace("\\", "/")
                    all_imgs.append((cat if cat != "." else "general", fp))

        logger.info(f"Total : {len(all_imgs)}  |  Remaining : {len(all_imgs) - len(done)}")
        remaining = [(c, p) for c, p in all_imgs if str(p) not in done]

        if not remaining:
            logger.info("Already complete.")
            continue

        meta_path = split_dir / "metadata.json"
        metadata: list = []
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read existing metadata; starting fresh.")

        failed = 0
        sample_logged = False

        for cat, img_path in tqdm(remaining, desc=split):
            try:
                img = Image.open(img_path).convert("RGB")
                img.load()
            except Exception as e:
                logger.warning(f"Bad image [{img_path.name}]: {e}")
                failed += 1
                continue

            try:
                results = captioner.caption(img)
                caption = captioner.build_caption(results, cat)
                if not sample_logged:
                    logger.info(f"Sample caption:\n  {caption[:300]}...")
                    sample_logged = True
            except Exception:
                logger.warning(f"Caption failed [{img_path.name}]:\n{traceback.format_exc()}")
                failed += 1
                continue

            img_path.with_suffix(".txt").write_text(caption, encoding="utf-8")
            metadata.append({
                "image":        str(img_path),
                "caption_file": str(img_path.with_suffix(".txt")),
                "caption":      caption,
                "visual":       results.get("visual", ""),
                "ocr":          results.get("ocr", ""),
                "regions":      results.get("regions", ""),
                "category":     cat,
                "width":        img.size[0],
                "height":       img.size[1],
                "timestamp":    datetime.now().isoformat(),
            })
            done.add(str(img_path))
            if len(done) % 50 == 0:
                save_checkpoint(ckpt_path, done)

        save_checkpoint(ckpt_path, done)
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Done — captioned: {len(metadata)}, failed/skipped: {failed}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All splits complete.")


def main():
    p = argparse.ArgumentParser(description="Florence-2 Caption Generator")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    run_captioning(load_config(args.config), args.splits)


if __name__ == "__main__":
    main()