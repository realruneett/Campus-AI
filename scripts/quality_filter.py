#!/usr/bin/env python3
"""
Image Quality Filter (GPU-Accelerated)
Filters raw scraped images based on resolution, sharpness, aspect ratio,
file size, and deduplication. Uses GPU for batch sharpness and color analysis.
Outputs high-quality images to data/processed/.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import cv2
import numpy as np
import imagehash
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ─── SM120 (Blackwell) CUDA optimizations ───────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-Accelerated Quality Checker
# ─────────────────────────────────────────────────────────────────────────────
class ImageQualityChecker:
    """
    Evaluate image quality using GPU-accelerated sharpness and color analysis.
    Falls back to CPU if no CUDA device is available.
    """

    # Laplacian kernel for GPU sharpness detection
    LAPLACIAN_KERNEL = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    def __init__(
        self,
        min_resolution: int = 512,
        min_sharpness: float = 50.0,
        min_aspect_ratio: float = 0.4,
        max_aspect_ratio: float = 2.5,
        min_file_size_kb: int = 20,
        max_file_size_mb: int = 50,
        device: str = "auto",
    ):
        self.min_resolution = min_resolution
        self.min_sharpness = min_sharpness
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_file_size_bytes = min_file_size_kb * 1024
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # GPU setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._kernel = self.LAPLACIAN_KERNEL.to(self.device)
        logger.info(f"Quality checker using device: {self.device}")

    def _gpu_sharpness(self, img_array: np.ndarray) -> float:
        """Compute sharpness using Laplacian on GPU."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Move to GPU as torch tensor
        tensor = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        # Apply Laplacian convolution on GPU
        laplacian = F.conv2d(tensor, self._kernel, padding=1)
        sharpness = laplacian.var().item()

        return sharpness

    def _gpu_color_std(self, img_array: np.ndarray) -> float:
        """Compute color standard deviation on GPU."""
        tensor = torch.from_numpy(img_array.astype(np.float32)).to(self.device)
        return tensor.std().item()

    def check(self, image_path: Path) -> tuple[bool, dict]:
        """
        Check image quality. Returns (passed, metrics_dict).
        Sharpness and color checks run on GPU.
        """
        metrics = {
            "path": str(image_path),
            "passed": False,
            "reason": None,
        }

        # File size check (CPU — trivial)
        file_size = image_path.stat().st_size
        metrics["file_size_bytes"] = file_size
        if file_size < self.min_file_size_bytes:
            metrics["reason"] = "file_too_small"
            return False, metrics
        if file_size > self.max_file_size_bytes:
            metrics["reason"] = "file_too_large"
            return False, metrics

        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            metrics["reason"] = "unreadable"
            return False, metrics

        w, h = img.size
        metrics["width"] = w
        metrics["height"] = h

        # Resolution check (CPU — trivial)
        if min(w, h) < self.min_resolution:
            metrics["reason"] = "low_resolution"
            return False, metrics

        # Aspect ratio check (CPU — trivial)
        aspect = w / h
        metrics["aspect_ratio"] = round(aspect, 3)
        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            metrics["reason"] = "bad_aspect_ratio"
            return False, metrics

        img_array = np.array(img)

        # Sharpness check (GPU-accelerated Laplacian)
        try:
            sharpness = self._gpu_sharpness(img_array)
            metrics["sharpness"] = round(sharpness, 2)
            if sharpness < self.min_sharpness:
                metrics["reason"] = "too_blurry"
                return False, metrics
        except Exception:
            metrics["reason"] = "sharpness_check_failed"
            return False, metrics

        # Color variance check (GPU-accelerated)
        std = self._gpu_color_std(img_array)
        metrics["color_std"] = round(float(std), 2)
        if std < 15.0:
            metrics["reason"] = "too_uniform"
            return False, metrics

        metrics["passed"] = True
        return True, metrics

    def check_batch(self, image_paths: list[Path]) -> list[tuple[bool, dict]]:
        """
        Batch quality check — processes multiple images with GPU acceleration.
        Pre-filters by file size and resolution on CPU, then batches
        GPU operations for remaining images.
        """
        results = []
        for path in image_paths:
            results.append(self.check(path))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Deduplicator
# ─────────────────────────────────────────────────────────────────────────────
class Deduplicator:
    """Remove near-duplicate images using perceptual hashing."""

    def __init__(self, hash_size: int = 8, threshold: int = 5):
        self.hash_size = hash_size
        self.threshold = threshold
        self.hashes: dict[str, "imagehash.ImageHash"] = {}

    def is_duplicate(self, image_path: Path) -> bool:
        try:
            img = Image.open(image_path).convert("RGB")
            h = imagehash.phash(img, hash_size=self.hash_size)
            for existing_path, existing_hash in self.hashes.items():
                if abs(h - existing_hash) <= self.threshold:
                    return True
            self.hashes[str(image_path)] = h
            return False
        except Exception:
            return True  # Can't hash → treat as duplicate


class GPUHasher:
    """
    GPU-accelerated Perceptual Hashing (pHash).
    Strictly forces GPU usage.
    """
    def __init__(self, device="cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available! GPUHasher requires a GPU.")
        
        self.device = device
        logger.info(f"⚡ GPUHasher initialized on: {str(self.device).upper()}")
        self.dct_matrix = self._get_dct_matrix(32).to(self.device)

    def _get_dct_matrix(self, N):
        """Standard DCT-II matrix."""
        dct_m = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                dct_m[k, n] = np.cos(np.pi / N * (n + 0.5) * k)
        return torch.from_numpy(dct_m).float()

    def compute_hashes(self, image_paths: list[Path], batch_size=64) -> dict[str, int]:
        """
        Compute pHash for a list of image paths using GPU acceleration.
        Returns dictionary {path_str: hash_int}
        """
        results = {}
        
        # Use tqdm for progress bar
        with tqdm(total=len(image_paths), desc="  Computing hashes (GPU)", unit="img") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_tensors = []
                valid_paths = []

                for p in batch_paths:
                    try:
                        # Open (L = grayscale)
                        # We avoid PIL.resize here to save CPU
                        img = Image.open(p).convert("L")
                        
                        # Convert to tensor [1, H, W] directly
                        t = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
                        batch_tensors.append(t)
                        valid_paths.append(str(p))
                    except Exception:
                        pass
                
                # Update pbar for the batch processed
                pbar.update(len(batch_paths))

                if not batch_tensors:
                    continue
                
                # GPU Processing
                try:
                    gpu_tensors = []
                    for t in batch_tensors:
                        # Move to GPU
                        t_gpu = t.to(self.device, non_blocking=True).unsqueeze(0) # [1, 1, H, W]
                        # Resize on GPU
                        t_resized = F.interpolate(t_gpu, size=(32, 32), mode='bilinear', align_corners=False)
                        gpu_tensors.append(t_resized.squeeze(0)) # [1, 32, 32]
                    
                    # Stack: [B, 32, 32]
                    pixel_batch = torch.stack(gpu_tensors).squeeze(1)
                    
                    # Compute DCT: D * I * D^T
                    # [32, 32] @ [B, 32, 32] @ [32, 32] -> [B, 32, 32]
                    dct = torch.matmul(self.dct_matrix, pixel_batch)
                    dct = torch.matmul(dct, self.dct_matrix.T)
        
                    # Extract top-left 8x8 (excluding DC term at 0,0)
                    # Flatten to [B, 64]
                    dct_low = dct[:, :8, :8].reshape(-1, 64)
        
                    # Compute median per image
                    medians = dct_low.median(dim=1, keepdim=True).values
        
                    # Generate hash: 1 if > median, 0 otherwise
                    bits = (dct_low > medians).long()
        
                    # Convert 64 bits to integer
                    # Powers of 2 vector: [2^0, 2^1, ... 2^63]
                    powers = (2 ** torch.arange(64, device=self.device)).long()
                    hashes = (bits * powers).sum(dim=1).cpu().numpy()
        
                    for p, h in zip(valid_paths, hashes):
                        results[p] = int(h)
                        
                except Exception as e:
                    logger.debug(f"GPU Hash batch failed: {e}")
                    continue
    
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_quality_filter(config: dict) -> dict:
    """Main quality filter pipeline (GPU-accelerated) with Auto-Scrape Top-Up."""
    from pinterest_scraper import PinterestScraper, DEFAULT_QUERIES  # Lazy import to avoid circular deps
    
    raw_dir = Path(config["paths"]["data"]["raw"])
    processed_dir = Path(config["paths"]["data"]["processed"])
    
    TARGET_COUNT = 1300

    if not raw_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_dir}")
        sys.exit(1)

    # Quality settings from config
    quality_cfg = config.get("dataset", {}).get("quality", {})
    
    checker = ImageQualityChecker(
        min_resolution=quality_cfg.get("min_resolution", 512),
        min_sharpness=quality_cfg.get("min_sharpness", 50.0),
        min_aspect_ratio=quality_cfg.get("min_aspect_ratio", 0.4),
        max_aspect_ratio=quality_cfg.get("max_aspect_ratio", 2.5),
    )
    dedup = Deduplicator()
    
    # Initialize scraper (but don't start driver yet)
    scraper = PinterestScraper(config, str(raw_dir))

    # Log GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🎮 GPU detected: {gpu_name}. Total memory: {gpu_mem:.2f} GB")
    else:
        logger.info("🖥️ No GPU detected — running on CPU (slower)")

    # Stats
    stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "duplicates": 0})
    
    # 1. LOAD ALL EXISTING PROCESSED IMAGES (Global Deduplication)
    logger.info("🧠 Learning ALL existing images to prevent duplicates...")
    all_processed_files = []
    for root, _, files in os.walk(processed_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_processed_files.append(Path(root) / file)
                
    existing_hashes = 0
    if all_processed_files:
        hasher = GPUHasher()
        # Compute hashes for everything currently in processed
        batch_hashes = hasher.compute_hashes(all_processed_files, batch_size=128)
        dedup.hashes.update(batch_hashes)
        existing_hashes = len(batch_hashes)
        
    logger.info(f"✅ Memorized {existing_hashes} unique images in processed dataset.")

    # Collect all leaf directories (directories that contain images, not just parents)
    leaf_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        root_path = Path(root)
        # Check if this is a leaf node we want to process
        # (It might be empty now but was scraped before, or we want to scrape it)
        # For now, rely on existing folders in raw.
        rel_path = root_path.relative_to(raw_dir)
        
        # Skip the root directory itself (files directly in data/raw)
        if str(rel_path) == ".":
            continue
            
        leaf_dirs.append((rel_path, root_path))

    if not leaf_dirs:
        logger.warning("No directories found in raw data.")
        return {}

    logger.info(f"Found {len(leaf_dirs)} theme directories to process")

    for rel_path, dir_path in sorted(leaf_dirs):
        category = str(rel_path).replace("\\", "/")
        out_dir = processed_dir / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # We assume leaf dir if it has no subdirs with images? 
        # Simpler: just process if we found it.

        while True:
            # Check current status in processed folder
            processed_images = [f for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            current_count = len(processed_images)
            
            # If we met the target, break loop and move to next category
            if current_count >= TARGET_COUNT:
                logger.info(f"✅ {category}: Target met ({current_count} images).")
                break
                
            needed = TARGET_COUNT - current_count
            logger.info(f"\nCategory: {category}")
            logger.info(f"  Current: {current_count} | Needed: {needed}")

            # Get raw images
            raw_images = sorted([
                dir_path / f for f in os.listdir(dir_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
            ])
            logger.info(f"  Raw images available: {len(raw_images)}")
            
            # Identify candidates (raw images NOT yet in processed folder by filename)
            existing_filenames = set(processed_images)
            candidates = [p for p in raw_images if p.name not in existing_filenames]
            
            added_this_round = 0
            
            if candidates:
                logger.info(f"  Processing {len(candidates)} new candidates...")
                pbar = tqdm(candidates, desc=f"  {category} (Filter)", unit="img")
                for img_path in pbar:
                    if added_this_round >= needed:
                        break
                    
                    stats[category]["total"] += 1
        
                    # Quality check (GPU-accelerated sharpness + color)
                    passed, metrics = checker.check(img_path)
                    if not passed:
                        stats[category]["failed"] += 1
                        # logger.debug(f"  REJECTED {img_path.name}: {metrics['reason']}")
                        continue
        
                    # Dedup check (Hash-based)
                    if dedup.is_duplicate(img_path):
                        stats[category]["duplicates"] += 1
                        # logger.debug(f"  DUPLICATE {img_path.name}")
                        continue
        
                    # Copy to processed
                    dest = out_dir / img_path.name
                    shutil.copy2(img_path, dest)
                    stats[category]["passed"] += 1
                    added_this_round += 1
                    
                pbar.close()
                current_count += added_this_round
                
                if current_count >= TARGET_COUNT:
                    continue # Re-evaluate loop condition (which will break)
            
            # If still short, trigger scraper
            needed = TARGET_COUNT - current_count
            if needed > 0:
                logger.warning(f"  ⚠️ Short by {needed} images! Launching Scraper to fetch more...")
                
                # Fetch query list
                queries = DEFAULT_QUERIES.get(category)
                if not queries:
                    # Fallback queries
                    theme = category.split("/")[-1]
                    queries = [f"{theme} poster", f"{theme} design", f"{theme} advertisement"]
                
                # Scrape 2x what we need
                scrape_target = len(raw_images) + (needed * 2) 
                # Ensure we at least target 2800 if we are really low
                scrape_target = max(scrape_target, 2800)
                
                scraper.TARGET_PER_THEME = scrape_target
                logger.info(f"  🕷️ Scraping target set to {scrape_target} for {category}...")
                
                try:
                    # scraper.scrape_category downloads to raw_dir/{category}
                    # It returns total downloaded count
                    new_total = scraper.scrape_category(category, queries)
                    logger.info(f"  ✅ Scraping finished. Raw total is now {new_total}. Rescanning...")
                except Exception as e:
                    logger.error(f"  ❌ Scraper failed: {e}")
                    break # Stop trying for this category if scraper fails
            else:
                break # Should be caught by top check, but safe fallback

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(stats)


def print_summary(stats: dict):
    """Print a summary table."""
    # ... existing print_summary code ...
    print("\n" + "=" * 60)
    print(f"{'Category':<35} | {'Total':<8} | {'Pass':<6} | {'Fail':<6} | {'Dupes':<6}")
    print("-" * 60)
    
    total_passed = 0
    for cat, data in sorted(stats.items()):
        print(f"{cat:<35} | {data['total']:<8} | {data['passed']:<6} | {data['failed']:<6} | {data['duplicates']:<6}")
        total_passed += data['passed']
        
    print("-" * 60)
    print(f"Total High-Quality Images: {total_passed}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quality Filter with Auto-Scrape")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Run pipeline
    stats = run_quality_filter(config)
    print_summary(stats)

    logger.info("\n" + "=" * 80)
    logger.info("QUALITY FILTER SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  {'Category':35s} {'Total':>7s} {'Passed':>7s} {'Failed':>7s} {'Dupes':>7s} {'Rate':>7s}")
    logger.info(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    grand_total = grand_passed = 0
    for cat, s in sorted(stats.items()):
        rate = f"{s['passed']/max(s['total'],1)*100:.1f}%"
        logger.info(
            f"  {cat:35s} {s['total']:7d} {s['passed']:7d} "
            f"{s['failed']:7d} {s['duplicates']:7d} {rate:>7s}"
        )
        grand_total += s["total"]
        grand_passed += s["passed"]

    rate = f"{grand_passed/max(grand_total,1)*100:.1f}%"
    logger.info(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    logger.info(f"  {'TOTAL':35s} {grand_total:7d} {grand_passed:7d}{'':>17s} {rate:>7s}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Image Quality Filter (GPU-Accelerated)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = run_quality_filter(config)
    print_summary(stats)


if __name__ == "__main__":
    main()
