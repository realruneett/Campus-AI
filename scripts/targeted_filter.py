import logging
import shutil
import sys
import os
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

# Add current directory to path so we can import sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quality_filter import ImageQualityChecker, Deduplicator, GPUHasher, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COUNT = 1300

def main():
    logger.info("🚀 Starting Targeted Top-Up Filter (v2)")
    logger.info(f"🎯 Goal: Ensure every category has >= {TARGET_COUNT} unique, high-quality images")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    config = load_config(config_path)
    
    raw_dir = Path(config["paths"]["data"]["raw"])
    processed_dir = Path(config["paths"]["data"]["processed"])
    
    # Initialize checkers
    checker = ImageQualityChecker(config)
    dedup = Deduplicator()
    
    if torch.cuda.is_available():
        logger.info(f"⚡ Using GPU: {torch.cuda.get_device_name(0)}")
    
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

    # 2. IDENTIFY CATEGORIES NEEDING TOP-UP
    categories_to_process = []
    for root, dirs, files in os.walk(raw_dir):
        if not dirs: # Leaf node
            rel_path = Path(root).relative_to(raw_dir)
            proc_path = processed_dir / rel_path
            
            # Count images in processed
            if proc_path.exists():
                curr_count = len([f for f in os.listdir(proc_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                curr_count = 0
            
            if curr_count < TARGET_COUNT:
                categories_to_process.append((rel_path, Path(root), proc_path, curr_count))
            else:
                pass # Already meets target
                
    if not categories_to_process:
        logger.info("✨ All categories meet the target of 1300! No work needed.")
        return

    logger.info(f"📋 Found {len(categories_to_process)} categories below target.")
    
    # 3. PROCESS MISSING CATEGORIES
    for rel_path, raw_category_path, proc_category_path, current_count in categories_to_process:
        needed = TARGET_COUNT - current_count
        category_name = str(rel_path).replace("\\", "/")
        
        logger.info(f"\n🔸 Processing: {category_name}")
        logger.info(f"   Current: {current_count} | Needed: {needed}")
        
        proc_category_path.mkdir(parents=True, exist_ok=True)
        
        # Get all raw files
        raw_files = sorted([
            raw_category_path / f 
            for f in os.listdir(raw_category_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        added = 0
        skipped_dupe = 0
        skipped_quality = 0
        
        # Batch process raw files for efficiency? 
        # Actually, since we need to copy them one by one based on check, 
        # we can batch quality check/hash check if we want, but sequential loop is clearer for "stop when satisfied".
        # Let's use GPUHasher on raw files in chunks to speed up the dedup check at least.
        
        # Optimization: Filter out filenames that already exist (exact match)
        existing_filenames = set(os.listdir(proc_category_path))
        candidates = [f for f in raw_files if f.name not in existing_filenames]
        
        if not candidates:
            logger.warning("   ❌ No new raw files available to scan!")
            continue

        # Progress bar
        pbar = tqdm(total=needed, desc=f"   Filling {category_name}", unit="img")
        
        # Iterate through candidates
        for raw_img_path in candidates:
            if added >= needed:
                break
            
            # 1. Deduplication Check (Fastest check first? No, Quality is cleaner but slower. Dedup is fast with hash)
            # Actually we need hash to check dedup.
            
            # We'll calculate hash for individual image (slower than batch but we need decision per image)
            # OR we could batch hash all candidates first. 
            # Let's batch hash candidates first!
            
            # Wait, let's just do it sequentially for simplicity unless it's too slow.
            # With GPUHasher, we can compute hash quickly.
            
            try:
                # 1. Quality Check (GPU)
                passed, metrics = checker.check(raw_img_path)
                if not passed:
                    skipped_quality += 1
                    continue
                
                # 2. Dedup Check (needs hash)
                if dedup.is_duplicate(raw_img_path):
                    skipped_dupe += 1
                    continue
                
                # 3. Copy
                shutil.copy2(raw_img_path, proc_category_path / raw_img_path.name)
                added += 1
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing {raw_img_path}: {e}")
                continue
                
        pbar.close()
        
        final_count = current_count + added
        if final_count >= TARGET_COUNT:
             logger.info(f"   ✅ Reached target! ({final_count})")
        else:
             logger.warning(f"   ⚠️ Finished scanning raw files. Ended with {final_count} (Still short by {TARGET_COUNT - final_count})")
             
    logger.info("\n🎉 Top-Up Complete!")

if __name__ == "__main__":
    main()

