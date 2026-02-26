
import os
import shutil
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
TARGET_PER_CATEGORY = 1000
SPLIT_RATIO = (0.8, 0.1, 0.1)  # Train, Val, Test

DATA_ROOT = Path("data")
PROCESSED_DIR = DATA_ROOT / "processed"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

def get_image_files(directory):
    """Recursively get all image files in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return [f for f in directory.rglob("*") if f.suffix.lower() in extensions and f.is_file()]

def clear_directory(path):
    """Deletes a directory and its contents if it exists."""
    if path.exists():
        logger.warning(f"Deleting existing directory: {path}")
        shutil.rmtree(path)

def main():
    logger.info("🚀 Starting Dataset Resplit (v2)")
    logger.info(f"🎯 Target: {TARGET_PER_CATEGORY} images/category | Split: {SPLIT_RATIO}")

    # 1. Clear existing splits
    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)
    clear_directory(TEST_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Iterate through categories in processed
    # We assume 'processed' has subfolders like 'workshops/coding', 'workshops/design', etc.
    # We walk to find leaf directories that contain images.
    
    # Optimized walker: Only look at files in the current directory
    categories = []
    for root, dirs, files in os.walk(PROCESSED_DIR):
        current_path = Path(root)
        
        # Check files in current dir only
        local_images = []
        for f in files:
            if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                local_images.append(current_path / f)
        
        if local_images:
            # It's a category folder
            rel_path = current_path.relative_to(PROCESSED_DIR)
            categories.append((rel_path, local_images))

    if not categories:
        logger.error("❌ No categories found in data/processed!")
        return

    logger.info(f"📂 Found {len(categories)} categories to process.")

    for rel_path, images in categories:
        category_name = str(rel_path).replace("\\", "/")
        logger.info(f"\n🔹 Processing: {category_name}")
        
        # Shuffle and Select
        random.shuffle(images)
        selected_images = images[:TARGET_PER_CATEGORY]
        count = len(selected_images)
        
        if count < TARGET_PER_CATEGORY:
            logger.warning(f"   ⚠️ Only found {count} images (Target: {TARGET_PER_CATEGORY})")
        else:
            logger.info(f"   ✅ Selected 1000 images from {len(images)} available.")

        # Calculate Splits
        n_train = int(count * SPLIT_RATIO[0])
        n_val = int(count * SPLIT_RATIO[1])
        # Give remainder to test to ensure sum == count (or fix strictly if required, but remainder is safer)
        n_test = count - n_train - n_val

        train_set = selected_images[:n_train]
        val_set = selected_images[n_train : n_train + n_val]
        test_set = selected_images[n_train + n_val :]

        logger.info(f"   Splitting: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

        # Copy Files
        for dataset, split_name, dest_root in [
            (train_set, "Train", TRAIN_DIR),
            (val_set, "Val", VAL_DIR),
            (test_set, "Test", TEST_DIR)
        ]:
            if not dataset:
                continue
                
            dest_category_dir = dest_root / rel_path
            dest_category_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in dataset:
                try:
                    shutil.copy2(img_path, dest_category_dir / img_path.name)
                    # Try to copy caption text file if it exists
                    txt_path = img_path.with_suffix(".txt")
                    if txt_path.exists():
                        shutil.copy2(txt_path, dest_category_dir / txt_path.name)
                except Exception as e:
                    logger.error(f"Failed to copy {img_path.name}: {e}")

    logger.info("\n🎉 Resplit Complete.")
    
    # Verification stats
    logger.info("📊 Final Counts:")
    for d, name in [(TRAIN_DIR, "TRAIN"), (VAL_DIR, "VAL"), (TEST_DIR, "TEST")]:
        total = len(list(d.rglob("*.*")))  # Approx count all files
        # Better to count images
        img_count = len(get_image_files(d))
        logger.info(f"   {name}: {img_count} images")

if __name__ == "__main__":
    main()

