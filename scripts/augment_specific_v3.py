
import os
import shutil
import logging
from pathlib import Path
from collections import defaultdict
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("⚠️ PIL (Pillow) not found. Image validation will be skipped (only file extension check).")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configuration
TARGET_COUNT = 1300  # Safety margin above 1000
TARGET_CATEGORIES = [
    "workshops/coding",
    "workshops/design"
]

DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

def get_image_files(directory):
    """Recursively get all image files in a directory."""
    extensions = {'*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'}
    files = []
    if not directory.exists():
        return files
    
    for ext in extensions:
        # Case insensitive search would be better but glob is case sensitive on Linux/WSL usually.
        # We will try both cases or just standarize. 
        # Walking is safer for case insensitivity if needed, but glob is faster.
        files.extend(directory.glob(f"**/{ext}"))
        files.extend(directory.glob(f"**/{ext.upper()}"))
    return sorted(list(set(files)))

def check_image_quality(file_path):
    """
    Basic quality check using PIL (if available).
    Returns (Passed: bool, Message: str)
    """
    if not PIL_AVAILABLE:
        # If PIL is missing, we assume file is okay if it exists and has size
        if file_path.stat().st_size < 5120: # < 5KB is suspect
            return False, "File too small"
        return True, "No PIL check"

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < 256 or height < 256:
                return False, f"Low resolution: {width}x{height}"
            
            # Aspect ratio check
            aspect = width / height
            if aspect < 0.4 or aspect > 2.5:
                return False, f"Extreme aspect ratio: {aspect:.2f}"
                
            return True, "OK"
    except Exception as e:
        return False, f"Corrupt image: {str(e)}"

def process_category(relative_path):
    """Process a single category."""
    category_name = str(relative_path).replace("\\", "/")
    logger.info(f"🔍 Checking category: {category_name}")

    raw_path = RAW_ROOT / relative_path
    processed_path = PROCESSED_ROOT / relative_path

    # Ensure processed directory exists
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. Count current Processed
    processed_files = get_image_files(processed_path)
    current_count = len(processed_files)
    processed_filenames = {f.name for f in processed_files}
    
    logger.info(f"   Existing processed images: {current_count}")

    if current_count >= TARGET_COUNT:
        logger.info(f"   ✅ Already met target of {TARGET_COUNT}. Skipping.")
        return

    needed = TARGET_COUNT - current_count
    logger.info(f"   ⚠️ Need {needed} more images.")

    # 2. Get Raw Candidates
    raw_files = get_image_files(raw_path)
    logger.info(f"   Found {len(raw_files)} raw images available.")

    # Filter out files that are already in processed (by filename)
    candidates = [f for f in raw_files if f.name not in processed_filenames]
    logger.info(f"   {len(candidates)} new unique candidates available to process.")

    if not candidates:
        logger.warning("   ❌ No new candidates found in raw folder!")
        return

    # 3. Copy Candidates
    added_count = 0
    passed_check = 0
    failed_check = 0

    # Progress bar setup
    iterator = tqdm(candidates, unit="img") if TQDM_AVAILABLE else candidates

    for src_file in iterator:
        if added_count >= needed:
            break

        # Quality Check
        is_ok, msg = check_image_quality(src_file)
        if not is_ok:
            failed_check += 1
            continue
        
        # Copy
        dst_file = processed_path / src_file.name
        try:
            shutil.copy2(src_file, dst_file)
            added_count += 1
            passed_check += 1
        except Exception as e:
            logger.error(f"Failed to copy {src_file.name}: {e}")

    logger.info(f"   🎉 Added {added_count} images.")
    logger.info(f"   Final Count: {current_count + added_count}")
    logger.info("-" * 40)

def main():
    logger.info("🚀 Starting targeted dataset augmentation...")
    logger.info(f"📂 Data Root: {DATA_ROOT.absolute()}")
    logger.info(f"🎯 Target: {TARGET_COUNT} images per category")

    for cat in TARGET_CATEGORIES:
        process_category(Path(cat))

    logger.info("✨ Done.")

if __name__ == "__main__":
    main()
