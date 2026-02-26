import os
import sqlite3
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class GlobalImageDeduplicator:
    """
    Globally tracks perceptual hashes of all images in the data directory
    to prevent downloading duplicates across all subfolders and phases.
    Uses an SQLite database for persistent caching to speed up initialization.
    """
    def __init__(self, data_dir: str, db_path: str = None, hash_size: int = 8, threshold: int = 5):
        self.data_dir = Path(data_dir)
        if db_path is None:
            # Store at root/data/phash_cache.db
            self.db_path = self.data_dir / "phash_cache.db"
        else:
            self.db_path = Path(db_path)
            
        self.hash_size = hash_size
        self.threshold = threshold
        self.hashes = [] # List of (filepath, imagehash.ImageHash)
        
        logger.info(f"Initializing Global Image Deduplicator using DB: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._load_and_sync()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS phashes (
                    filepath TEXT PRIMARY KEY,
                    mtime REAL,
                    hash_str TEXT
                )
            ''')
            
    def _load_and_sync(self):
        logger.info(f"Scanning {self.data_dir} for images...")
        all_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            all_files.extend(self.data_dir.rglob(ext))
            
        # Get existing from DB
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, mtime, hash_str FROM phashes")
        db_records = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        to_hash = []
        to_delete = []
        
        # Determine what needs hashing
        current_files = set(str(f) for f in all_files)
        
        for f in all_files:
            f_str = str(f)
            mtime = os.path.getmtime(f)
            if f_str in db_records:
                # If modified time changed, rehash
                if db_records[f_str][0] < mtime:
                    to_hash.append((f_str, f, mtime))
            else:
                to_hash.append((f_str, f, mtime))
                
        for db_file in db_records:
            if db_file not in current_files:
                to_delete.append(db_file)
                
        # Delete missing files from DB
        if to_delete:
            logger.info(f"Removing {len(to_delete)} deleted files from cache.")
            with self.conn:
                self.conn.executemany("DELETE FROM phashes WHERE filepath = ?", [(f,) for f in to_delete])
                
        # Hash new or modified files
        if to_hash:
            logger.info(f"Hashing {len(to_hash)} new/modified images. This might take a while...")
            
            def compute_hash(args):
                f_str, f, mtime = args
                try:
                    with Image.open(f) as img:
                        # Convert to RGB to be safe and avoid issues with alpha channels
                        conv_img = img.convert("RGB")
                        h = imagehash.phash(conv_img, hash_size=self.hash_size)
                        return f_str, mtime, str(h)
                except Exception as e:
                    logger.debug(f"Error hashing {f}: {e}")
                    return None
            
            results = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for res in tqdm(executor.map(compute_hash, to_hash), total=len(to_hash), desc="Hashing"):
                    if res is not None:
                        results.append(res)
            
            # Save new hashes to DB
            with self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO phashes (filepath, mtime, hash_str) VALUES (?, ?, ?)", results)
                
        # Load all hashes into memory for fast comparison
        cursor.execute("SELECT filepath, hash_str FROM phashes")
        
        for filepath, hash_str in cursor.fetchall():
            self.hashes.append((filepath, imagehash.hex_to_hash(hash_str)))
            
        logger.info(f"Loaded {len(self.hashes)} image hashes for deduplication.")

    def is_duplicate(self, img: Image.Image, save_path: str = None) -> bool:
        """
        Check if an image is a duplicate of any globally known image.
        If save_path is provided, and it's NOT a duplicate, it adds the hash to the in-memory 
        cache immediately so we don't download the same duplicate in the same session.
        """
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        h = imagehash.phash(img, hash_size=self.hash_size)
        
        for existing_path, existing_hash in self.hashes:
            if abs(h - existing_hash) <= self.threshold:
                # logger.debug(f"Duplicate found! Matches {existing_path}")
                return True
                
        if save_path:
            self.hashes.append((str(save_path), h))
            
        return False
        
    def add_to_disk_cache(self, filepath: str, img: Image.Image):
        """
        Manually add an image to the DB cache. Use this after saving an image to disk
        so next time we run, it's already in the DB.
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
        h = imagehash.phash(img, hash_size=self.hash_size)
        # Wait slightly to ensure mtime is written
        time.sleep(0.01)
        mtime = os.path.getmtime(filepath)
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO phashes (filepath, mtime, hash_str) VALUES (?, ?, ?)", 
                            (str(filepath), mtime, str(h)))
