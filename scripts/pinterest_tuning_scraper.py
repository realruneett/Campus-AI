import os
import sys
import re
import time
import random
import hashlib
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

import yaml
import requests
from requests.adapters import HTTPAdapter
from PIL import Image
from tqdm import tqdm
from image_deduplicator import GlobalImageDeduplicator
from tuning_dataset import CATEGORIES

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("WARNING: selenium/webdriver_manager not installed.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PINIMG_RE = re.compile(r'https://[a-z0-9]+\.pinimg\.com/[^\s"\'<>]+\.jpg')


class PinterestTuningScraper:
    """Scrape specific tuning poster images from Pinterest using Selenium."""

    PINTEREST_SEARCH_URL = "https://www.pinterest.com/search/pins/?q={query}"

    def __init__(self, config: dict, output_dir: str, target_per_theme: int = 20):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_theme = target_per_theme

        scraping_cfg = config.get("scraping", {}).get("pinterest", {})
        self.scroll_pause = scraping_cfg.get("scroll_pause_seconds", 2.0)
        self.download_timeout = scraping_cfg.get("download_timeout", 15)
        self.min_resolution = scraping_cfg.get("min_resolution", 512)

        data_root = self.config.get("paths", {}).get("data", {}).get("root", "data")
        self.dedup = GlobalImageDeduplicator(data_dir=data_root)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        adapter = HTTPAdapter(
            pool_connections=16,
            pool_maxsize=16,
            max_retries=1
        )
        self.session.mount("https://i.pinimg.com", adapter)
        self.session.mount("https://v1.pinimg.com", adapter)

    def _create_driver(self):
        import undetected_chromedriver as uc
        import random

        opts = uc.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--mute-audio")
        opts.add_argument("--no-first-run")
        opts.add_argument("--disable-background-networking")

        driver = uc.Chrome(options=opts, version_main=145, headless=True)
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(10)
        return driver

    def _nuke_modals(self, driver):
        try:
            driver.execute_script('''
                document.querySelectorAll(
                    '[data-test-id="giftWrap"],[data-test-id="signup"],'
                    '[data-test-id="unauthModal"],.Modal__overlay'
                ).forEach(e => e.remove());
                document.body.style.overflow = "auto";
                document.documentElement.style.overflow = "auto";
            ''')
        except Exception:
            pass

    def _scroll_and_collect_urls(self, driver, query: str, max_images: int) -> list[str]:
        url = self.PINTEREST_SEARCH_URL.format(query=query.replace(" ", "+"))

        try:
            driver.get(url)
        except Exception:
            pass

        # Wait up to 15s for React to hydrate
        for _ in range(15):
            if "pinimg.com" in driver.page_source:
                break
            time.sleep(1)

        self._nuke_modals(driver)

        image_urls = set()
        last_height = 0
        scroll_step = 400          # smaller steps — triggers lazy loader reliably
        current_pos = 0
        max_scroll_pos = 80000     # ~80 screens worth, Pinterest never goes deeper
        no_new_count = 0           # stall on CONTENT not page height
        height_stall_count = 0

        pbar = tqdm(total=max_images, desc=f"  Scrolling: {query[:40]}")

        while len(image_urls) < max_images and no_new_count < 8 and current_pos < max_scroll_pos:
            try:
                current_pos += scroll_step
                driver.execute_script(f"window.scrollTo(0, {current_pos});")
                time.sleep(self.scroll_pause + random.uniform(0.3, 1.2))
            except Exception:
                no_new_count += 1
                continue

            prev_count = len(image_urls)

            try:
                page_source = driver.page_source
                found = PINIMG_RE.findall(page_source)
                for src in found:
                    if "profile_images" in src or "75x75_RS" in src:
                        continue
                    # 736x resolution keeps download fast but high-quality enough
                    src = (src.replace("/236x/", "/736x/")
                              .replace("/474x/", "/736x/")
                              .replace("/originals/", "/736x/"))
                    image_urls.add(src)
            except Exception:
                pass

            new_found = len(image_urls) - prev_count
            if new_found == 0:
                no_new_count += 1   # count scrolls with ZERO new images
            else:
                no_new_count = 0    # reset whenever new images found

            pbar.update(max(0, len(image_urls) - pbar.n))

            try:
                new_height = driver.execute_script("return document.body.scrollHeight")
                if current_pos >= new_height:
                    self._nuke_modals(driver)
                    if new_height == last_height:
                        height_stall_count += 1
                        if height_stall_count >= 3:
                            # Truly at bottom of page, nothing more to load
                            break
                    else:
                        height_stall_count = 0
                        last_height = new_height
                    current_pos = new_height
            except Exception:
                no_new_count += 1

        pbar.close()
        return list(image_urls)[:max_images]

    def _is_valid_url(self, url: str) -> bool:
        skip = ["profile_images", "75x75", "30x30", "user_images", "avatars"]
        return not any(s in url for s in skip)

    def _download_image(self, url: str, save_path: Path) -> bool:
        try:
            resp = self.session.get(url, timeout=(2, 4))
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if min(img.size) < self.min_resolution:
                return False
            if self.dedup.is_duplicate(img, save_path=str(save_path)):
                return False
            img.save(save_path, "JPEG", quality=95)
            self.dedup.add_to_disk_cache(str(save_path), img)
            return True
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return False

    def scrape_category(self, subcategory_path: str, queries: list[str]) -> int:
        cat_dir = self.output_dir / subcategory_path
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing_files = set(f.name for f in cat_dir.glob("*.jpg"))
        existing_count = len(existing_files)
        logger.info(f"Subcategory '{subcategory_path}': {existing_count} existing images")

        if existing_count >= self.target_per_theme:
            logger.info(f"  ✓ Already at target ({self.target_per_theme}), skipping!")
            return existing_count

        if not HAS_SELENIUM:
            logger.error("Selenium not available.")
            return 0

        driver = self._create_driver()
        total_downloaded = existing_count
        all_seen_urls = set()
        queries = list(queries)

        try:
            query_cycle = 0
            query_fail_counts = {}

            while total_downloaded < self.target_per_theme:
                for query in list(queries):
                    if total_downloaded >= self.target_per_theme:
                        break
                    # Mutate query to break pagination bounds and prioritize design aesthetics
                    active_query = query
                    if query_cycle > 0:
                        modifiers = [" poster layout", " graphic design", " aesthetic", " template", " typography"]
                        active_query = f"{query}{modifiers[query_cycle % len(modifiers)]}"

                    logger.info(f"  Query: '{active_query}' (Cycle {query_cycle + 1})")
                    target_to_fetch = self.target_per_theme * (query_cycle + 2)

                    try:
                        urls = self._scroll_and_collect_urls(driver, active_query, target_to_fetch)
                        query_fail_counts[query] = 0
                    except Exception as scroll_err:
                        logger.warning(f"  WebDriver failed/timed out on '{query}': {scroll_err}")
                        query_fail_counts[query] = query_fail_counts.get(query, 0) + 1

                        if query_fail_counts[query] >= 1:
                            logger.error(f"  Skipping query '{query}' permanently.")
                            queries = [q for q in queries if q != query]
                            if not queries:
                                logger.error("  All queries failed. Breaking out of category.")
                                break

                        logger.warning("  Rebooting Chrome driver and retrying...")
                        time.sleep(random.uniform(3, 6))
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        driver = self._create_driver()
                        continue

                    # Reboot driver if session returned near-zero results (blacklisted)
                    if len(urls) < 10 and total_downloaded < self.target_per_theme:
                        logger.warning("  Session returned <10 URLs — rebooting driver.")
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        time.sleep(random.uniform(3, 6))
                        driver = self._create_driver()
                    new_urls = [u for u in urls if u not in all_seen_urls]
                    all_seen_urls.update(urls)

                    # FIX 7: parallel downloads — 16 workers instead of sequential
                    needed = self.target_per_theme - total_downloaded
                    candidates = [
                        u for u in new_urls
                        if self._is_valid_url(u)
                        and f"tuning_{hashlib.md5(u.encode()).hexdigest()[:12]}.jpg"
                           not in existing_files
                    ][:needed * 4]

                    def _dl(u, _cat_dir=cat_dir):
                        fname = f"tuning_{hashlib.md5(u.encode()).hexdigest()[:12]}.jpg"
                        sp = _cat_dir / fname
                        if sp.exists():
                            return None
                        return (fname, self._download_image(u, sp))

                    with ThreadPoolExecutor(max_workers=16) as pool:
                        futures = {pool.submit(_dl, u): u for u in candidates}
                        pbar_dl = tqdm(total=min(needed, len(candidates)),
                                       desc="  Downloading", leave=False)
                        for fut in as_completed(futures):
                            if total_downloaded >= self.target_per_theme:
                                pool.shutdown(wait=True, cancel_futures=True)
                                break
                            result = fut.result()
                            if result:
                                fname, ok = result
                                if ok:
                                    total_downloaded += 1
                                    existing_files.add(fname)
                                    pbar_dl.update(1)
                        pbar_dl.close()

                if total_downloaded < self.target_per_theme:
                    if not queries:
                        break
                    logger.warning(
                        f"  Only at {total_downloaded}/{self.target_per_theme}. "
                        f"Cycling queries again and scrolling deeper."
                    )
                    query_cycle += 1
                    max_cycles = max(5, len(queries))  # exhaust full query pool
                    if query_cycle >= max_cycles:
                        logger.error(
                            f"  Exhausted all {max_cycles} query cycles. "
                            f"Stuck at {total_downloaded}/{self.target_per_theme}. Breaking."
                        )
                        break

        finally:
            try:
                driver.quit()
            except Exception:
                pass

        logger.info(f"  ✓ Downloaded {total_downloaded} images for {subcategory_path}.")
        return total_downloaded


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Tuning Dataset Pinterest Scraper")
    parser.add_argument("--target", default="data/tuning", help="Root directory for tuning data")
    parser.add_argument("--per-category", type=int, default=100, help="Images per subcategory")
    args = parser.parse_args()

    config = load_config()
    target_dir = Path(args.target)

    logger.info("🚀 Starting Pinterest Tuning Scraper")
    logger.info(f"🎯 Target Count: {args.per_category} images per subcategory")

    scraper = PinterestTuningScraper(config, output_dir=str(target_dir), target_per_theme=args.per_category)

    for subcat, queries in CATEGORIES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {subcat}")
        logger.info(f"{'='*60}")
        try:
            count = scraper.scrape_category(subcat, queries)
            logger.info(f"✅ Finished {subcat}: {count} total images")
        except Exception as e:
            logger.error(f"❌ Failed processing {subcat}: {e}")
        time.sleep(2)

    logger.info("\n🎉 All tuning categories processed safely without duplicates!")


if __name__ == "__main__":
    main()