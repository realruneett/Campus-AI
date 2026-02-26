#!/usr/bin/env python3
"""
Pinterest Poster Image Scraper
Config-driven scraper using Selenium + BeautifulSoup.
Reads queries from config.yaml, downloads poster images to data/raw/{category}/
"""

import os
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
from io import BytesIO
from urllib.parse import urljoin
import yaml
import requests
import imagehash
from PIL import Image
from tqdm import tqdm

from image_deduplicator import GlobalImageDeduplicator

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("WARNING: selenium/webdriver_manager not installed. Install with:")
    print("  pip install selenium webdriver-manager")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load master config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Default search queries (per category) – can be overridden in config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QUERIES = {
    # ══════════════════════════════════════════════════════════════
    # TECH FEST  (parent: tech_fest/)
    # ══════════════════════════════════════════════════════════════
    "tech_fest/hackathon": [
        "hackathon poster design",
        "24 hour hackathon event poster",
        "code sprint competition poster",
        "startup hackathon poster design",
        "programming hackathon poster",
        "hackathon flyer template",
        "university hackathon invite",
        "tech hackathon banner",
        "coding marathon event poster",
        "hackathon winner announcement",
        "virtual hackathon poster",
        "hackathon timeline graphic",
        "innovate hackathon poster",
        "hackathon ideas poster",
        "hackathon challenge flyer"
    ],
    "tech_fest/coding_competition": [
        "coding competition poster design",
        "competitive programming poster",
        "code challenge event poster",
        "algorithm contest poster",
        "debug code competition poster",
        "bug bounty event poster",
        "coding battle flyer",
        "programming contest flyer",
        "code war poster design",
        "coding tournament bracket"
    ],
    "tech_fest/ai_ml": [
        "artificial intelligence conference poster",
        "machine learning workshop poster",
        "deep learning summit poster",
        "data science event poster",
        "AI summit poster design",
    ],
    "tech_fest/robotics": [
        "robotics event poster design",
        "robot competition poster",
        "robotics workshop poster",
        "drone racing event poster",
    ],
    "tech_fest/cybersecurity": [
        "cyber security event poster",
        "ethical hacking workshop poster",
        "CTF competition poster",
        "cybersecurity conference poster",
    ],
    "tech_fest/web_app_dev": [
        "web development bootcamp poster",
        "app development workshop poster",
        "full stack developer event poster",
        "software engineering meetup poster",
    ],
    "tech_fest/general": [
        "tech fest poster design",
        "technology conference poster",
        "tech expo poster design",
        "tech summit poster design",
        "innovation challenge poster",
        "tech symposium poster",
        "engineering college fest poster",
        "tech week event flyer",
        "future tech event poster",
        "technology showcase poster",
        "IT fest poster design"
    ],

    # ══════════════════════════════════════════════════════════════
    # CULTURAL FEST  (parent: cultural_fest/)
    # ══════════════════════════════════════════════════════════════
    "cultural_fest/dance": [
        "dance competition poster design",
        "classical dance event poster",
        "hip hop dance poster",
        "bollywood dance night poster",
        "dance festival poster design",
    ],
    "cultural_fest/music": [
        "music concert poster design",
        "live music event poster",
        "DJ night poster design",
        "band performance poster",
        "acoustic night event poster",
        "indie music festival poster",
    ],
    "cultural_fest/drama_theatre": [
        "theatre play poster design",
        "drama festival poster",
        "street play nukkad natak poster",
        "stage performance poster",
    ],
    "cultural_fest/art_exhibition": [
        "art exhibition poster design",
        "painting exhibition poster",
        "modern art show poster",
        "sculpture exhibition poster",
        "photography exhibition poster",
    ],
    "cultural_fest/fashion_show": [
        "fashion show poster design",
        "college fashion event poster",
        "runway show poster design",
        "fashion week poster design",
    ],
    "cultural_fest/literary": [
        "literary festival poster",
        "poetry slam event poster",
        "book launch poster design",
        "debate competition poster",
        "storytelling event poster",
        "quiz competition poster",
    ],
    "cultural_fest/standup_comedy": [
        "standup comedy show poster",
        "open mic night poster",
        "comedy night poster design",
        "improv comedy poster",
    ],
    "cultural_fest/general": [
        "cultural fest poster design",
        "college cultural event poster",
        "cultural night poster India",
        "talent show poster design",
    ],

    # ══════════════════════════════════════════════════════════════
    # SPORTS  (parent: sports/)
    # ══════════════════════════════════════════════════════════════
    "sports/cricket": [
        "cricket tournament poster",
        "IPL fan event poster",
        "cricket match poster design",
        "T20 cricket championship poster",
    ],
    "sports/football": [
        "football tournament poster design",
        "soccer championship poster",
        "inter-college football poster",
        "futsal tournament poster",
    ],
    "sports/basketball": [
        "basketball tournament poster design",
        "3x3 basketball event poster",
        "college basketball championship poster",
    ],
    "sports/badminton_tennis": [
        "badminton tournament poster",
        "tennis championship poster",
        "table tennis tournament poster",
        "squash competition poster",
    ],
    "sports/athletics": [
        "athletics meet poster design",
        "track and field event poster",
        "marathon poster design",
        "fun run event poster",
    ],
    "sports/esports": [
        "esports tournament poster",
        "gaming event poster design",
        "BGMI tournament poster",
        "valorant tournament poster",
        "FIFA tournament poster",
    ],
    "sports/kabaddi_kho": [
        "kabaddi tournament poster India",
        "kho kho competition poster",
        "traditional Indian sports poster",
    ],
    "sports/yoga_fitness": [
        "yoga day event poster",
        "fitness challenge poster",
        "gym event poster design",
        "wellness camp poster",
        "cycling event poster",
    ],
    "sports/general": [
        "sports tournament poster design",
        "college sports day poster",
        "inter-college sports poster",
        "sports carnival poster",
        "annual sports meet poster",
        "sports championship flyer",
        "athletic meet event poster",
        "intramural sports poster",
        "sports league banner",
        "team sports event poster"
    ],

    # ══════════════════════════════════════════════════════════════
    # COLLEGE EVENTS  (parent: college_events/)
    # ══════════════════════════════════════════════════════════════
    "college_events/annual_fest": [
        "college fest poster India",
        "university festival poster",
        "college annual day poster",
        "campus fest poster design",
    ],
    "college_events/freshers": [
        "freshers party poster design",
        "freshers welcome poster India",
        "welcome party poster design",
        "fresher orientation poster",
    ],
    "college_events/farewell": [
        "farewell party poster college",
        "goodbye seniors poster design",
        "senior farewell poster",
        "farewell ceremony poster",
    ],
    "college_events/graduation": [
        "graduation ceremony poster",
        "convocation poster design",
        "degree ceremony poster",
        "graduation day poster",
    ],
    "college_events/clubs_recruitment": [
        "student club poster design",
        "college society recruitment poster",
        "club recruitment drive poster",
        "join our club poster design",
    ],
    "college_events/alumni_reunion": [
        "alumni meet poster design",
        "class reunion poster",
        "homecoming event poster",
        "alumni networking event poster",
    ],

    # ══════════════════════════════════════════════════════════════
    # FESTIVALS  (parent: festivals/)
    # ══════════════════════════════════════════════════════════════
    "festivals/diwali": [
        "Diwali celebration poster",
        "Diwali event poster design",
        "Diwali festival poster",
        "Deepavali poster design",
        "Diwali mela poster",
        "Diwali night event poster",
    ],
    "festivals/holi": [
        "Holi festival poster design",
        "Holi event poster colorful",
        "Holi party poster design",
        "Holi DJ night poster",
        "Holi splash event poster",
    ],
    "festivals/navratri_garba": [
        "Navratri celebration poster",
        "Navratri garba night poster",
        "dandiya event poster",
        "Navratri festival poster design",
        "garba night pass design",
        "dandiya raas invitation",
        "navratri dandiya night flyer",
        "gujarati garba night poster",
        "navratri utsav poster",
        "dandiya night ticket design"
    ],
    "festivals/durga_puja": [
        "Durga puja poster design",
        "Durga puja pandal poster",
        "Durga puja celebration poster",
    ],
    "festivals/ganesh_chaturthi": [
        "Ganesh Chaturthi poster design",
        "Ganpati festival poster",
        "Ganesh utsav poster",
        "eco friendly Ganpati poster",
    ],
    "festivals/eid": [
        "Eid celebration poster design",
        "Eid mubarak event poster",
        "Eid ul fitr poster",
        "Ramadan event poster",
        "iftar party poster",
    ],
    "festivals/christmas": [
        "Christmas party poster design",
        "Christmas celebration event poster",
        "Christmas carnival poster",
    ],
    "festivals/new_year": [
        "new year celebration poster",
        "new year eve party poster",
        "new year countdown poster",
    ],
    "festivals/onam": [
        "Onam festival poster design",
        "Onam celebration poster",
        "Kerala Onam poster",
    ],
    "festivals/pongal_sankranti": [
        "Pongal celebration poster",
        "Makar Sankranti poster design",
        "Lohri celebration poster",
        "harvest festival poster India",
    ],
    "festivals/independence_republic": [
        "independence day poster India",
        "republic day poster design",
        "15 August celebration poster",
        "26 January event poster",
        "patriotic event poster India",
    ],

    # ══════════════════════════════════════════════════════════════
    # WORKSHOPS & ACADEMIC  (parent: workshops/)
    # ══════════════════════════════════════════════════════════════
    "workshops/coding": [
        "coding workshop poster",
        "python workshop poster",
        "programming workshop poster design",
        "hackathon coding workshop poster",
        "web dev bootcamp poster",
        "learn to code event poster",
        "java programming workshop poster",
        "c++ workshop poster design",
        "react js workshop poster",
        "machine learning workshop poster design",
        "app development workshop poster",
        "coding bootcamp flyer design",
        "programming contest poster",
        "software engineering workshop poster",
        "game development workshop poster",
        "data structures workshop poster",
        "coding marathon poster design",
        "algorithm workshop poster",
        "backend development workshop poster",
        "frontend workshop poster design"
    ],
    "workshops/design": [
        "graphic design workshop poster",
        "UI UX design workshop poster",
        "video editing workshop poster",
        "photography workshop poster",
        "logo design workshop poster",
        "poster design workshop flyer",
        "typography workshop poster",
        "adobe photoshop workshop poster",
        "adobe illustrator workshop poster",
        "digital art workshop poster",
        "creative design workshop poster",
        "branding workshop poster design",
        "product design workshop poster",
        "animation workshop poster design",
        "3d design workshop poster",
        "figma workshop poster",
        "canva design workshop poster",
        "sketching workshop poster design",
        "motion graphics workshop poster",
        "visual design workshop poster"
    ],
    "workshops/business": [
        "entrepreneurship seminar poster",
        "startup workshop poster",
        "business plan competition poster",
        "marketing workshop poster",
        "business strategy workshop flyer",
        "startup weekend poster",
        "business model canvas workshop",
        "digital marketing seminar poster",
        "finance workshop poster",
        "MBA event poster design"
    ],
    "workshops/soft_skills": [
        "public speaking workshop poster",
        "leadership workshop poster",
        "communication skills seminar poster",
        "resume building workshop poster",
    ],
    "workshops/seminar": [
        "seminar poster template professional",
        "webinar event poster",
        "guest lecture poster design",
        "research paper workshop poster",
    ],
    "workshops/conference": [
        "academic conference poster",
        "research symposium poster",
        "TEDx event poster design",
        "panel discussion poster",
        "keynote speaker event poster",
    ],
    "workshops/placement": [
        "placement drive poster design",
        "career fair poster",
        "campus hiring poster design",
        "internship drive poster",
        "job recruitment poster",
    ],

    # ══════════════════════════════════════════════════════════════
    # SOCIAL & AWARENESS  (parent: social/)
    # ══════════════════════════════════════════════════════════════
    "social/blood_donation": [
        "blood donation camp poster",
        "blood donation drive poster",
        "donate blood save life poster",
    ],
    "social/environment": [
        "environment day poster design",
        "tree planting event poster",
        "cleanliness drive poster",
        "earth day poster design",
    ],
    "social/charity": [
        "charity event poster design",
        "fundraiser poster",
        "NGO event poster",
        "donation drive poster design",
    ],
    "social/awareness": [
        "health awareness camp poster",
        "women empowerment event poster",
        "mental health awareness poster",
        "road safety awareness poster",
    ],

    # ══════════════════════════════════════════════════════════════
    # FOOD & ENTERTAINMENT  (parent: entertainment/)
    # ══════════════════════════════════════════════════════════════
    "entertainment/food_fest": [
        "food festival poster design",
        "food carnival poster",
        "street food event poster",
        "bake sale poster design",
        "cooking competition poster",
    ],
    "entertainment/movie_night": [
        "movie night event poster",
        "film screening poster design",
        "cinema night poster",
        "short film festival poster",
    ],
    "entertainment/gaming": [
        "gaming night poster design",
        "LAN party poster",
        "board game event poster",
        "game jam poster design",
    ],

    # ══════════════════════════════════════════════════════════════
    # DESIGN STYLES  (parent: styles/)
    # ══════════════════════════════════════════════════════════════
    "styles/minimalist": [
        "minimalist event poster design",
        "clean modern poster layout",
        "simple elegant poster design",
        "white space poster design",
    ],
    "styles/neon_glow": [
        "neon glow party poster design",
        "glowing neon event poster",
        "cyberpunk poster design",
        "neon lights party poster",
    ],
    "styles/retro_vintage": [
        "retro vintage poster design",
        "80s style event poster",
        "vintage college event poster",
        "retro music poster design",
    ],
    "styles/3d_futuristic": [
        "3D event poster design",
        "futuristic poster design",
        "sci-fi event poster",
        "holographic poster design",
    ],
    "styles/watercolor": [
        "watercolor event poster design",
        "hand painted poster design",
        "artistic poster illustration",
        "brush stroke poster design",
    ],
    "styles/gradient": [
        "gradient poster design modern",
        "colorful gradient event poster",
        "vibrant gradient poster",
        "modern abstract poster design",
    ],
    "styles/dark_theme": [
        "dark theme poster design",
        "black background event poster",
        "dark mode poster design",
        "dark elegant poster",
    ],
    "styles/typography": [
        "typography poster design",
        "bold text poster design",
        "kinetic typography poster",
        "lettering poster design",
    ],
    "styles/illustration": [
        "illustrated event poster",
        "cartoon style poster design",
        "hand drawn poster design",
        "vector illustration poster",
    ],

    # ══════════════════════════════════════════════════════════════
    # GENERAL  (catch-all)
    # ══════════════════════════════════════════════════════════════
    "general": [
        "event poster design modern",
        "professional poster layout",
        "modern event flyer design",
        "creative poster design 2024",
        "minimalist event poster",
    ],
}



# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Hash Dedup  (Moved to image_deduplicator.py)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Pinterest Scraper
# ─────────────────────────────────────────────────────────────────────────────
class PinterestScraper:
    """Scrape poster images from Pinterest using Selenium."""

    PINTEREST_SEARCH_URL = "https://www.pinterest.com/search/pins/?q={query}"
    TARGET_PER_THEME = 1900  # Download extra to ensure 1300+ survive quality filtering

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _create_driver(self) -> "webdriver.Chrome":
        """Create a headless Chrome driver."""
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)

    def _scroll_and_collect_urls(self, driver, query: str, max_images: int) -> list[str]:
        """Scroll Pinterest search page and collect image URLs."""
        url = self.PINTEREST_SEARCH_URL.format(query=query.replace(" ", "+"))
        driver.get(url)
        time.sleep(3)

        image_urls: set[str] = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        stall_count = 0

        pbar = tqdm(total=max_images, desc=f"  Scrolling: {query[:40]}")
        while len(image_urls) < max_images and stall_count < 8:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.scroll_pause)

            # Parse page for image URLs
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                # Pinterest uses /originals/ for full-res or /736x/ for medium
                if "pinimg.com" in src:
                    # Try to get highest resolution
                    full_url = src.replace("/236x/", "/originals/").replace("/474x/", "/originals/").replace("/736x/", "/originals/")
                    image_urls.add(full_url)

            pbar.update(len(image_urls) - pbar.n)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                stall_count += 1
            else:
                stall_count = 0
            last_height = new_height

        pbar.close()
        return list(image_urls)[:max_images]

    def _download_image(self, url: str, save_path: Path) -> bool:
        """Download a single image, validate, and dedup."""
        try:
            resp = self.session.get(url, timeout=self.download_timeout)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content)).convert("RGB")

            # Check minimum resolution
            if min(img.size) < self.min_resolution:
                return False

            # Check duplicate against global corpus cache
            if self.dedup.is_duplicate(img, save_path=str(save_path)):
                return False

            img.save(save_path, "JPEG", quality=95)
            self.dedup.add_to_disk_cache(str(save_path), img)
            return True

        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return False

    def scrape_category(self, category: str, queries: list[str]) -> int:
        """
        Scrape images for one category/theme.
        Keeps going until TARGET_PER_THEME (1000) is reached.
        Cycles through queries multiple rounds with increasing scroll depth.
        Skips already-downloaded images.
        """
        cat_dir = self.output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images (skip already-downloaded)
        existing_files = set(f.name for f in cat_dir.glob("*.jpg"))
        existing_count = len(existing_files)
        logger.info(f"Category '{category}': {existing_count} existing images")

        if existing_count >= self.TARGET_PER_THEME:
            logger.info(f"  ✓ Already at target ({self.TARGET_PER_THEME}), skipping!")
            return existing_count

        remaining = self.TARGET_PER_THEME - existing_count
        logger.info(f"  Need {remaining} more images to reach {self.TARGET_PER_THEME}")

        if not HAS_SELENIUM:
            logger.error("Selenium not available — cannot scrape Pinterest.")
            return 0

        driver = self._create_driver()
        total_downloaded = existing_count
        all_seen_urls: set[str] = set()  # Track all URLs across rounds

        try:
            round_num = 0
            max_rounds = 5  # Try up to 5 rounds of cycling through queries

            while total_downloaded < self.TARGET_PER_THEME and round_num < max_rounds:
                round_num += 1
                round_new = 0
                # Increase scroll depth each round to find deeper content
                scroll_target = 300 + (round_num * 200)

                logger.info(f"\n  ── Round {round_num}/{max_rounds} (scroll depth: {scroll_target}) ──")

                for query_idx, query in enumerate(queries):
                    if total_downloaded >= self.TARGET_PER_THEME:
                        break

                    # Add variation to queries in later rounds
                    if round_num > 1:
                        variations = [
                            f"{query} HD",
                            f"{query} professional",
                            f"{query} creative",
                            f"{query} inspiration",
                            f"best {query}",
                        ]
                        actual_query = variations[(round_num - 2) % len(variations)]
                    else:
                        actual_query = query

                    logger.info(f"  Query [{query_idx+1}/{len(queries)}]: '{actual_query}'")
                    urls = self._scroll_and_collect_urls(driver, actual_query, scroll_target)

                    # Filter out already-seen URLs
                    new_urls = [u for u in urls if u not in all_seen_urls]
                    all_seen_urls.update(urls)
                    logger.info(f"  Found {len(urls)} URLs ({len(new_urls)} new)")

                    for url in tqdm(new_urls, desc=f"  Downloading", leave=False):
                        if total_downloaded >= self.TARGET_PER_THEME:
                            break

                        fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
                        save_path = cat_dir / fname

                        # Skip if already downloaded
                        if fname in existing_files or save_path.exists():
                            continue

                        if self._download_image(url, save_path):
                            total_downloaded += 1
                            round_new += 1
                            existing_files.add(fname)

                    # Rate-limit between queries
                    time.sleep(3)

                logger.info(f"  Round {round_num} complete: +{round_new} new images, {total_downloaded} total")

                # If no new images found this round, stop early
                if round_new == 0:
                    logger.info(f"  No new images found in round {round_num}, moving on.")
                    break

        finally:
            driver.quit()

        new_count = len(list(cat_dir.glob("*.jpg")))
        logger.info(
            f"\nCategory '{category}': {new_count}/{self.TARGET_PER_THEME} images "
            f"({new_count - existing_count} new this session)"
        )
        return new_count

    def scrape_all(self, queries_map: dict[str, list[str]] | None = None) -> dict[str, int]:
        """Scrape all categories."""
        if queries_map is None:
            queries_map = DEFAULT_QUERIES

        results = {}
        for category, queries in queries_map.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping category: {category}")
            logger.info(f"{'='*60}")
            count = self.scrape_category(category, queries)
            results[category] = count

        return results



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pinterest Poster Image Scraper")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--category", default=None, help="Scrape a single category only")
    parser.add_argument("--target", type=int, default=None, help="Override target image count (default: 1900)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Override target if specified
    if args.target:
        PinterestScraper.TARGET_PER_THEME = args.target
        logger.info(f"🎯 Target count overridden to {args.target} images per category")

    raw_dir = config["paths"]["data"]["raw"]

    scraper = PinterestScraper(config, raw_dir)

    if args.category:
        queries = DEFAULT_QUERIES.get(args.category, [f"{args.category} poster design"])
        results = {args.category: scraper.scrape_category(args.category, queries)}
    else:
        results = scraper.scrape_all()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCRAPING SUMMARY")
    logger.info("=" * 60)
    total = 0
    for cat, count in results.items():
        logger.info(f"  {cat:20s}: {count:5d} images")
        total += count
    logger.info(f"  {'TOTAL':20s}: {total:5d} images")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
