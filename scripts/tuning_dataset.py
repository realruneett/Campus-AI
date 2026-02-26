#!/usr/bin/env python3
"""
Tuning Dataset Builder
======================
Downloads high-quality poster images for Phase 3 fine-tuning.
Uses Google Custom Search (free tier) and Bing image search as fallback.
Images are saved into data/tuning/<category>/<subcategory>/.

Usage:
    python scripts/tuning_dataset.py
    python scripts/tuning_dataset.py --per-category 20
    python scripts/tuning_dataset.py --dry-run
"""

import os
import sys
import json
import time
import hashlib
import argparse
import re
import requests
from pathlib import Path
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# SEARCH QUERIES — curated for each of the 55 subcategories
# ============================================================================

CATEGORIES = {
    # ---- College Events ----
    "college_events/farewell": [
        "college farewell party poster design HD",
        "farewell event invitation poster aesthetic",
        "farewell night celebration poster elegant design",
    ],
    "college_events/freshers": [
        "freshers party welcome poster design neon",
        "freshers day college poster vibrant colorful",
        "fresher welcome event poster creative",
    ],
    "college_events/alumni_reunion": [
        "alumni reunion event poster design elegant",
        "alumni meet invitation poster university",
    ],
    "college_events/graduation": [
        "graduation ceremony poster elegant gold",
        "convocation celebration poster university beautiful",
    ],

    # ---- Cultural Fest ----
    "cultural_fest/art_exhibition": [
        "art exhibition poster design modern gallery",
        "art gallery opening event poster minimal beautiful",
    ],
    "cultural_fest/dance": [
        "dance competition poster vibrant colorful HD",
        "dance festival event poster aesthetic beautiful",
        "classical dance performance poster Indian design",
    ],
    "cultural_fest/drama_theatre": [
        "theatre drama play poster artistic dark elegant",
        "stage play event poster design creative",
    ],
    "cultural_fest/fashion_show": [
        "fashion show event poster elegant luxury design",
        "college fashion gala poster premium aesthetic",
    ],
    "cultural_fest/general": [
        "cultural fest poster college India vibrant",
        "annual cultural festival poster colorful design",
    ],
    "cultural_fest/literary": [
        "literary fest poster book reading event design",
        "poetry slam event poster creative typography",
    ],
    "cultural_fest/music": [
        "music concert poster design neon glow HD",
        "live music event poster rock band stage",
        "college music festival poster vibrant DJ",
    ],
    "cultural_fest/standup_comedy": [
        "standup comedy show poster design microphone",
        "open mic comedy night poster neon creative",
        "comedy event poster funny stage spotlight",
    ],

    # ---- Entertainment ----
    "entertainment/food_fest": [
        "food festival poster design appetizing HD",
        "street food fest poster colorful delicious",
        "college food carnival poster warm inviting",
    ],
    "entertainment/gaming": [
        "gaming tournament poster esports neon RGB",
        "video game competition poster futuristic glowing",
    ],
    "entertainment/movie_night": [
        "movie night poster cinema event retro",
        "outdoor movie screening poster vintage film",
    ],

    # ---- Festivals ----
    "festivals/christmas": [
        "christmas celebration poster festive red green",
        "merry christmas event poster elegant snowflakes",
    ],
    "festivals/diwali": [
        "diwali celebration poster beautiful golden diya HD",
        "deepavali festival poster vibrant rangoli colors",
        "diwali night event poster fireworks sparkle",
    ],
    "festivals/durga_puja": [
        "durga puja poster beautiful artistic HD",
        "durga puja celebration poster traditional bengali",
    ],
    "festivals/eid": [
        "eid celebration poster beautiful crescent moon",
        "eid mubarak event poster elegant islamic design",
    ],
    "festivals/ganesh_chaturthi": [
        "ganesh chaturthi poster design vibrant festival",
        "ganpati celebration poster traditional colorful",
    ],
    "festivals/holi": [
        "holi festival poster colorful splash paint HD",
        "holi celebration party poster vibrant gulal",
    ],
    "festivals/independence_republic": [
        "india independence day poster tricolor patriotic",
        "republic day celebration poster 26 january",
    ],
    "festivals/navratri_garba": [
        "navratri garba poster design colorful dandiya",
        "dandiya night event poster festive vibrant",
        "garba raas festival poster traditional Gujarat",
    ],
    "festivals/new_year": [
        "new year celebration poster party fireworks",
        "new year eve event poster glowing golden",
    ],
    "festivals/onam": [
        "onam festival poster kathakali traditional Kerala",
        "onam celebration poster pookalam floral boat",
    ],
    "festivals/pongal_sankranti": [
        "pongal festival poster traditional Tamil Nadu",
        "makar sankranti poster kite festival colorful",
    ],

    # ---- Social ----
    "social/awareness": [
        "social awareness campaign poster design impactful",
        "mental health awareness poster college creative",
    ],
    "social/blood_donation": [
        "blood donation camp poster design red heart",
        "donate blood save lives poster minimal clean",
    ],
    "social/charity": [
        "charity event poster design heartfelt giving",
        "fundraiser event poster college community",
    ],
    "social/environment": [
        "environment day poster tree planting green earth",
        "eco friendly campaign poster sustainability",
    ],

    # ---- Sports ----
    "sports/athletics": [
        "athletics sports day poster dynamic running",
        "track and field event poster energy motion",
    ],
    "sports/badminton_tennis": [
        "badminton tournament poster design sports action",
        "tennis competition poster athletic dynamic",
    ],
    "sports/basketball": [
        "basketball tournament poster dynamic slam dunk HD",
        "basketball championship poster sports energy",
    ],
    "sports/cricket": [
        "cricket tournament poster design India stadium HD",
        "cricket match poster IPL style vibrant action",
        "cricket championship poster batsman dynamic",
    ],
    "sports/esports": [
        "esports tournament poster gaming neon cyberpunk",
        "valorant tournament poster aggressive design",
        "gaming championship poster RGB glowing dark",
    ],
    "sports/football": [
        "football tournament poster design action dynamic",
        "soccer championship event poster stadium energy",
    ],
    "sports/general": [
        "sports day poster college event medals trophy",
        "annual sports meet poster design vibrant",
    ],
    "sports/kabaddi_kho": [
        "kabaddi tournament poster Indian sports action",
        "kho kho competition poster dynamic traditional",
    ],
    "sports/yoga_fitness": [
        "yoga day poster peaceful sunrise meditation",
        "fitness challenge poster gym workout energy",
    ],

    # ---- Styles ----
    "styles/3d_futuristic": [
        "futuristic 3D poster design abstract technology",
        "3D event poster sci-fi hologram aesthetic",
    ],
    "styles/dark_theme": [
        "dark theme poster design moody elegant",
        "dark aesthetic event poster premium black gold",
    ],
    "styles/gradient": [
        "gradient poster design smooth mesh colors",
        "gradient background poster modern vibrant",
    ],
    "styles/illustration": [
        "illustrated event poster hand drawn artistic",
        "illustration poster design flat vector creative",
    ],
    "styles/minimalist": [
        "minimalist poster design clean modern white",
        "minimal event poster elegant white space",
    ],
    "styles/neon_glow": [
        "neon glow poster design vibrant dark",
        "neon lights event poster cyberpunk glowing",
    ],
    "styles/retro_vintage": [
        "retro vintage poster design grunge old school",
        "vintage event poster classic typography worn",
    ],
    "styles/typography": [
        "typography poster design bold text art creative",
        "typographic event poster lettering experimental",
    ],
    "styles/watercolor": [
        "watercolor poster design soft artistic floral",
        "watercolor painting poster pastel dreamy",
    ],

    # ---- Tech Fest ----
    "tech_fest/ai_ml": [
        "AI machine learning event poster futuristic neural",
        "artificial intelligence conference poster technology",
    ],
    "tech_fest/coding_competition": [
        "coding competition poster hacker developer dark",
        "code challenge event poster programming terminal",
    ],
    "tech_fest/cybersecurity": [
        "cybersecurity event poster hacker CTF dark",
        "cyber security awareness poster digital lock",
    ],
    "tech_fest/general": [
        "tech fest poster college futuristic innovation",
        "technology festival poster digital modern",
    ],
    "tech_fest/hackathon": [
        "hackathon event poster design code developer",
        "36 hour hackathon poster startup tech vibrant",
        "hack day poster creative developer community",
    ],
    "tech_fest/robotics": [
        "robotics competition poster futuristic mechanical",
        "robot challenge event poster technology modern",
    ],
    "tech_fest/web_app_dev": [
        "web development workshop poster modern code",
        "app development event poster mobile technology",
    ],

    # ---- Workshops ----
    "workshops/business": [
        "business workshop poster corporate professional",
        "entrepreneurship event poster startup modern",
    ],
    "workshops/coding": [
        "coding workshop poster developer bootcamp",
        "programming workshop poster technology education",
    ],
    "workshops/conference": [
        "conference event poster professional academic",
        "academic conference poster modern clean",
    ],
    "workshops/design": [
        "design workshop poster UI UX creative",
        "graphic design event poster artistic colorful",
    ],
    "workshops/placement": [
        "placement drive poster campus recruitment",
        "career fair poster professional job event",
    ],
    "workshops/seminar": [
        "seminar event poster professional academic clean",
        "guest lecture poster university speaker modern",
    ],
    "workshops/soft_skills": [
        "soft skills workshop poster leadership training",
        "communication skills event poster professional",
    ],
}


# ============================================================================
# IMAGE SEARCH ENGINE  (DuckDuckGo — no API key needed)
# ============================================================================

def search_images(query, max_results=8):
    """Search for images using DuckDuckGo. Returns list of image URLs."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        # Get search token
        resp = requests.get(
            f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images",
            headers=headers, timeout=10
        )
        vqd = None
        match = re.search(r"vqd=([\d-]+)", resp.text)
        if match:
            vqd = match.group(1)
        if not vqd:
            # Try alternative pattern
            match = re.search(r"vqd=['\"]?([\d-]+)", resp.text)
            if match:
                vqd = match.group(1)
        if not vqd:
            return []

        # Fetch image results
        params = {
            "l": "us-en", "o": "json", "q": query,
            "vqd": vqd, "f": ",,,,,", "p": "1",
        }
        resp = requests.get(
            "https://duckduckgo.com/i.js",
            headers=headers, params=params, timeout=10
        )
        data = resp.json()

        urls = []
        for result in data.get("results", [])[:max_results * 2]:
            url = result.get("image", "")
            if url and url.startswith("http"):
                # Prefer larger images
                width = result.get("width", 0)
                height = result.get("height", 0)
                if width >= 400 and height >= 400:
                    urls.append(url)
                elif len(urls) < max_results // 2:
                    urls.append(url)  # Accept smaller ones if few results
            if len(urls) >= max_results:
                break

        return urls[:max_results]

    except Exception as e:
        return []


# ============================================================================
# IMAGE DOWNLOADER with validation
# ============================================================================

def download_image(url, save_path, min_size_kb=15, timeout=12):
    """Download and validate a single image. Returns True on success."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not any(
            url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
        ):
            return False

        data = resp.content

        # Skip tiny/broken images
        if len(data) < min_size_kb * 1024:
            return False

        # Quick header check — verify it's actually an image
        if not (data[:2] == b'\xff\xd8' or      # JPEG
                data[:4] == b'\x89PNG' or         # PNG
                data[:4] == b'RIFF' or            # WebP
                data[:3] == b'GIF'):              # GIF
            return False

        with open(save_path, "wb") as f:
            f.write(data)
        return True

    except Exception:
        return False


def get_filename(url, folder):
    """Generate a unique, deterministic filename from the URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return os.path.join(folder, f"tuning_{url_hash}.jpg")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tuning Dataset Builder — download fresh poster images for Phase 3"
    )
    parser.add_argument("--target", default="data/tuning",
                        help="Root directory to save images into")
    parser.add_argument("--per-category", type=int, default=15,
                        help="Target new images per subcategory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview searches without downloading")
    args = parser.parse_args()

    total_cats = len(CATEGORIES)
    print("=" * 60)
    print("  TUNING DATASET BUILDER — Phase 3")
    print("=" * 60)
    print(f"  Target folder : {args.target}")
    print(f"  Per subcategory: {args.per_category} images")
    print(f"  Subcategories : {total_cats}")
    print(f"  Est. total    : ~{total_cats * args.per_category} images")
    print("=" * 60)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for i, (subcat, queries) in enumerate(CATEGORIES.items(), 1):
        folder = os.path.join(args.target, subcat)
        os.makedirs(folder, exist_ok=True)

        existing = len([f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])

        print(f"\n[{i:02d}/{total_cats}] 📁 {subcat}  ({existing} existing)")

        if args.dry_run:
            for q in queries:
                print(f"  🔍 Would search: '{q}'")
            continue

        downloaded = 0
        per_query = max(3, (args.per_category + len(queries) - 1) // len(queries))

        for query in queries:
            if downloaded >= args.per_category:
                break

            print(f"  🔍 '{query}'")
            urls = search_images(query, max_results=per_query + 3)

            if not urls:
                print(f"     ⚠️  No results")
                continue

            for url in urls:
                if downloaded >= args.per_category:
                    break

                filepath = get_filename(url, folder)
                if os.path.exists(filepath):
                    stats["skipped"] += 1
                    continue

                if download_image(url, filepath):
                    downloaded += 1
                    stats["downloaded"] += 1
                    print(f"     ✅ {downloaded}/{args.per_category}")
                else:
                    stats["failed"] += 1

            # Rate limit — be respectful
            time.sleep(1.5)

        print(f"  → {downloaded} new images saved")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  ✅ Downloaded : {stats['downloaded']}")
    print(f"  ⏭️  Skipped   : {stats['skipped']} (duplicates)")
    print(f"  ❌ Failed     : {stats['failed']}")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Caption the new images:")
    print("     python scripts/caption_generator.py --input data/tuning")
    print("  2. Run Phase 3 training:")
    print("     cd ai-toolkit && python run.py ../configs/train_sdxl_lora_phase3.yaml")
    print()


if __name__ == "__main__":
    main()
