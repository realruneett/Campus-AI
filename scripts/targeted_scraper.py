import logging
import sys
import os
import time

# Add current directory to path so we can import sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinterest_scraper import PinterestScraper, load_config, DEFAULT_QUERIES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of categories that need more images (Target: 2800 raw to get ~1300 clean)
TARGET_CATEGORIES = [
    "workshops/design",
    "workshops/coding",
    "workshops/business",
    "tech_fest/hackathon",
    "tech_fest/general",
    "tech_fest/coding_competition",
    "tech_fest/web_app_dev",
    "tech_fest/cybersecurity",
    "festivals/navratri_garba",
    "sports/general"
]

TARGET_COUNT = 2800

def main():
    logger.info("🚀 Starting Targeted Scraper for Low-Data Categories")
    logger.info(f"🎯 Target Count: {TARGET_COUNT} images per category")
    
    # Load config from parent directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    config = load_config(config_path)
    
    raw_dir = config["paths"]["data"]["raw"]
    
    # Initialize scraper
    scraper = PinterestScraper(config, raw_dir)
    
    # Override global target
    scraper.TARGET_PER_THEME = TARGET_COUNT
    
    for category in TARGET_CATEGORIES:
        logger.info(f"\n============================================================")
        logger.info(f"Processing: {category}")
        logger.info(f"============================================================")
        
        # Get queries for this category
        queries = DEFAULT_QUERIES.get(category)
        if not queries:
            logger.warning(f"⚠️ No specific queries found for {category}, generating generic ones.")
            # Fallback if no specific queries exist (though they should based on our previous edits)
            theme = category.split("/")[-1]
            queries = [f"{theme} poster design", f"{theme} event flyer", f"creative {theme} poster"]
            
        try:
            count = scraper.scrape_category(category, queries)
            logger.info(f"✅ Finished {category}: {count} total images")
        except Exception as e:
            logger.error(f"❌ Failed processing {category}: {e}")
            
        # Small break between categories
        time.sleep(2)

    logger.info("\n🎉 All targeted categories processed!")

if __name__ == "__main__":
    main()
