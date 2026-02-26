#!/usr/bin/env python3
"""Monitor download progress across all subcategories."""
import os
import time
from pathlib import Path

RAW_DIR = Path("data/raw")
TARGET_PER_SUBFOLDER = 1900

def count_images():
    """Count images in each subfolder and show progress."""
    os.system("cls" if os.name == "nt" else "clear")

    total_images = 0
    total_target = 0
    rows = []

    for parent in sorted(RAW_DIR.iterdir()):
        if not parent.is_dir():
            continue
        for sub in sorted(parent.iterdir()):
            if not sub.is_dir():
                continue
            count = sum(
                1 for f in sub.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
            remaining = max(0, TARGET_PER_SUBFOLDER - count)
            pct = min(100, count / TARGET_PER_SUBFOLDER * 100)
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            status = "✅" if count >= TARGET_PER_SUBFOLDER else "⏳"

            category = f"{parent.name}/{sub.name}"
            rows.append((category, count, remaining, pct, bar, status))
            total_images += count
            total_target += TARGET_PER_SUBFOLDER

    # Print header
    total_remaining = max(0, total_target - total_images)
    total_pct = total_images / total_target * 100 if total_target > 0 else 0
    print(f"{'='*80}")
    print(f"  📊 DOWNLOAD MONITOR  |  {total_images:,} / {total_target:,} images  "
          f"({total_pct:.1f}%)  |  {total_remaining:,} remaining")
    print(f"{'='*80}")
    print(f"  {'Category':<35} {'Count':>6} {'Left':>6} {'Progress':<24} ")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*24}")

    for category, count, remaining, pct, bar, status in rows:
        print(f"  {category:<35} {count:>6} {remaining:>6} {bar} {pct:5.1f}% {status}")

    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*24}")
    total_bar = "█" * int(total_pct // 5) + "░" * (20 - int(total_pct // 5))
    print(f"  {'TOTAL':<35} {total_images:>6} {total_remaining:>6} {total_bar} {total_pct:5.1f}%")
    print(f"\n  Last updated: {time.strftime('%H:%M:%S')}  |  Refreshing every 1s  |  Ctrl+C to stop")

if __name__ == "__main__":
    while True:
        try:
            count_images()
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n  Monitoring stopped.")
            break
