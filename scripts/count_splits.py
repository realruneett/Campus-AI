
import os
from pathlib import Path

# Config
data_root = Path("data")
train_dir = data_root / "train"
val_dir = data_root / "val"
test_dir = data_root / "test"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def count_images_in_dir(d: Path) -> int:
    if not d.exists():
        return 0
    return len([f for f in os.listdir(d) if Path(f).suffix.lower() in IMG_EXTENSIONS])

# Find all categories from processed dir (source of truth)
processed_dir = data_root / "processed"
categories = set()

if processed_dir.exists():
    for root, dirs, files in os.walk(processed_dir):
        if any(Path(f).suffix.lower() in IMG_EXTENSIONS for f in files):
            rel = Path(root).relative_to(processed_dir)
            categories.add(str(rel).replace("\\", "/"))
else:
    # Fallback: finding categories from splits directly
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            for root, dirs, files in os.walk(d):
                if any(Path(f).suffix.lower() in IMG_EXTENSIONS for f in files):
                    rel = Path(root).relative_to(d)
                    categories.add(str(rel).replace("\\", "/"))

print(f"{'Category':<40} | {'Train':<6} | {'Val':<5} | {'Test':<5} | {'Total':<6} | {'% Train':<8}")
print("-" * 100)

grand_totals = {"train": 0, "val": 0, "test": 0, "total": 0}

for cat in sorted(list(categories)):
    c_train = count_images_in_dir(train_dir / cat)
    c_val = count_images_in_dir(val_dir / cat)
    c_test = count_images_in_dir(test_dir / cat)
    total = c_train + c_val + c_test
    
    grand_totals["train"] += c_train
    grand_totals["val"] += c_val
    grand_totals["test"] += c_test
    grand_totals["total"] += total

    pct_train = (c_train / total * 100) if total > 0 else 0.0
    
    print(f"{cat:<40} | {c_train:<6} | {c_val:<5} | {c_test:<5} | {total:<6} | {pct_train:.1f}%")

print("-" * 100)
t_train = grand_totals['train']
t_total = grand_totals['total']
t_pct = (t_train / t_total * 100) if t_total > 0 else 0
print(f"{'TOTAL':<40} | {t_train:<6} | {grand_totals['val']:<5} | {grand_totals['test']:<5} | {t_total:<6} | {t_pct:.1f}%")
