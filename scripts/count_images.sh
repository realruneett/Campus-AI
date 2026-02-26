#!/bin/bash
# Count images in data/processed subdirectories
# Usage: bash scripts/count_images.sh

TARGET=1300
DATA_DIR="data/processed"

echo "=================================================="
echo "  PROCESSED IMAGE COUNT REPORT (Target: $TARGET)"
echo "=================================================="
printf "%-40s %6s %10s\n" "CATEGORY" "COUNT" "STATUS"
echo "--------------------------------------------------------"

total_imgs=0
pass_count=0
fail_count=0

# Find all subdirectories that contain images
# Using find to get directories, then counting files inside
find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type d | sort | while read -r dir; do
    # Count image files (case insensitive extensions)
    count=$(find "$dir" -maxdepth 1 -type f | grep -iE "\.(jpg|jpeg|png|webp|bmp)$" | wc -l)
    
    # Get relative path (category/subcategory)
    rel_path=${dir#$DATA_DIR/}
    
    if [ "$count" -ge "$TARGET" ]; then
        status="✅ PASS"
        ((pass_count++))
    else
        status="❌ FAIL"
        ((fail_count++))
    fi
    
    if [ "$count" -gt 0 ]; then
        printf "%-40s %6d %10s\n" "$rel_path" "$count" "$status"
        total_imgs=$((total_imgs + count))
    fi
done

echo "--------------------------------------------------------"
# Recalculate total because of pipe subshell scope issue in bash
grand_total=$(find "$DATA_DIR" -type f | grep -iE "\.(jpg|jpeg|png|webp|bmp)$" | wc -l)
echo "TOTAL: $grand_total images across all processed categories"
echo "=================================================="

# Check for failures (Need a separate loop or temp file to persist fail_count if strict, 
# but for visual report this is fine)
# To actually return bad exit code if failed:
failures=$(find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type d | while read -r d; do 
  c=$(find "$d" -maxdepth 1 -type f | grep -iE "\.(jpg|jpeg|png|webp|bmp)$" | wc -l); 
  if [ "$c" -lt "$TARGET" ] && [ "$c" -gt 0 ]; then echo "fail"; fi; 
done | wc -l)

if [ "$failures" -gt 0 ]; then
  echo "⚠️  $failures categories are below target ($TARGET)."
  echo "    Run 'python scripts/targeted_filter_v2.py' to fix."
else
  echo "🎉 All categories meet the target goal!"
fi
