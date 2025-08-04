#!/bin/bash
# 学習用画像の高速コピースクリプト

set -e

IMAGES_DIR="${1:-images}"
TRAINING_DIR="${2:-temp_training_data}"
NORMAL_DIR="${TRAINING_DIR}/normal"

# ディレクトリ作成
mkdir -p "$NORMAL_DIR"

total_images=0

# grid_XX ディレクトリから画像をコピー
for i in {00..15}; do
    grid_dir="${IMAGES_DIR}/grid_${i}"
    echo "DEBUG: Checking directory: $grid_dir" >&2
    if [ -d "$grid_dir" ]; then
        echo "DEBUG: Directory exists: $grid_dir" >&2
        image_count=0
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            if [ -f "$img" ]; then
                dest_name="grid_${i}_$(basename "$img")"
                cp "$img" "$NORMAL_DIR/$dest_name"
                ((image_count++))
                ((total_images++))
            fi
        done
        if [ $image_count -gt 0 ]; then
            echo "grid_${i}: ${image_count} 画像をコピー"
        fi
    fi
done

# no_person ディレクトリから画像をコピー
no_person_dir="${IMAGES_DIR}/no_person"
if [ -d "$no_person_dir" ]; then
    image_count=0
    for img in "$no_person_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        if [ -f "$img" ]; then
            dest_name="no_person_$(basename "$img")"
            cp "$img" "$NORMAL_DIR/$dest_name"
            ((image_count++))
            ((total_images++))
        fi
    done
    if [ $image_count -gt 0 ]; then
        echo "no_person: ${image_count} 画像をコピー"
    fi
fi

echo "統合された学習用正常画像: ${total_images} 枚"
echo "$total_images"