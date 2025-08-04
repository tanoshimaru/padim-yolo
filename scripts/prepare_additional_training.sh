#!/bin/bash
# 追加学習用データ準備高速スクリプト

set -e

IMAGES_DIR="${1:-images}"
TRAIN_DATA_DIR="${2:-tmp/train_data}"
NORMAL_DIR="${TRAIN_DATA_DIR}/normal"
ABNORMAL_DIR="${TRAIN_DATA_DIR}/abnormal"

# ディレクトリ作成
mkdir -p "$NORMAL_DIR" "$ABNORMAL_DIR"

copied_count=0

# grid_XX ディレクトリから学習用画像をコピー
for i in {00..15}; do
    grid_dir="${IMAGES_DIR}/grid_${i}"
    if [ -d "$grid_dir" ]; then
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            if [ -f "$img" ]; then
                target_path="${NORMAL_DIR}/$(printf '%06d' $copied_count).png"
                cp "$img" "$target_path"
                ((copied_count++))
            fi
        done
    fi
done

# no_person ディレクトリから学習用画像をコピー
no_person_dir="${IMAGES_DIR}/no_person"
if [ -d "$no_person_dir" ]; then
    for img in "$no_person_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        if [ -f "$img" ]; then
            target_path="${NORMAL_DIR}/$(printf '%06d' $copied_count).png"
            cp "$img" "$target_path"
            ((copied_count++))
        fi
    done
fi

echo "学習用画像を準備: ${copied_count}枚 -> ${NORMAL_DIR}"
echo "$copied_count"