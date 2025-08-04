#!/bin/bash
# 学習データ準備用高速スクリプト

set -e

SOURCE_DIR="${1:-./images}"
TARGET_DIR="${2:-./training_data}"
NORMAL_RATIO="${3:-0.8}"
COPY_MODE="${4:-true}"
RANDOM_SEED="${5:-42}"

echo "ソースディレクトリ: $SOURCE_DIR"
echo "ターゲットディレクトリ: $TARGET_DIR"
echo "正常画像比率: $NORMAL_RATIO"
echo "コピーモード: $COPY_MODE"

# ターゲットディレクトリ作成
mkdir -p "$TARGET_DIR/normal" "$TARGET_DIR/abnormal"

# 一時ファイル作成
temp_normal=$(mktemp)
temp_no_person=$(mktemp)

# grid_XX ディレクトリから正常画像を収集
normal_count=0
for grid_dir in "$SOURCE_DIR"/grid_*; do
    if [ -d "$grid_dir" ]; then
        grid_name=$(basename "$grid_dir")
        image_count=0
        for img in "$grid_dir"/*.{png,jpg,jpeg}; do
            if [ -f "$img" ]; then
                echo "$img" >> "$temp_normal"
                ((image_count++))
                ((normal_count++))
            fi
        done
        if [ $image_count -gt 0 ]; then
            echo "$grid_name: ${image_count} 枚の画像を発見"
        fi
    fi
done

# no_person ディレクトリから画像を収集
no_person_count=0
no_person_dir="$SOURCE_DIR/no_person"
if [ -d "$no_person_dir" ]; then
    for img in "$no_person_dir"/*.{png,jpg,jpeg}; do
        if [ -f "$img" ]; then
            echo "$img" >> "$temp_no_person"
            ((no_person_count++))
        fi
    done
    if [ $no_person_count -gt 0 ]; then
        echo "no_person: ${no_person_count} 枚の画像を発見"
    fi
fi

echo "総正常画像数: $normal_count"
echo "総no_person画像数: $no_person_count"

# シャッフルと分割のPythonスクリプトを作成・実行
python3 << EOF
import random
import shutil
import os
import math

random.seed($RANDOM_SEED)

def process_images(file_list, ratio, target_prefix):
    if not os.path.exists(file_list) or os.path.getsize(file_list) == 0:
        return 0, 0
    
    with open(file_list) as f:
        images = [line.strip() for line in f if line.strip()]
    
    if not images:
        return 0, 0
    
    random.shuffle(images)
    split_point = int(len(images) * ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]
    
    # 学習用画像をコピー
    train_count = 0
    for i, img_path in enumerate(train_images):
        if os.path.exists(img_path):
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            target_path = f"$TARGET_DIR/normal/{target_prefix}_{name}_{i:04d}{ext}"
            if "$COPY_MODE" == "true":
                shutil.copy2(img_path, target_path)
            else:
                shutil.move(img_path, target_path)
            train_count += 1
    
    # 検証用画像をコピー
    val_count = 0
    for i, img_path in enumerate(val_images):
        if os.path.exists(img_path):
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            target_path = f"$TARGET_DIR/abnormal/{target_prefix}_val_{name}_{i:04d}{ext}"
            if "$COPY_MODE" == "true":
                shutil.copy2(img_path, target_path)
            else:
                shutil.move(img_path, target_path)
            val_count += 1
    
    return train_count, val_count

# 正常画像処理
train_normal, val_normal = process_images("$temp_normal", $NORMAL_RATIO, "normal")
print(f"正常画像分割: 学習用 {train_normal} 枚, 検証用 {val_normal} 枚")

# no_person画像処理
train_no_person, val_no_person = process_images("$temp_no_person", $NORMAL_RATIO, "no_person")
print(f"no_person画像分割: 学習用 {train_no_person} 枚, 検証用 {val_no_person} 枚")

total_train = train_normal + train_no_person
total_val = val_normal + val_no_person
print(f"正常画像（学習用）: {total_train} 枚を配置")
print(f"検証用画像: {total_val} 枚を配置")
print(f"{total_train}|{total_val}")
EOF

# 一時ファイル削除
rm -f "$temp_normal" "$temp_no_person"