#!/bin/bash

# EfficientAd用データセット構成作成スクリプト
# 使用法: ./create_dataset_structure.sh <source_images_dir> <target_dataset_dir> <image_size>

set -e

SOURCE_DIR="$1"
TARGET_DIR="$2"
IMAGE_SIZE="$3"

if [ "$#" -ne 3 ]; then
    echo "使用法: $0 <source_images_dir> <target_dataset_dir> <image_size>"
    echo "例: $0 images dataset 256x256"
    exit 1
fi

echo "=== EfficientAd用データセット構成作成 ==="
echo "ソースディレクトリ: $SOURCE_DIR"
echo "ターゲットディレクトリ: $TARGET_DIR"
echo "画像サイズ: $IMAGE_SIZE"

# ターゲットディレクトリの作成
mkdir -p "$TARGET_DIR/train/good"
mkdir -p "$TARGET_DIR/test/good"
mkdir -p "$TARGET_DIR/test/defect"

total_images=0

# train/good: 正常な学習用画像 (grid_XX + no_person)
echo "train/good に正常画像をコピー中..."
train_count=0

# grid_XX ディレクトリから学習用正常画像をコピー
for grid_dir in "$SOURCE_DIR"/grid_*; do
    if [ -d "$grid_dir" ]; then
        echo "処理中: $grid_dir"
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            if [ -f "$img" ]; then
                filename=$(basename "$img")
                grid_name=$(basename "$grid_dir")
                target_name="${grid_name}_${filename}"
                convert "$img" -resize "$IMAGE_SIZE!" "$TARGET_DIR/train/good/$target_name"
                ((train_count++))
            fi
        done
    fi
done

# no_person ディレクトリから学習用正常画像をコピー
if [ -d "$SOURCE_DIR/no_person" ]; then
    echo "処理中: $SOURCE_DIR/no_person"
    for img in "$SOURCE_DIR/no_person"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            target_name="no_person_${filename}"
            convert "$img" -resize "$IMAGE_SIZE!" "$TARGET_DIR/train/good/$target_name"
            ((train_count++))
        fi
    done
fi

echo "train/good: $train_count 画像"
total_images=$((total_images + train_count))

# test/good: 正常なテスト用画像 (images/test/normal)
test_good_count=0
if [ -d "$SOURCE_DIR/test/normal" ]; then
    echo "test/good に正常テスト画像をコピー中..."
    for img in "$SOURCE_DIR/test/normal"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            convert "$img" -resize "$IMAGE_SIZE!" "$TARGET_DIR/test/good/$filename"
            ((test_good_count++))
        fi
    done
fi

echo "test/good: $test_good_count 画像"
total_images=$((total_images + test_good_count))

# test/defect: 異常なテスト用画像 (images/test/anomaly)
test_defect_count=0
if [ -d "$SOURCE_DIR/test/anomaly" ]; then
    echo "test/defect に異常テスト画像をコピー中..."
    for img in "$SOURCE_DIR/test/anomaly"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            convert "$img" -resize "$IMAGE_SIZE!" "$TARGET_DIR/test/defect/$filename"
            ((test_defect_count++))
        fi
    done
fi

echo "test/defect: $test_defect_count 画像"
total_images=$((total_images + test_defect_count))

echo "=== データセット作成完了 ==="
echo "データセット構成:"
echo "  dataset/"
echo "  ├── train/"
echo "  │   └── good/ ($train_count 画像)"
echo "  └── test/"
echo "      ├── good/ ($test_good_count 画像)"
echo "      └── defect/ ($test_defect_count 画像)"
echo ""
echo "総画像数: $total_images"

# 最後の行に総画像数を出力（Pythonスクリプトが読み取る）
echo "$total_images"