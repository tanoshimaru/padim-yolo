#!/bin/bash
# 学習用画像の高速コピースクリプト

# グロブパターンがマッチしない場合でもエラーにならないようにする
shopt -s nullglob

IMAGES_DIR="${1:-images}"
TRAINING_DIR="${2:-temp_training_data}"
NORMAL_DIR="${TRAINING_DIR}/normal"

# ディレクトリ作成
mkdir -p "$NORMAL_DIR"

total_images=0

# grid_XX ディレクトリから画像をコピー
echo "画像コピー処理開始..."
for i in {00..15}; do
    grid_dir="${IMAGES_DIR}/grid_${i}"
    if [ -d "$grid_dir" ]; then
        image_count=0
        echo "処理中: grid_${i}..."
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            # グロブパターンがマッチしない場合をスキップ
            [ -f "$img" ] || continue
            dest_name="grid_${i}_$(basename "$img")"
            if cp "$img" "$NORMAL_DIR/$dest_name"; then
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
    echo "処理中: no_person..."
    for img in "$no_person_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        # グロブパターンがマッチしない場合をスキップ
        [ -f "$img" ] || continue
        dest_name="no_person_$(basename "$img")"
        if cp "$img" "$NORMAL_DIR/$dest_name"; then
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