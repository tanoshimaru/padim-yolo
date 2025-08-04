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
for i in {00..15}; do
    grid_dir="${IMAGES_DIR}/grid_${i}"
    echo "DEBUG: Checking directory: $grid_dir" >&2
    if [ -d "$grid_dir" ]; then
        echo "DEBUG: Directory exists: $grid_dir" >&2
        image_count=0
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            echo "DEBUG: Checking file: $img" >&2
            # グロブパターンがマッチしない場合をスキップ
            [ -f "$img" ] || continue
            echo "DEBUG: File exists, copying: $img" >&2
            dest_name="grid_${i}_$(basename "$img")"
            if cp "$img" "$NORMAL_DIR/$dest_name"; then
                ((image_count++))
                ((total_images++))
                echo "DEBUG: Successfully copied: $img" >&2
            else
                echo "DEBUG: Failed to copy: $img (exit code: $?)" >&2
                echo "DEBUG: Source exists: $(ls -la "$img")" >&2
                echo "DEBUG: Target dir writable: $(ls -ld "$NORMAL_DIR")" >&2
            fi
        done
        echo "DEBUG: Finished processing $grid_dir, copied $image_count files" >&2
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