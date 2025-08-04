#!/bin/bash
# 学習用画像の高速コピー&リサイズスクリプト
# 640x480画像を224x224にリサイズしながらコピー

# グロブパターンがマッチしない場合でもエラーにならないようにする
shopt -s nullglob

IMAGES_DIR="${1:-images}"
TRAINING_DIR="${2:-temp_training_data}"
TARGET_SIZE="${3:-224x224}"
NORMAL_DIR="${TRAINING_DIR}/normal"

# ImageMagickの存在確認
if ! command -v convert &> /dev/null; then
    echo "エラー: ImageMagickが見つかりません。インストールしてください。"
    exit 1
fi

# ディレクトリ作成
mkdir -p "$NORMAL_DIR"

total_images=0

echo "画像リサイズ&コピー処理開始... (ターゲットサイズ: ${TARGET_SIZE})"

# grid_XX ディレクトリから画像をリサイズ&コピー
for i in {00..15}; do
    grid_dir="${IMAGES_DIR}/grid_${i}"
    if [ -d "$grid_dir" ]; then
        image_count=0
        echo "処理中: grid_${i}... (リサイズ: ${TARGET_SIZE})"
        for img in "$grid_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
            # グロブパターンがマッチしない場合をスキップ
            [ -f "$img" ] || continue
            dest_name="grid_${i}_$(basename "$img")"
            dest_path="$NORMAL_DIR/$dest_name"
            
            # ImageMagickでリサイズしながらコピー
            if convert "$img" -resize "${TARGET_SIZE}!" "$dest_path" 2>/dev/null; then
                ((image_count++))
                ((total_images++))
            else
                echo "警告: $img のリサイズに失敗しました"
            fi
        done
        if [ $image_count -gt 0 ]; then
            echo "grid_${i}: ${image_count} 画像をリサイズ&コピー"
        fi
    fi
done

# no_person ディレクトリから画像をリサイズ&コピー
no_person_dir="${IMAGES_DIR}/no_person"
if [ -d "$no_person_dir" ]; then
    image_count=0
    echo "処理中: no_person... (リサイズ: ${TARGET_SIZE})"
    for img in "$no_person_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
        # グロブパターンがマッチしない場合をスキップ
        [ -f "$img" ] || continue
        dest_name="no_person_$(basename "$img")"
        dest_path="$NORMAL_DIR/$dest_name"
        
        # ImageMagickでリサイズしながらコピー
        if convert "$img" -resize "${TARGET_SIZE}!" "$dest_path" 2>/dev/null; then
            ((image_count++))
            ((total_images++))
        else
            echo "警告: $img のリサイズに失敗しました"
        fi
    done
    if [ $image_count -gt 0 ]; then
        echo "no_person: ${image_count} 画像をリサイズ&コピー"
    fi
fi

echo "統合された学習用正常画像: ${total_images} 枚 (サイズ: ${TARGET_SIZE})"
echo "$total_images"