#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def classify_images():
    """
    images/checkedフォルダの画像をprefixに応じて分類する
    - grid_で始まる画像 → images/grid_xx/
    - no_personで始まる画像 → images/no_person/
    """
    checked_dir = Path("images/checked")
    
    if not checked_dir.exists():
        print(f"エラー: {checked_dir} フォルダが存在しません")
        return
    
    # 出力ディレクトリを作成
    grid_dirs = {}
    no_person_dir = Path("images/no_person")
    no_person_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    for image_file in checked_dir.iterdir():
        if not image_file.is_file():
            continue
            
        filename = image_file.name
        
        if filename.startswith("grid_"):
            # grid_xx形式のprefixを抽出
            parts = filename.split("_")
            if len(parts) >= 2:
                grid_prefix = f"grid_{parts[1]}"
                
                # 対応するgrid_xxディレクトリを作成
                if grid_prefix not in grid_dirs:
                    grid_dirs[grid_prefix] = Path(f"images/{grid_prefix}")
                    grid_dirs[grid_prefix].mkdir(parents=True, exist_ok=True)
                
                # ファイルを移動
                dest_path = grid_dirs[grid_prefix] / filename
                shutil.move(str(image_file), str(dest_path))
                print(f"移動: {filename} → {grid_prefix}/")
                moved_count += 1
                
        elif filename.startswith("no_person"):
            # no_personディレクトリに移動
            dest_path = no_person_dir / filename
            shutil.move(str(image_file), str(dest_path))
            print(f"移動: {filename} → no_person/")
            moved_count += 1
        else:
            print(f"スキップ: {filename} (該当するprefixなし)")
    
    print(f"\n分類完了: {moved_count} 個のファイルを移動しました")

if __name__ == "__main__":
    classify_images()