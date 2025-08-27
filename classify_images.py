#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def classify_images():
    """
    images/checkedフォルダの画像をprefixに応じて分類する
    - defect_grid_で始まる画像 → images/defect/
    - grid_で始まる画像 → images/grid_xx/
    - no_personで始まる画像 → images/no_person/
    - error_で始まる画像 → 削除
    """
    checked_dir = Path("images/checked")
    
    if not checked_dir.exists():
        print(f"エラー: {checked_dir} フォルダが存在しません")
        return
    
    # 出力ディレクトリを作成
    grid_dirs = {}
    no_person_dir = Path("images/no_person")
    no_person_dir.mkdir(parents=True, exist_ok=True)
    defect_dir = Path("images/defect")
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    for image_file in checked_dir.iterdir():
        if not image_file.is_file():
            continue
            
        filename = image_file.name
        
        if filename.startswith("defect_grid_"):
            # defect_grid_で始まる画像はdefectディレクトリに移動（prefixを削除）
            # defect_grid_XX_YYYYmmdd_HHMMSS.ext → YYYYmmdd_HHMMSS.ext
            parts = filename.split("_", 3)  # defect, grid, XX, 残り
            if len(parts) >= 4:
                new_filename = parts[3]  # タイムスタンプ部分
            else:
                new_filename = filename  # フォールバック
            dest_path = defect_dir / new_filename
            shutil.move(str(image_file), str(dest_path))
            print(f"移動: {filename} → defect/{new_filename}")
            moved_count += 1
            
        elif filename.startswith("grid_"):
            # grid_xx形式のprefixを抽出してディレクトリ名に使用、ファイル名からはprefix削除
            # grid_XX_YYYYmmdd_HHMMSS.ext → grid_XX/ に YYYYmmdd_HHMMSS.ext として保存
            parts = filename.split("_", 2)  # grid, XX, 残り
            if len(parts) >= 3:
                grid_prefix = f"grid_{parts[1]}"
                new_filename = parts[2]  # タイムスタンプ部分
                
                # 対応するgrid_xxディレクトリを作成
                if grid_prefix not in grid_dirs:
                    grid_dirs[grid_prefix] = Path(f"images/{grid_prefix}")
                    grid_dirs[grid_prefix].mkdir(parents=True, exist_ok=True)
                
                # ファイルを移動
                dest_path = grid_dirs[grid_prefix] / new_filename
                shutil.move(str(image_file), str(dest_path))
                print(f"移動: {filename} → {grid_prefix}/{new_filename}")
                moved_count += 1
                
        elif filename.startswith("no_person"):
            # no_personディレクトリに移動（prefixを削除）
            # no_person_YYYYmmdd_HHMMSS.ext → YYYYmmdd_HHMMSS.ext
            parts = filename.split("_", 1)  # no_person, 残り
            if len(parts) >= 2:
                new_filename = parts[1]  # タイムスタンプ部分
            else:
                new_filename = filename  # フォールバック
            dest_path = no_person_dir / new_filename
            shutil.move(str(image_file), str(dest_path))
            print(f"移動: {filename} → no_person/{new_filename}")
            moved_count += 1
        elif filename.startswith("error_"):
            # error_で始まる画像は削除
            image_file.unlink()
            print(f"削除: {filename}")
            moved_count += 1
        else:
            print(f"スキップ: {filename} (該当するprefixなし)")
    
    print(f"\n分類完了: {moved_count} 個のファイルを処理しました")

if __name__ == "__main__":
    classify_images()