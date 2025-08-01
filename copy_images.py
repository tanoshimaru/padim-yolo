import os
import shutil
import random
from pathlib import Path


def copy_random_images(src_dir, dst_dir, images_per_folder=200):
    """
    各フォルダから指定枚数の画像をランダムにコピーする
    
    Args:
        src_dir (str): ソースディレクトリのパス
        dst_dir (str): 出力先ディレクトリのパス
        images_per_folder (int): 各フォルダからコピーする画像数
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 出力先ディレクトリが存在しない場合は作成
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルの拡張子
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # ソースディレクトリ内の各フォルダを処理
    for folder in src_path.iterdir():
        if not folder.is_dir():
            continue
            
        print(f"処理中のフォルダ: {folder.name}")
        
        # フォルダ内の画像ファイルを取得
        image_files = []
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        print(f"  - 見つかった画像ファイル数: {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"  - スキップ: 画像ファイルがありません")
            continue
        
        # 出力先フォルダを作成
        dst_folder = dst_path / folder.name
        dst_folder.mkdir(exist_ok=True)
        
        # ランダムに選択する枚数を決定
        copy_count = min(images_per_folder, len(image_files))
        selected_images = random.sample(image_files, copy_count)
        
        print(f"  - コピーする画像数: {copy_count}")
        
        # 選択された画像をコピー
        copied_count = 0
        for img_file in selected_images:
            try:
                dst_file = dst_folder / img_file.name
                shutil.copy2(img_file, dst_file)
                copied_count += 1
            except Exception as e:
                print(f"  - エラー: {img_file.name} のコピーに失敗しました: {e}")
        
        print(f"  - 完了: {copied_count}枚の画像をコピーしました\n")


def main():
    # パスを設定
    source_dir = "/source/path"  # ここを実際のソースディレクトリに変更
    destination_dir = "/destination/path"  # ここを実際のディスティネーションディレクトリに変更
    images_per_folder = 200
    
    print(f"ソースディレクトリ: {source_dir}")
    print(f"出力先ディレクトリ: {destination_dir}")
    print(f"各フォルダからコピーする画像数: {images_per_folder}\n")
    
    # ソースディレクトリの確認
    if not os.path.exists(source_dir):
        print(f"エラー: ソースディレクトリが存在しません: {source_dir}")
        return
    
    try:
        copy_random_images(source_dir, destination_dir, images_per_folder)
        print("すべての処理が完了しました！")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()