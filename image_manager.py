#!/usr/bin/env python3
"""
画像管理モジュール

機能:
- 各フォルダの画像数制限（200枚）
- 古い画像の自動削除
- ファイル名の統一（YYYYMMDD_HHMMSS.png）
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class ImageManager:
    """画像管理クラス"""

    def __init__(self, base_dir: str = "images", max_images_per_folder: int = 200):
        self.base_dir = Path(base_dir)
        self.max_images_per_folder = max_images_per_folder
        self.logger = logging.getLogger(__name__)

        # ベースディレクトリを作成
        self.base_dir.mkdir(exist_ok=True)

        # 必要なサブディレクトリを作成
        self._create_directories()

    def _create_directories(self):
        """必要なディレクトリを作成"""
        directories = ["no_person"]

        # grid_00 から grid_15 まで作成
        for i in range(16):
            directories.append(f"grid_{i:02d}")

        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)

    def save_image(self, source_path: str, category: str, timestamp: str = None) -> str:
        """画像を指定カテゴリに保存

        Args:
            source_path: 元画像のパス
            category: 保存先カテゴリ ('no_person', 'grid_00', 'grid_01', ...)
            timestamp: タイムスタンプ文字列（YYYYMMDD_HHMMSS形式）

        Returns:
            str: 保存先のパス
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存先ディレクトリ
        target_dir = self.base_dir / category
        target_dir.mkdir(exist_ok=True)

        # ファイル名を統一形式に
        filename = f"{timestamp}.png"
        target_path = target_dir / filename

        try:
            # 画像をコピー
            shutil.copy2(source_path, target_path)
            self.logger.info(f"画像を保存: {target_path}")

            # フォルダ内の画像数制限チェック
            self._cleanup_old_images(category)

            return str(target_path)

        except Exception as e:
            self.logger.error(f"画像保存エラー: {e}")
            raise

    def _cleanup_old_images(self, category: str):
        """古い画像を削除して制限数以内に保つ"""
        target_dir = self.base_dir / category

        if not target_dir.exists():
            return

        # PNG ファイルのみ対象
        image_files = list(target_dir.glob("*.png"))

        if len(image_files) <= self.max_images_per_folder:
            return

        # ファイル名（タイムスタンプ）でソート
        image_files.sort(key=lambda x: x.stem)

        # 削除する古いファイル数
        files_to_delete = len(image_files) - self.max_images_per_folder

        for i in range(files_to_delete):
            old_file = image_files[i]
            try:
                old_file.unlink()
                self.logger.info(f"古い画像を削除: {old_file}")
            except Exception as e:
                self.logger.error(f"古い画像の削除に失敗: {old_file}, エラー: {e}")

    def get_folder_info(self, category: str = None) -> dict:
        """フォルダの画像数情報を取得

        Args:
            category: 特定のカテゴリ（Noneの場合は全カテゴリ）

        Returns:
            dict: カテゴリ別の画像数情報
        """
        info = {}

        if category:
            categories = [category]
        else:
            # 全カテゴリを取得
            categories = ["no_person"]
            categories.extend([f"grid_{i:02d}" for i in range(16)])

        for cat in categories:
            cat_dir = self.base_dir / cat
            if cat_dir.exists():
                image_count = len(list(cat_dir.glob("*.png")))
                info[cat] = {
                    "count": image_count,
                    "max": self.max_images_per_folder,
                    "remaining": max(0, self.max_images_per_folder - image_count),
                }
            else:
                info[cat] = {
                    "count": 0,
                    "max": self.max_images_per_folder,
                    "remaining": self.max_images_per_folder,
                }

        return info

    def get_training_images(self) -> List[str]:
        """学習用の画像パスリストを取得

        Returns:
            List[str]: 学習用画像のパスリスト
        """
        training_images = []

        # no_person と全グリッドの画像を収集
        categories = ["no_person"]
        categories.extend([f"grid_{i:02d}" for i in range(16)])

        for category in categories:
            cat_dir = self.base_dir / category
            if cat_dir.exists():
                image_files = list(cat_dir.glob("*.png"))
                training_images.extend([str(f) for f in image_files])

        self.logger.info(f"学習用画像数: {len(training_images)}")
        return training_images

    def cleanup_empty_directories(self):
        """空のディレクトリを削除"""
        for item in self.base_dir.iterdir():
            if item.is_dir():
                try:
                    # ディレクトリが空の場合削除を試行
                    item.rmdir()
                    self.logger.info(f"空のディレクトリを削除: {item}")
                except OSError:
                    # ディレクトリが空でない場合は何もしない
                    pass


def main():
    """テスト用のメイン関数"""
    import tempfile
    import cv2
    import numpy as np

    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # 画像マネージャーを初期化
    manager = ImageManager()

    # テスト用画像を作成
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        # ダミー画像作成
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(tmp_file.name, test_image)

        # 画像を保存
        saved_path = manager.save_image(tmp_file.name, "grid_00", "20250131_120000")
        print(f"テスト画像を保存: {saved_path}")

        # ファイル情報を取得
        info = manager.get_folder_info("grid_00")
        print(f"フォルダ情報: {info}")

        # 一時ファイルを削除
        os.unlink(tmp_file.name)


if __name__ == "__main__":
    main()
