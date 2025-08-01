#!/usr/bin/env python3
"""
PaDiM学習用データ準備スクリプト

images/ ディレクトリ内の画像を training_data/ に分散配置し、
PaDiMの学習データセットを準備します。
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import random


def setup_logging():
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = (
        f"prepare_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


class TrainingDataPreparer:
    """学習データ準備クラス"""

    def __init__(
        self,
        source_dir: str = "./images",
        target_dir: str = "./training_data",
        normal_ratio: float = 0.8,
        copy_mode: bool = True,
        random_seed: int = 42,
    ):
        self.logger = setup_logging()
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.normal_ratio = normal_ratio
        self.copy_mode = copy_mode  # True: コピー, False: 移動
        self.random_seed = random_seed

        # ランダムシード設定
        random.seed(random_seed)

    def collect_images(self) -> Tuple[List[Path], List[Path]]:
        """画像ファイルを収集"""
        normal_images = []
        no_person_images = []

        try:
            if not self.source_dir.exists():
                self.logger.error(
                    f"ソースディレクトリが見つかりません: {self.source_dir}"
                )
                return [], []

            # grid_XX ディレクトリから正常画像を収集
            for grid_dir in self.source_dir.glob("grid_*"):
                if grid_dir.is_dir():
                    images = list(grid_dir.glob("*.png")) + list(grid_dir.glob("*.jpg"))
                    normal_images.extend(images)
                    self.logger.info(f"{grid_dir.name}: {len(images)} 枚の画像を発見")

            # no_person ディレクトリから画像を収集
            no_person_dir = self.source_dir / "no_person"
            if no_person_dir.exists() and no_person_dir.is_dir():
                images = list(no_person_dir.glob("*.png")) + list(
                    no_person_dir.glob("*.jpg")
                )
                no_person_images.extend(images)
                self.logger.info(f"no_person: {len(images)} 枚の画像を発見")

            self.logger.info(f"総正常画像数: {len(normal_images)}")
            self.logger.info(f"総no_person画像数: {len(no_person_images)}")

            return normal_images, no_person_images

        except Exception as e:
            self.logger.error(f"画像収集でエラー: {e}")
            return [], []

    def split_images(
        self, images: List[Path], ratio: float
    ) -> Tuple[List[Path], List[Path]]:
        """画像を指定比率で分割"""
        if not images:
            return [], []

        # シャッフル
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        # 分割
        split_point = int(len(shuffled_images) * ratio)
        first_part = shuffled_images[:split_point]
        second_part = shuffled_images[split_point:]

        return first_part, second_part

    def copy_or_move_files(self, source_files: List[Path], target_dir: Path) -> int:
        """ファイルをコピーまたは移動"""
        target_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0

        for i, source_file in enumerate(source_files, 1):
            try:
                target_file = (
                    target_dir / f"{source_file.stem}_{i:04d}{source_file.suffix}"
                )

                if self.copy_mode:
                    shutil.copy2(source_file, target_file)
                else:
                    shutil.move(str(source_file), target_file)

                success_count += 1

            except Exception as e:
                self.logger.error(f"ファイル操作エラー ({source_file}): {e}")

        return success_count

    def prepare_training_data(self) -> bool:
        """学習データを準備"""
        try:
            self.logger.info("=== 学習データ準備開始 ===")

            # 画像収集
            normal_images, no_person_images = self.collect_images()

            if not normal_images and not no_person_images:
                self.logger.error("学習用画像が見つかりません")
                return False

            # 正常画像を学習用と検証用に分割
            if normal_images:
                train_normal, val_normal = self.split_images(
                    normal_images, self.normal_ratio
                )
                self.logger.info(
                    f"正常画像分割: 学習用 {len(train_normal)} 枚, 検証用 {len(val_normal)} 枚"
                )
            else:
                train_normal, val_normal = [], []

            # no_person画像を学習用と検証用に分割
            if no_person_images:
                train_no_person, val_no_person = self.split_images(
                    no_person_images, self.normal_ratio
                )
                self.logger.info(
                    f"no_person画像分割: 学習用 {len(train_no_person)} 枚, 検証用 {len(val_no_person)} 枚"
                )
            else:
                train_no_person, val_no_person = [], []

            # 正常画像（学習用）
            normal_dir = self.target_dir / "normal"
            total_train_normal = train_normal + train_no_person
            if total_train_normal:
                count = self.copy_or_move_files(total_train_normal, normal_dir)
                self.logger.info(f"正常画像（学習用）: {count} 枚を配置")

            # 異常画像（検証用）- 必要に応じて
            # 今回は検証用画像も正常として扱う
            abnormal_dir = self.target_dir / "abnormal"
            total_val = val_normal + val_no_person
            if total_val:
                count = self.copy_or_move_files(total_val, abnormal_dir)
                self.logger.info(f"検証用画像: {count} 枚を配置")

            # データセット情報ファイル作成
            self.create_dataset_info(len(total_train_normal), len(total_val))

            self.logger.info("=== 学習データ準備完了 ===")
            return True

        except Exception as e:
            self.logger.error(f"学習データ準備でエラー: {e}")
            return False

    def create_dataset_info(self, train_count: int, val_count: int):
        """データセット情報ファイルを作成"""
        info_content = f"""# PaDiM学習データセット情報

## 作成日時
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## データ統計
- 学習用正常画像: {train_count} 枚
- 検証用画像: {val_count} 枚
- 総画像数: {train_count + val_count} 枚

## ソース設定
- ソースディレクトリ: {self.source_dir}
- ターゲットディレクトリ: {self.target_dir}
- 正常画像比率: {self.normal_ratio}
- 操作モード: {"コピー" if self.copy_mode else "移動"}
- ランダムシード: {self.random_seed}

## 学習開始コマンド
```bash
python train_padim.py --data_root {self.target_dir}
```
"""

        info_path = self.target_dir / "dataset_info.md"
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(info_content)

        self.logger.info(f"データセット情報を作成: {info_path}")

    def clean_target_directory(self):
        """ターゲットディレクトリをクリーン"""
        if self.target_dir.exists():
            self.logger.info(f"既存のターゲットディレクトリを削除: {self.target_dir}")
            shutil.rmtree(self.target_dir)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="PaDiM学習用データセット準備")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./images",
        help="ソース画像ディレクトリ (default: ./images)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./training_data",
        help="学習データ出力ディレクトリ (default: ./training_data)",
    )
    parser.add_argument(
        "--normal_ratio",
        type=float,
        default=0.8,
        help="学習用データの比率 (default: 0.8)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="ファイルを移動（デフォルトはコピー）",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="ランダムシード (default: 42)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ターゲットディレクトリを事前にクリーン",
    )

    args = parser.parse_args()

    try:
        preparer = TrainingDataPreparer(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            normal_ratio=args.normal_ratio,
            copy_mode=not args.move,
            random_seed=args.random_seed,
        )

        if args.clean:
            preparer.clean_target_directory()

        success = preparer.prepare_training_data()

        if success:
            print(f"\n学習データの準備が完了しました: {args.target_dir}")
            print(f"\n次のコマンドで学習を開始できます:")
            print(f"python train_padim.py --data_root {args.target_dir}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n処理が中断されました")
        return 1
    except Exception as e:
        print(f"エラーが発生: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
