#!/usr/bin/env python3
"""
PaDiM 異常検知モデル学習スクリプト

学習データ:
- images/grid_XX (00-15): 正常画像 (人が写っている正常なグリッド別画像)
- images/no_person: 正常画像 (人が写っていない画像)

検証データ:
- images/test: 検証用画像

学習は正常画像のみで行い、異常検知モデルを構築します。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import argparse

import lightning.pytorch as pl
from anomalib.models import Padim
from anomalib.data import Folder


def setup_logging() -> logging.Logger:
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = f"train_padim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def check_data_structure(images_dir: str) -> dict:
    """データセット構造の確認"""
    images_path = Path(images_dir)

    # グリッドディレクトリの確認
    grid_dirs = []
    for i in range(16):  # grid_00 から grid_15
        grid_dir = images_path / f"grid_{i:02d}"
        if grid_dir.exists() and grid_dir.is_dir():
            grid_dirs.append(grid_dir)

    # no_personディレクトリの確認
    no_person_dir = images_path / "no_person"

    # testディレクトリの確認
    test_dir = images_path / "test"

    return {
        "grid_dirs": grid_dirs,
        "no_person_dir": no_person_dir if no_person_dir.exists() else None,
        "test_dir": test_dir if test_dir.exists() else None,
    }


def count_images_in_directory(directory: Path) -> int:
    """ディレクトリ内の画像ファイル数をカウント"""
    if not directory.exists():
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            count += 1

    return count


def get_training_info(images_dir: str) -> tuple:
    """学習用データの情報を取得"""
    logger = logging.getLogger(__name__)

    data_structure = check_data_structure(images_dir)

    # 正常画像数をカウント
    total_normal_images = 0

    # grid_XX ディレクトリから正常画像をカウント
    for grid_dir in data_structure["grid_dirs"]:
        image_count = count_images_in_directory(grid_dir)
        logger.info(f"{grid_dir.name}: {image_count} 画像")
        total_normal_images += image_count

    # no_person ディレクトリから正常画像をカウント
    if data_structure["no_person_dir"]:
        no_person_count = count_images_in_directory(data_structure["no_person_dir"])
        logger.info(f"no_person: {no_person_count} 画像")
        total_normal_images += no_person_count

    # test ディレクトリの画像数をカウント
    test_images = 0
    if data_structure["test_dir"]:
        test_images = count_images_in_directory(data_structure["test_dir"])
        logger.info(f"test: {test_images} 画像")

    logger.info(f"学習用正常画像: {total_normal_images} 枚")
    logger.info(f"検証用画像: {test_images} 枚")

    # 実際のディレクトリパスを返す
    normal_dirs = [str(d) for d in data_structure["grid_dirs"]]
    if data_structure["no_person_dir"]:
        normal_dirs.append(str(data_structure["no_person_dir"]))

    test_dir = str(data_structure["test_dir"]) if data_structure["test_dir"] else None

    return normal_dirs, test_dir, total_normal_images, test_images


def create_padim_model(
    image_size: tuple = (256, 256),
    backbone: str = "resnet18",
    layers: List[str] | None = None,
) -> Padim:
    """PaDiMモデルの作成"""
    if layers is None:
        layers = ["layer1", "layer2", "layer3"]

    pre_processor = Padim.configure_pre_processor(image_size=image_size)

    model = Padim(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        pre_processor=pre_processor,
    )

    return model


def train_padim_model(
    images_dir: str,
    model_save_path: str = "models/padim_model.ckpt",
    image_size: tuple = (256, 256),
    max_epochs: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    """PaDiMモデルの学習"""
    logger = logging.getLogger(__name__)

    # モデル保存ディレクトリを作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("PaDiMモデルの学習を開始します")
    logger.info(f"画像サイズ: {image_size}")
    logger.info(f"最大エポック数: {max_epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"ワーカー数: {num_workers}")

    # データモジュールの準備
    # grid_XX, no_personを正常画像として使用
    datamodule = Folder(
        root=images_dir,
        normal_dir="grid_00",  # まず単一ディレクトリで試す
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        val_split_ratio=0.2,  # 正常画像の20%を検証に使用
    )

    # データモジュールをセットアップ
    datamodule.setup()

    # モデルの準備
    model = create_padim_model(image_size=image_size)

    # Trainerの準備
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir="lightning_logs",
    )

    # 学習実行
    logger.info("学習を開始します...")
    trainer.fit(model=model, datamodule=datamodule)

    # モデル保存
    logger.info(f"モデルを保存します: {model_save_path}")
    trainer.save_checkpoint(model_save_path)

    # 追加で.save()形式でも保存
    save_dir = Path(model_save_path).parent / "padim_saved_model"
    save_dir.mkdir(exist_ok=True)
    model.model.save(str(save_dir))
    logger.info(f"モデル（.save()形式）を保存しました: {save_dir}")

    logger.info("学習が完了しました")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="PaDiM異常検知モデルの学習")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="画像データディレクトリのパス (default: images)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/padim_model.ckpt",
        help="モデル保存パス (default: models/padim_model.ckpt)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="画像サイズ (width height) (default: 256 256)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="最大エポック数 (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="バッチサイズ (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="データローダーのワーカー数 (default: 4)",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="データ構造のチェックのみ実行"
    )

    args = parser.parse_args()

    # ログ設定
    logger = setup_logging()

    try:
        logger.info("PaDiM学習スクリプトを開始します")
        logger.info(f"画像ディレクトリ: {args.images_dir}")

        # データ構造の確認
        data_structure = check_data_structure(args.images_dir)

        logger.info("=== データセット構造 ===")
        logger.info(f"グリッドディレクトリ数: {len(data_structure['grid_dirs'])}")
        for grid_dir in data_structure["grid_dirs"]:
            count = count_images_in_directory(grid_dir)
            logger.info(f"  {grid_dir.name}: {count} 画像")

        if data_structure["no_person_dir"]:
            no_person_count = count_images_in_directory(data_structure["no_person_dir"])
            logger.info(f"no_person: {no_person_count} 画像")
        else:
            logger.warning("no_personディレクトリが見つかりません")

        if data_structure["test_dir"]:
            test_count = count_images_in_directory(data_structure["test_dir"])
            logger.info(f"test: {test_count} 画像")
        else:
            logger.warning("testディレクトリが見つかりません")

        # データ構造チェックのみの場合は終了
        if args.check_only:
            logger.info("データ構造チェックが完了しました")
            return 0

        # 学習データが存在するかチェック
        total_grid_images = sum(
            count_images_in_directory(grid_dir)
            for grid_dir in data_structure["grid_dirs"]
        )
        no_person_images = (
            count_images_in_directory(data_structure["no_person_dir"])
            if data_structure["no_person_dir"]
            else 0
        )
        total_normal_images = total_grid_images + no_person_images

        if total_normal_images == 0:
            logger.error("学習用の正常画像が見つかりません")
            return 1

        logger.info(f"合計正常画像数: {total_normal_images}")

        # 学習用データの情報を取得
        normal_dirs, test_dir, normal_count, test_count = get_training_info(
            args.images_dir
        )

        if normal_count == 0:
            logger.error("正常画像が見つかりません")
            return 1

        # モデル学習の実行
        train_padim_model(
            images_dir=args.images_dir,
            model_save_path=args.model_path,
            image_size=tuple(args.image_size),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        logger.info("すべての処理が完了しました")
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
