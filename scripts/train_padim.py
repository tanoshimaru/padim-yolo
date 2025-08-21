#!/usr/bin/env python3
"""
PaDiM 異常検知モデル学習スクリプト

学習データ:
- images/grid_XX (00-15): 正常画像 (人が写っている正常なグリッド別画像)
- images/no_person: 正常画像 (人が写っていない画像)

検証データ:
- images/test/normal: 正常な検証画像
- images/test/anomaly: 異常な検証画像

学習は正常画像のみで行い、異常検知モデルを構築します。
testディレクトリ内のnormal/anomalyサブディレクトリは推論評価時に使用されます。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import argparse
import torch


from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim
import shutil

# Jetson Orin GPU向けの最適化設定
torch.set_float32_matmul_precision("high")


def create_unified_training_dir(
    images_dir: str,
    training_dir: str = "dataset",
    image_size: tuple = (224, 224),
) -> tuple:
    """全ての正常画像を統合した学習ディレクトリを作成（シェルスクリプト使用）"""
    import subprocess

    logger = logging.getLogger(__name__)

    training_path = Path(training_dir)
    train_good_dir = training_path / "train" / "good"
    test_good_dir = training_path / "test" / "good"
    test_defect_dir = training_path / "test" / "defect"

    # 既存ディレクトリがある場合、リサイズ済みかどうかを確認
    if train_good_dir.exists():
        existing_images = list(train_good_dir.glob("*"))
        if test_good_dir.exists():
            existing_images.extend(list(test_good_dir.glob("*")))
        if test_defect_dir.exists():
            existing_images.extend(list(test_defect_dir.glob("*")))
        
        existing_image_files = [
            f
            for f in existing_images
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        ]

        # 既存画像のサイズをチェック（最初の画像で判定）
        if existing_image_files:
            from PIL import Image

            try:
                sample_image = Image.open(existing_image_files[0])
                current_size = sample_image.size
                target_size = (image_size[0], image_size[1])

                if current_size == target_size and len(existing_image_files) >= 10:
                    logger.info(
                        f"既存の画像は既に{target_size}にリサイズ済みです: {len(existing_image_files)} 画像"
                    )
                    return str(training_path), len(existing_image_files)
                else:
                    logger.info(
                        f"既存画像サイズ: {current_size} → ターゲット: {target_size} - 再リサイズが必要"
                    )
                    # ファイルを削除して再作成
                    for file in existing_image_files:
                        file.unlink()
            except Exception as e:
                logger.warning(f"既存画像のサイズ確認エラー: {e} - 再作成します")
                # エラー時は既存ファイルを削除
                for file in existing_image_files:
                    file.unlink()
        else:
            logger.info("既存ディレクトリは空です - 新規作成")
    else:
        logger.info("新規でdatasetディレクトリを作成")

    # シェルスクリプトで高速リサイズ&コピーを実行
    script_path = Path(__file__).parent / "create_dataset_structure.sh"
    target_size = f"{image_size[0]}x{image_size[1]}"  # 224x224形式
    try:
        result = subprocess.run(
            [str(script_path), images_dir, training_dir, target_size],
            capture_output=True,
            text=True,
            check=True,
        )

        # 最後の行から画像数を取得
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines[:-1]:  # 最後の行以外を出力
            logger.info(line)

        total_images = int(output_lines[-1]) if output_lines else 0

    except subprocess.CalledProcessError as e:
        logger.error(f"シェルスクリプト実行エラー: {e.stderr}")
        return str(training_path), 0
    except (ValueError, IndexError):
        logger.error("シェルスクリプトからの出力解析エラー")
        return str(training_path), 0

    logger.info(f"データセット作成完了: {total_images} 枚")
    return str(training_path), total_images


def cleanup_training_dir(training_dir: str):
    """学習用データセットディレクトリを削除"""
    logger = logging.getLogger(__name__)

    training_path = Path(training_dir)
    if training_path.exists():
        shutil.rmtree(training_path)
        logger.info(f"データセットディレクトリを削除しました: {training_dir}")


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
    for i in range(8):  # grid_00 から grid_15
        grid_dir = images_path / f"grid_{i:02d}"
        if grid_dir.exists() and grid_dir.is_dir():
            grid_dirs.append(grid_dir)

    # no_personディレクトリの確認
    no_person_dir = images_path / "no_person"

    # testディレクトリの確認（normal/anomalyサブディレクトリ含む）
    test_dir = images_path / "test"
    test_normal_dir = test_dir / "normal" if test_dir.exists() else None
    test_anomaly_dir = test_dir / "anomaly" if test_dir.exists() else None

    return {
        "grid_dirs": grid_dirs,
        "no_person_dir": no_person_dir if no_person_dir.exists() else None,
        "test_dir": test_dir if test_dir.exists() else None,
        "test_normal_dir": test_normal_dir
        if test_normal_dir and test_normal_dir.exists()
        else None,
        "test_anomaly_dir": test_anomaly_dir
        if test_anomaly_dir and test_anomaly_dir.exists()
        else None,
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
    image_size: tuple = (224, 224),  # ResNet標準サイズ（最適な処理効率）
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
        n_features=None,  # 自動で特徴量数を決定
    )

    return model




def train_padim_model(
    images_dir: str,
    model_save_path: str = "models/padim_trained.ckpt",
    image_size: tuple = (224, 224),  # ResNet標準サイズ（最適な処理効率）
    max_epochs: int = 10,
    batch_size: int = 4,  # Jetson向けに削減
    num_workers: int = 2,  # Jetson向けに削減
) -> None:
    """PaDiMモデルの学習"""
    logger = logging.getLogger(__name__)

    # モデル保存ディレクトリを作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("PaDiMモデルの学習を開始します")
    logger.info("元画像サイズ: 640x480 (カメラ解像度)")
    logger.info(f"リサイズ後サイズ: {image_size} (処理効率のため)")
    logger.info(f"最大エポック数: {max_epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"ワーカー数: {num_workers}")

    # データセットディレクトリを作成または再利用
    training_dir = "dataset"

    try:
        # データセットを作成（既存の場合は再利用）
        training_root, total_images = create_unified_training_dir(
            images_dir, training_dir, image_size
        )

        if total_images == 0:
            logger.error("学習に使用できる画像が見つかりません")
            logger.info("以下のディレクトリに画像を配置してください:")
            logger.info("  - images/grid_00 〜 images/grid_15 (人が写っている正常画像)")
            logger.info("  - images/no_person (人が写っていない正常画像)")
            logger.info("  - images/test/normal (正常なテスト画像)")
            logger.info("  - images/test/anomaly (異常なテスト画像)")
            cleanup_training_dir(training_dir)
            return

        # Folderデータモジュールを使用（dataset/train/good、dataset/test/good、dataset/test/defect）
        datamodule = Folder(
            name="padim_training",
            root=training_root,
            normal_dir="train/good",
            abnormal_dir="test/defect",
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            test_split_mode="from_dir",
            test_split_ratio=0.0,
            val_split_mode="from_dir", 
            val_split_ratio=0.0,
        )
        logger.info(f"Folderデータモジュールを作成しました (num_workers={num_workers})")

        # 実際のファイル数を先に確認
        train_files = len(
            [f for f in (Path(training_root) / "train" / "good").iterdir() if f.is_file()]
        )
        test_good_files = len(
            [f for f in (Path(training_root) / "test" / "good").iterdir() if f.is_file()]
        ) if (Path(training_root) / "test" / "good").exists() else 0
        test_defect_files = len(
            [f for f in (Path(training_root) / "test" / "defect").iterdir() if f.is_file()]
        ) if (Path(training_root) / "test" / "defect").exists() else 0
        
        logger.info(f"dataset/train/good内の実際のファイル数: {train_files}")
        logger.info(f"dataset/test/good内の実際のファイル数: {test_good_files}")
        logger.info(f"dataset/test/defect内の実際のファイル数: {test_defect_files}")

        # データモジュールをセットアップ
        logger.info("データモジュールをセットアップ中...")
        datamodule.setup()
        logger.info("データモジュールのセットアップが完了しました")

    except Exception as e:
        logger.error(f"データセットの準備に失敗: {e}")
        raise

    # モデルの準備（画像サイズを明示的に指定）
    logger.info(f"PaDiMモデルを作成中（画像サイズ: {image_size}）")
    model = create_padim_model(image_size=image_size)
    engine = Engine(max_epochs=max_epochs)

    # 学習実行
    logger.info("=" * 50)
    logger.info("PaDiMモデル学習開始")
    logger.info("=" * 50)
    logger.info(f"学習開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("モデルバックボーン: resnet18")
    # logger.info(
    #     f"使用デバイス: {engine.trainer.device_ids if hasattr(engine.trainer, 'device_ids') else 'auto'}"
    # )
    # logger.info(f"アクセラレータ: {engine.trainer.accelerator}")
    logger.info("特徴抽出レイヤー: ['layer1', 'layer2', 'layer3']")
    logger.info("=" * 50)

    try:
        engine.fit(model=model, datamodule=datamodule)
        logger.info("=" * 50)
        logger.info("学習が正常に完了しました")
        logger.info(f"学習完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error("=" * 50)
        logger.error("学習中にエラーが発生しました")
        logger.error(f"エラー詳細: {e}")
        logger.error("=" * 50)
        raise

    # モデル保存
    logger.info("=" * 30)
    logger.info("モデル保存開始")
    logger.info("=" * 30)
    logger.info(f"保存パス: {model_save_path}")

    try:
        engine.trainer.save_checkpoint(model_save_path)
        model_size = Path(model_save_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"チェックポイント保存完了: {model_size:.2f} MB")
        logger.info("=" * 30)
        logger.info("モデル保存完了")
        logger.info("=" * 30)

    except Exception as e:
        logger.error("=" * 30)
        logger.error("モデル保存エラー")
        logger.error(f"エラー詳細: {e}")
        logger.error("=" * 30)
        raise

    # テストは既存のdatamoduleで実行
    logger.info("=" * 30)
    logger.info("テスト開始")
    logger.info("=" * 30)

    try:
        # testを実行
        logger.info("テスト実行中...")
        test_results = engine.test(model=model, datamodule=datamodule)

        logger.info("=" * 30)
        logger.info("テスト完了")
        logger.info(f"テスト結果: {test_results}")
        logger.info("=" * 30)

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        logger.warning("テストに失敗しましたが、学習は正常に完了しています")

    logger.info("🎉 PaDiMモデル学習が正常に完了しました 🎉")


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
        default="models/padim_trained.ckpt",
        help="モデル保存パス (default: models/padim_trained.ckpt)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="リサイズ後の画像サイズ (width height) (default: 224 224)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="最大エポック数 (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="バッチサイズ (default: 4 - Jetson最適化)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="データローダーのワーカー数 (default: 2 - Jetson最適化)",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="データ構造のチェックのみ実行"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="学習後にdatasetディレクトリを削除",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="既存のdatasetディレクトリを強制的に再作成",
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
            # testディレクトリ内のサブディレクトリごとに画像数を表示
            test_normal_count = 0
            test_anomaly_count = 0

            if data_structure["test_normal_dir"]:
                test_normal_count = count_images_in_directory(
                    data_structure["test_normal_dir"]
                )
                logger.info(f"test/normal: {test_normal_count} 画像")

            if data_structure["test_anomaly_dir"]:
                test_anomaly_count = count_images_in_directory(
                    data_structure["test_anomaly_dir"]
                )
                logger.info(f"test/anomaly: {test_anomaly_count} 画像")

            # 全体のtest画像数を計算（サブディレクトリの合計）
            test_count = test_normal_count + test_anomaly_count
            logger.info(f"test (合計): {test_count} 画像")
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

        # 学習用データの情報を取得（ログ重複回避のため簡略化）
        normal_count = total_normal_images

        if normal_count == 0:
            logger.error("正常画像が見つかりません")
            return 1

        # --force-recreateオプションが指定された場合は既存ディレクトリを削除
        if args.force_recreate and Path("dataset").exists():
            cleanup_training_dir("dataset")
            logger.info(
                "--force-recreateオプションにより、既存のdatasetディレクトリを削除しました"
            )

        # モデル学習の実行
        train_padim_model(
            images_dir=args.images_dir,
            model_save_path=args.model_path,
            image_size=tuple(args.image_size),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # --cleanupオプションが指定された場合のみディレクトリを削除
        if args.cleanup:
            cleanup_training_dir("dataset")
            logger.info(
                "--cleanupオプションにより、datasetディレクトリを削除しました"
            )

        logger.info("すべての処理が完了しました")
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
