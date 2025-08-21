#!/usr/bin/env python3
"""
EfficientAd 異常検知モデル学習スクリプト

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
import argparse
import torch


from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import shutil

# Jetson Orin GPU向けの最適化設定
torch.set_float32_matmul_precision("high")


def create_unified_training_dir(
    images_dir: str,
    training_dir: str = "temp_training_data_256x256",
    image_size: tuple = (256, 256),
) -> tuple:
    """全ての正常画像を統合した一時的な学習ディレクトリを作成（シェルスクリプト使用）"""
    import subprocess

    logger = logging.getLogger(__name__)

    training_path = Path(training_dir)
    normal_dir = training_path / "normal"

    # 既存ディレクトリがある場合、リサイズ済みかどうかを確認
    if normal_dir.exists():
        existing_images = list(normal_dir.glob("*"))
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
        logger.info("新規でtemp_training_data_256x256ディレクトリを作成")

    # シェルスクリプトで高速リサイズ&コピーを実行
    script_path = Path(__file__).parent / "copy_training_images_resized.sh"
    target_size = f"{image_size[0]}x{image_size[1]}"  # 256x256形式
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

    logger.info(f"統合された学習用正常画像: {total_images} 枚")
    return str(training_path), total_images


def cleanup_training_dir(training_dir: str):
    """学習用一時ディレクトリを削除"""
    logger = logging.getLogger(__name__)

    training_path = Path(training_dir)
    if training_path.exists():
        shutil.rmtree(training_path)
        logger.info(f"一時学習ディレクトリを削除しました: {training_dir}")


def setup_logging() -> logging.Logger:
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = f"train_efficient_ad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def create_efficientad_model(
) -> EfficientAd:
    """EfficientAdモデルの作成"""
    model = EfficientAd()
    model.prepare_pretrained_model()
    return model


def create_test_datamodule(
    images_dir: str, batch_size: int = 1
) -> Folder:
    """test用のdatamoduleを作成"""
    logger = logging.getLogger(__name__)

    images_path = Path(images_dir)
    test_dir = images_path / "test"

    if not test_dir.exists():
        logger.warning("testディレクトリが見つかりません")
        return None

    # test用datamoduleを作成
    test_datamodule = Folder(
        name="efficientad_test",
        root=str(test_dir),
        normal_dir="normal",
        abnormal_dir="anomaly",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        val_split_ratio=0.0,  # testデータなので分割しない
        test_split_ratio=1.0,  # 全てをtestデータとして使用
    )

    logger.info(f"test用datamoduleを作成: {test_dir}")
    return test_datamodule


def train_efficientad_model(
    images_dir: str,
    model_save_path: str = "models/efficientad_trained.ckpt",
    image_size: tuple = (256, 256),  # EfficientAd推奨サイズ
    max_epochs: int = 10,
    batch_size: int = 1,  # EfficientAd最適化
    num_workers: int = 2,  # Jetson向けに削減
) -> None:
    """EfficientAdモデルの学習"""
    logger = logging.getLogger(__name__)

    # モデル保存ディレクトリを作成
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("EfficientAdモデルの学習を開始します")
    logger.info("元画像サイズ: 640x480 (カメラ解像度)")
    logger.info(f"リサイズ後サイズ: {image_size} (EfficientAd推奨サイズ)")
    logger.info(f"最大エポック数: {max_epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"ワーカー数: {num_workers}")

    # 統合学習ディレクトリを作成または再利用
    training_dir = "temp_training_data_256x256"

    try:
        # 全画像を統合した一時ディレクトリを作成（既存の場合は再利用）
        training_root, total_images = create_unified_training_dir(
            images_dir, training_dir, image_size
        )

        if total_images == 0:
            logger.error("学習に使用できる正常画像が見つかりません")
            logger.info("以下のディレクトリに画像を配置してください:")
            logger.info("  - images/grid_00 〜 images/grid_15 (人が写っている正常画像)")
            logger.info("  - images/no_person (人が写っていない正常画像)")
            cleanup_training_dir(training_dir)
            return

        # Folderデータモジュールを使用（学習は正常画像のみ）
        datamodule = Folder(
            name="efficientad_training",
            root=training_root,
            normal_dir="normal",
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            val_split_ratio=0.2,
            test_split_ratio=0.0,  # 学習用なのでtestは使用しない
        )
        logger.info(f"Folderデータモジュールを作成しました (num_workers={num_workers})")

        # 実際のファイル数を先に確認
        actual_files = len(
            [f for f in (Path(training_root) / "normal").iterdir() if f.is_file()]
        )
        logger.info(f"temp_training_data_256x256/normal内の実際のファイル数: {actual_files}")

        # データモジュールをセットアップ
        logger.info("データモジュールをセットアップ中...")
        datamodule.setup()
        logger.info("データモジュールのセットアップが完了しました")

    except Exception as e:
        logger.error(f"データセットの準備に失敗: {e}")
        raise

    # モデルの準備（画像サイズを明示的に指定）
    logger.info(f"EfficientAdモデルを作成中（画像サイズ: {image_size}）")
    model = create_efficientad_model()
    engine = Engine(max_epochs=max_epochs)

    # 学習実行
    logger.info("=" * 50)
    logger.info("EfficientAdモデル学習開始")
    logger.info("=" * 50)
    logger.info(f"学習開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("モデル: EfficientAd")
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

    # test実行
    logger.info("=" * 30)
    logger.info("テスト開始")
    logger.info("=" * 30)

    try:
        # test用datamoduleを作成
        test_datamodule = create_test_datamodule(images_dir, batch_size)

        if test_datamodule is not None:
            # testデータをセットアップ
            test_datamodule.setup()

            # testを実行
            logger.info("テスト実行中...")
            model = EfficientAd.load_from_checkpoint(model_save_path)
            test_results = engine.test(model=model, datamodule=test_datamodule)

            logger.info("=" * 30)
            logger.info("テスト完了")
            logger.info(f"テスト結果: {test_results}")
            logger.info("=" * 30)
        else:
            logger.warning(
                "test用datamoduleの作成に失敗したため、テストをスキップします"
            )

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        logger.warning("テストに失敗しましたが、学習は正常に完了しています")

    logger.info("🎉 EfficientAdモデル学習が正常に完了しました 🎉")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="EfficientAd異常検知モデルの学習")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="画像データディレクトリのパス (default: images)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/efficientad_trained.ckpt",
        help="モデル保存パス (default: models/efficientad_trained.ckpt)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="リサイズ後の画像サイズ (width height) (default: 256 256)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="最大エポック数 (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="バッチサイズ (default: 1 - EfficientAd最適化)",
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
        help="学習後にtemp_training_data_256x256ディレクトリを削除",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="既存のtemp_training_data_256x256ディレクトリを強制的に再作成",
    )

    args = parser.parse_args()

    # ログ設定
    logger = setup_logging()

    try:
        logger.info("EfficientAd学習スクリプトを開始します")
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
        if args.force_recreate and Path("temp_training_data_256x256").exists():
            cleanup_training_dir("temp_training_data_256x256")
            logger.info(
                "--force-recreateオプションにより、既存のtemp_training_data_256x256ディレクトリを削除しました"
            )

        # モデル学習の実行
        train_efficientad_model(
            images_dir=args.images_dir,
            model_save_path=args.model_path,
            image_size=tuple(args.image_size),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # --cleanupオプションが指定された場合のみディレクトリを削除
        if args.cleanup:
            cleanup_training_dir("temp_training_data_256x256")
            logger.info(
                "--cleanupオプションにより、temp_training_data_256x256ディレクトリを削除しました"
            )

        logger.info("すべての処理が完了しました")
        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
