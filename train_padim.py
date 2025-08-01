#!/usr/bin/env python3
"""
PaDiM 異常検知モデルの学習スクリプト

このスクリプトは以下の処理を行います：
1. 学習データセットの準備
2. PaDiMモデルの初期化
3. モデルの学習
4. 学習済みモデルの保存
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from anomalib.models import Padim
    from anomalib.engine import Engine
    from anomalib.data import Folder
    from anomalib.metrics import AUROC, F1Score
    from anomalib.callbacks import ModelCheckpoint
except ImportError as e:
    print(f"必要なライブラリのインポートに失敗しました: {e}")
    print("pip install anomalib で anomalib をインストールしてください")
    sys.exit(1)


def setup_logging():
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


class PaDiMTrainer:
    """PaDiM学習クラス"""

    def __init__(
        self,
        data_root: str = "./images/train",
        normal_dir: str = "normal",
        abnormal_dir: str = "abnormal",
        model_save_path: str = "./models/padim_model.ckpt",
        image_size: tuple = (256, 256),
        batch_size: int = 32,
        max_epochs: int = 1,
        backbone: str = "resnet18",
        layers: list = None,
    ):
        self.logger = setup_logging()
        self.data_root = Path(data_root)
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.model_save_path = Path(model_save_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.backbone = backbone
        self.layers = layers or ["layer1", "layer2", "layer3"]

        # モデル保存ディレクトリを作成
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self) -> bool:
        """データセットの準備と検証"""
        try:
            # データセットディレクトリの存在確認
            if not self.data_root.exists():
                self.logger.error(f"データセットディレクトリが見つかりません: {self.data_root}")
                return False

            normal_path = self.data_root / self.normal_dir
            abnormal_path = self.data_root / self.abnormal_dir

            if not normal_path.exists():
                self.logger.error(f"正常画像ディレクトリが見つかりません: {normal_path}")
                return False

            if not abnormal_path.exists():
                self.logger.warning(f"異常画像ディレクトリが見つかりません: {abnormal_path}")
                self.logger.warning("異常画像なしで学習を続行します")

            # 画像ファイル数をカウント
            normal_images = list(normal_path.glob("*.png")) + list(normal_path.glob("*.jpg"))
            abnormal_images = []
            if abnormal_path.exists():
                abnormal_images = list(abnormal_path.glob("*.png")) + list(abnormal_path.glob("*.jpg"))

            self.logger.info(f"正常画像数: {len(normal_images)}")
            self.logger.info(f"異常画像数: {len(abnormal_images)}")

            if len(normal_images) == 0:
                self.logger.error("正常画像が見つかりません")
                return False

            return True

        except Exception as e:
            self.logger.error(f"データセット準備でエラー: {e}")
            return False

    def create_datamodule(self) -> Optional[Folder]:
        """データモジュールの作成"""
        try:
            # abnormal_dirが存在しない場合はNoneに設定
            abnormal_dir = self.abnormal_dir
            abnormal_path = self.data_root / self.abnormal_dir
            if not abnormal_path.exists() or len(list(abnormal_path.glob("*.png")) + list(abnormal_path.glob("*.jpg"))) == 0:
                abnormal_dir = None
                self.logger.info("異常画像がないため、正常画像のみで学習します")

            datamodule = Folder(
                name="padim_training",
                root=str(self.data_root),
                normal_dir=self.normal_dir,
                abnormal_dir=abnormal_dir,
                task="classification",
                image_size=self.image_size,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                num_workers=4,
                val_split_ratio=0.2,
                test_split_ratio=0.2,
            )

            self.logger.info("データモジュールを作成しました")
            return datamodule

        except Exception as e:
            self.logger.error(f"データモジュール作成でエラー: {e}")
            return None

    def create_model(self) -> Optional[Padim]:
        """PaDiMモデルの作成"""
        try:
            # プリプロセッサーの設定
            pre_processor = Padim.configure_pre_processor(image_size=self.image_size)

            model = Padim(
                backbone=self.backbone,
                layers=self.layers,
                pre_trained=True,
                pre_processor=pre_processor,
            )

            self.logger.info(f"PaDiMモデルを作成しました (backbone: {self.backbone}, layers: {self.layers})")
            return model

        except Exception as e:
            self.logger.error(f"モデル作成でエラー: {e}")
            return None

    def create_engine(self) -> Optional[Engine]:
        """学習エンジンの作成"""
        try:
            # チェックポイントコールバックの設定
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(self.model_save_path.parent),
                filename="padim_best",
                monitor="val_AUROC",
                mode="max",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )

            engine = Engine(
                task="classification",
                image_size=self.image_size,
                max_epochs=self.max_epochs,
                accelerator="auto",
                devices="auto",
                logger=True,
                image_metrics=["AUROC", "F1Score"],
                pixel_metrics=["AUROC"],
                callbacks=[checkpoint_callback],
            )

            self.logger.info(f"学習エンジンを作成しました (max_epochs: {self.max_epochs})")
            return engine

        except Exception as e:
            self.logger.error(f"エンジン作成でエラー: {e}")
            return None

    def train(self) -> bool:
        """学習の実行"""
        try:
            self.logger.info("=== PaDiM学習開始 ===")

            # データセット準備
            if not self.prepare_dataset():
                return False

            # データモジュール作成
            datamodule = self.create_datamodule()
            if datamodule is None:
                return False

            # モデル作成
            model = self.create_model()
            if model is None:
                return False

            # エンジン作成
            engine = self.create_engine()
            if engine is None:
                return False

            # 学習実行
            self.logger.info("学習を開始します...")
            engine.fit(model=model, datamodule=datamodule)

            # 最終的なモデル保存
            final_model_path = self.model_save_path
            model.save(final_model_path)
            self.logger.info(f"学習済みモデルを保存しました: {final_model_path}")

            # テスト実行（テストデータがある場合）
            try:
                self.logger.info("テストを実行します...")
                test_results = engine.test(model=model, datamodule=datamodule)
                self.logger.info("テスト完了")
                if test_results:
                    for metric, value in test_results.items():
                        self.logger.info(f"Test {metric}: {value}")
            except Exception as e:
                self.logger.warning(f"テスト実行でエラー（学習は成功）: {e}")

            self.logger.info("=== PaDiM学習完了 ===")
            return True

        except Exception as e:
            self.logger.error(f"学習でエラー: {e}")
            return False

    def create_sample_dataset_structure(self):
        """サンプルデータセット構造の作成"""
        self.logger.info("サンプルデータセット構造を作成します...")
        
        # ディレクトリ作成
        normal_path = self.data_root / self.normal_dir
        abnormal_path = self.data_root / self.abnormal_dir
        
        normal_path.mkdir(parents=True, exist_ok=True)
        abnormal_path.mkdir(parents=True, exist_ok=True)
        
        # README作成
        readme_content = """# PaDiM学習用データセット


## ディレクトリ構造
```
images/
└── train/      # 学習用データ
    ├── normal/     # 正常画像（学習用）
    │   ├── normal_001.png
    │   ├── normal_002.png
    │   └── ...
    └── abnormal/   # 異常画像（検証・テスト用、オプション）
        ├── abnormal_001.png
        ├── abnormal_002.png
        └── ...
```

## 使用方法
1. normal/ ディレクトリに正常画像を配置してください
2. abnormal/ ディレクトリに異常画像を配置してください（オプション）
3. 学習スクリプトを実行してください

## 注意事項
- 画像形式: PNG, JPG
- 推奨画像サイズ: 256x256 ピクセル（自動リサイズされます）
- 最低でも正常画像が必要です
"""
        
        readme_path = self.data_root / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        self.logger.info(f"サンプルデータセット構造を作成しました: {self.data_root}")
        self.logger.info(f"README.mdを作成しました: {readme_path}")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="PaDiM異常検知モデルの学習")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./images/train",
        help="学習データのルートディレクトリ (default: ./images/train)",
    )
    parser.add_argument(
        "--normal_dir",
        type=str,
        default="normal",
        help="正常画像のディレクトリ名 (default: normal)",
    )
    parser.add_argument(
        "--abnormal_dir",
        type=str,
        default="abnormal", 
        help="異常画像のディレクトリ名 (default: abnormal)",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models/padim_model.ckpt",
        help="学習済みモデルの保存パス (default: ./models/padim_model.ckpt)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="入力画像サイズ [height width] (default: 256 256)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="バッチサイズ (default: 32)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="最大エポック数 (default: 1)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"],
        help="バックボーンモデル (default: resnet18)",
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="サンプルデータセット構造を作成",
    )

    args = parser.parse_args()

    try:
        trainer = PaDiMTrainer(
            data_root=args.data_root,
            normal_dir=args.normal_dir,
            abnormal_dir=args.abnormal_dir,
            model_save_path=args.model_save_path,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            backbone=args.backbone,
        )

        if args.create_sample:
            trainer.create_sample_dataset_structure()
            print(f"\nサンプルデータセット構造を作成しました: {args.data_root}")
            print("正常画像を normal/ ディレクトリに配置してから学習を実行してください。")
            return 0

        # 学習実行
        success = trainer.train()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n学習が中断されました")
        return 1
    except Exception as e:
        print(f"学習でエラーが発生: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
