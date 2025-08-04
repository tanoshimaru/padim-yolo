#!/usr/bin/env python3
"""
PaDiM 追加学習スクリプト（土曜日実行）

機能:
- images/no_person, images/grid_00~grid_15 の画像を使用してPaDiMの追加学習
- 学習結果を models に保存
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import Folder

from image_manager import ImageManager


def setup_logging():
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = f"train_additional_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


class AdditionalTrainer:
    """追加学習クラス"""

    def __init__(self):
        self.logger = setup_logging()
        self.image_manager = ImageManager()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def prepare_training_data(self) -> str:
        """学習用データの準備（シェルスクリプト使用）

        Returns:
            str: 学習用データディレクトリのパス
        """
        import subprocess

        # 学習用データディレクトリを作成
        train_data_dir = Path("tmp/train_data")

        # シェルスクリプトで高速コピーを実行
        script_path = (
            Path(__file__).parent / "scripts" / "prepare_additional_training.sh"
        )
        try:
            result = subprocess.run(
                [
                    str(script_path),
                    "images",  # 画像ディレクトリ
                    str(train_data_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # 出力をログに記録
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[:-1]:  # 最後の行以外を出力
                self.logger.info(line)

            # 最後の行からコピー数を取得
            copied_count = int(output_lines[-1]) if output_lines else 0

            if copied_count == 0:
                self.logger.warning("学習用画像が見つかりません")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"シェルスクリプト実行エラー: {e.stderr}")
            raise RuntimeError(f"学習データ準備に失敗しました: {e.stderr}")
        except (ValueError, IndexError) as e:
            self.logger.error(f"シェルスクリプトからの出力解析エラー: {e}")
            raise RuntimeError(f"シェルスクリプト出力の解析に失敗しました: {e}")

        return str(train_data_dir)

    def create_model(self) -> Padim:
        """PaDiMモデルの作成"""
        # 既存モデルがある場合は読み込み、ない場合は新規作成
        existing_model_path = self.models_dir / "padim_model.ckpt"

        if existing_model_path.exists():
            try:
                self.logger.info(f"既存モデルを読み込み: {existing_model_path}")
                model = Padim.load_from_checkpoint(str(existing_model_path))
            except Exception as e:
                self.logger.warning(
                    f"既存モデルの読み込みに失敗: {e}。新規モデルを作成します。"
                )
                model = self._create_new_model()
        else:
            self.logger.info("新規PaDiMモデルを作成")
            model = self._create_new_model()

        return model

    def _create_new_model(self) -> Padim:
        """新規PaDiMモデルの作成"""
        pre_processor = Padim.configure_pre_processor(image_size=(224, 224))
        return Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            n_features=None,
            pre_processor=pre_processor,
        )

    def train_model(self, data_dir: str) -> bool:
        """モデルの学習実行

        Args:
            data_dir: 学習データディレクトリ

        Returns:
            bool: 学習成功可否
        """
        try:
            self.logger.info("PaDiM追加学習を開始")

            # データモジュールの作成
            datamodule = Folder(
                root=data_dir,
                normal_dir="normal",
                abnormal_dir="abnormal",
                image_size=[224, 224],  # 640x480からResNet標準サイズにリサイズ
                train_batch_size=32,  # リサイズにより標準バッチサイズに戻す
                eval_batch_size=32,
                num_workers=0,
            )

            # モデルの作成
            model = self.create_model()

            # エンジンの作成
            engine = Engine(
                max_epochs=10,
                accelerator="auto",
                devices=1,
                logger=False,  # MLflow等のロガーを無効化
                callbacks=None,
            )

            # 学習実行
            self.logger.info("=" * 50)
            self.logger.info("PaDiM追加学習開始")
            self.logger.info("=" * 50)
            self.logger.info(f"学習開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"エポック数: {engine.max_epochs}")
            self.logger.info(f"元画像サイズ: 640x480 → リサイズ後: 224x224 (ResNet標準)")
            self.logger.info(f"バッチサイズ: 32")
            self.logger.info(f"データディレクトリ: {data_dir}")
            self.logger.info("=" * 50)
            
            try:
                engine.fit(model=model, datamodule=datamodule)
                self.logger.info("=" * 50)
                self.logger.info("追加学習が正常に完了しました")
                self.logger.info(f"学習完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info("=" * 50)
            except Exception as e:
                self.logger.error("=" * 50)
                self.logger.error("追加学習中にエラーが発生しました")
                self.logger.error(f"エラー詳細: {e}")
                self.logger.error("=" * 50)
                raise

            # モデル保存
            self.logger.info("=" * 30)
            self.logger.info("追加学習モデル保存開始")
            self.logger.info("=" * 30)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = self.models_dir / f"padim_model_{timestamp}.ckpt"
            latest_model_path = self.models_dir / "padim_model.ckpt"
            
            try:
                # チェックポイント保存
                engine.trainer.save_checkpoint(str(model_save_path))
                model_size = model_save_path.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"タイムスタンプ付きモデル保存完了: {model_size:.2f} MB")

                # 最新モデルとしてもコピー
                import shutil
                shutil.copy2(str(model_save_path), str(latest_model_path))
                
                self.logger.info(f"学習完了。モデルを保存: {model_save_path}")
                self.logger.info(f"最新モデルとして更新: {latest_model_path}")
                self.logger.info(f"保存日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.logger.info("=" * 30)
                self.logger.info("追加学習モデル保存完了")
                self.logger.info("=" * 30)
                
            except Exception as e:
                self.logger.error("=" * 30)
                self.logger.error("追加学習モデル保存エラー")
                self.logger.error(f"エラー詳細: {e}")
                self.logger.error("=" * 30)
                raise

            return True

        except Exception as e:
            self.logger.error(f"学習中にエラーが発生: {e}")
            return False

    def cleanup_temp_data(self, data_dir: str):
        """一時データディレクトリの削除"""
        try:
            import shutil

            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                self.logger.info(f"一時データディレクトリを削除: {data_dir}")
        except Exception as e:
            self.logger.error(f"一時データ削除エラー: {e}")

    def get_model_info(self) -> dict:
        """保存されているモデル情報を取得"""
        info = {"models": [], "latest_model": None, "total_models": 0}

        if not self.models_dir.exists():
            return info

        # チェックポイントファイルを取得
        ckpt_files = list(self.models_dir.glob("*.ckpt"))

        for ckpt_file in ckpt_files:
            stat = ckpt_file.stat()
            model_info = {
                "name": ckpt_file.name,
                "path": str(ckpt_file),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            info["models"].append(model_info)

        # 最新モデルファイルがあるかチェック
        latest_model = self.models_dir / "padim_model.ckpt"
        if latest_model.exists():
            info["latest_model"] = str(latest_model)

        info["total_models"] = len(ckpt_files)

        # 更新日時でソート
        info["models"].sort(key=lambda x: x["modified"], reverse=True)

        return info


def main():
    """メイン関数"""
    try:
        trainer = AdditionalTrainer()

        print("=== PaDiM 追加学習開始 ===")

        # 現在のフォルダ情報を表示
        folder_info = trainer.image_manager.get_folder_info()
        total_images = sum(info["count"] for info in folder_info.values())

        print(f"学習用画像数: {total_images}枚")
        for category, info in folder_info.items():
            if info["count"] > 0:
                print(f"  {category}: {info['count']}枚")

        if total_images == 0:
            print("学習用画像がありません。学習をスキップします。")
            return 0

        # 学習データ準備
        print("\n学習データを準備中...")
        data_dir = trainer.prepare_training_data()

        try:
            # モデル学習実行
            print("\nモデル学習中...")
            success = trainer.train_model(data_dir)

            if success:
                print("✅ 学習が完了しました")

                # モデル情報を表示
                model_info = trainer.get_model_info()
                print(f"\n保存されているモデル数: {model_info['total_models']}")

                if model_info["latest_model"]:
                    print(f"最新モデル: {model_info['latest_model']}")

                print("\n=== モデル一覧 ===")
                for model in model_info["models"][:5]:  # 最新5個まで表示
                    print(
                        f"  {model['name']} ({model['size_mb']}MB) - {model['modified']}"
                    )

                return 0
            else:
                print("❌ 学習に失敗しました")
                return 1

        finally:
            # 一時データ削除
            trainer.cleanup_temp_data(data_dir)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
