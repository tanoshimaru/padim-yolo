#!/usr/bin/env python3
"""
PaDiM学習用データ準備スクリプト

images/ ディレクトリ内の画像を training_data/ に分散配置し、
PaDiMの学習データセットを準備します。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path


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


    def prepare_training_data(self) -> bool:
        """学習データを準備（シェルスクリプト使用）"""
        import subprocess

        try:
            self.logger.info("=== 学習データ準備開始（高速化） ===")

            # シェルスクリプトで高速処理を実行
            script_path = Path(__file__).parent / "prepare_data.sh"
            try:
                result = subprocess.run(
                    [
                        str(script_path),
                        str(self.source_dir),
                        str(self.target_dir),
                        str(self.normal_ratio),
                        "true" if self.copy_mode else "false",
                        str(self.random_seed),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # 出力をログに記録
                output_lines = result.stdout.strip().split("\n")
                for line in output_lines[:-1]:  # 最後の行以外を出力
                    self.logger.info(line)

                # 最後の行から統計情報を取得
                stats_line = output_lines[-1]
                if "|" in stats_line:
                    train_count, val_count = map(int, stats_line.split("|"))

                    # データセット情報ファイル作成
                    self.create_dataset_info(train_count, val_count)

                    self.logger.info("=== 学習データ準備完了 ===")
                    return True
                else:
                    self.logger.error("シェルスクリプトからの統計情報取得エラー")
                    return False

            except subprocess.CalledProcessError as e:
                self.logger.error(f"シェルスクリプト実行エラー: {e.stderr}")
                raise RuntimeError(f"学習データ準備に失敗しました: {e.stderr}")

        except Exception as e:
            self.logger.error(f"学習データ準備でエラー: {e}")
            raise

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
        import shutil
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
            print("\n次のコマンドで学習を開始できます:")
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
