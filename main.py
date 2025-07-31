#!/usr/bin/env python3
"""
PaDiM + YOLO 異常検知メインスクリプト

平日の処理:
1. 撮影
2. YOLO で人の検出
3. PaDiM で異常検知
4. 結果の記録と画像保存
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

try:
    import cv2
    from anomalib.models import Padim
    from anomalib.pre_processing import Preprocessor
    from anomalib.engine import Engine
    from anomalib.data import Folder
    from person_detector import detect_person_and_get_grid
    from image_manager import ImageManager
except ImportError as e:
    print(f"必要なライブラリのインポートに失敗しました: {e}")
    print(
        "uv run --with opencv-python --with anomalib python main.py で実行してください"
    )
    sys.exit(1)


def setup_logging():
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = f"main_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


class PaDiMAnomalyDetector:
    """PaDiM 異常検知クラス"""

    def __init__(self, model_path: str = "models/padim_model.ckpt"):
        self.model_path = model_path
        self.model = None
        self.engine = None
        self._load_model()

    def _load_model(self):
        """PaDiMモデルの読み込み"""
        try:
            if os.path.exists(self.model_path):
                # 学習済みモデルがある場合
                self.model = Padim.load_from_checkpoint(self.model_path)
                self.model.eval()
                logging.info(f"学習済みPaDiMモデルを読み込みました: {self.model_path}")
            else:
                pre_processor = Preprocessor(image_size=(256, 256))
                self.model = Padim(
                    backbone="resnet18",
                    layers=["layer1", "layer2", "layer3"],
                    pre_trained=True,
                    pre_processor=pre_processor,
                )
                logging.warning(
                    "学習済みモデルが見つかりません。初期モデルを使用します。"
                )

            self.engine = Engine()

        except Exception as e:
            logging.error(f"PaDiMモデルの読み込みに失敗: {e}")
            raise

    def predict(self, image_path: str) -> Dict[str, Any]:
        """異常検知の実行"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

            # 単一画像の推論用データモジュール
            datamodule = Folder(
                root="./", normal_dir="./tmp_normal", abnormal_dir="./tmp_abnormal"
            )

            # 一時ディレクトリ作成
            Path("./tmp_normal").mkdir(exist_ok=True)
            Path("./tmp_abnormal").mkdir(exist_ok=True)

            # 画像を一時的にコピー
            import shutil

            temp_image_path = "./tmp_normal/temp.png"
            shutil.copy2(image_path, temp_image_path)

            try:
                # 推論実行
                predictions = self.engine.predict(
                    model=self.model, datamodule=datamodule
                )

                if predictions and len(predictions) > 0:
                    pred = predictions[0]

                    result = {
                        "anomaly_score": float(
                            pred.pred_score.item()
                            if hasattr(pred, "pred_score")
                            else 0.0
                        ),
                        "is_anomaly": bool(
                            pred.pred_label.item()
                            if hasattr(pred, "pred_label")
                            else False
                        ),
                        "anomaly_map": pred.anomaly_map.cpu().numpy()
                        if hasattr(pred, "anomaly_map")
                        else None,
                    }

                    logging.info(
                        f"PaDiM推論完了: score={result['anomaly_score']:.4f}, anomaly={result['is_anomaly']}"
                    )
                    return result
                else:
                    logging.warning("PaDiM推論結果が空です")
                    return {
                        "anomaly_score": 0.0,
                        "is_anomaly": False,
                        "anomaly_map": None,
                    }

            finally:
                # 一時ファイル削除
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

        except Exception as e:
            logging.error(f"PaDiM異常検知エラー: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_map": None,
                "error": str(e),
            }


class MainProcessor:
    """メイン処理クラス"""

    def __init__(self):
        self.logger = setup_logging()
        self.image_manager = ImageManager()
        self.padim_detector = PaDiMAnomalyDetector()
        self.yolo_model = None
        self._load_yolo_model()

    def _load_yolo_model(self):
        """YOLOモデルの読み込み"""
        try:
            from ultralytics import YOLO

            if os.path.exists("models/yolo11n.engine"):
                self.yolo_model = YOLO("models/yolo11n.engine", task="detect")
            elif os.path.exists("models/yolo11n.pt"):
                self.yolo_model = YOLO("models/yolo11n.pt")
            else:
                self.logger.error("YOLOモデルファイルが見つかりません")
                raise FileNotFoundError("YOLOモデルファイルが見つかりません")

            self.logger.info("YOLOモデルを読み込みました")

        except Exception as e:
            self.logger.error(f"YOLOモデルの読み込みに失敗: {e}")
            raise

    def capture_image(self, output_path: str, rtsp_url: str | None = None) -> bool:
        """RTSP カメラから画像撮影"""
        try:
            # RTSP URL が指定されていない場合のデフォルト値
            if rtsp_url is None:
                # 環境変数から個別に取得してURL構築
                username = os.getenv("RTSP_USERNAME", "username")
                password = os.getenv("RTSP_PASSWORD", "password")
                ip = os.getenv("RTSP_IP", "ip_address")
                port = os.getenv("RTSP_PORT", "554")
                rtsp_url = (
                    f"rtsp://{username}:{password}@{ip}:{port}/profile2/media.smp"
                )

            self.logger.info(f"RTSP接続を試行: {rtsp_url}")

            # RTSPストリームに接続
            cap = cv2.VideoCapture(rtsp_url)

            # 接続タイムアウト設定 (5秒)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                self.logger.error(f"RTSPストリームに接続できません: {rtsp_url}")
                return False

            # フレームを読み取り
            ret, frame = cap.read()

            if not ret or frame is None:
                self.logger.error("RTSPストリームからフレームを取得できません")
                cap.release()
                return False

            # 画像を保存
            success = cv2.imwrite(output_path, frame)
            cap.release()

            if success:
                self.logger.info(f"RTSPから画像を撮影しました: {output_path}")
                return True
            else:
                self.logger.error(f"画像の保存に失敗: {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"RTSP画像撮影エラー: {e}")
            return False

    def process_weekday(self) -> Dict[str, Any]:
        """平日の処理メイン"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 撮影
        temp_image_path = f"tmp/captured_{timestamp}.png"
        os.makedirs("tmp", exist_ok=True)

        if not self.capture_image(temp_image_path):
            return {"error": "画像撮影に失敗しました"}

        try:
            # 2. YOLO で人検出
            person_result = detect_person_and_get_grid(temp_image_path, self.yolo_model)

            result = {
                "timestamp": timestamp,
                "image_path": temp_image_path,
                "person_detected": person_result is not None,
                "person_info": person_result,
                "padim_result": None,
                "final_decision": "normal",
                "saved_path": None,
            }

            if person_result is None:
                # 人が写っていない場合
                saved_path = self.image_manager.save_image(
                    temp_image_path, "no_person", timestamp
                )
                result["saved_path"] = saved_path
                result["final_decision"] = "no_person"
                self.logger.info("人が検出されませんでした。images/no_person に保存")

            else:
                # 人が写っている場合 - PaDiM で異常検知
                padim_result = self.padim_detector.predict(temp_image_path)
                result["padim_result"] = padim_result

                grid_num = person_result["grid_index"]

                if padim_result.get("is_anomaly", False):
                    # 異常検出
                    result["final_decision"] = "anomaly"
                    self.logger.warning(
                        f"異常が検出されました (Grid {grid_num:02d}): score={padim_result.get('anomaly_score', 0):.4f}"
                    )
                else:
                    # 正常 - 該当グリッドに保存
                    saved_path = self.image_manager.save_image(
                        temp_image_path, f"grid_{grid_num:02d}", timestamp
                    )
                    result["saved_path"] = saved_path
                    result["final_decision"] = "normal"
                    self.logger.info(f"正常画像として保存: {saved_path}")

            # 結果をJSONで記録
            self._save_result_log(result)

            return result

        except Exception as e:
            self.logger.error(f"処理中にエラーが発生: {e}")
            return {"error": str(e), "timestamp": timestamp}

        finally:
            # 一時ファイル削除
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    def _save_result_log(self, result: Dict[str, Any]):
        """結果をJSONログとして保存"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d')}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"結果ログの保存に失敗: {e}")


def main():
    """メイン関数"""
    try:
        processor = MainProcessor()

        # 平日処理を実行
        result = processor.process_weekday()

        print("=== 処理結果 ===")
        print(f"タイムスタンプ: {result.get('timestamp', 'N/A')}")
        print(f"人検出: {result.get('person_detected', False)}")

        if result.get("person_info"):
            person_info = result["person_info"]
            print(
                f"検出位置: Grid {person_info['grid_index']:02d} (x={person_info['grid_x']}, y={person_info['grid_y']})"
            )
            print(f"信頼度: {person_info['confidence']:.4f}")

        if result.get("padim_result"):
            padim_result = result["padim_result"]
            print(f"異常スコア: {padim_result.get('anomaly_score', 0):.4f}")
            print(f"異常判定: {padim_result.get('is_anomaly', False)}")

        print(f"最終判定: {result.get('final_decision', 'unknown')}")

        if result.get("saved_path"):
            print(f"保存先: {result['saved_path']}")

        if result.get("error"):
            print(f"エラー: {result['error']}")
            return 1

        return 0

    except Exception as e:
        print(f"メイン処理でエラーが発生: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
