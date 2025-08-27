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
import numpy as np

import cv2
from anomalib.models import Padim
from anomalib.engine import Engine
from dotenv import load_dotenv
from person_detector import detect_person_and_get_grid


# 環境変数を明示的に読み込み
try:
    load_dotenv()
except ImportError:
    pass


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

    def __init__(self, model_path: str = "models/padim_trained.ckpt", anomaly_threshold: float = 0.5):
        self.model_path = model_path
        self.anomaly_threshold = anomaly_threshold
        self.model = None
        self.engine = None
        self._load_model()

    def _load_model(self):
        """PaDiMモデルの読み込み"""
        try:
            if os.path.exists(self.model_path):
                # 学習済みモデルがある場合
                try:
                    # 新しい形式（.save()で保存されたモデル）を試行
                    self.model = Padim.load(self.model_path)
                    self.model.eval()
                    logging.info(
                        f"学習済みPaDiMモデルを読み込みました（新形式）: {self.model_path}"
                    )
                except Exception:
                    try:
                        # 古い形式（チェックポイント）を試行
                        self.model = Padim.load_from_checkpoint(self.model_path)
                        self.model.eval()
                        logging.info(
                            f"学習済みPaDiMモデルを読み込みました（チェックポイント）: {self.model_path}"
                        )
                    except Exception as e:
                        logging.warning(f"モデル読み込みに失敗、初期モデルを使用: {e}")
                        self._create_initial_model()
            else:
                logging.warning(
                    "学習済みモデルが見つかりません。初期モデルを使用します。"
                )
                self._create_initial_model()

            self.engine = Engine()

        except Exception as e:
            logging.error(f"PaDiMモデルの読み込みに失敗: {e}")
            raise

    def _create_initial_model(self):
        """初期モデルの作成"""
        pre_processor = Padim.configure_pre_processor(image_size=(224, 224))
        self.model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            pre_processor=pre_processor,
        )

    def predict(self, image_path: str) -> Dict[str, Any]:
        """異常検知の実行（推論専用）"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

            # 直接画像を読み込んで推論
            import torch
            from torchvision import transforms
            from PIL import Image

            # 画像を読み込み
            image = Image.open(image_path).convert("RGB")
            
            # 前処理（PaDiMの標準前処理）
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加
            
            # GPUが利用可能な場合はGPUを使用
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                if hasattr(self.model, 'cuda'):
                    self.model = self.model.cuda()
            
            # 推論モードに設定
            self.model.eval()
            
            with torch.no_grad():
                # 直接モデルで推論
                output = self.model(image_tensor)
                
                # anomalib v0.7.0以降の場合
                if hasattr(output, 'pred_score'):
                    anomaly_score = float(output.pred_score.cpu().item())
                    anomaly_map = output.anomaly_map.cpu().numpy() if hasattr(output, 'anomaly_map') else None
                # 辞書形式の場合
                elif isinstance(output, dict):
                    anomaly_score = float(output.get('pred_score', 0.0))
                    anomaly_map = output.get('anomaly_map', None)
                    if anomaly_map is not None and hasattr(anomaly_map, 'cpu'):
                        anomaly_map = anomaly_map.cpu().numpy()
                # フォールバック: 出力テンソルから直接スコアを取得
                else:
                    if hasattr(output, 'cpu'):
                        anomaly_score = float(torch.mean(output).cpu().item())
                        anomaly_map = output.cpu().numpy()
                    else:
                        anomaly_score = float(torch.mean(torch.tensor(output)).item())
                        anomaly_map = output
                
                result = {
                    "anomaly_score": anomaly_score,
                    "is_anomaly": anomaly_score > self.anomaly_threshold,
                    "anomaly_map": anomaly_map,
                    "threshold": self.anomaly_threshold,
                }

                # 最も異常度が高い座標区画を特定
                if anomaly_map is not None:
                    max_anomaly_coords = self._find_max_anomaly_coordinates(
                        anomaly_map
                    )
                    result["max_anomaly_coordinates"] = max_anomaly_coords

                logging.info(
                    f"PaDiM推論完了: score={result['anomaly_score']:.4f}, threshold={self.anomaly_threshold:.4f}, anomaly={result['is_anomaly']}"
                )
                return result

        except Exception as e:
            logging.error(f"PaDiM異常検知エラー: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_map": None,
                "max_anomaly_coordinates": None,
                "error": str(e),
            }

    def _find_max_anomaly_coordinates(self, anomaly_map: np.ndarray) -> Dict[str, Any]:
        """異常マップから最も異常度が高い座標区画を特定"""
        try:
            if anomaly_map.ndim == 3:
                # チャンネル次元がある場合は最初のチャンネルを使用
                anomaly_map = anomaly_map[0]
            elif anomaly_map.ndim == 4:
                # バッチ次元とチャンネル次元がある場合
                anomaly_map = anomaly_map[0, 0]

            # 最大異常値とその座標を取得
            max_anomaly_value = float(np.max(anomaly_map))
            max_coords = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)

            # 座標を画像サイズに正規化（0-1の範囲）
            height, width = anomaly_map.shape
            normalized_y = float(max_coords[0]) / height
            normalized_x = float(max_coords[1]) / width

            # 4x4グリッドでの区画番号を計算
            grid_y = int(normalized_y * 4)
            grid_x = int(normalized_x * 4)
            grid_index = grid_y * 4 + grid_x

            # 範囲チェック
            grid_y = min(max(grid_y, 0), 3)
            grid_x = min(max(grid_x, 0), 3)
            grid_index = min(max(grid_index, 0), 15)

            return {
                "max_anomaly_value": max_anomaly_value,
                "pixel_coordinates": {"x": int(max_coords[1]), "y": int(max_coords[0])},
                "normalized_coordinates": {"x": normalized_x, "y": normalized_y},
                "grid_coordinates": {"x": grid_x, "y": grid_y},
                "grid_index": grid_index,
                "anomaly_map_shape": list(anomaly_map.shape),
            }

        except Exception as e:
            logging.error(f"最大異常座標の特定でエラー: {e}")
            return {
                "error": str(e),
                "max_anomaly_value": 0.0,
                "pixel_coordinates": {"x": 0, "y": 0},
                "normalized_coordinates": {"x": 0.0, "y": 0.0},
                "grid_coordinates": {"x": 0, "y": 0},
                "grid_index": 0,
                "anomaly_map_shape": [],
            }


class MainProcessor:
    def process_image_file(self, image_path: str) -> Dict[str, Any]:
        """指定画像ファイルで推論を実行し、全画像をimages/checkedにprefix付きで保存"""
        try:
            person_result = detect_person_and_get_grid(image_path, self.yolo_model)
            result = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "image_path": image_path,
                "person_detected": person_result is not None,
                "person_info": person_result,
                "padim_result": None,
                "final_decision": None,
                "saved_path": None,
            }
            prefix = ""
            if person_result is None:
                prefix = "no_person_"
                result["final_decision"] = "no_person"
            else:
                padim_result = self.padim_detector.predict(image_path)
                result["padim_result"] = padim_result
                grid_num = person_result.get("grid_index", 0)
                if padim_result.get("error"):
                    prefix = "error_"
                    result["final_decision"] = "error"
                else:
                    if padim_result.get("is_anomaly", False):
                        prefix = f"defect_grid_{grid_num:02d}_"
                        result["final_decision"] = "anomaly"
                    else:
                        prefix = f"grid_{grid_num:02d}_"
                        result["final_decision"] = "normal"
            # 保存先ディレクトリ
            checked_dir = Path("images/checked")
            checked_dir.mkdir(parents=True, exist_ok=True)
            # 保存ファイル名
            ts = result["timestamp"]
            ext = Path(image_path).suffix or ".png"
            save_name = f"{prefix}{ts}{ext}"
            save_path = checked_dir / save_name
            import shutil
            shutil.copy2(image_path, save_path)
            result["saved_path"] = str(save_path)
            self.logger.info(f"画像を {save_path} に保存 (prefix={prefix})")
            self._save_result_log(result)
            return result
        except Exception as e:
            self.logger.error(f"画像ファイル処理中にエラーが発生: {e}")
            # エラー時もerror_で保存
            checked_dir = Path("images/checked")
            checked_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(image_path).suffix or ".png"
            save_name = f"error_{ts}{ext}"
            save_path = checked_dir / save_name
            try:
                import shutil
                shutil.copy2(image_path, save_path)
            except Exception:
                pass
            return {"error": str(e), "image_path": image_path, "saved_path": str(save_path)}
    """メイン処理クラス"""

    def __init__(self, anomaly_threshold: float = None):
        self.logger = setup_logging()

        
        # 閾値設定（環境変数またはパラメータから取得）
        if anomaly_threshold is None:
            anomaly_threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
        
        self.padim_detector = PaDiMAnomalyDetector(anomaly_threshold=anomaly_threshold)
        self.yolo_model = None
        self._load_yolo_model()
        
        self.logger.info(f"異常検出閾値を設定: {anomaly_threshold}")

    def _load_yolo_model(self):
        """YOLOモデルの読み込み"""
        try:
            from ultralytics import YOLO

            # modelsディレクトリを作成
            os.makedirs("models", exist_ok=True)

            if not os.path.exists("models/yolo11n.engine"):
                self.logger.info("yolo11n.engineファイルが見つかりません。")
                # なければYOLOモデルを自動ダウンロード
                self.yolo_model = YOLO("models/yolo11n.pt")
                # TensorRTエンジンを生成
                self.yolo_model.export(format="engine", task="detect")
            self.logger.info("yolo11n.engineファイルを使用")
            self.yolo_model = YOLO("models/yolo11n.engine", task="detect")
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

            # RTSPストリームに接続（FFmpegバックエンドを明示的に指定）
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # 接続設定を最適化
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
            cap.set(cv2.CAP_PROP_FPS, 15)  # フレームレートを下げて安定化
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # タイムアウト設定（OpenCVのプロパティとして）
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10秒でタイムアウト
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 読み取りタイムアウト5秒

            if not cap.isOpened():
                self.logger.error(f"RTSPストリームに接続できません: {rtsp_url}")
                cap.release()
                return False

            # フレームを読み取り（複数回試行）
            frame = None
            for attempt in range(3):  # 最大3回試行
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                else:
                    if attempt < 2:  # 最後の試行でなければ少し待機
                        import time

                        time.sleep(0.5)

            if not ret or frame is None:
                self.logger.error(
                    "RTSPストリームからフレームを取得できません（全試行失敗）"
                )
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
        """平日の処理メイン: 撮影画像もcheckedにprefix付きで保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"tmp/captured_{timestamp}.png"
        os.makedirs("tmp", exist_ok=True)
        if not self.capture_image(temp_image_path):
            return {"error": "画像撮影に失敗しました"}
        try:
            # 推論・保存もprocess_image_fileで一元化
            result = self.process_image_file(temp_image_path)
            return result
        finally:
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

        # コマンドライン引数で画像パス指定があればその画像で推論
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            result = processor.process_image_file(image_path)
        else:
            # 平日処理を実行（従来通りカメラ撮影）
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
            print(f"異常閾値: {padim_result.get('threshold', 0):.4f}")
            print(f"異常判定: {padim_result.get('is_anomaly', False)}")

            # 最も異常度が高い座標区画の情報を出力
            max_anomaly_coords = padim_result.get("max_anomaly_coordinates")
            if max_anomaly_coords and not max_anomaly_coords.get("error"):
                print("=== 最も異常度が高い座標区画 ===")
                print(f"最大異常値: {max_anomaly_coords['max_anomaly_value']:.6f}")
                print(
                    f"ピクセル座標: x={max_anomaly_coords['pixel_coordinates']['x']}, y={max_anomaly_coords['pixel_coordinates']['y']}"
                )
                print(
                    f"正規化座標: x={max_anomaly_coords['normalized_coordinates']['x']:.4f}, y={max_anomaly_coords['normalized_coordinates']['y']:.4f}"
                )
                print(
                    f"グリッド座標: x={max_anomaly_coords['grid_coordinates']['x']}, y={max_anomaly_coords['grid_coordinates']['y']}"
                )
                print(f"グリッド番号: Grid {max_anomaly_coords['grid_index']:02d}")
                print(f"異常マップサイズ: {max_anomaly_coords['anomaly_map_shape']}")

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
