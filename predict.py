import cv2
import os
import shutil
from datetime import datetime
from typing import Dict, Any
from person_detector import detect_person_and_get_grid
from distribute_images import distribute_image, setup_directories
from ultralytics import YOLO


def capture_image() -> str:
    """カメラから画像を撮影してimagesディレクトリに保存

    Returns:
        str: 撮影した画像のファイルパス
    """
    # imagesディレクトリが存在しない場合は作成
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # 現在の日時を取得してファイル名に使用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = os.path.join(images_dir, f"{timestamp}.jpg")

    # カメラから撮影
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("画像を撮影できませんでした")

    # 画像を保存
    cv2.imwrite(image_file, frame)
    print(f"画像を撮影しました: {image_file}")

    return image_file


def detect_anomaly_with_padim(image_path: str, grid_index: int) -> bool:
    """PADIMで異常検知を実行

    Args:
        image_path: 画像ファイルのパス
        grid_index: グリッド位置のインデックス

    Returns:
        bool: 異常が検出された場合True
    """
    # PADIMのモデルファイルパス（区画ごとに異なるモデルを想定）
    model_path = f"padim_models/grid_{grid_index:02d}.ckpt"

    if not os.path.exists(model_path):
        print(f"警告: PADIM モデルが見つかりません: {model_path}")
        return False

    try:
        # anomalibのPADIMを使用した異常検知
        # 実際の実装では、anomalibのAPIを使用
        # ここでは簡易的な実装例

        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"エラー: 画像を読み込めません: {image_path}")
            return False

        # TODO: 実際のPADIM推論処理を実装
        # from anomalib.models import Padim
        # from anomalib.data import Folder
        #
        # model = Padim.load_from_checkpoint(model_path)
        # result = model.predict(image_path)
        # return result.pred_score > threshold

        # 仮の実装（常にFalseを返す）
        print(f"PADIM異常検知実行（仮実装）: {image_path}, グリッド{grid_index}")
        return False

    except Exception as e:
        print(f"PADIM異常検知エラー: {e}")
        return False


def move_image_to_grid(image_path: str, grid_index: int) -> None:
    """画像を指定されたグリッドディレクトリに移動

    Args:
        image_path: 画像ファイルのパス
        grid_index: グリッドインデックス
    """
    # グリッドディレクトリのパス
    grid_dir = f"images/grid_{grid_index:02d}"

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    # ファイル名を取得
    filename = os.path.basename(image_path)
    dest_path = os.path.join(grid_dir, filename)

    # 画像を移動
    shutil.move(image_path, dest_path)
    print(f"画像をgrid_{grid_index:02d}ディレクトリに移動しました: {dest_path}")


def process_no_person_image(image_path: str) -> None:
    """人が検出されなかった画像をdistribute_imagesで処理

    Args:
        image_path: 画像ファイルのパス
    """
    print("人が検出されませんでした。distribute_imagesで処理します。")

    # YOLOモデルをロード
    if os.path.exists("models/yolo11n.engine"):
        trt_model = YOLO("models/yolo11n.engine", task="detect")
    else:
        print("エラー: YOLOモデルが見つかりません")
        return

    # ディレクトリセットアップ
    input_dir = "images"
    grid_dirs, no_person_dir, output_dir = setup_directories(input_dir)

    # 画像を処理
    _grid_info, _grid_index = distribute_image(
        image_path, trt_model, grid_dirs, no_person_dir, output_dir
    )


def main() -> Dict[str, Any]:
    """メイン処理関数

    Returns:
        Dict[str, Any]: 処理結果の情報
    """
    result = {
        "image_path": None,
        "person_detected": False,
        "grid_index": None,
        "anomaly_detected": False,
        "action_taken": None,
    }

    try:
        # 1. 画像を撮影してimagesに保存
        image_path = capture_image()
        result["image_path"] = image_path

        # 2. person_detectorで人検出
        detection_result = detect_person_and_get_grid(image_path)

        if detection_result is None:
            # 3a. 人が検出されなかった場合
            result["person_detected"] = False
            result["action_taken"] = "distributed_no_person"
            process_no_person_image(image_path)

        else:
            # 3b. 人が検出された場合
            result["person_detected"] = True
            grid_index = detection_result["grid_index"]
            result["grid_index"] = grid_index

            print(f"人を検出しました。グリッド位置: {grid_index}")

            # 4. PADIMで異常検知
            anomaly_detected = detect_anomaly_with_padim(image_path, grid_index)
            result["anomaly_detected"] = anomaly_detected

            if anomaly_detected:
                # 5a. 異常が検出された場合
                result["action_taken"] = "anomaly_detected"
                print("異常が検出されました！")
                # 異常検出時の処理（アラート、ログ記録など）

            else:
                # 5b. 異常が検出されなかった場合
                result["action_taken"] = f"moved_to_grid{grid_index}"
                move_image_to_grid(image_path, grid_index)

        return result

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        result["action_taken"] = "error"
        return result


if __name__ == "__main__":
    result = main()
    print(f"処理結果: {result}")
