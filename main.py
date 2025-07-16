from ultralytics import YOLO
import cv2
import os
import shutil


def get_grid_position(center_x, center_y, image_width, image_height):
    """画像を4x4の16分割した時の位置を取得"""
    # 各グリッドのサイズ
    grid_width = image_width / 4
    grid_height = image_height / 4

    # グリッドの位置を計算 (0-3の範囲)
    grid_x = min(int(center_x / grid_width), 3)
    grid_y = min(int(center_y / grid_height), 3)

    # 16分割のインデックス (0-15)
    grid_index = grid_y * 4 + grid_x

    return grid_index, grid_x, grid_y


def main():
    if not os.path.exists("models/yolo11n.engine"):
        # Load a YOLO11n PyTorch model
        model = YOLO("models/yolo11n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo11n.engine'

    # Load the exported TensorRT model
    trt_model = YOLO("models/yolo11n.engine", task="detect")

    # 入力ディレクトリ
    input_dir = "images"

    # 人が検出されなかった画像の振り分け先ディレクトリ
    no_person_dir = os.path.join(input_dir, "no-person")

    # 16分割のグリッド位置別ディレクトリを作成
    grid_dirs = {}
    for i in range(16):
        grid_dir = os.path.join(input_dir, f"grid_{i:02d}")
        grid_dirs[i] = grid_dir
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)

    # no-personディレクトリも作成
    if not os.path.exists(no_person_dir):
        os.makedirs(no_person_dir)

    # 出力ディレクトリがなければ作成
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像ファイルのリストを取得
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_files = []

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(
            file.lower().endswith(ext) for ext in image_extensions
        ):
            image_files.append(file_path)

    print(f"{len(image_files)}枚の画像が見つかりました")

    # 振り分けカウンター
    distribution_count = {i: 0 for i in range(16)}
    no_person_count = 0

    # 各画像に対して推論を実行
    for image_path in image_files:
        print(f"処理中: {image_path}")

        # 画像のサイズを取得
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # 推論実行
        results = trt_model(image_path)
        result = results[0]

        # 検出結果を画像に描画
        result_image = result.plot()

        # 出力ファイルパスを設定
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{output_filename}")

        # 画像を保存
        cv2.imwrite(output_path, result_image)
        print(f"検出結果を保存しました: {output_path}")

        # 人が検出されたかどうかを確認し、座標を取得
        person_detected = False
        person_positions = []

        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            if class_name.lower() == "person":
                person_detected = True

                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 中央座標を計算
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # グリッド位置を取得
                grid_index, grid_x, grid_y = get_grid_position(
                    center_x, center_y, image_width, image_height
                )
                person_positions.append(
                    (grid_index, grid_x, grid_y, center_x, center_y)
                )

                print(
                    f"人を検出: 中央座標({center_x:.1f}, {center_y:.1f}) -> グリッド位置({grid_x}, {grid_y}) インデックス{grid_index}"
                )

        # 画像を振り分け
        if person_detected:
            # 複数の人が検出された場合は最初の人の位置を使用
            grid_index = person_positions[0][0]
            dest_dir = grid_dirs[grid_index]
            distribution_count[grid_index] += 1
            grid_info = f"grid_{grid_index:02d}"
        else:
            dest_dir = no_person_dir
            no_person_count += 1
            grid_info = "no-person"

        dest_path = os.path.join(dest_dir, os.path.basename(image_path))

        # 画像を移動
        shutil.move(image_path, dest_path)
        print(f"画像を{grid_info}ディレクトリに移動しました")

    # 振り分け結果をテキストファイルに出力
    result_text_path = os.path.join(output_dir, "distribution_result.txt")
    with open(result_text_path, "w", encoding="utf-8") as f:
        f.write("画像振り分け結果\n")
        f.write("=" * 30 + "\n\n")

        # グリッド別の振り分け結果
        f.write("グリッド別振り分け結果:\n")
        total_person_images = 0
        for i in range(16):
            count = distribution_count[i]
            if count > 0:
                grid_x = i % 4
                grid_y = i // 4
                f.write(f"grid_{i:02d} (位置: {grid_x}, {grid_y}): {count}枚\n")
                total_person_images += count

        f.write(f"\nno-person: {no_person_count}枚\n")
        f.write(f"\n合計: {total_person_images + no_person_count}枚\n")
        f.write(f"人検出画像: {total_person_images}枚\n")
        f.write(f"人未検出画像: {no_person_count}枚\n")

    # コンソールにも結果を表示
    print("\n" + "=" * 50)
    print("振り分け結果:")
    print("=" * 50)
    for i in range(16):
        count = distribution_count[i]
        if count > 0:
            grid_x = i % 4
            grid_y = i // 4
            print(f"grid_{i:02d} (位置: {grid_x}, {grid_y}): {count}枚")

    print(f"no-person: {no_person_count}枚")
    print(f"\n結果をファイルに保存しました: {result_text_path}")


if __name__ == "__main__":
    main()
