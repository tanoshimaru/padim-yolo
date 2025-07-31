# PaDiM + YOLO 異常検知システム

anomalib を使用して PaDiM で異常検知を行い、YOLO で人の検出を行うシステムです。

## 機能概要

### 平日処理（月〜金）

1. **撮影**: カメラから画像を撮影
2. **人検出**: YOLO で人が写っているかを検出
3. **異常検知**: 人が写っている場合、PaDiM で異常検知を実行
4. **画像保存**: 結果に基づいて適切なフォルダに画像を保存
5. **ログ記録**: YOLO・PaDiM の結果と画像保存先を記録

### 土曜日処理

- **追加学習**: `images/no_person`、`images/grid_00~grid_15`の画像を使用して PaDiM の追加学習
- **モデル保存**: 学習結果を`models`に保存

### 日曜日

- 処理休止

## ディレクトリ構造

```
padim-yolo/
├── main.py                    # メイン処理（平日実行）
├── train_additional.py        # 追加学習（土曜実行）
├── run_daily.py              # 日次実行スクリプト
├── person_detector.py        # YOLO人検出モジュール
├── image_manager.py          # 画像管理モジュール
├── Dockerfile                # Dockerコンテナ設定
├── docker-compose.yml        # Docker Compose設定
├── docker-start.sh           # コンテナ起動スクリプト
├── docker-crontab.txt        # コンテナ用cron設定
├── models/                   # YOLOモデル
│   ├── yolo11n.pt
│   └── yolo11n.onnx
├── images/                   # 画像保存ディレクトリ
│   ├── no_person/           # 人が写っていない画像（最大200枚）
│   ├── grid_00/             # グリッド0の正常画像（最大200枚）
│   ├── grid_01/             # グリッド1の正常画像（最大200枚）
│   ├── ...                  # grid_02 ~ grid_15
│   └── models/              # PaDiMモデル保存先
└── logs/                    # ログファイル
    ├── main_YYYYMMDD.log
    ├── train_additional_YYYYMMDD.log
    ├── results_YYYYMMDD.jsonl
    └── cron.log              # cronスケジュール実行ログ
```

## グリッド分割

画像を 4×4 の 16 分割し、人が検出された位置を特定：

```
grid_00  grid_01  grid_02  grid_03
grid_04  grid_05  grid_06  grid_07
grid_08  grid_09  grid_10  grid_11
grid_12  grid_13  grid_14  grid_15
```

## 処理フロー

### 平日の処理フロー

1. **撮影** → 画像取得
2. **YOLO 検出**
   - 人が写っていない → `images/no_person`に保存
   - 人が写っている → 3 へ
3. **グリッド位置特定** → 人の位置から grid_XX 番号を取得
4. **PaDiM 異常検知**
   - 異常検出 → 結果を異常として記録（画像は保存しない）
   - 正常 → `images/grid_XX`に保存

### 画像管理

- 各フォルダの最大画像数: **200 枚**
- ファイル名形式: `YYYYMMDD_HHMMSS.png`
- 200 枚を超えた場合、古い画像から自動削除

## 実行方法

### 1. Docker コンテナでの自動実行（推奨）

#### 起動

```bash
# コンテナをビルドして起動
docker compose up -d --build
```

#### 停止

```bash
# コンテナを停止
docker compose down
```

#### ログ確認

```bash
# cronログを確認
tail -f logs/cron.log

# コンテナログを確認
docker compose logs -f
```

**スケジュール**:

- **平日（月-金）8:00-17:00**: 30 秒間隔で自動実行
- **土曜 9:00**: 追加学習を 1 回実行
- **日曜**: 処理休止

### 2. 単発実行

#### コンテナ内で手動実行

```bash
# コンテナに入る
docker compose exec app bash

# 平日処理
python main.py

# 追加学習
python train_additional.py
```

### 3. 日次自動実行（手動）

```bash
# コンテナ内で曜日判定による自動実行
python run_daily.py
```

自動的に曜日を判定して適切な処理を実行します。

## 設定

### YOLO モデル

- `models/yolo11n.pt` または `models/yolo11n.engine` を配置
- 人（person）クラスの検出に使用

### PaDiM モデル

- 初回実行時は未学習モデルを使用
- 土曜日の追加学習でモデルが更新される
- 学習済みモデルは`models/padim_model.ckpt`に保存

### カメラ設定

RTSP カメラの設定は環境変数で行います：

```bash
# .envファイルを作成してカメラ情報を設定
RTSP_USERNAME=admin
RTSP_PASSWORD=password123
RTSP_IP=192.168.1.100
RTSP_PORT=554
```

現在はダミー画像を生成しています。実際のカメラを使用する場合は`main.py`の`capture_image`関数を修正してください。

## ログ

### 実行ログ

- `logs/main_YYYYMMDD.log`: 平日処理のログ
- `logs/train_additional_YYYYMMDD.log`: 追加学習のログ
- `logs/cron.log`: cron スケジュール実行ログ（Docker 使用時）

### 結果ログ

- `logs/results_YYYYMMDD.jsonl`: JSON 形式の処理結果
  ```json
  {
    "timestamp": "20250131_120000",
    "person_detected": true,
    "person_info": {
      "grid_index": 5,
      "grid_x": 1,
      "grid_y": 1,
      "confidence": 0.95
    },
    "padim_result": {
      "anomaly_score": 0.12,
      "is_anomaly": false
    },
    "final_decision": "normal",
    "saved_path": "images/grid_05/20250131_120000.png"
  }
  ```

## トラブルシューティング

### YOLO モデルが見つからない

```bash
# YOLOモデルをダウンロード
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt -O models/yolo11n.pt
```

### PaDiM 学習がうまくいかない場合

- 学習用画像が少ない可能性があります
- `images/no_person`と`images/grid_XX`に十分な画像が蓄積されてから追加学習を実行してください

## 必要なライブラリ

主要な依存関係：

- `anomalib==2.0.0`: PaDiM 異常検知
- `opencv-python`: 画像処理・カメラ操作
- `ultralytics`: YOLO 物体検出（requirements.txt でコメントアウト済み）
- `torch`, `torchvision`: PyTorch バックエンド

詳細は`requirements.txt`を参照してください。
