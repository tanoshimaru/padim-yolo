# PaDiM + YOLO 異常検知システム

RTSP カメラから取得した映像を使って、YOLO による人物検出と PaDiM による異常検知を行うシステムです。

## クイックスタート

```bash
# 1. RTSPカメラの設定（.envファイルを作成）
cp .env.example .env  # または手動作成
# RTSP_IP=192.168.1.100 など実際の値に変更

# 2. システム起動
./run_container.sh

# 3. ログ確認
./run_container.sh logs

# 4. RTSP接続テスト
./run_container.sh test
```

## 機能概要

### 平日処理（月〜金）

1. **RTSP 撮影**: RTSP カメラから画像を撮影
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
├── run_container.sh          # コンテナ管理スクリプト
├── person_detector.py        # YOLO人検出モジュール
├── image_manager.py          # 画像管理モジュール
├── test_rtsp.py              # RTSP接続テスト用スクリプト
├── test_rtsp_simple.py       # シンプルRTSP接続テスト
├── .env                      # 環境変数設定（RTSP設定など）
├── Dockerfile                # Dockerコンテナ設定
├── docker-compose.yml        # Docker Compose設定
├── docker-start.sh           # コンテナ起動スクリプト
├── docker-crontab.txt        # コンテナ用cron設定
├── models/                   # モデル格納ディレクトリ
│   ├── yolo11n.pt           # YOLO11モデル（自動ダウンロード）
│   └── yolo11n.onnx
├── images/                   # 画像保存ディレクトリ
│   ├── no_person/           # 人が写っていない画像（最大200枚）
│   ├── grid_00/             # グリッド0の正常画像（最大200枚）
│   ├── grid_01/             # グリッド1の正常画像（最大200枚）
│   └── ...                  # grid_02 ~ grid_15
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

### 1. 簡単な起動・管理（推奨）

`run_container.sh` スクリプトを使用してコンテナを簡単に管理できます：

```bash
# コンテナ起動
./run_container.sh

# または明示的に
./run_container.sh start

# コンテナ停止
./run_container.sh stop

# コンテナ再起動
./run_container.sh restart

# ログ表示（リアルタイム）
./run_container.sh logs

# RTSP接続テスト
./run_container.sh test

# ヘルプ表示
./run_container.sh help
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
- 学習済みモデルは`models/padim_trained.ckpt`に保存

### RTSP カメラ設定

RTSP カメラの設定は`.env`ファイルで行います：

```bash
# .envファイルの例（プロジェクトルートに作成）
RTSP_USERNAME=admin
RTSP_PASSWORD=your_password
RTSP_IP=192.168.1.100
RTSP_PORT=554
```

#### RTSP 接続のテスト

```bash
# RTSP接続をテスト
./run_container.sh test
```

**注意**: RTSP カメラが利用できない場合、システムは自動的にテスト用のダミー画像を生成して処理を継続します。

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

### YOLO モデルの自動ダウンロード

初回実行時に`models/yolo11n.pt`が存在しない場合、自動的にダウンロードされます。手動でダウンロードする場合：

```bash
mkdir -p models
# モデルは初回実行時に自動ダウンロードされます
```

### RTSP 接続の問題

```bash
# RTSP接続をテスト
./run_container.sh test

# 詳細なテストの場合
docker compose exec app python test_rtsp_simple.py
```

### PaDiM 学習がうまくいかない場合

- 学習用画像が少ない可能性があります
- `images/no_person`と`images/grid_XX`に十分な画像が蓄積されてから追加学習を実行してください

### ログの確認

```bash
# リアルタイムでログを確認
./run_container.sh logs

# または個別ファイルを確認
tail -f logs/cron.log
tail -f logs/main_$(date +%Y%m%d).log
```

## 必要なライブラリ

主要な依存関係：

- `anomalib`: PaDiM 異常検知
- `opencv-python`: 画像処理・RTSP 接続
- `ultralytics`: YOLO 物体検出
- `torch`, `torchvision`: PyTorch バックエンド
- `python-dotenv`: 環境変数読み込み

詳細は`requirements.txt`を参照してください。

## 主な特徴

- **自動化**: cron による完全自動運用
- **RTSP 対応**: ネットワークカメラからの映像取得
- **フォールバック**: RTSP 接続失敗時のダミー画像生成
- **画像管理**: 自動的な古いファイルの削除（各フォルダ最大 200 枚）
- **モデル自動ダウンロード**: YOLO モデルの自動取得
- **簡単管理**: `run_container.sh`による統合管理
- **テスト機能**: RTSP 接続テスト機能内蔵
- **権限管理**: ホストユーザーの UID/GID でコンテナ実行（ファイル権限問題なし）

## Docker 環境での権限管理

このシステムは、ホストユーザーと同一の UID/GID で Docker コンテナを実行するため、ファイル権限の問題が発生しません。

### 自動設定の仕組み

```bash
# run_container.sh が自動的に設定
UID=$(id -u)        # ホストのUID
GID=$(id -g)        # ホストのGID
USERNAME=$(whoami)  # ホストのユーザー名
```

コンテナ内で同一の UID/GID を持つユーザーが作成され、ホスト側でファイルの編集・削除が可能です。

### 利点

- ✅ **ファイル権限問題なし**: ホストとコンテナ間でファイル編集が自由
- ✅ **セキュリティ**: root ではなく一般ユーザーで実行
- ✅ **自動設定**: 手動での UID/GID 指定は不要
- ✅ **ポータビリティ**: どの Linux 環境でも同じように動作

### 動作確認

起動時に以下のようなメッセージが表示されます：

```
=== RTSPカメラ異常検知システム起動 ===
ユーザー設定: USERNAME=tano, UID=1000, GID=1000
RTSP設定: rtsp://admin:***@192.168.1.100:554/profile2/media.smp
```
