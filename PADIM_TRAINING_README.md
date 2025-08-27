# PaDiM 学習スクリプトの使用方法

PaDiM 異常検知モデルの学習用スクリプトを作成しました。以下の手順で学習を実行できます。

## 作成されたファイル

1. **`scripts/train_padim.py`** - メインの学習スクリプト
2. **`scripts/prepare_dataset.py`** - データセット準備スクリプト  
3. **`scripts/distribute_images.py`** - 画像分散スクリプト
4. **`main.py`** - 学習済みモデル対応に修正済み
5. **`train_additional.py`** - 現在保留中（Jetson学習の課題）

## 使用手順

### Step 1: 画像の分類

まず`images/checked`内の画像をprefixに応じて分類：

```bash
# 画像を分類してディレクトリに分配
uv run classify_images.py
```

### Step 2: 学習データの準備

分類された`images/`ディレクトリから学習用データセットを準備：

```bash
# 基本的な準備（コピーモード）
python scripts/prepare_dataset.py

# 詳細オプション付き
python scripts/prepare_dataset.py \
    --source_dir ./images \
    --target_dir ./training_data \
    --normal_ratio 0.8 \
    --random_seed 42
```

**オプション説明:**

- `--source_dir`: ソース画像ディレクトリ（デフォルト: `./images`）
- `--target_dir`: 出力ディレクトリ（デフォルト: `./training_data`）
- `--normal_ratio`: 学習用データの比率（デフォルト: 0.8）
- `--move`: ファイルを移動（デフォルトはコピー）
- `--clean`: 既存の出力ディレクトリをクリーン
- `--random_seed`: ランダムシード（デフォルト: 42）

### Step 3: PaDiM モデルの学習

```bash
# 基本的な学習
python scripts/train_padim.py

# 詳細オプション付き
python scripts/train_padim.py \
    --data_root ./training_data \
    --model_save_path ./models/padim_trained.ckpt \
    --batch_size 32 \
    --max_epochs 10 \
    --backbone resnet18
```

**オプション説明:**

- `--data_root`: 学習データディレクトリ（デフォルト: `./training_data`）
- `--normal_dir`: 正常画像ディレクトリ名（デフォルト: `normal`）
- `--abnormal_dir`: 異常画像ディレクトリ名（デフォルト: `abnormal`）
- `--model_save_path`: モデル保存パス（デフォルト: `./models/padim_trained.ckpt`）
- `--image_size`: 入力画像サイズ（デフォルト: 256 256）
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--max_epochs`: 最大エポック数（デフォルト: 1）
- `--backbone`: バックボーンモデル（resnet18, resnet34, resnet50, wide_resnet50_2）

### Step 4: サンプルデータセット構造の作成（オプション）

```bash
# サンプル構造を作成
python scripts/train_padim.py --create_sample
```

## データセット構造

学習スクリプトは以下のデータ構造を期待します：

```
training_data/
├── normal/          # 正常画像（必須）
│   ├── image_001.png  # images/no_person, images/grid_XX から自動収集
│   ├── image_002.png
│   └── ...
└── abnormal/        # 異常画像（オプション）
    ├── anomaly_001.png  # images/defect から自動収集
    ├── anomaly_002.png
    └── ...
```

**注意**: `images/defect`フォルダに異常画像が蓄積されてから学習を実行することで、より精度の高いモデルを作成できます。

## 学習されたモデルの使用

学習が完了すると、`models/padim_trained.ckpt`にモデルが保存されます。
`main.py`は自動的にこのモデルを読み込んで使用します。

```bash
# 学習済みモデルを使用して異常検知を実行
python main.py
```

## ログとモニタリング

- 学習ログ: `logs/train_padim_YYYYMMDD_HHMMSS.log`
- データ準備ログ: `logs/prepare_training_data_YYYYMMDD_HHMMSS.log`
- TensorBoard ログ: `lightning_logs/`（学習中に自動生成）

```bash
# TensorBoardでモニタリング
tensorboard --logdir lightning_logs
```

## 注意事項

1. **メモリ使用量**: バッチサイズが大きいとメモリ不足になる場合があります
2. **GPU 使用**: 自動的に GPU が検出・使用されます
3. **画像形式**: PNG、JPG 形式をサポート
4. **最小データ量**: 正常画像は最低でも数十枚以上推奨

## トラブルシューティング

### エラー: "CUDA out of memory"

```bash
# バッチサイズを小さくして再実行
python scripts/train_padim.py --batch_size 16
```

### エラー: "No normal images found"

```bash
# データ準備スクリプトを再実行
python scripts/prepare_dataset.py --clean
```

### 学習が進まない場合

```bash
# エポック数を増やして再実行
python scripts/train_padim.py --max_epochs 10
```

## 高度な使用方法

### カスタムデータセットで学習

```bash
# 独自のデータセット構造で学習
python scripts/train_padim.py \
    --data_root /path/to/custom/dataset \
    --normal_dir good \
    --abnormal_dir defect
```

### 異なるバックボーンで学習

```bash
# より強力なバックボーンを使用
python scripts/train_padim.py --backbone resnet50 --batch_size 16
```

### 複数エポックで精度向上

```bash
# より長時間学習して精度を向上
python scripts/train_padim.py --max_epochs 20 --batch_size 16
```
