# padim-yolo

参考：[https://docs.ultralytics.com/ja/guides/nvidia-jetson/](https://docs.ultralytics.com/ja/guides/nvidia-jetson/)

## Usage

まず，人の写り込みの有無で振り分けたい画像を`images`フォルダに配置した後，

```bash
docker compose up -d --build
```

このコマンドにより：

- `images`フォルダ内の画像に対して人検出推論を実行
- 検出結果画像を`results`フォルダに保存
- 人が検出された画像を 16 分割グリッド位置に基づいて振り分け（`images/grid_00`〜`images/grid_15`）
- 人が検出されなかった画像を`images/no-person`に移動
- 振り分け結果を`results/distribution_result.txt`に保存
