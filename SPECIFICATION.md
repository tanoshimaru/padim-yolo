# anomalib を用いて PaDiM 異常検知

- 平日：撮影，推論，画像保存
  1. 撮影
  2. 画像に人が写っているかを YOLO で検出
     - 人が写っていない場合：撮影した画像を images/checked に `no_person_` prefix で保存
     - 人が写っている場合：写っている区画（grid_00~grid_15）を取得，撮影した画像を PaDiM で異常検知
       - 異常が検出された場合：結果を異常とし，撮影した画像を images/checked に `defect_grid_<grid_number>_` prefix で保存
       - 異常が検出されなかった場合：撮影した画像を images/checked に `grid_<grid_number>_` prefix で保存
       - エラーが発生した場合：撮影した画像を images/checked に `error_` prefix で保存
  3. YOLO の結果と PaDiM の結果と画像の保存先を記録
  4. 画像分類処理（classify_images.py）
     - `defect_grid_*` → images/defect/ にprefixを削除して保存
     - `grid_XX_*` → images/grid_XX/ にprefixを削除して保存  
     - `no_person_*` → images/no_person/ にprefixを削除して保存
     - `error_*` → 削除
  - 画像保存枚数は各分類フォルダごとに 200 枚までで，200 枚を超えた場合は古い画像から削除，画像名は YYYYMMDD_HHMMSS.png の形式
- 土曜日：追加学習
  - 現在の実装では `train_additional.py` は保留中（Jetson上での学習が困難）
  - 代替として `scripts/train_padim.py` を使用した手動学習が可能
  - 学習時は images/no_person, images/grid_00~grid_15, images/defect の画像を活用
  - images/defect の画像は異常サンプルとして使用され、より精度の高い異常検知モデルの構築に活用
  - 学習結果は models に保存
