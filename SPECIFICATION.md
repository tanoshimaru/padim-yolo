# anomalib を用いて PaDiM 異常検知

- 平日：撮影，推論
  1. 撮影
  2. 画像に人が写っているかを YOLO で検出
     - 人が写っていない場合：撮影した画像を images/no_person に保存
     - 人が写っている場合：写っている区画（grid_00~grid_15）を取得，撮影した画像を PaDiM で異常検知
       - 異常が人が写っている区画で検出された場合：結果を異常とする
       - 異常が人が写っていない区画で検出された場合：結果を正常とし，撮影した画像を images/grid\_<grid_number>に保存
       - 異常が検出されなかった場合：撮影した画像を images/grid\_<grid_number>に保存
  3. YOLO の結果と PaDiM の結果と画像の保存先を記録
  - 画像保存枚数は各フォルダごとに 200 枚までで，200 枚を超えた場合は古い画像から削除，画像名は YYYYMMDD_HHMMSS.png の形式
- 土曜日：追加学習
  - images/no_person, images/grid_00~grid_15 の画像を用いて PaDiM の追加学習を行う
  - 追加学習の結果は images/models に保存
