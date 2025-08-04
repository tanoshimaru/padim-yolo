#!/bin/bash
# Docker コンテナ起動スクリプト

echo "=== PaDiM-YOLO コンテナ起動 ==="
echo "日時: $(date)"

# cronサービスを開始
echo "cronサービスを開始中..."
sudo service cron start

# cron設定を確認
echo "cron設定確認:"
crontab -l

# ログディレクトリを作成
mkdir -p /app/logs

# 初期化メッセージをログに記録
echo "[$(date)] PaDiM-YOLO コンテナが起動しました" >> /app/logs/cron.log

echo "=== 起動完了 ==="
echo "ログファイル: /app/logs/cron.log"
echo "平日8-17時に30秒間隔で実行されます"
echo "土曜9時に追加学習が実行されます"
echo ""

# コンテナを実行し続けるために無限ループ
while true; do
    sleep 3600  # 1時間ごとにチェック
    
    # cronサービスが停止していた場合は再起動
    if ! pgrep cron > /dev/null; then
        echo "[$(date)] cronサービスが停止していたため再起動します" >> /app/logs/cron.log
        sudo service cron start
    fi
    echo "自動実行停止中..."
done
