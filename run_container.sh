#!/bin/bash

# RTSPカメラ設定を.envファイルから読み込み
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "RTSPカメラ異常検知システムを開始します..."
echo "RTSP設定: rtsp://${RTSP_USERNAME}:***@${RTSP_IP}:${RTSP_PORT}/profile2/media.smp"

# Dockerコンテナでアプリケーションを実行
docker compose up --build