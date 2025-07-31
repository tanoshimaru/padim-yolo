#!/bin/bash

# ホストユーザーのUID/GIDを取得してDockerコンテナを起動
export UID=$(id -u)
export GID=$(id -g)

echo "UID: $UID, GID: $GID でコンテナを起動します"

# 既存のコンテナを停止・削除
docker-compose down

# 新しいコンテナを起動
docker-compose up -d

echo "コンテナが起動しました"