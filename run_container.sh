#!/bin/bash

# 使用方法を表示する関数
show_usage() {
    echo "使用方法: $0 [ACTION]"
    echo ""
    echo "ACTIONS:"
    echo "  start     コンテナを起動 (デフォルト)"
    echo "  stop      コンテナを停止"
    echo "  restart   コンテナを再起動"
    echo "  logs      ログを表示"
    echo "  test      RTSP接続をテスト"
    echo "  help      このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0              # コンテナ起動"
    echo "  $0 start        # コンテナ起動"
    echo "  $0 stop         # コンテナ停止"
    echo "  $0 logs         # ログ表示"
    echo "  $0 test         # RTSP接続テスト"
}

# デフォルトアクション
ACTION=${1:-start}

# .envファイルから設定を読み込み
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# ホストのUID/GIDを設定
export UID=$(id -u)
export GID=$(id -g)
export USERNAME=$(whoami)

# アクションに応じて実行
case $ACTION in
    "start")
        echo "=== RTSPカメラ異常検知システム起動 ==="
        echo "ユーザー設定: USERNAME=$USERNAME, UID=$UID, GID=$GID"
        echo "RTSP設定: rtsp://${RTSP_USERNAME:-admin}:***@${RTSP_IP:-192.168.1.100}:${RTSP_PORT:-554}/profile2/media.smp"
        echo ""
        echo "バックグラウンドモードで起動中..."
        docker compose up -d --build
        echo "✅ コンテナが起動しました"
        echo "📋 ログを確認: $0 logs"
        ;;
        
    "stop")
        echo "コンテナを停止中..."
        docker compose down
        echo "✅ コンテナが停止しました"
        ;;
        
    "restart")
        echo "コンテナを再起動中..."
        docker compose down
        docker compose up -d --build
        echo "✅ コンテナが再起動しました"
        ;;
        
    "logs")
        echo "=== コンテナログ ==="
        docker compose logs -f
        ;;
        
    "test")
        echo "=== RTSP接続テスト ==="
        if ! docker compose ps | grep -q "Up"; then
            echo "⚠️  コンテナが起動していません。まず起動してください: $0 start"
            exit 1
        fi
        docker compose exec app python test_rtsp_simple.py
        ;;
        
    "help"|"-h"|"--help")
        show_usage
        ;;
        
    *)
        echo "❌ 不明なアクション: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac