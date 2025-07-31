#!/bin/bash

# 使用方法を表示する関数
show_usage() {
    echo "使用方法: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -u, --username USERNAME    コンテナ内のユーザー名を指定 (デフォルト: 現在のユーザー名)"
    echo "  -d, --daemon              バックグラウンドで実行"
    echo "  -b, --build               強制的に再ビルド"
    echo "  -s, --stop                コンテナを停止"
    echo "  -r, --restart             コンテナを再起動"
    echo "  -l, --logs                ログを表示"
    echo "  -t, --test-rtsp           RTSP接続をテスト"
    echo "  -h, --help                このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0                        # デフォルト設定で起動"
    echo "  $0 -u myuser -d           # 指定ユーザーでバックグラウンド起動"
    echo "  $0 --test-rtsp            # RTSP接続テスト"
    echo "  $0 --stop                 # コンテナ停止"
}

# デフォルト値設定
USERNAME=$(whoami)
DAEMON_MODE=false
FORCE_BUILD=false
ACTION="start"

# コマンドライン引数を解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            USERNAME="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        -b|--build)
            FORCE_BUILD=true
            shift
            ;;
        -s|--stop)
            ACTION="stop"
            shift
            ;;
        -r|--restart)
            ACTION="restart"
            shift
            ;;
        -l|--logs)
            ACTION="logs"
            shift
            ;;
        -t|--test-rtsp)
            ACTION="test-rtsp"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# .envファイルから設定を読み込み
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 環境変数を設定
export USERNAME=$USERNAME
export UID=$(id -u)
export GID=$(id -g)

# アクションに応じて実行
case $ACTION in
    "start")
        echo "=== RTSPカメラ異常検知システム起動 ==="
        echo "USERNAME: $USERNAME"
        echo "UID: $UID, GID: $GID"
        echo "RTSP設定: rtsp://${RTSP_USERNAME:-admin}:***@${RTSP_IP:-192.168.1.100}:${RTSP_PORT:-554}/profile2/media.smp"
        echo ""
        
        # ビルドオプション
        BUILD_ARGS=""
        if [ "$FORCE_BUILD" = true ]; then
            BUILD_ARGS="--build"
        fi
        
        # 実行モード
        if [ "$DAEMON_MODE" = true ]; then
            echo "バックグラウンドモードで起動中..."
            docker-compose up -d $BUILD_ARGS
            echo "コンテナがバックグラウンドで起動しました"
            echo "ログを確認: $0 --logs"
        else
            echo "フォアグラウンドモードで起動中..."
            docker-compose up $BUILD_ARGS
        fi
        ;;
        
    "stop")
        echo "コンテナを停止中..."
        docker-compose down
        echo "コンテナが停止しました"
        ;;
        
    "restart")
        echo "コンテナを再起動中..."
        docker-compose down
        docker-compose up -d --build
        echo "コンテナが再起動しました"
        ;;
        
    "logs")
        echo "=== コンテナログ ==="
        docker-compose logs -f
        ;;
        
    "test-rtsp")
        echo "=== RTSP接続テスト ==="
        docker-compose exec -T app python test_rtsp_simple.py
        ;;
esac