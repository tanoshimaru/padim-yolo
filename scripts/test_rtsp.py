#!/usr/bin/env python3
"""
RTSP接続テスト用スクリプト
"""

import cv2
import os
import sys
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()


def test_rtsp_connection():
    """RTSP接続をテスト"""

    # 環境変数から接続情報を取得
    username = os.getenv("RTSP_USERNAME", "admin")
    password = os.getenv("RTSP_PASSWORD", "password")
    ip = os.getenv("RTSP_IP", "192.168.1.100")
    port = os.getenv("RTSP_PORT", "554")

    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/profile2/media.smp"

    print(f"RTSP URL: {rtsp_url}")
    print("接続テスト開始...")

    # VideoCapture で接続テスト
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 接続設定
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    if not cap.isOpened():
        print("❌ RTSP接続に失敗しました")
        return False

    print("✅ RTSP接続成功")

    # フレーム取得テスト
    ret, frame = cap.read()

    if ret and frame is not None:
        print(f"✅ フレーム取得成功 - サイズ: {frame.shape}")

        # テスト画像を保存
        cv2.imwrite("rtsp_test_frame.jpg", frame)
        print("📸 テスト画像を rtsp_test_frame.jpg に保存しました")

        cap.release()
        return True
    else:
        print("❌ フレーム取得に失敗しました")
        cap.release()
        return False


if __name__ == "__main__":
    success = test_rtsp_connection()
    sys.exit(0 if success else 1)
