#!/usr/bin/env python3
"""
RTSP接続テスト用スクリプト（依存関係最小版）
"""

import os
import subprocess
import sys

# .envファイルを手動で読み込む関数
def load_env_file(env_path=".env"):
    """手動で.envファイルを読み込む"""
    if not os.path.exists(env_path):
        print(f"⚠️  .envファイルが見つかりません: {env_path}")
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

def test_rtsp_with_ffmpeg():
    """FFmpegを使ったRTSP接続テスト"""
    
    # 環境変数から接続情報を取得
    username = os.getenv("RTSP_USERNAME", "admin")
    password = os.getenv("RTSP_PASSWORD", "password")
    ip = os.getenv("RTSP_IP", "192.168.1.100")
    port = os.getenv("RTSP_PORT", "554")
    
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/profile2/media.smp"
    
    print(f"RTSP URL: {rtsp_url}")
    print("FFmpegでRTSP接続テスト中...")
    
    try:
        # FFmpegで5秒間テスト
        cmd = [
            "ffmpeg", 
            "-i", rtsp_url,
            "-t", "5",
            "-f", "null",
            "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ RTSP接続成功")
            return True
        else:
            print("❌ RTSP接続失敗")
            print(f"エラー: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ RTSP接続タイムアウト")
        return False
    except FileNotFoundError:
        print("❌ FFmpegが見つかりません")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_network_connectivity():
    """ネットワーク接続テスト"""
    ip = os.getenv("RTSP_IP", "192.168.1.100")
    port = os.getenv("RTSP_PORT", "554")
    
    print(f"\n=== ネットワークテスト ===")
    
    # Ping テスト
    try:
        result = subprocess.run(["ping", "-c", "3", ip], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ping {ip} 成功")
        else:
            print(f"❌ Ping {ip} 失敗")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"❌ Ping {ip} エラー: {e}")
    
    # ポート接続テスト（netcatまたはtelnet）
    try:
        # netcat を試す
        result = subprocess.run(["nc", "-z", "-v", ip, port], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ ポート {port} 接続可能")
        else:
            print(f"❌ ポート {port} 接続不可")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        print("⚠️ ポートテストをスキップ（netcat未インストール）")

if __name__ == "__main__":
    print("=== RTSP接続診断ツール ===")
    
    # .envファイルを読み込み
    load_env_file()
    
    # 環境変数の確認
    print(f"RTSP_USERNAME: {os.getenv('RTSP_USERNAME', 'admin')}")
    print(f"RTSP_IP: {os.getenv('RTSP_IP', '192.168.1.100')}")
    print(f"RTSP_PORT: {os.getenv('RTSP_PORT', '554')}")
    
    # ネットワークテスト
    test_network_connectivity()
    
    # RTSPテスト
    success = test_rtsp_with_ffmpeg()
    
    sys.exit(0 if success else 1)