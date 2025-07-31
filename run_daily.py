#!/usr/bin/env python3
"""
日次実行スクリプト

平日: main.py を実行（撮影・YOLO検出・PaDiM異常検知）
土曜: train_additional.py を実行（追加学習）
日曜: 休止
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path


def get_day_type() -> str:
    """曜日の種類を取得

    Returns:
        str: 'weekday', 'saturday', 'sunday'
    """
    today = datetime.now()
    weekday = today.weekday()  # 0=月曜, 6=日曜

    if weekday == 5:  # 土曜日
        return "saturday"
    elif weekday == 6:  # 日曜日
        return "sunday"
    else:  # 平日（月〜金）
        return "weekday"


def run_command(command: list, description: str) -> bool:
    """コマンドを実行

    Args:
        command: 実行するコマンドのリスト
        description: コマンドの説明

    Returns:
        bool: 実行成功可否
    """
    print(f"=== {description} ===")
    print(f"コマンド: {' '.join(command)}")

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        # 標準出力を表示
        if result.stdout:
            print("--- 標準出力 ---")
            print(result.stdout)

        # エラー出力を表示
        if result.stderr:
            print("--- エラー出力 ---")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} 完了")
            return True
        else:
            print(f"❌ {description} 失敗 (終了コード: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ {description} 実行中にエラー: {e}")
        return False


def main():
    """メイン関数"""
    print(f"=== 日次処理開始 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")

    day_type = get_day_type()
    print(f"今日のタイプ: {day_type}")

    if day_type == "weekday":
        # 平日: メイン処理
        print("\n📷 平日処理を実行します")
        command = ["python", "main.py"]
        success = run_command(command, "平日処理（撮影・YOLO検出・PaDiM異常検知）")

        if success:
            print("\n✅ 平日処理が正常に完了しました")
            return 0
        else:
            print("\n❌ 平日処理でエラーが発生しました")
            return 1

    elif day_type == "saturday":
        # 土曜日: 追加学習
        print("\n🎓 土曜日処理（追加学習）を実行します")
        command = ["python", "train_additional.py"]
        success = run_command(command, "追加学習処理")

        if success:
            print("\n✅ 追加学習が正常に完了しました")
            return 0
        else:
            print("\n❌ 追加学習でエラーが発生しました")
            return 1

    elif day_type == "sunday":
        # 日曜日: 休止
        print("\n😴 日曜日のため処理を休止します")
        return 0

    else:
        print(f"\n❓ 不明な曜日タイプ: {day_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
