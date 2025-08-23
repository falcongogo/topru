#!/usr/bin/env python3
"""
実際のスリムスコア28S画像をテストするスクリプト
"""

import cv2
import numpy as np
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processor import ScoreImageProcessor

def test_real_image(image_path: str):
    """実際の画像をテストし、デバッグ情報を出力する"""
    print(f"画像ファイル: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ ファイルが見つかりません: {image_path}")
        return False
    
    try:
        processor = ScoreImageProcessor()
        print("✅ 画像処理クラスを初期化しました")
    except Exception as e:
        print(f"❌ 画像処理クラスの初期化に失敗: {e}")
        return False
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 画像を読み込めませんでした: {image_path}")
            return False
        print(f"✅ 画像を読み込みました (サイズ: {image.shape})")
    except Exception as e:
        print(f"❌ 画像読み込みエラー: {e}")
        return False
    
    try:
        print("\n--- 画像処理実行中 (デバッグモード) ---")
        result = processor.process_image(image, debug=True, shear_correction_method='manual', manual_shear_angle= -13.0)
        if not result:
            print("❌ 画像処理に失敗しました。")
            return False

        scores = result.get('scores', {})
        debug_bundle = result.get('debug_bundle', {})

        print("✅ 画像処理が完了しました。")
        
        # 認識された点数を表示
        if scores:
            print("\n--- 認識された点数 ---")
            for player, score in scores.items():
                print(f"  {player}: {score}点")
        else:
            print("\n--- 認識された点数 ---")
            print("［警告］ 点数を読み取れませんでした。")

        # 主要なデバッグ画像を保存
        if not debug_bundle:
            print("\n--- デバッグ情報 ---")
            print("［警告］ デバッグバンドルが生成されませんでした。")
            return True

        print("\n--- デバッグ画像の保存 ---")
        base_name, ext = os.path.splitext(image_path)
        os.makedirs("debug", exist_ok=True) # Ensure debug directory exists

        images_to_save = {
            'warped_screen': '1_warped_screen',
            'shear_corrected_screen': '2_shear_corrected',
        }

        for key, name_suffix in images_to_save.items():
            if key in debug_bundle and debug_bundle[key] is not None and debug_bundle[key].size > 0:
                debug_path = os.path.join("debug", f"{os.path.basename(base_name)}_{name_suffix}{ext}")
                cv2.imwrite(debug_path, debug_bundle[key])
                print(f"✅ 「{key}」を保存しました: {debug_path}")
            else:
                print(f"ℹ️ 「{key}」は利用できませんでした。")

        # Player name mapping for ASCII-safe filenames
        player_name_map = {
            '上家': 'kamicha',
            '対面': 'toimen',
            '自家': 'jicha',
            '下家': 'shimocha'
        }

        # 4分割された領域を保存
        if 'split_region_images' in debug_bundle and debug_bundle['split_region_images']:
            print("\n--- 4分割領域の保存 ---")
            for player, region_image in debug_bundle['split_region_images'].items():
                 if region_image is not None and region_image.size > 0:
                    player_en = player_name_map.get(player, 'unknown')
                    debug_path = os.path.join("debug", f"{os.path.basename(base_name)}_3_split_{player_en}{ext}")
                    cv2.imwrite(debug_path, region_image)
                    print(f"✅ 「{player}」領域を保存しました: {debug_path}")

        # 認識された数字画像を保存
        if 'split_digits_by_player' in debug_bundle and debug_bundle['split_digits_by_player']:
            print("\n--- 認識された数字の保存 ---")
            for player, digits in debug_bundle['split_digits_by_player'].items():
                if digits:
                    player_en = player_name_map.get(player, 'unknown')
                    player_digits_img = np.hstack(digits)
                    debug_path = os.path.join("debug", f"{os.path.basename(base_name)}_4_digits_{player_en}{ext}")
                    cv2.imwrite(debug_path, player_digits_img)
                    print(f"✅ 「{player}」の数字を保存しました: {debug_path}")

    except Exception as e:
        print(f"❌ デバッグ処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    """メイン関数"""
    if len(sys.argv) != 2:
        print("使用方法: python test_real_image.py <画像ファイルパス>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 60)
    print("スリムスコア28S画像テスト開始")
    print("=" * 60)
    
    success = test_real_image(image_path)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ テストが正常に完了しました")
    else:
        print("❌ テスト中にエラーが発生しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
