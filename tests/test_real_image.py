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
        print("\n--- デバッグバンドル生成中 ---")
        debug_bundle = processor.get_full_debug_bundle(image, shear_correction_method='zeros')
        if not debug_bundle:
            print("❌ デバッグバンドルの生成に失敗しました。")
            return False
        print("✅ デバッグバンドルを生成しました。")
        
        # 認識された点数を表示
        if 'scores' in debug_bundle and debug_bundle['scores']:
            print("\n--- 認識された点数 ---")
            for player, score in debug_bundle['scores'].items():
                print(f"  {player}: {score}点")
        else:
            print("\n--- 認識された点数 ---")
            print("［警告］ 点数を読み取れませんでした。")

        # 主要なデバッグ画像を保存
        print("\n--- デバッグ画像の保存 ---")
        base_name, ext = os.path.splitext(image_path)

        images_to_save = {
            'warped_screen': '1_warped_screen',
            'shear_corrected_screen': '2_shear_corrected',
        }

        for key, name_suffix in images_to_save.items():
            if key in debug_bundle and debug_bundle[key] is not None and debug_bundle[key].size > 0:
                debug_path = f"{base_name}_{name_suffix}{ext}"
                cv2.imwrite(debug_path, debug_bundle[key])
                print(f"✅ 「{key}」を保存しました: {debug_path}")
            else:
                print(f"ℹ️ 「{key}」は利用できませんでした。")

        # 4分割された領域を保存
        if 'split_region_images' in debug_bundle and debug_bundle['split_region_images']:
            print("\n--- 4分割領域の保存 ---")
            for player, region_image in debug_bundle['split_region_images'].items():
                 if region_image is not None and region_image.size > 0:
                    debug_path = f"{base_name}_3_split_{player}{ext}"
                    cv2.imwrite(debug_path, region_image)
                    print(f"✅ 「{player}」領域を保存しました: {debug_path}")

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
