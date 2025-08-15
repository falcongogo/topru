#!/usr/bin/env python3
"""
実際のスリムスコア28S画像をテストするスクリプト
"""

import cv2
import numpy as np
from image_processor import ScoreImageProcessor
import os
import sys

def test_real_image(image_path: str):
    """実際の画像をテスト"""
    print(f"画像ファイル: {image_path}")
    
    # ファイルの存在確認
    if not os.path.exists(image_path):
        print(f"❌ ファイルが見つかりません: {image_path}")
        return False
    
    # 画像処理クラスの初期化
    try:
        processor = ScoreImageProcessor()
        print("✅ 画像処理クラスを初期化しました")
    except Exception as e:
        print(f"❌ 画像処理クラスの初期化に失敗: {e}")
        return False
    
    # 画像の読み込み
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 画像を読み込めませんでした: {image_path}")
            return False
        
        print(f"✅ 画像を読み込みました (サイズ: {image.shape})")
    except Exception as e:
        print(f"❌ 画像読み込みエラー: {e}")
        return False
    
    # 前処理のテスト
    try:
        processed_image = processor.preprocess_image(image)
        print("✅ 画像の前処理が完了しました")
        
        # 前処理結果を保存
        processed_path = image_path.replace('.', '_processed.')
        cv2.imwrite(processed_path, processed_image)
        print(f"前処理結果を保存: {processed_path}")
    except Exception as e:
        print(f"❌ 前処理エラー: {e}")
        return False
    
    # 文字領域検出のテスト
    try:
        text_regions = processor.detect_text_regions(processed_image)
        print(f"✅ 文字領域を検出しました (検出数: {len(text_regions)})")

        # 検出された領域を表示
        for i, (x1, y1, x2, y2) in enumerate(text_regions):
            print(f"  領域{i+1}: ({x1}, {y1}) - ({x2}, {y2}) [幅: {x2-x1}, 高さ: {y2-y1}]")
    except Exception as e:
        print(f"❌ 文字領域検出エラー: {e}")
        return False

    # 重複領域マージのテスト
    try:
        merged_regions = processor.merge_overlapping_regions(text_regions)
        print(f"✅ 重複領域をマージしました (マージ後: {len(merged_regions)}個)")

        for i, (x1, y1, x2, y2) in enumerate(merged_regions):
            print(f"  マージ後領域{i+1}: ({x1}, {y1}) - ({x2}, {y2})")
    except Exception as e:
        print(f"❌ 重複領域マージエラー: {e}")
        return False

    # 点数表示領域検出のテスト
    try:
        score_regions = processor.detect_score_regions(image)
        print(f"✅ 点数表示領域を検出しました (検出数: {len(score_regions)})")
        
        for player, (x1, y1, x2, y2) in score_regions.items():
            print(f"  {player}: ({x1}, {y1}) - ({x2}, {y2})")
    except Exception as e:
        print(f"❌ 点数表示領域検出エラー: {e}")
        return False
    
    # 点数読み取りのテスト
    try:
        scores = processor.process_score_image(image_path)
        print(f"✅ 点数読み取りが完了しました")
        
        if scores:
            print("読み取られた点数:")
            for player, score in scores.items():
                print(f"  {player}: {score}点")
        else:
            print("⚠️ 点数を読み取れませんでした")
    except Exception as e:
        print(f"❌ 点数読み取りエラー: {e}")
        return False
    
    # デバッグ画像の生成
    try:
        debug_image, regions = processor.debug_detection(image)
        debug_path = image_path.replace('.', '_debug.')
        cv2.imwrite(debug_path, debug_image)
        print(f"✅ デバッグ画像を保存しました: {debug_path}")
    except Exception as e:
        print(f"❌ デバッグ画像生成エラー: {e}")
        return False
    
    return True

def main():
    """メイン関数"""
    if len(sys.argv) != 2:
        print("使用方法: python test_real_image.py <画像ファイルパス>")
        print("例: python test_real_image.py slimscore28s.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 60)
    print("スリムスコア28S画像テスト開始")
    print("=" * 60)
    
    success = test_real_image(image_path)
    
    print("=" * 60)
    if success:
        print("✅ テストが正常に完了しました")
    else:
        print("❌ テスト中にエラーが発生しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
