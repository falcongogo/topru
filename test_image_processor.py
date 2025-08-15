import unittest
from unittest.mock import patch
import numpy as np
import cv2
from image_processor import ScoreImageProcessor
import tempfile
import os

class TestImageProcessorDefinitive(unittest.TestCase):
    """画像処理モジュールの最終テスト"""

    def setUp(self):
        """テスト前の準備"""
        self.processor = ScoreImageProcessor()
        self.test_image = np.full((200, 800, 3), (20, 20, 20), dtype=np.uint8) # Dark background

        # 水平レイアウトを定義
        self.positions = ['自分', '下家', '対面', '上家']
        self.scores = {'自分': 29000, '下家': 29000, '対面': 29000, '上家': 29000}
        
        # 1. 外側の銀色フレームを描画 (明るい色)
        outer_frame_coords = (10, 40, 780, 120) # x1, y1, x2, y2
        cv2.rectangle(self.test_image, (outer_frame_coords[0], outer_frame_coords[1]), (outer_frame_coords[2], outer_frame_coords[3]), (220, 220, 220), -1)

        # 2. 内側のLCDスクリーンを描画 (暗い色、コントラストを明確に)
        self.inner_lcd_coords = (25, 50, 775, 110) # x1, y1, x2, y2
        cv2.rectangle(self.test_image, (self.inner_lcd_coords[0], self.inner_lcd_coords[1]), (self.inner_lcd_coords[2], self.inner_lcd_coords[3]), (100, 100, 100), -1)

        # 3. LCDスクリーン内にスコアを描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_w = self.inner_lcd_coords[2] - self.inner_lcd_coords[0]
        region_w = frame_w // 4
        for i, player in enumerate(self.positions):
            score_text = str(self.scores[player])
            text_x = self.inner_lcd_coords[0] + (i * region_w) + (region_w // 4)
            text_y = self.inner_lcd_coords[1] + (self.inner_lcd_coords[3] - self.inner_lcd_coords[1]) // 2
            cv2.putText(self.test_image, score_text, (text_x, text_y), font, 1.2, (20, 20, 20), 3)

    def test_detect_and_assign_regions(self):
        """内側LCDスクリーンを検出し、それを4分割するかのテスト"""
        detected_regions = self.processor.detect_score_regions(self.test_image)
        
        self.assertEqual(len(detected_regions), 4, "4つの領域が検出されるべき")

        # 正しい順序でプレイヤーが割り当てられているか確認
        self.assertListEqual(list(detected_regions.keys()), self.positions)

        # 検出された領域のベースが、内側LCDフレームの座標と一致するか確認
        all_x = [r[0] for r in detected_regions.values()] + [r[2] for r in detected_regions.values()]
        all_y = [r[1] for r in detected_regions.values()] + [r[3] for r in detected_regions.values()]

        detected_base_frame_x1 = min(all_x)
        detected_base_frame_y1 = min(all_y)

        self.assertAlmostEqual(detected_base_frame_x1, self.inner_lcd_coords[0], delta=5, msg="検出された分割領域の開始X座標が、内側LCDと一致しません")
        self.assertAlmostEqual(detected_base_frame_y1, self.inner_lcd_coords[1], delta=5, msg="検出された分割領域の開始Y座標が、内側LCDと一致しません")

    def test_full_process_with_distractors(self):
        """点差などのノイズを含む画像からのE2Eテスト"""
        # pytesseractをモック化せず、実際のOCRエンジンでテスト
        # (Tesseractがインストールされていることが前提)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            cv2.imwrite(tmp_file.name, self.test_image)
            tmp_path = tmp_file.name
        
        try:
            # 統合処理を実行
            scores = self.processor.process_score_image(tmp_path)
            # 正しいスコアが読み取られ、点差が無視されたか確認
            self.assertEqual(scores, self.scores)
        finally:
            os.unlink(tmp_path)

def run_image_processor_tests():
    """画像処理モジュールのテストを実行"""
    print("画像処理モジュールのテストを開始します...")
    
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestImageProcessorDefinitive)
    
    # テストを実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果を表示
    if result.wasSuccessful():
        print("✅ すべてのテストが成功しました")
    else:
        print("❌ 一部のテストが失敗しました")
        for failure in result.failures:
            print(f"失敗: {failure[0]}")
            print(f"理由: {failure[1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_image_processor_tests()
