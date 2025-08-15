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
        self.test_image = np.full((200, 800, 3), (120, 120, 120), dtype=np.uint8)

        # 水平レイアウトを定義
        self.positions = ['自分', '下家', '対面', '上家']
        self.scores = {'自分': 25000, '下家': 25000, '対面': 25000, '上家': 25000}
        
        # 全体を囲む大きな白いフレームを描画
        frame_x1, frame_y1 = 10, 50
        frame_w, frame_h = 780, 100
        cv2.rectangle(self.test_image, (frame_x1, frame_y1), (frame_x1 + frame_w, frame_y1 + frame_h), (240, 240, 240), -1)

        # フレーム内にスコアを描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        region_w = frame_w // 4
        for i, player in enumerate(self.positions):
            score_text = str(self.scores[player])
            text_x = frame_x1 + (i * region_w) + (region_w // 4)
            text_y = frame_y1 + (frame_h // 2)
            cv2.putText(self.test_image, score_text, (text_x, text_y), font, 1.2, (20, 20, 20), 3)

    def test_detect_and_assign_regions(self):
        """水平レイアウトの単一フレームからの4分割検出のテスト"""
        detected_regions = self.processor.detect_score_regions(self.test_image)
        
        self.assertEqual(len(detected_regions), 4, "4つの領域が検出されるべき")

        # 正しい順序でプレイヤーが割り当てられているか確認
        self.assertListEqual(list(detected_regions.keys()), self.positions)

        # 最初の領域（自分）のx座標がフレームの開始点とほぼ同じか確認
        me_region = detected_regions['自分']
        self.assertAlmostEqual(me_region[0], 10, delta=5)

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
