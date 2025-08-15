import unittest
from unittest.mock import patch
import numpy as np
import cv2
from image_processor import ScoreImageProcessor
import tempfile
import os

class TestImageProcessor(unittest.TestCase):
    """画像処理モジュールのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.processor = ScoreImageProcessor()
        
        # グレー背景に大きな白いスコア枠を持つテスト画像を作成
        self.test_image = np.full((400, 800, 3), (128, 128, 128), dtype=np.uint8)
        self.main_frame = (50, 100, 750, 200)  # x1, y1, x2, y2

        # 白い枠を描画
        cv2.rectangle(self.test_image, (self.main_frame[0], self.main_frame[1]), (self.main_frame[2], self.main_frame[3]), (255, 255, 255), -1)

        self.scores = {
            '自分': 28000,
            '下家': 35000,
            '対面': 30000,
            '上家': 27000,
        }
        
        # 4分割された各領域に点数を描画
        x1, y1, x2, y2 = self.main_frame
        w = x2 - x1
        h = y2 - y1
        region_w = w // 4

        font = cv2.FONT_HERSHEY_SIMPLEX
        positions = ['自分', '下家', '対面', '上家']
        for i, player in enumerate(positions):
            text = str(self.scores[player])
            region_x_start = x1 + i * region_w

            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            text_x = region_x_start + (region_w - text_size[0]) // 2
            text_y = y1 + (h + text_size[1]) // 2
            cv2.putText(self.test_image, text, (text_x, text_y), font, 0.8, (0, 0, 0), 2)

        # 期待される分割後の領域を計算
        self.expected_regions = {}
        for i, player in enumerate(positions):
            rx1 = x1 + i * region_w
            ry1 = y1
            rx2 = x1 + (i + 1) * region_w
            ry2 = y2
            if i == 3: # last region
                rx2 = x2
            self.expected_regions[player] = (rx1, ry1, rx2, ry2)

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.min_score, 1000)
        self.assertEqual(self.processor.max_score, 99999)
        self.assertEqual(self.processor.expected_digits, 5)
    
    def test_preprocess_image(self):
        """画像前処理テスト"""
        processed = self.processor.preprocess_image(self.test_image)
        self.assertEqual(len(processed.shape), 2)  # グレースケール
        self.assertEqual(processed.dtype, np.uint8)
    
    def test_extract_score_region(self):
        """点数表示領域の切り出しテスト"""
        region = self.expected_regions['自分']
        extracted = self.processor.extract_score_region(self.test_image, region)
        self.assertIsNotNone(extracted)
        h = self.main_frame[3] - self.main_frame[1]
        w = (self.main_frame[2] - self.main_frame[0]) // 4
        self.assertEqual(extracted.shape, (h, w, 3))

    def test_detect_score_regions(self):
        """点数表示領域の検出と4分割のテスト"""
        regions = self.processor.detect_score_regions(self.test_image)
        
        self.assertEqual(len(regions), 4)
        
        # 検出された領域が期待される領域と一致するかチェック
        positions = ['自分', '下家', '対面', '上家']
        for player in positions:
            self.assertIn(player, regions)
            detected = regions[player]
            expected = self.expected_regions[player]
            self.assertTrue(np.allclose(detected, expected, atol=5),
                            msg=f"Player {player} - Detected: {detected}, Expected: {expected}")

    def test_is_valid_score(self):
        """点数妥当性チェックテスト"""
        self.assertTrue(self.processor._is_valid_score(28000))
        self.assertFalse(self.processor._is_valid_score(999))

    @patch('image_processor.pytesseract.image_to_string')
    def test_process_score_image(self, mock_image_to_string):
        """画像処理の統合テスト"""
        # detect_score_regionsが返す辞書の順序に依存
        mock_image_to_string.side_effect = [
            str(self.scores['自分']),
            str(self.scores['下家']),
            str(self.scores['対面']),
            str(self.scores['上家']),
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, self.test_image)
            tmp_path = tmp_file.name
        
        try:
            scores = self.processor.process_score_image(tmp_path)
            self.assertEqual(len(scores), 4)
            self.assertEqual(scores, self.scores)
        finally:
            os.unlink(tmp_path)
    
    @patch('image_processor.pytesseract.image_to_string')
    def test_read_score_from_region(self, mock_image_to_string):
        """点数読み取りテスト"""
        mock_image_to_string.return_value = str(self.scores['自分'])
        
        # 「自分」の領域を切り出してテスト
        region_coords = self.expected_regions['自分']
        region_image = self.processor.extract_score_region(self.test_image, region_coords)
        
        score = self.processor.read_score_from_region(region_image)

        self.assertIsNotNone(score)
        self.assertEqual(score, self.scores['自分'])
        mock_image_to_string.assert_called_once()

def run_image_processor_tests():
    """画像処理モジュールのテストを実行"""
    print("画像処理モジュールのテストを開始します...")
    
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestImageProcessor)
    
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
