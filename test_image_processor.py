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
        
        # より現実に近いテスト画像を作成（グレー背景に白いスコア枠）
        self.test_image = np.full((400, 800, 3), (128, 128, 128), dtype=np.uint8)

        self.score_regions = {
            '自分': (50, 50, 250, 100),
            '下家': (450, 50, 650, 100),
            '対面': (50, 250, 250, 300),
            '上家': (450, 250, 650, 300),
        }

        self.scores = {
            '自分': 28000,
            '下家': 35000,
            '対面': 30000,
            '上家': 27000,
        }
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for player, (x1, y1, x2, y2) in self.score_regions.items():
            # 白い枠を描画
            cv2.rectangle(self.test_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            # 黒い文字で点数を描画
            text = str(self.scores[player])
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(self.test_image, text, (text_x, text_y), font, 0.8, (0, 0, 0), 2)

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
        region = self.score_regions['自分']
        extracted = self.processor.extract_score_region(self.test_image, region)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.shape, (50, 200, 3))

    def test_detect_score_regions(self):
        """点数表示領域の検出テスト"""
        regions = self.processor.detect_score_regions(self.test_image)
        
        self.assertEqual(len(regions), 4)
        
        # 検出された領域が設定した領域と一致するかチェック
        detected_regions_sorted = sorted(regions.values(), key=lambda r: (r[1], r[0]))
        expected_regions_sorted = sorted(self.score_regions.values(), key=lambda r: (r[1], r[0]))
        
        for detected, expected in zip(detected_regions_sorted, expected_regions_sorted):
            self.assertTrue(np.allclose(detected, expected, atol=2),
                            msg=f"Detected: {detected}, Expected: {expected}")

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
        region_coords = self.score_regions['自分']
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
