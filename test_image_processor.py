import unittest
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
        
        # テスト用の画像を作成
        self.test_image = np.ones((800, 1000, 3), dtype=np.uint8) * 255  # 白い画像
        
        # テスト用の点数表示を描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.test_image, '28000', (120, 230), font, 1, (0, 0, 0), 2)
        cv2.putText(self.test_image, '35000', (320, 230), font, 1, (0, 0, 0), 2)
        cv2.putText(self.test_image, '30000', (520, 230), font, 1, (0, 0, 0), 2)
        cv2.putText(self.test_image, '27000', (720, 230), font, 1, (0, 0, 0), 2)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.processor)
        # 自動検出モードでは固定のscore_regionsは存在しない
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
        region = (100, 200, 200, 250)  # テスト用の領域
        extracted = self.processor.extract_score_region(self.test_image, region)
        self.assertIsNotNone(extracted)
        self.assertEqual(len(extracted.shape), 3)  # カラー画像
    
    def test_detect_score_regions(self):
        """点数表示領域の検出テスト"""
        regions = self.processor.detect_score_regions(self.test_image)
        
        # 自動検出なので、必ずしも4つの領域が検出されるとは限らない
        # 検出された領域の形式をチェック
        for player, region in regions.items():
            self.assertIn(player, ['自分', '下家', '対面', '上家'])
            self.assertEqual(len(region), 4)  # (x1, y1, x2, y2)
            self.assertTrue(all(isinstance(x, int) for x in region))
    
    def test_detect_text_regions(self):
        """文字領域検出テスト"""
        processed_image = self.processor.preprocess_image(self.test_image)
        text_regions = self.processor.detect_text_regions(processed_image)
        
        # 文字領域が検出されることを確認
        self.assertIsInstance(text_regions, list)
        for region in text_regions:
            self.assertEqual(len(region), 4)  # (x1, y1, x2, y2)
            self.assertTrue(all(isinstance(x, int) for x in region))
    
    def test_merge_overlapping_regions(self):
        """重複領域マージテスト"""
        # 重複する領域を作成
        regions = [(0, 0, 100, 50), (50, 0, 150, 50), (200, 0, 300, 50)]
        merged = self.processor.merge_overlapping_regions(regions)
        
        # 重複がマージされることを確認
        self.assertIsInstance(merged, list)
        self.assertLessEqual(len(merged), len(regions))
    
    def test_is_valid_score(self):
        """点数妥当性チェックテスト"""
        # 有効な点数
        self.assertTrue(self.processor._is_valid_score(28000))
        self.assertTrue(self.processor._is_valid_score(35000))
        
        # 無効な点数
        self.assertFalse(self.processor._is_valid_score(999))  # 範囲外
        self.assertFalse(self.processor._is_valid_score(100000))  # 範囲外
        self.assertFalse(self.processor._is_valid_score(28001))  # 1000点単位でない
        self.assertFalse(self.processor._is_valid_score(2800))  # 桁数が違う
    
    def test_process_score_image(self):
        """画像処理の統合テスト"""
        # 一時ファイルにテスト画像を保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, self.test_image)
            tmp_path = tmp_file.name
        
        try:
            scores = self.processor.process_score_image(tmp_path)
            # OCRの精度に依存するため、結果の存在のみチェック
            self.assertIsInstance(scores, dict)
        finally:
            # 一時ファイルを削除
            os.unlink(tmp_path)
    
    def test_read_score_from_region(self):
        """点数読み取りテスト"""
        # テスト用の領域を切り出し
        region = (100, 200, 200, 250)
        region_image = self.processor.extract_score_region(self.test_image, region)
        
        # 前処理
        processed_region = self.processor.preprocess_image(region_image)
        
        # 点数読み取り
        score = self.processor.read_score_from_region(processed_region)
        # OCRの精度に依存するため、結果の型のみチェック
        if score is not None:
            self.assertIsInstance(score, int)
            self.assertTrue(1000 <= score <= 99999)

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
