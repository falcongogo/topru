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
        self.test_image = np.full((600, 800, 3), (120, 120, 120), dtype=np.uint8)

        # 2x2グリッドのレイアウトを定義
        rects = {
            'top_left': (100, 50, 300, 200),
            'top_right': (500, 50, 700, 200),
            'bottom_left': (100, 300, 350, 500),  # 自分（大きい）
            'bottom_right': (500, 300, 700, 450)
        }

        # このテストケースでのプレイヤー配置 (自分が左下)
        self.expected_layout = {
            '自分': rects['bottom_left'],
            '対面': rects['top_right'],
            '上家': rects['top_left'],
            '下家': rects['bottom_right']
        }
        
        self.scores = {
            '自分': 42000, '対面': 18000, '上家': 35000, '下家': 25000
        }

        # 画像に領域と点数を描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        for player, rect in self.expected_layout.items():
            (x1, y1, x2, y2) = rect
            cv2.rectangle(self.test_image, (x1, y1), (x2, y2), (240, 240, 240), -1)
            score_text = str(self.scores[player])
            cv2.putText(self.test_image, score_text, (x1 + 20, y1 + 80), font, 1.2, (0, 0, 0), 3)
            if player != '自分':
                diff_text = f"-{self.scores['自分'] - self.scores[player]}"
                cv2.putText(self.test_image, diff_text, (x1 + 50, y1 + 120), font, 0.5, (40, 40, 40), 1)

    def test_detect_and_assign_regions(self):
        """2x2非対称レイアウトの検出と役割割り当てのテスト"""
        # processor.detect_score_regions のロジックを直接テスト
        detected_regions = self.processor.detect_score_regions(self.test_image)
        
        self.assertEqual(len(detected_regions), 4, "4つの領域が検出されるべき")

        # 各プレイヤーの領域が期待通りかチェック
        for player, expected_rect in self.expected_layout.items():
            self.assertIn(player, detected_regions, f"{player}が検出結果にない")
            detected_rect = detected_regions[player]
            self.assertTrue(np.allclose(detected_rect, expected_rect, atol=5),
                            f"{player}の領域が不一致: 検出={detected_rect}, 期待={expected_rect}")

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
