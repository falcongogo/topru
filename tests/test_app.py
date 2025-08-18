import unittest
from unittest.mock import MagicMock
import sys
import os

# StreamlitのUI要素に依存せずにロジックをテストできるよう、モックを設定
sys.modules['streamlit'] = MagicMock()

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui import validate_inputs, get_condition_style

class TestAppFunctions(unittest.TestCase):

    def test_validate_inputs_valid(self):
        """有効な入力値の検証テスト"""
        scores = {'自分': 25000, '下家': 25000, '対面': 25000, '上家': 25000}
        self.assertTrue(validate_inputs(scores, 1, 0))

    def test_validate_inputs_invalid_scores(self):
        """無効な点数の検証テスト"""
        scores = {'自分': -100, '下家': 25000, '対面': 25000, '上家': 25000}
        self.assertFalse(validate_inputs(scores, 1, 0))
        sys.modules['streamlit'].error.assert_called_with("点数は0以上で入力してください")

    def test_validate_inputs_invalid_tsumibo_kyotaku(self):
        """無効な積み棒・供託棒の検証テスト"""
        scores = {'自分': 25000, '下家': 25000, '対面': 25000, '上家': 25000}
        self.assertFalse(validate_inputs(scores, -1, 0))
        sys.modules['streamlit'].error.assert_called_with("積み棒・供託棒は0以上で入力してください")
        self.assertFalse(validate_inputs(scores, 0, -1))
        sys.modules['streamlit'].error.assert_called_with("積み棒・供託棒は0以上で入力してください")

    def test_get_condition_style_impossible(self):
        """不可能な条件のスタイルテスト"""
        result = {'rank': '不可能', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffd6d6')
        self.assertEqual(badge, '❌')
        self.assertEqual(weight_class, '')

    def test_get_condition_style_mangan(self):
        """満貫のスタイルテスト"""
        result = {'rank': '満貫', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffe566')
        self.assertEqual(badge, '🌟')
        self.assertEqual(weight_class, 'bold')

    def test_get_condition_style_yakuman(self):
        """役満のスタイルテスト"""
        result = {'rank': '役満', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffd700')
        self.assertEqual(badge, '💎')
        self.assertEqual(weight_class, 'bold')

    def test_get_condition_style_direct(self):
        """直撃のスタイルテスト"""
        result = {'rank': '20符2翻', 'is_direct': True}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#e0f7fa')
        self.assertEqual(badge, '直撃')
        self.assertEqual(weight_class, 'bold')

    def test_get_condition_style_normal(self):
        """通常の条件のスタイルテスト"""
        result = {'rank': '30符1翻', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#fff6e6')
        self.assertEqual(badge, '')
        self.assertEqual(weight_class, '')

    def test_get_condition_style_jump_mangan(self):
        """跳満のスタイルテスト"""
        result = {'rank': '跳満', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffd700')
        self.assertEqual(badge, '💎')
        self.assertEqual(weight_class, 'bold')

    def test_get_condition_style_baiman(self):
        """倍満のスタイルテスト"""
        result = {'rank': '倍満', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffd700')
        self.assertEqual(badge, '💎')
        self.assertEqual(weight_class, 'bold')

    def test_get_condition_style_sanbaiman(self):
        """三倍満のスタイルテスト"""
        result = {'rank': '三倍満', 'is_direct': False}
        bgcolor, badge, weight_class = get_condition_style(result)
        self.assertEqual(bgcolor, '#ffd700')
        self.assertEqual(badge, '💎')
        self.assertEqual(weight_class, 'bold')

if __name__ == '__main__':
    unittest.main()
