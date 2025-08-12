import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# app.pyの関数をテストするために、Streamlitの依存関係をモック
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestAppFunctions(unittest.TestCase):
    
    def test_validate_inputs_valid(self):
        """有効な入力値の検証テスト"""
        from app import validate_inputs
        
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        self.assertTrue(validate_inputs(scores, 0, 0))
        self.assertTrue(validate_inputs(scores, 5, 2))
    
    def test_validate_inputs_invalid_scores(self):
        """無効な点数の検証テスト"""
        from app import validate_inputs
        
        # 負の点数
        scores = {'自分': -1000, '下家': 30000, '対面': 28000, '上家': 27000}
        self.assertFalse(validate_inputs(scores, 0, 0))
        
        # 0点は有効
        scores = {'自分': 0, '下家': 30000, '対面': 28000, '上家': 27000}
        self.assertTrue(validate_inputs(scores, 0, 0))
    
    def test_validate_inputs_invalid_tsumibo_kyotaku(self):
        """無効な積み棒・供託棒の検証テスト"""
        from app import validate_inputs
        
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        
        # 負の積み棒
        self.assertFalse(validate_inputs(scores, -1, 0))
        
        # 負の供託棒
        self.assertFalse(validate_inputs(scores, 0, -1))
        
        # 両方負
        self.assertFalse(validate_inputs(scores, -1, -1))
    
    def test_get_condition_style_impossible(self):
        """不可能な条件のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '不可能', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffd6d6')
        self.assertEqual(style['badge'], "❌")
        self.assertEqual(style['style'], '')
    
    def test_get_condition_style_mangan(self):
        """満貫のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '満貫', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffe566')
        self.assertEqual(style['badge'], "🌟")
        self.assertEqual(style['style'], 'font-weight:700;')
    
    def test_get_condition_style_yakuman(self):
        """役満のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '役満', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffd700')
        self.assertEqual(style['badge'], "💎")
        self.assertEqual(style['style'], 'font-weight:700;')
    
    def test_get_condition_style_direct(self):
        """直撃のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '30符1翻', 'is_direct': True}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#e0f7fa')
        self.assertEqual(style['badge'], "直撃")
        self.assertEqual(style['style'], 'font-weight:700;')
    
    def test_get_condition_style_normal(self):
        """通常の条件のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '30符1翻', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#fff6e6')
        self.assertEqual(style['badge'], "")
        self.assertEqual(style['style'], '')
    
    def test_get_condition_style_jump_mangan(self):
        """跳満のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '跳満', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffd700')
        self.assertEqual(style['badge'], "💎")
        self.assertEqual(style['style'], 'font-weight:700;')
    
    def test_get_condition_style_baiman(self):
        """倍満のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '倍満', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffd700')
        self.assertEqual(style['badge'], "💎")
        self.assertEqual(style['style'], 'font-weight:700;')
    
    def test_get_condition_style_sanbaiman(self):
        """三倍満のスタイルテスト"""
        from app import get_condition_style
        
        result = {'rank': '三倍満', 'is_direct': False}
        style = get_condition_style(result)
        
        self.assertEqual(style['bgcolor'], '#ffd700')
        self.assertEqual(style['badge'], "💎")
        self.assertEqual(style['style'], 'font-weight:700;')

if __name__ == '__main__':
    unittest.main()
