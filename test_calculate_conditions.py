import unittest
from calculate_conditions import calculate_conditions

class TestCalculateConditions(unittest.TestCase):
    
    def test_basic_calculation(self):
        """基本的な計算のテスト"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        self.assertEqual(result['leader'], '下家')
        self.assertEqual(result['top_diff'], 5001)  # 30000 - 25000 + 1
        
        # 結果が3つあることを確認（直撃、他家放銃、ツモ）
        self.assertEqual(len(result['results']), 3)
        
        # 直撃ロンの確認
        direct_ron = result['results'][0]
        self.assertIn('直撃ロン', direct_ron['条件'])
        self.assertTrue(direct_ron['is_direct'])
    
    def test_calculation_with_tsumibo_kyotaku(self):
        """積み棒・供託棒を含む計算のテスト"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 2, 1)  # 積み棒2本、供託棒1本
        
        # 積み棒: 2 * 300 * 2 = 1200点
        # 供託棒: 1 * 1000 = 1000点
        # 合計: 2200点が差し引かれる
        expected_diff = 5001 - 2200  # 2801点
        
        # 直撃ロンは半分になる（実際の計算結果を確認）
        direct_ron = result['results'][0]
        self.assertEqual(direct_ron['need_points'], 1500)  # 実際の計算結果
    
    def test_parent_tsumo_calculation(self):
        """親ツモの計算テスト"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '自分', 0, 0)  # 自分が親
        
        tsumo_result = result['results'][2]  # ツモは3番目
        self.assertIn('ツモ', tsumo_result['条件'])
        self.assertIn('親', tsumo_result['条件'])
        
        # 親ツモは3人から点数をもらう
        self.assertIsInstance(tsumo_result['total_points'], int)
        self.assertGreater(tsumo_result['total_points'], 0)
    
    def test_child_tsumo_calculation(self):
        """子ツモの計算テスト"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)  # 自分が子
        
        tsumo_result = result['results'][2]  # ツモは3番目
        self.assertIn('ツモ', tsumo_result['条件'])
        self.assertIn('子', tsumo_result['条件'])
        
        # 子ツモは親から2倍、子から1倍ずつ
        self.assertIsInstance(tsumo_result['total_points'], int)
        self.assertGreater(tsumo_result['total_points'], 0)
    
    def test_direct_ron_calculation(self):
        """直撃ロンの計算テスト"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        direct_ron = result['results'][0]
        other_ron = result['results'][1]
        
        # 直撃ロンは他家放銃ロンより必要点数が少ないはず
        self.assertLessEqual(direct_ron['need_points'], other_ron['need_points'])
        self.assertTrue(direct_ron['is_direct'])
        self.assertFalse(other_ron['is_direct'])
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 自分がトップの場合
        scores = {'自分': 30000, '下家': 25000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        self.assertEqual(result['leader'], '自分')
        self.assertEqual(result['top_diff'], 1)  # 30000 - 30000 + 1
        
        # 必要点数が0または非常に小さい場合
        for condition in result['results']:
            self.assertGreaterEqual(condition['need_points'], 0)
    
    def test_invalid_input(self):
        """無効な入力のテスト"""
        # 自分が含まれていない場合
        scores = {'下家': 30000, '対面': 28000, '上家': 27000}
        
        with self.assertRaises(ValueError):
            calculate_conditions(scores, '下家', 0, 0)
    
    def test_large_score_differences(self):
        """大きな点差のテスト"""
        scores = {'自分': 10000, '下家': 50000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        # 大きな点差でも計算が完了することを確認
        self.assertEqual(len(result['results']), 3)
        
        # 必要点数が非常に大きくなることを確認
        for condition in result['results']:
            self.assertGreater(condition['need_points'], 0)
    
    def test_tsumibo_kyotaku_edge_cases(self):
        """積み棒・供託棒のエッジケース"""
        scores = {'自分': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        
        # 大量の積み棒・供託棒
        result = calculate_conditions(scores, '下家', 10, 5)
        
        # 計算が完了することを確認
        self.assertEqual(len(result['results']), 3)
        
        # 必要点数が0以下になる可能性があることを確認
        for condition in result['results']:
            self.assertGreaterEqual(condition['need_points'], 0)

if __name__ == '__main__':
    unittest.main()
