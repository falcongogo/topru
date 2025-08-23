import unittest
from calculate_conditions import calculate_conditions

class TestCalculateConditions(unittest.TestCase):
    
    def test_basic_calculation(self):
        """基本的な計算のテスト"""
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
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
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 2, 1)  # 積み棒2本、供託棒1本
        
        # 積み棒: 2 * 300 * 2 = 1200点
        # 供託棒: 1 * 1000 = 1000点
        # 合計: 2200点が差し引かれる
        expected_diff = 5001 - 2200  # 2801点
        
        # 直撃ロンは半分になる（実際の計算結果を確認）
        direct_ron = result['results'][0]
        self.assertEqual(direct_ron['need_points'], 1800)
    
    def test_parent_tsumo_calculation(self):
        """親ツモの計算テスト"""
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '自家', 0, 0)  # 自家が親
        
        tsumo_result = result['results'][2]  # ツモは3番目
        self.assertIn('ツモ', tsumo_result['条件'])
        self.assertIn('親', tsumo_result['条件'])
        
        # 親ツモは3人から点数をもらう
        self.assertIsInstance(tsumo_result['total_points'], int)
        self.assertGreater(tsumo_result['total_points'], 0)
    
    def test_child_tsumo_calculation(self):
        """子ツモの計算テスト"""
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)  # 自家が子
        
        tsumo_result = result['results'][2]  # ツモは3番目
        self.assertIn('ツモ', tsumo_result['条件'])
        self.assertIn('子', tsumo_result['条件'])
        
        # 子ツモは親から2倍、子から1倍ずつ
        self.assertIsInstance(tsumo_result['total_points'], int)
        self.assertGreater(tsumo_result['total_points'], 0)
    
    def test_direct_ron_calculation(self):
        """直撃ロンの計算テスト"""
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        direct_ron = result['results'][0]
        other_ron = result['results'][1]
        
        # 直撃ロンは他家放銃ロンより必要点数が少ないはず
        self.assertLessEqual(direct_ron['need_points'], other_ron['need_points'])
        self.assertTrue(direct_ron['is_direct'])
        self.assertFalse(other_ron['is_direct'])

        # 他家放銃ロンの場合、トップの失点は0のはず
        self.assertEqual(other_ron['opponent_loss'], 0, "他家放銃ロンの場合、トップの失点は0であるべきです")
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 自家がトップの場合
        scores = {'自家': 30000, '下家': 25000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        self.assertEqual(result['leader'], '自家')
        self.assertEqual(result['top_diff'], 1)  # 30000 - 30000 + 1
        
        # 必要点数が0または非常に小さい場合
        for condition in result['results']:
            self.assertGreaterEqual(condition['need_points'], 0)
    
    def test_invalid_input(self):
        """無効な入力のテスト"""
        # 自家が含まれていない場合
        scores = {'下家': 30000, '対面': 28000, '上家': 27000}
        
        with self.assertRaises(ValueError):
            calculate_conditions(scores, '下家', 0, 0)
    
    def test_large_score_differences(self):
        """大きな点差のテスト"""
        scores = {'自家': 10000, '下家': 50000, '対面': 28000, '上家': 27000}
        result = calculate_conditions(scores, '下家', 0, 0)
        
        # 大きな点差でも計算が完了することを確認
        self.assertEqual(len(result['results']), 3)
        
        # 必要点数が非常に大きくなることを確認
        for condition in result['results']:
            self.assertGreater(condition['need_points'], 0)
    
    def test_tsumibo_kyotaku_edge_cases(self):
        """積み棒・供託棒のエッジケース"""
        scores = {'自家': 25000, '下家': 30000, '対面': 28000, '上家': 27000}
        
        # 大量の積み棒・供託棒
        result = calculate_conditions(scores, '下家', 10, 5)
        
        # 計算が完了することを確認
        self.assertEqual(len(result['results']), 3)
        
        # 必要点数が0以下になる可能性があることを確認
        for condition in result['results']:
            self.assertGreaterEqual(condition['need_points'], 0)

    def test_tsumo_calculation_details(self):
        """ツモ和了の具体的な計算ロジックを検証するテスト"""

        # ケース1: 自家が親、トップが子
        # 点差10001点、供託1、積み棒1本
        # 期待される支払い(1人あたり): (10001 - 1 - 1000 - 4*100) / 4 = 8600 / 4 = 2150点より大きい -> 2151点が必要
        scores_p = {'自家': 30000, '下家': 40000, '対面': 20000, '上家': 10000}
        result_p = calculate_conditions(scores_p, oya='自家', tsumibo=1, kyotaku=1)
        tsumo_p = result_p['results'][2]
        self.assertEqual(tsumo_p['need_points'], 2151, "親ツモの必要点数計算が不正確です")
        self.assertEqual(tsumo_p['opponent_loss'], 2600, "親ツモの相手失点計算が不正確です")

        # ケース2: 自家が子、トップが親
        # 点差20001点、供託1、積み棒1本
        # 期待される子の支払い: (20001 - 1 - 1000 - 4*100) / 6 = 18600 / 6 = 3100点より大きい -> 3101点が必要
        scores_c_top_p = {'自家': 20000, '下家': 10000, '対面': 40000, '上家': 30000}
        result_c_top_p = calculate_conditions(scores_c_top_p, oya='対面', tsumibo=1, kyotaku=1)
        tsumo_c_top_p = result_c_top_p['results'][2]
        self.assertEqual(tsumo_c_top_p['need_points'], 3101, "子ツモ（トップが親）の必要点数計算が不正確です")
        self.assertEqual(tsumo_c_top_p['opponent_loss'], 8000, "子ツモ（トップが親）の相手失点計算が不正確です")

        # ケース3: 自家が子、トップも子
        # 点差10001点、供託1、積み棒1本
        # 期待される子の支払い: (10001 - 1 - 1000 - 4*100) / 5 = 8600 / 5 = 1720点より大きい -> 1721点が必要
        scores_c_top_c = {'自家': 20000, '下家': 30000, '対面': 25000, '上家': 25000}
        result_c_top_c = calculate_conditions(scores_c_top_c, oya='対面', tsumibo=1, kyotaku=1)
        tsumo_c_top_c = result_c_top_c['results'][2]
        self.assertEqual(tsumo_c_top_c['need_points'], 1721, "子ツモ（トップが子）の必要点数計算が不正確です")
        self.assertEqual(tsumo_c_top_c['opponent_loss'], 2000, "子ツモ（トップが子）の相手失点計算が不正確です")


if __name__ == '__main__':
    unittest.main()
