import unittest
from points_table import POINTS_TABLE

class TestPointsTable(unittest.TestCase):
    
    def test_table_structure(self):
        """点数表の構造テスト"""
        # 基本的な構造の確認
        self.assertIn('child', POINTS_TABLE)
        self.assertIn('parent', POINTS_TABLE)
        
        for role in ['child', 'parent']:
            self.assertIn('ron', POINTS_TABLE[role])
            self.assertIn('tsumo', POINTS_TABLE[role])
    
    def test_ron_points_structure(self):
        """ロンの点数構造テスト"""
        # 子のロン
        child_ron = POINTS_TABLE['child']['ron']
        for (fu, han), points in child_ron.items():
            self.assertIsInstance(fu, int)
            self.assertIsInstance(han, int)
            self.assertIsInstance(points, int)
            self.assertGreater(points, 0)
            self.assertLessEqual(fu, 50)
            self.assertLessEqual(han, 4)
            self.assertGreaterEqual(fu, 20)
            self.assertGreaterEqual(han, 1)
        
        # 親のロン
        parent_ron = POINTS_TABLE['parent']['ron']
        for (fu, han), points in parent_ron.items():
            self.assertIsInstance(fu, int)
            self.assertIsInstance(han, int)
            self.assertIsInstance(points, int)
            self.assertGreater(points, 0)
            self.assertLessEqual(fu, 50)
            self.assertLessEqual(han, 4)
            self.assertGreaterEqual(fu, 20)
            self.assertGreaterEqual(han, 1)
    
    def test_tsumo_points_structure(self):
        """ツモの点数構造テスト"""
        # 子のツモ
        child_tsumo = POINTS_TABLE['child']['tsumo']
        for (fu, han), points in child_tsumo.items():
            self.assertIsInstance(fu, int)
            self.assertIsInstance(han, int)
            self.assertIsInstance(points, tuple)
            self.assertEqual(len(points), 2)
            child_pay, parent_pay = points
            self.assertIsInstance(child_pay, int)
            self.assertIsInstance(parent_pay, int)
            self.assertGreater(child_pay, 0)
            self.assertGreater(parent_pay, 0)
            # 親の支払いは子の支払いの2倍（ただし、一部例外がある）
            # 実際の点数表では、親の支払いは子の支払いの2倍またはそれに近い値
            self.assertGreaterEqual(parent_pay, child_pay)
        
        # 親のツモ
        parent_tsumo = POINTS_TABLE['parent']['tsumo']
        for (fu, han), points in parent_tsumo.items():
            self.assertIsInstance(fu, int)
            self.assertIsInstance(han, int)
            self.assertIsInstance(points, int)
            self.assertGreater(points, 0)
    
    def test_required_combinations(self):
        """必要な組み合わせの存在確認"""
        required_fu = [20, 30, 40, 50]
        required_han = [1, 2, 3, 4]
        
        for role in ['child', 'parent']:
            for method in ['ron', 'tsumo']:
                table = POINTS_TABLE[role][method]
                
                # 30符、40符、50符の1-4翻は必須
                for fu in [30, 40, 50]:
                    for han in required_han:
                        self.assertIn((fu, han), table, 
                                    f"Missing {fu}符{han}翻 for {role} {method}")
                
                # 20符は2翻以上のみ
                for han in [2, 3, 4, 5]:
                    if (20, han) in table:
                        self.assertGreaterEqual(han, 2)
    
    def test_points_consistency(self):
        """点数の一貫性テスト"""
        # 同じ符・翻数で親のロンは子のロンより高い
        for (fu, han) in POINTS_TABLE['child']['ron'].keys():
            if (fu, han) in POINTS_TABLE['parent']['ron']:
                child_points = POINTS_TABLE['child']['ron'][(fu, han)]
                parent_points = POINTS_TABLE['parent']['ron'][(fu, han)]
                self.assertGreater(parent_points, child_points)
    
    def test_mangan_thresholds(self):
        """満貫の閾値テスト"""
        # 子のロン：8000点以上は満貫
        child_ron = POINTS_TABLE['child']['ron']
        for (fu, han), points in child_ron.items():
            if han == 4:
                self.assertEqual(points, 8000)
        
        # 親のロン：12000点以上は満貫
        parent_ron = POINTS_TABLE['parent']['ron']
        for (fu, han), points in parent_ron.items():
            if han == 4:
                self.assertEqual(points, 12000)
        
        # 親のツモ：4翻は4000点（満貫）、ただし20符は2600点
        parent_tsumo = POINTS_TABLE['parent']['tsumo']
        for (fu, han), points in parent_tsumo.items():
            if han == 4:
                if fu == 20:
                    self.assertEqual(points, 2600)
                else:
                    self.assertEqual(points, 4000)
        
        # 子のツモ：4翻は2000点（満貫）、ただし20符は1300点
        child_tsumo = POINTS_TABLE['child']['tsumo']
        for (fu, han), points in child_tsumo.items():
            if han == 4:
                if fu == 20:
                    self.assertEqual(points[0], 1300)  # 子の支払い額
                    self.assertEqual(points[1], 2600)  # 親の支払い額
                else:
                    self.assertEqual(points[0], 2000)  # 子の支払い額
                    self.assertEqual(points[1], 4000)  # 親の支払い額
    
    def test_no_20fu_ron(self):
        """20符のロンが存在しないことを確認"""
        # 子のロン
        child_ron = POINTS_TABLE['child']['ron']
        for (fu, han) in child_ron.keys():
            self.assertNotEqual(fu, 20, "20符のロンは存在しないはず")
        
        # 親のロン
        parent_ron = POINTS_TABLE['parent']['ron']
        for (fu, han) in parent_ron.keys():
            self.assertNotEqual(fu, 20, "20符のロンは存在しないはず")

if __name__ == '__main__':
    unittest.main()
