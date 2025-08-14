import unittest
from points_lookup import ceil100, reverse_lookup

class TestPointsLookup(unittest.TestCase):
    
    def test_ceil100(self):
        """ceil100関数のテスト"""
        self.assertEqual(ceil100(0), 0)
        self.assertEqual(ceil100(100), 100)
        self.assertEqual(ceil100(150), 200)
        self.assertEqual(ceil100(200), 200)
        self.assertEqual(ceil100(250), 300)
        self.assertEqual(ceil100(999), 1000)
        self.assertEqual(ceil100(1000), 1000)
    
    def test_reverse_lookup_ron_child(self):
        """子のロンの逆引きテスト"""
        # 基本ケース
        result = reverse_lookup(1000, 'ron', False)
        self.assertEqual(result['rank'], '30符1翻')
        self.assertEqual(result['points'], 1000)
        
        result = reverse_lookup(2000, 'ron', False)
        self.assertEqual(result['rank'], '30符2翻')
        self.assertEqual(result['points'], 2000)
        
        # 満貫ケース
        result = reverse_lookup(8000, 'ron', False)
        self.assertEqual(result['rank'], '満貫')
        self.assertEqual(result['points'], 8000)
        
        # 跳満ケース
        result = reverse_lookup(12000, 'ron', False)
        self.assertEqual(result['rank'], '跳満')
        self.assertEqual(result['points'], 12000)
            
    def test_reverse_lookup_ron_parent(self):
        """親のロンの逆引きテスト"""
        # 基本ケース
        result = reverse_lookup(1500, 'ron', True)
        self.assertEqual(result['rank'], '30符1翻')
        self.assertEqual(result['points'], 1500)
        
        # 満貫ケース
        result = reverse_lookup(12000, 'ron', True)
        self.assertEqual(result['rank'], '満貫')
        self.assertEqual(result['points'], 12000)
    
    def test_reverse_lookup_tsumo_child(self):
        """子のツモの逆引きテスト"""
        result = reverse_lookup(300, 'tsumo', False)
        self.assertEqual(result['rank'], '30符1翻')
        self.assertEqual(result['display'], '300-500') 

        # 満貫ケース
        result = reverse_lookup(2000, 'tsumo', False)
        self.assertEqual(result['rank'], '満貫')
        self.assertEqual(result['display'], '2000-4000') 

    def test_reverse_lookup_tsumo_parent(self):
        """親のツモの逆引きテスト"""
        result = reverse_lookup(500, 'tsumo', True)
        self.assertEqual(result['rank'], '30符1翻')
        self.assertEqual(result['display'], '500オール') 

        # 満貫ケース
        result = reverse_lookup(4000, 'tsumo', True)
        self.assertEqual(result['rank'], '満貫')
        self.assertEqual(result['display'], '4000オール') 
    
    def test_reverse_lookup_edge_cases(self):
        """エッジケースのテスト"""
        # 0点のケース
        result = reverse_lookup(0, 'ron', False)
        self.assertEqual(result['rank'], '不要')
        self.assertEqual(result['points'], 0)
        
        # 負の点数のケース
        result = reverse_lookup(-100, 'ron', False)
        self.assertEqual(result['rank'], '不要')
        self.assertEqual(result['points'], 0)
        
        # 非常に大きな点数のケース
        result = reverse_lookup(50000, 'ron', False)
        self.assertEqual(result['rank'], '役満')
        self.assertEqual(result['points'], 50000)

if __name__ == '__main__':
    unittest.main()
