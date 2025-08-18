#!/usr/bin/env python3
"""
TOPる プロジェクトの全単体テストを実行するスクリプト
"""

import unittest
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_all_tests():
    """全てのテストを実行"""
    # テストファイルのリスト
    test_files = [
        'test_points_lookup.py',
        'test_calculate_conditions.py',
        'test_app.py',
        'test_points_table.py',
        'test_image_processor.py'
    ]
    
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 各テストファイルをスイートに追加
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                # テストファイルをインポートしてスイートに追加
                module_name = test_file.replace('.py', '')
                module = __import__(module_name)
                suite.addTests(loader.loadTestsFromModule(module))
                print(f"✓ {test_file} を読み込みました")
            except ImportError as e:
                print(f"✗ {test_file} の読み込みに失敗: {e}")
        else:
            print(f"✗ {test_file} が見つかりません")
    
    # テストを実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果を表示
    print("\n" + "="*50)
    print("テスト実行結果")
    print("="*50)
    print(f"実行したテスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # 終了コードを設定
    if result.failures or result.errors:
        sys.exit(1)
    else:
        print("\n🎉 全てのテストが成功しました！")
        sys.exit(0)

if __name__ == '__main__':
    run_all_tests()
