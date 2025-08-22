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
    # 'tests' ディレクトリからテストを自動検出
    loader = unittest.TestLoader()
    # プロジェクトルートからの相対パスとして 'tests' を指定
    test_dir = os.path.join(os.path.dirname(__file__))
    suite = loader.discover(test_dir)
    
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
