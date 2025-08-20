import sys
from image_processor import ScoreImageProcessor

def main():
    """
    修正が正しく機能するかを検証するためのシンプルなテストスクリプト。
    test_data内の画像を処理し、認識結果を出力する。
    """
    # テスト画像のパスを指定
    # 他の画像で試す場合はここを変更する
    image_path = "test_data/test3.jpg"
    print(f"--- テスト実行: {image_path} ---")

    try:
        # 画像処理クラスを初期化
        processor = ScoreImageProcessor()

        # 点数読み取りを実行
        scores = processor.process_score_image(image_path)

        # 結果を出力
        if scores:
            print("✅ 認識成功:")
            for player, score in scores.items():
                print(f"  - {player}: {score}")
        else:
            print("［警告］ 認識失敗: 点数を読み取れませんでした。")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
