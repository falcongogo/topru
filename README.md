# TOPる – 麻雀オーラス逆転条件計算ツール

麻雀のオーラスで逆転するために必要な条件を計算するツールです。

## 機能

- 現在の点数状況から逆転に必要な条件を計算
- 直撃ロン、他家放銃ロン、ツモの3つのパターンを表示
- 積み棒・供託棒を考慮した計算
- 親・子の違いを考慮した計算
- **スリムスコア28S画像からの自動点数入力**（新機能）

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
streamlit run app.py
```

### 画像からの自動点数入力

1. スリムスコア28Sの点数表示をスマホで撮影
2. アプリの「📷 スリムスコア28S画像から自動入力」セクションで画像をアップロード
3. 「🔍 画像から点数を読み取り」ボタンをクリック
4. 自動読み取りされた点数が手動入力欄に反映されます
5. 必要に応じて手動で調整してから「計算」ボタンをクリック

**自動検出機能**:
- 固定座標ではなく、画像から自動的に点数表示領域を検出
- MSER（Maximally Stable Extremal Regions）を使用した文字領域検出
- OCRによる文字認識と点数妥当性チェック
- 重複領域の自動マージ機能

**デバッグ機能**:
- 「🔧 デバッグモード」を有効にすると検出された領域を可視化
- 検出結果の詳細確認が可能

**注意**: 画像の明るさや角度によって読み取り精度が変わる場合があります。読み取りに失敗した場合は、デバッグモードで検出状況を確認するか、画像を再撮影してください。

## テスト

### 全テストの実行

```bash
python run_tests.py
```

### 個別テストの実行

```bash
# points_lookup.pyのテスト
python -m unittest test_points_lookup.py

# calculate_conditions.pyのテスト
python -m unittest test_calculate_conditions.py

# app.pyのテスト
python -m unittest test_app.py

# points_table.pyのテスト
python -m unittest test_points_table.py
```

### テストカバレッジ

現在のテストは以下の機能をカバーしています：

#### points_lookup.py
- `ceil100()` 関数の動作確認
- `reverse_lookup()` 関数の各種ケース
- 満貫以上の判定ロジック
- エッジケース（0点、負の点数、非常に大きな点数）

#### calculate_conditions.py
- 基本的な条件計算
- 積み棒・供託棒を含む計算
- 親・子のツモ計算
- 直撃ロンの計算
- エッジケース（自分がトップ、大きな点差）

#### app.py
- 入力値検証機能
- 条件スタイル設定機能
- 各種役種のスタイル確認

#### points_table.py
- 点数表のデータ構造検証
- 必要な組み合わせの存在確認
- 点数の一貫性確認
- 満貫閾値の確認

#### image_processor.py
- 画像処理機能の動作確認
- OCR文字認識の精度確認
- 点数表示領域の検出確認
- 画像前処理の効果確認

## ファイル構成

- `app.py` - Streamlitアプリケーション
- `calculate_conditions.py` - 条件計算ロジック
- `points_lookup.py` - 点数逆引きロジック
- `points_table.py` - 点数表データ
- `image_processor.py` - 画像処理・OCR機能
- `test_*.py` - 単体テストファイル
- `run_tests.py` - テスト実行スクリプト
