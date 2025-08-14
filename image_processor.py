import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from typing import Dict, Optional, Tuple, List

class ScoreImageProcessor:
    """スリムスコア28Sの点数表示を読み取るクラス"""
    
    def __init__(self):
        # OCR設定
        self.ocr_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        
        # 点数表示の特徴
        self.min_score = 1000
        self.max_score = 99999
        self.expected_digits = 5  # 28000のような5桁の数字
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """画像の前処理"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去
        denoised = cv2.medianBlur(gray, 3)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 二値化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """画像から文字領域を検出"""
        # MSER（Maximally Stable Extremal Regions）を使用して文字領域を検出
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(image)
        
        # 文字領域の候補を抽出
        text_regions = []
        for region in regions:
            # 領域の境界ボックスを取得
            x, y, w, h = cv2.boundingRect(region)
            
            # 文字領域として適切なサイズかチェック
            if self._is_valid_text_region(w, h, image.shape):
                text_regions.append((x, y, x + w, y + h))
        
        return text_regions
    
    def _is_valid_text_region(self, width: int, height: int, image_shape: Tuple[int, int, int]) -> bool:
        """文字領域として適切なサイズかチェック"""
        img_height, img_width = image_shape[:2]
        
        # 最小・最大サイズの制限
        min_width = img_width * 0.02  # 画像幅の2%
        max_width = img_width * 0.15  # 画像幅の15%
        min_height = img_height * 0.02  # 画像高さの2%
        max_height = img_height * 0.08  # 画像高さの8%
        
        # アスペクト比の制限（横長の文字）
        aspect_ratio = width / height if height > 0 else 0
        min_aspect = 1.5  # 最小アスペクト比
        max_aspect = 8.0  # 最大アスペクト比
        
        return (min_width <= width <= max_width and
                min_height <= height <= max_height and
                min_aspect <= aspect_ratio <= max_aspect)
    
    def merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """重複する領域をマージ"""
        if not regions:
            return []
        
        # 領域をx座標でソート
        sorted_regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current = sorted_regions[0]
        
        for region in sorted_regions[1:]:
            # 重複チェック
            if self._regions_overlap(current, region):
                # マージ
                x1 = min(current[0], region[0])
                y1 = min(current[1], region[1])
                x2 = max(current[2], region[2])
                y2 = max(current[3], region[3])
                current = (x1, y1, x2, y2)
            else:
                merged.append(current)
                current = region
        
        merged.append(current)
        return merged
    
    def _regions_overlap(self, region1: Tuple[int, int, int, int], region2: Tuple[int, int, int, int]) -> bool:
        """2つの領域が重複しているかチェック"""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def detect_score_regions(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """画像から点数表示領域を自動検出"""
        # 前処理
        processed_image = self.preprocess_image(image)
        
        # 文字領域を検出
        text_regions = self.detect_text_regions(processed_image)
        
        # 重複する領域をマージ
        merged_regions = self.merge_overlapping_regions(text_regions)
        
        # 各領域でOCRを試行して点数を検出
        score_regions = []
        for region in merged_regions:
            region_image = self.extract_score_region(processed_image, region)
            score = self.read_score_from_region(region_image)
            
            if score is not None:
                score_regions.append((region, score))
        
        # 点数が検出された領域を4つまで選択（4プレイヤー分）
        score_regions.sort(key=lambda x: x[1], reverse=True)  # 点数でソート
        selected_regions = score_regions[:4]
        
        # プレイヤー名を割り当て（位置に基づいて）
        result = {}
        if len(selected_regions) >= 4:
            # 4つの領域を位置でソート（左から右、上から下）
            selected_regions.sort(key=lambda x: (x[0][1], x[0][0]))  # y座標、x座標でソート
            
            # 位置に基づいてプレイヤー名を割り当て
            positions = ['自分', '下家', '対面', '上家']
            for i, (region, score) in enumerate(selected_regions):
                result[positions[i]] = region
        
        return result
    
    def extract_score_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """指定された領域を切り出し"""
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]
    
    def read_score_from_region(self, region_image: np.ndarray) -> Optional[int]:
        """指定された領域から点数を読み取り"""
        try:
            # 領域が小さすぎる場合はスキップ
            if region_image.shape[0] < 10 or region_image.shape[1] < 20:
                return None
            
            # OCRで文字認識
            text = pytesseract.image_to_string(region_image, config=self.ocr_config)
            
            # 数字のみを抽出
            numbers = re.findall(r'\d+', text)
            
            if numbers:
                # 各数字をチェック
                for number in numbers:
                    score = int(number)
                    
                    # 点数として妥当かチェック
                    if self._is_valid_score(score):
                        return score
            
            return None
            
        except Exception as e:
            print(f"OCR読み取りエラー: {e}")
            return None
    
    def _is_valid_score(self, score: int) -> bool:
        """点数として妥当かチェック"""
        # 範囲チェック
        if not (self.min_score <= score <= self.max_score):
            return False
        
        # 桁数チェック（5桁の数字）
        if len(str(score)) != self.expected_digits:
            return False
        
        # 麻雀の点数として妥当かチェック（1000点単位）
        if score % 1000 != 0:
            return False
        
        return True
    
    def process_score_image(self, image_path: str) -> Dict[str, int]:
        """画像から全プレイヤーの点数を読み取り"""
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        
        # 前処理
        processed_image = self.preprocess_image(image)
        
        # 点数表示領域の検出
        regions = self.detect_score_regions(image)
        
        # 各プレイヤーの点数を読み取り
        scores = {}
        for player, region in regions.items():
            region_image = self.extract_score_region(processed_image, region)
            score = self.read_score_from_region(region_image)
            
            if score is not None:
                scores[player] = score
            else:
                print(f"警告: {player}の点数を読み取れませんでした")
        
        return scores
    
    def process_uploaded_image(self, uploaded_file) -> Dict[str, int]:
        """Streamlitでアップロードされた画像を処理"""
        # アップロードされたファイルをOpenCV形式に変換
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("アップロードされた画像を読み込めませんでした")
        
        # 前処理
        processed_image = self.preprocess_image(image)
        
        # 点数表示領域の検出
        regions = self.detect_score_regions(image)
        
        # 各プレイヤーの点数を読み取り
        scores = {}
        for player, region in regions.items():
            region_image = self.extract_score_region(processed_image, region)
            score = self.read_score_from_region(region_image)
            
            if score is not None:
                scores[player] = score
            else:
                print(f"警告: {player}の点数を読み取れませんでした")
        
        return scores
    
    def debug_detection(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """デバッグ用：検出された領域を可視化"""
        debug_image = image.copy()
        processed_image = self.preprocess_image(image)
        regions = self.detect_score_regions(image)
        
        # 検出された領域を描画
        for player, (x1, y1, x2, y2) in regions.items():
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, player, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_image, list(regions.values())

def test_image_processor():
    """画像処理モジュールのテスト"""
    processor = ScoreImageProcessor()
    print("画像処理モジュールが正常に初期化されました")
    print("使用可能な機能:")
    print("- process_score_image(image_path): ファイルパスから画像を処理")
    print("- process_uploaded_image(uploaded_file): Streamlitアップロードファイルを処理")
    print("- detect_score_regions(image): 点数表示領域を自動検出")
    print("- debug_detection(image): 検出結果を可視化（デバッグ用）")

if __name__ == "__main__":
    test_image_processor()
