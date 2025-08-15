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
    
    def _find_score_rectangles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """画像からスコア表示と思われる白い長方形の領域を検出する"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 白色の範囲を定義 (低彩度・高明度) - パラメータを緩和
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 70, 255])
        
        # マスクを作成
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = image.shape[0] * image.shape[1]
        candidate_rects = []

        for cnt in contours:
            # 輪郭を多角形で近似
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 4つの頂点を持つ輪郭（四角形）のみを対象
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)

                # 面積とアスペクト比でフィルタリング
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                # スコア表示領域らしいかどうかの条件 (パラメータを緩和)
                # 面積: 画像全体の0.5%～20%
                # アスペクト比: 1.5～10.0 (横長)
                is_candidate = (img_area * 0.005 < area < img_area * 0.20 and
                                1.5 < aspect_ratio < 10.0)

                if is_candidate:
                    candidate_rects.append((x, y, x + w, y + h))
        
        return candidate_rects
    
    def detect_score_regions(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """画像から点数表示領域を自動検出"""
        # 白い長方形の領域を検出
        rects = self._find_score_rectangles(image)
        
        # 4つの領域が検出されなかった場合は空の結果を返す
        if len(rects) < 4:
            print(f"警告: スコア領域が4つ未満しか検出できませんでした (検出数: {len(rects)})")
            return {}

        # 4つ以上見つかった場合、面積が大きい順にソートして上位4つを選択
        # これにより、ノイズで小さな四角形が検出された場合に対応
        sorted_rects = sorted(rects, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
        top_four_rects = sorted_rects[:4]

        # プレイヤー名を割り当て（位置に基づいて）
        # y座標、x座標でソート
        top_four_rects.sort(key=lambda r: (r[1], r[0]))

        result = {}
        positions = ['自分', '下家', '対面', '上家']
        for i, region in enumerate(top_four_rects):
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

            # 余白をカットして枠線の影響を除去 (上下左右8%)
            h, w = region_image.shape[:2]
            margin_y, margin_x = int(h * 0.08), int(w * 0.08)
            roi = region_image[margin_y : h - margin_y, margin_x : w - margin_x]

            if roi.shape[0] < 5 or roi.shape[1] < 10:
                return None

            # OCRに適した前処理
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            # 背景を白、文字を黒に
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # OCRで文字認識
            text = pytesseract.image_to_string(binary, config=self.ocr_config)
            
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
        
        # 点数表示領域の検出
        regions = self.detect_score_regions(image)
        
        # 各プレイヤーの点数を読み取り
        scores = {}
        for player, region in regions.items():
            # 検出された領域を元のカラー画像から切り出す
            region_image = self.extract_score_region(image, region)
            # 領域内の画像をOCRで読み取り
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
        
        # 点数表示領域の検出
        regions = self.detect_score_regions(image)
        
        # 各プレイヤーの点数を読み取り
        scores = {}
        for player, region in regions.items():
            # 検出された領域を元のカラー画像から切り出す
            region_image = self.extract_score_region(image, region)
            # 領域内の画像をOCRで読み取り
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
