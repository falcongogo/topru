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
        self.ocr_config = '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
        
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
    
    def _find_all_candidate_rects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """画像からスコア表示の候補となる全ての長方形領域を検出する"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 白〜灰色の範囲を定義 (低彩度・中〜高輝度)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = image.shape[0] * image.shape[1]
        candidate_rects = []

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                # 個々のスコア表示領域らしいかどうかの条件
                is_candidate = (img_area * 0.005 < area < img_area * 0.15 and
                                1.0 < aspect_ratio < 8.0)

                if is_candidate:
                    candidate_rects.append((x, y, x + w, y + h))

        return candidate_rects

    def detect_score_regions(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """画像から4つの点数表示領域を検出し、プレイヤーを割り当てる"""
        candidates = self._find_all_candidate_rects(image)
        
        if len(candidates) < 4:
            print(f"警告: スコア領域が4つ未満しか検出できませんでした (検出数: {len(candidates)})")
            return {}

        # 面積でソートして上位4つを選択
        top_four = sorted(candidates, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:4]

        # 座標でソートして位置関係を把握
        # y座標でソート -> 上段と下段に分ける
        y_sorted = sorted(top_four, key=lambda r: r[1])
        top_row = sorted(y_sorted[:2], key=lambda r: r[0])
        bottom_row = sorted(y_sorted[2:], key=lambda r: r[0])

        top_left, top_right = top_row
        bottom_left, bottom_right = bottom_row

        # 最大領域が「自分」
        me_rect = max(top_four, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))

        # 自分(me)の位置に基づいて役割を割り当て
        # 一般的な麻雀卓のレイアウトを想定 (自分=下、対面=上、上家=左、下家=右)
        # カメラアングルによって回転する可能性があるため、自分の位置を基準とする
        if me_rect == bottom_left:
            return {'自分': bottom_left, '下家': bottom_right, '対面': top_right, '上家': top_left}
        elif me_rect == bottom_right:
            return {'自分': bottom_right, '下家': top_right, '対面': top_left, '上家': bottom_left}
        elif me_rect == top_right:
            return {'自分': top_right, '下家': top_left, '対面': bottom_left, '上家': bottom_right}
        elif me_rect == top_left:
            return {'自分': top_left, '下家': bottom_left, '対面': bottom_right, '上家': top_right}

        # 万が一、最大領域が4つの角のどれとも一致しない場合 (通常発生しない)
        return {}
    
    def extract_score_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """指定された領域を切り出し"""
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]
    
    def read_score_from_region(self, region_image: np.ndarray, player: str) -> Optional[int]:
        """指定された領域から点数を読み取り"""
        try:
            if region_image.shape[0] < 20 or region_image.shape[1] < 40:
                return None

            # グレースケールに変換
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)

            # 他家の場合は、下部の点差表示を無視するために上半分をクロップ
            if player != '自分':
                h, w = gray.shape
                gray = gray[0:int(h * 0.7)]

            # リサイズしてOCR精度を安定させる
            h, w = gray.shape
            if w == 0: return None
            resized = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

            # コントラスト強調
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(resized)

            # 二値化
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # OCRで文字認識
            text = pytesseract.image_to_string(binary, config=self.ocr_config)
            
            # 数字のみを抽出
            numbers = re.findall(r'\d+', text)
            
            if numbers:
                # 妥当なスコアが見つかったら、それを返す
                for number in numbers:
                    score = int(number)
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
        
        # 麻雀の点数として妥当かチェック（100点単位）
        if score % 100 != 0:
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
            score = self.read_score_from_region(region_image, player)
            
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
            score = self.read_score_from_region(region_image, player)
            
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

    def get_full_debug_bundle(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像処理の全ステップのデバッグ情報を生成する。
        """
        debug_bundle = {}
        original_image = image.copy()

        # 1. 色マスクの生成
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        debug_bundle['hsv_mask'] = mask

        # 2. 全ての候補領域を描画
        all_candidates_img = original_image.copy()
        # _find_all_candidate_rectsはx,y,w,hを返すので変換
        all_candidates_raw = self._find_all_candidate_rects(image)
        all_candidates = [(r[0], r[1], r[0]+r[2], r[1]+r[3]) for r in all_candidates_raw]

        for i, (x1, y1, x2, y2) in enumerate(all_candidates):
            cv2.rectangle(all_candidates_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(all_candidates_img, str(i), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        debug_bundle['all_candidates'] = all_candidates_img

        # 3. 上位4つの領域を描画
        top_four_img = original_image.copy()
        if len(all_candidates) >= 4:
            top_four_raw = sorted(all_candidates_raw, key=lambda r: r[2] * r[3], reverse=True)[:4]
            top_four = [(r[0], r[1], r[0]+r[2], r[1]+r[3]) for r in top_four_raw]
            for i, (x1, y1, x2, y2) in enumerate(top_four):
                cv2.rectangle(top_four_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        debug_bundle['top_four'] = top_four_img

        # 4. 最終的な割り当てを描画
        final_assignment_img = original_image.copy()
        regions = self.detect_score_regions(image)
        for player, (x1, y1, x2, y2) in regions.items():
            cv2.rectangle(final_assignment_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(final_assignment_img, player, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        debug_bundle['final_assignments'] = final_assignment_img

        # 5. 各プレイヤーのOCR前処理画像
        pre_ocr_images = {}
        for player, region in regions.items():
            try:
                region_image = self.extract_score_region(original_image, region)

                # read_score_from_regionのロジックをここに展開
                if region_image.shape[0] < 20 or region_image.shape[1] < 40: continue

                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                if player != '自分':
                    h, w = gray.shape
                    gray = gray[0:int(h * 0.7)]

                h, w = gray.shape
                if w == 0: continue
                resized = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(resized)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # OCRに渡す直前の画像を保存
                pre_ocr_images[player] = binary
            except Exception as e:
                print(f"デバッグ情報生成中にエラー: {player} - {e}")

        debug_bundle['pre_ocr_images'] = pre_ocr_images

        return debug_bundle

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
