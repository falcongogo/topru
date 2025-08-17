import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List, Any

class ScoreImageProcessor:
    """スリムスコア28Sの点数表示を読み取るクラス"""
    
    def __init__(self):
        # 点数表示の特徴
        self.min_score = 1000
        self.max_score = 99999
        self.expected_digits = 5  # 28000のような5桁の数字

        # 7セグメントディスプレイのパターン (a, b, c, d, e, f, g)
        #   a
        # f   b
        #   g
        # e   c
        #   d
        self.seven_segment_patterns = {
            (True, True, True, True, True, True, False): 0,  # 0
            (False, True, True, False, False, False, False): 1,  # 1
            (True, True, False, True, True, False, True): 2,  # 2
            (True, True, True, True, False, False, True): 3,  # 3
            (False, True, True, False, False, True, True): 4,  # 4
            (True, False, True, True, False, True, True): 5,  # 5
            (True, False, True, True, True, True, True): 6,  # 6
            (True, True, True, False, False, False, False): 7,  # 7
            (True, True, True, True, True, True, True): 8,  # 8
            (True, True, True, True, False, True, True): 9,  # 9
        }
    
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

    def _find_main_score_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """画像からスコア表示全体を囲む最も大きな白い長方形の領域を検出する"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 白〜灰色の範囲を定義 (低彩度・中〜高輝度)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        
        # マスクを作成
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 枠線の途切れを補完するための形態学的処理
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations = 2)
        eroded = cv2.erode(dilated, kernel, iterations = 2)

        # 輪郭を検出
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # 最も面積の大きい輪郭を見つける
        main_contour = max(contours, key=cv2.contourArea)

        # 輪郭の外接矩形を取得
        x, y, w, h = cv2.boundingRect(main_contour)

        # 画像全体の面積に対する割合でフィルタリング
        img_area = image.shape[0] * image.shape[1]
        contour_area = w * h

        # あまりに小さい、または大きすぎる領域は除外
        if not (img_area * 0.01 < contour_area < img_area * 0.9):
            print(f"警告: 検出されたメイン領域のサイズが不適切です (画像全体の{contour_area/img_area:.1%})")
            return None

        return (x, y, w, h)

    def _find_inner_lcd_screen_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        画像領域内から、特徴的な水色を頼りにLCDスクリーン領域の輪郭を見つけて返す。
        """
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # LCDスクリーンの水色の範囲を定義 (範囲を少し広げる)
        lower_lcd_blue = np.array([80, 50, 100])
        upper_lcd_blue = np.array([110, 255, 255])

        # マスクを作成
        mask = cv2.inRange(hsv, lower_lcd_blue, upper_lcd_blue)

        # マスクのノイズを除去し、穴を埋めるための形態学的処理（イテレーションを減らす）
        kernel = np.ones((5,5), np.uint8)
        morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 輪郭を検出
        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("警告: 水色の領域から輪郭を検出できませんでした。")
            return None

        # 最も面積の大きい輪郭を見つける
        main_contour = max(contours, key=cv2.contourArea)

        # 輪郭の面積チェックを削除（画像がタイトにクロップされている場合に対応）
        return main_contour

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """4つの点を top-left, top-right, bottom-right, bottom-left の順に並べ替える"""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference

        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
        """4つの点と出力サイズを受け取り、画像を正面から見たように補正する"""
        rect = self._order_points(pts)
        (w, h) = dst_size

        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (w, h))

        return warped

    def _split_digits(self, image: np.ndarray) -> List[np.ndarray]:
        """
        画像を固定幅で5分割する
        戻り値: 数字画像(np.ndarray)のリスト
        """
        h, w = image.shape[:2]
        digit_width = w // self.expected_digits

        results = []
        for i in range(self.expected_digits):
            x_start = i * digit_width
            x_end = (i + 1) * digit_width

            # 各桁の間にわずかな隙間を設ける
            margin = int(digit_width * 0.05)
            digit_slice = image[:, x_start + margin : x_end - margin]

            # スライス内で輪郭を見つけてタイトにクロップ
            contours, _ = cv2.findContours(digit_slice.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                final_contour = max(contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(final_contour)
                # 小さすぎるノイズは除去
                if w_b > 5 and h_b > 10:
                    results.append(digit_slice[y_b:y_b+h_b, x_b:x_b+w_b])
                else:
                    results.append(digit_slice) # ノイズならそのまま
            else:
                 results.append(digit_slice)

        if len(results) == self.expected_digits:
            return results
        else:
            print(f"警告: 固定幅分割で期待される桁数({self.expected_digits})を検出できませんでした。")
            return []

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        ハフ変換を用いて、傾いた画像をまっすぐに補正する。
        画像は黒背景(0)、白文字(255)の二値化画像であることを前提とする。
        """
        # Cannyエッジ検出
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # 確率的ハフ変換で直線を検出 (より厳しいパラメータ)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=20, maxLineGap=3)

        if lines is None:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                 angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return deskewed

    def _recognize_7_segment_digit(self, digit_image: np.ndarray) -> Optional[int]:
        """
        単一の数字画像から7セグメントのパターンを読み取り、数字を返す。
        画像は白文字(255)と黒い背景(0)に二値化されていることを前提とする。
        """
        if digit_image is None or digit_image.size == 0:
            return None

        h, w = digit_image.shape[:2]
        if h < 10 or w < 5: # 小さすぎる画像は処理しない
            return None

        rois = {
            'a': (0.2, 0.0, 0.8, 0.2), 'b': (0.7, 0.1, 1.0, 0.45),
            'c': (0.7, 0.55, 1.0, 0.9), 'd': (0.2, 0.8, 0.8, 1.0),
            'e': (0.0, 0.55, 0.3, 0.9), 'f': (0.0, 0.1, 0.3, 0.45),
            'g': (0.2, 0.4, 0.8, 0.6)
        }

        def is_on(segment_roi):
            x1, y1, x2, y2 = segment_roi
            roi_abs = digit_image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            if roi_abs.size == 0: return False
            active_pixels = np.count_nonzero(roi_abs)
            total_pixels = roi_abs.size
            return (active_pixels / total_pixels) > 0.1

        try:
            pattern = tuple(is_on(rois[seg]) for seg in ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        except (IndexError, ValueError):
            return None

        return self.seven_segment_patterns.get(pattern)

    def detect_and_warp_screen(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        画像からLCDスクリーンを検出し、傾きを補正した画像を返す。
        (外側フレーム検出をバイパスし、画像全体から直接LCDスクリーンを探すように修正)
        """
        contour = self._find_inner_lcd_screen_contour(image)
        if contour is None:
            print("警告: 内側のLCDスクリーンを検出できませんでした。")
            return None

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_abs = np.intp(box)

        (w, h) = rect[1]
        if w < h: (w, h) = (h, w)

        warped_screen = self._four_point_transform(image, box_abs, (int(w), int(h)))
        return warped_screen

    def split_screen_into_regions(self, screen_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        補正済みのスクリーン画像を4つのプレイヤー領域に分割し、
        それぞれの領域の画像を辞書で返す。
        """
        if screen_image is None or screen_image.size == 0:
            return {}

        h, w = screen_image.shape[:2]
        total_parts = 7
        part_w = w / total_parts
        x1_split = int(2 * part_w)
        x2_split = int(5 * part_w)

        left_img = screen_image[:, 0:x1_split]
        middle_img = screen_image[:, x1_split:x2_split]
        right_img = screen_image[:, x2_split:w]

        mid_h = middle_img.shape[0] // 2
        middle_top_img = middle_img[0:mid_h, :]
        middle_bottom_img = middle_img[mid_h:, :]

        return {
            '上家': left_img, '対面': middle_top_img,
            '自分': middle_bottom_img, '下家': right_img
        }
    
    def read_score_from_region(self, region_image: np.ndarray, player: str) -> Optional[int]:
        """指定された領域から点数を読み取り"""
        try:
            if region_image.shape[0] < 10 or region_image.shape[1] < 20: return None

            h, w = region_image.shape[:2]
            margin_y, margin_x = int(h * 0.08), int(w * 0.08)
            roi = region_image[margin_y : h - margin_y, margin_x : w - margin_x]
            if roi.shape[0] < 5 or roi.shape[1] < 10: return None

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if player != '自分':
                h_gray, w_gray = gray.shape
                gray = gray[0:int(h_gray * 0.7)]

            h_gray, w_gray = gray.shape
            if w_gray == 0: return None
            scale = 300 / w_gray
            resized = cv2.resize(gray, (300, int(h_gray * scale)))

            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 1. 領域全体の傾きを補正
            deskewed_region = self._deskew_image(binary)
            
            # 2. 傾き補正された画像から数字を切り出し
            digit_images = self._split_digits(deskewed_region)

            if len(digit_images) != self.expected_digits:
                return None

            recognized_digits = []
            for digit_img in digit_images:
                digit = self._recognize_7_segment_digit(digit_img)
                if digit is not None:
                    recognized_digits.append(str(digit))
                else:
                    print(f"警告: {player}の領域で数字の一部の認識に失敗しました。")
                    return None

            if len(recognized_digits) == self.expected_digits:
                if recognized_digits[-1] != '0' or recognized_digits[-2] != '0':
                    print(f"警告: {player}の領域で読み取った点数の下2桁が00ではありません: {''.join(recognized_digits)}")
                    return None
                score = int("".join(recognized_digits))
                if self._is_valid_score(score):
                    return score
            return None
        except Exception as e:
            print(f"OCR読み取りエラー: {player} - {e}")
            return None
    
    def _is_valid_score(self, score: int) -> bool:
        """点数として妥当かチェック"""
        if not (self.min_score <= score <= self.max_score): return False
        if len(str(score)) != self.expected_digits: return False
        return True
    
    def process_score_image(self, image_path: str) -> Dict[str, int]:
        """画像から全プレイヤーの点数を読み取り"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            return {}

        region_images = self.split_screen_into_regions(warped_screen)
        
        scores = {}
        for player, region_image in region_images.items():
            score = self.read_score_from_region(region_image, player)
            if score is not None:
                scores[player] = score
            else:
                print(f"警告: {player}の点数を読み取れませんでした")
        
        return scores
    
    def process_uploaded_image(self, uploaded_file) -> Dict[str, int]:
        """Streamlitでアップロードされた画像を処理"""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("アップロードされた画像を読み込めませんでした")
        
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            return {}

        region_images = self.split_screen_into_regions(warped_screen)
        
        scores = {}
        for player, region_image in region_images.items():
            score = self.read_score_from_region(region_image, player)
            if score is not None:
                scores[player] = score
            else:
                print(f"警告: {player}の点数を読み取れませんでした")
        
        return scores
    
    def debug_detection(self, image: np.ndarray) -> Optional[np.ndarray]:
        """デバッグ用：検出された領域と補正後のスクリーンを可視化"""
        debug_image = image.copy()
        
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            cv2.putText(debug_image, "Screen not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return debug_image

        h, w = warped_screen.shape[:2]
        total_parts = 7
        part_w = w / total_parts
        x1_split = int(2 * part_w)
        x2_split = int(5 * part_w)
        cv2.line(warped_screen, (x1_split, 0), (x1_split, h), (0, 0, 255), 1)
        cv2.line(warped_screen, (x2_split, 0), (x2_split, h), (0, 0, 255), 1)
        mid_h = h // 2
        cv2.line(warped_screen, (x1_split, mid_h), (x2_split, mid_h), (0, 0, 255), 1)

        h1, w1 = debug_image.shape[:2]
        h2, w2 = warped_screen.shape[:2]
        combined_width = w1
        combined_height = h1 + int(h2 * (w1 / w2))
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_image[0:h1, 0:w1] = debug_image
        resized_warped = cv2.resize(warped_screen, (w1, int(h2 * (w1 / w2))))
        combined_image[h1:, 0:w1] = resized_warped
        return combined_image

    def get_full_debug_bundle(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像処理の全ステップのデバッグ情報を生成する。
        """
        debug_bundle = {}
        debug_bundle['main_frame'] = image.copy()

        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            debug_bundle['warped_screen'] = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(debug_bundle['warped_screen'], "Not Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return debug_bundle
        debug_bundle['warped_screen'] = warped_screen

        region_images = self.split_screen_into_regions(warped_screen)
        debug_bundle['split_regions'] = region_images

        pre_ocr_images = {}
        deskewed_digits_by_player = {}

        for player, region_image in region_images.items():
            try:
                if region_image.shape[0] < 10 or region_image.shape[1] < 20: continue
                h, w = region_image.shape[:2]
                margin_y, margin_x = int(h * 0.08), int(w * 0.08)
                roi = region_image[margin_y : h - margin_y, margin_x : w - margin_x]
                if roi.shape[0] < 5 or roi.shape[1] < 10: continue
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if player != '自分':
                    h_gray, w_gray = gray.shape
                    gray = gray[0:int(h_gray * 0.7)]
                h_gray, w_gray = gray.shape
                if w_gray == 0: continue
                scale = 300 / w_gray
                resized = cv2.resize(gray, (300, int(h_gray * scale)))
                blurred = cv2.GaussianBlur(resized, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

                deskewed_region = self._deskew_image(binary)
                pre_ocr_images[player] = deskewed_region

                digit_images = self._split_digits(deskewed_region)
                if not digit_images: continue

                deskewed_digits_by_player[player] = digit_images
            except Exception as e:
                print(f"デバッグ情報生成中にエラー: {player} - {e}")

        debug_bundle['pre_ocr_images'] = pre_ocr_images
        debug_bundle['deskewed_digits'] = deskewed_digits_by_player
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
