import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from typing import Dict, Optional, Tuple, List, Any

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

        # LCDスクリーンの水色の範囲を定義
        lower_lcd_blue = np.array([85, 50, 100])
        upper_lcd_blue = np.array([105, 255, 255])

        # マスクを作成
        mask = cv2.inRange(hsv, lower_lcd_blue, upper_lcd_blue)

        # マスクのノイズを除去し、穴を埋めるための形態学的処理
        kernel = np.ones((5,5), np.uint8)
        morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 輪郭を検出
        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("警告: 水色の領域から輪郭を検出できませんでした。")
            return None

        # 最も面積の大きい輪郭を見つける
        main_contour = max(contours, key=cv2.contourArea)

        # 輪郭の面積が小さすぎる場合はノイズと判断
        min_area = image.shape[0] * image.shape[1] * 0.05 # 領域全体の5%未満は除外
        if cv2.contourArea(main_contour) < min_area:
            print(f"警告: 検出された水色領域が小さすぎます (面積: {cv2.contourArea(main_contour)})")
            return None

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

    def detect_and_warp_screen(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        画像からLCDスクリーンを検出し、傾きを補正した画像を返す。
        """
        # 1. 全体を囲む外側フレームを検出
        outer_frame_coords = self._find_main_score_frame(image)
        if outer_frame_coords is None:
            print("警告: スコア表示のメインフレームを検出できませんでした。")
            return None

        x_outer, y_outer, w_outer, h_outer = outer_frame_coords
        outer_frame_img = image[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer]

        # 2. 外側フレーム内で、内側のLCDスクリーンの輪郭情報を取得
        contour = self._find_inner_lcd_screen_contour(outer_frame_img)
        if contour is None:
            print("警告: 内側のLCDスクリーンを検出できませんでした。")
            return None

        # 3. 最小外接矩形の情報を取得
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        # 座標を絶対座標に変換
        box_abs = np.intp(box + (x_outer, y_outer))

        # 出力画像のサイズを決定 (width, height)
        (w, h) = rect[1]
        # ランドスケープモードを維持
        if w < h:
            (w, h) = (h, w)

        # 4. 検出した四隅を元に、傾きを補正した画像を取得
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

        # 水平に2:3:2の比率で分割
        total_parts = 7
        part_w = w / total_parts
        x1_split = int(2 * part_w)
        x2_split = int(5 * part_w)

        # 領域を定義
        left_img = screen_image[:, 0:x1_split]
        middle_img = screen_image[:, x1_split:x2_split]
        right_img = screen_image[:, x2_split:w]

        # 中央の領域を垂直に2分割
        mid_h = middle_img.shape[0] // 2
        middle_top_img = middle_img[0:mid_h, :]
        middle_bottom_img = middle_img[mid_h:, :]

        regions = {
            '上家': left_img,
            '対面': middle_top_img,
            '自分': middle_bottom_img,
            '下家': right_img
        }

        return regions
    
    def read_score_from_region(self, region_image: np.ndarray, player: str) -> Optional[int]:
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

            # 他家の場合は、下部の点差表示を無視するために上半分をクロップ
            if player != '自分':
                h, w = gray.shape
                gray = gray[0:int(h * 0.7)]

            # サイズ正規化（幅300pxに）
            h, w = gray.shape
            if w == 0: return None
            scale = 300 / w
            resized = cv2.resize(gray, (300, int(h * scale)))

            # ガウシアンブラーでノイズ除去
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)

            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
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
        
        # 麻雀の点数として妥当かチェック（100点単位）
        if score % 100 != 0:
            return False
        
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
        
        # 元の画像に検出したスクリーン領域の輪郭を描画
        outer_frame_coords = self._find_main_score_frame(image)
        if outer_frame_coords:
            x_outer, y_outer, w_outer, h_outer = outer_frame_coords
            outer_frame_img = image[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer]
            inner_lcd_corners_rel = self._find_inner_lcd_screen(outer_frame_img)

            if inner_lcd_corners_rel is not None:
                inner_lcd_corners_abs = inner_lcd_corners_rel + (x_outer, y_outer)
                cv2.drawContours(debug_image, [inner_lcd_corners_abs], -1, (0, 255, 0), 2)

        # 補正後のスクリーン画像を取得
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            # 失敗した場合は元の画像にテキストを描画
            cv2.putText(debug_image, "Screen not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return debug_image

        # 補正後の画像に分割線を描画
        h, w = warped_screen.shape[:2]
        total_parts = 7
        part_w = w / total_parts
        x1_split = int(2 * part_w)
        x2_split = int(5 * part_w)
        cv2.line(warped_screen, (x1_split, 0), (x1_split, h), (0, 0, 255), 1)
        cv2.line(warped_screen, (x2_split, 0), (x2_split, h), (0, 0, 255), 1)

        mid_w = x2_split - x1_split
        mid_h = h // 2
        cv2.line(warped_screen, (x1_split, mid_h), (x2_split, mid_h), (0, 0, 255), 1)

        # 2つの画像を結合して表示
        h1, w1 = debug_image.shape[:2]
        h2, w2 = warped_screen.shape[:2]
        combined_height = h1 + h2
        combined_width = max(w1, w2)

        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_image[0:h1, 0:w1] = debug_image
        combined_image[h1:h1+h2, 0:w2] = warped_screen
        
        return combined_image

    def get_full_debug_bundle(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像処理の全ステップのデバッグ情報を生成する。
        """
        debug_bundle = {}
        original_image = image.copy()

        # 1. HSVカラーマスク
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        debug_bundle['hsv_mask'] = mask

        # 2. 形態学的処理後のマスク
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations = 2)
        eroded = cv2.erode(dilated, kernel, iterations = 2)
        debug_bundle['morph_mask'] = eroded

        # 3. 検出されたメインフレーム
        main_frame_img = original_image.copy()
        main_frame = self._find_main_score_frame(image)
        if main_frame:
            x, y, w, h = main_frame
            cv2.rectangle(main_frame_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        debug_bundle['main_frame'] = main_frame_img

        # 4. 内側LCD検出用の連結成分解析の可視化
        inner_lcd_debug_img = original_image.copy()
        outer_frame = self._find_main_score_frame(image)
        if outer_frame:
            x_outer, y_outer, w_outer, h_outer = outer_frame
            outer_frame_img_cropped = image[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer]

            # 色マスク作成
            hsv = cv2.cvtColor(outer_frame_img_cropped, cv2.COLOR_BGR2HSV)
            lower_lcd_color = np.array([0, 0, 100])
            upper_lcd_color = np.array([180, 80, 255])
            mask = cv2.inRange(hsv, lower_lcd_color, upper_lcd_color)

            # 連結成分を解析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

            # 各連結成分を色付けして可視化
            if num_labels > 1:
                # 背景を除いたラベルにランダムな色を割り当て
                colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
                colors[0] = [0, 0, 0] # 背景は黒
                colored_labels = colors[labels]
                # 元画像とブレンドして見やすくする
                inner_lcd_debug_img[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer] = cv2.addWeighted(outer_frame_img_cropped, 0.6, colored_labels.astype(np.uint8), 0.4, 0)

            # 最終的なバウンディングボックスを描画
            inner_lcd_rel = self._find_inner_lcd_screen(outer_frame_img_cropped)
            if inner_lcd_rel:
                x_rel, y_rel, w_rel, h_rel = inner_lcd_rel
                cv2.rectangle(inner_lcd_debug_img, (x_outer + x_rel, y_outer + y_rel), (x_outer + x_rel + w_rel, y_outer + y_rel + h_rel), (0, 255, 255), 2) # 黄色

        debug_bundle['inner_lcd_components'] = inner_lcd_debug_img

        # 6. 最終的な割り当て（4分割）
        final_assignment_img = original_image.copy()
        regions = self.detect_score_regions(image)
        for player, (x1, y1, x2, y2) in regions.items():
            cv2.rectangle(final_assignment_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(final_assignment_img, player, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        debug_bundle['final_assignments'] = final_assignment_img

        # 6. 各プレイヤーのOCR前処理画像
        pre_ocr_images = {}
        for player, region in regions.items():
            try:
                region_image = self.extract_score_region(original_image, region)

                # read_score_from_regionのロジックをここに展開して、中間画像を生成
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
                blurred = cv2.GaussianBlur(resized, (3, 3), 0)
                enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                pre_ocr_images[player] = binary
            except Exception as e:
                print(f"デバッグ情報（OCR画像）生成中にエラー: {player} - {e}")

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
