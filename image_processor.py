import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List, Any

class ScoreImageProcessor:
    """スリムスコア28Sの点数表示を読み取るクラス"""
    
    def __init__(self):
        self.min_score = 1000
        self.max_score = 99999
        self.expected_digits = 5
        self.seven_segment_patterns = {
            (True, True, True, True, True, True, False): 0, (False, True, True, False, False, False, False): 1,
            (True, True, False, True, True, False, True): 2, (True, True, True, True, False, False, True): 3,
            (False, True, True, False, False, True, True): 4, (True, False, True, True, False, True, True): 5,
            (True, False, True, True, True, True, True): 6, (True, True, True, False, False, False, False): 7,
            (True, True, True, True, True, True, True): 8, (True, True, True, True, False, True, True): 9,
        }

    def _find_main_score_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations = 2)
        eroded = cv2.erode(dilated, kernel, iterations = 2)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        img_area = image.shape[0] * image.shape[1]
        contour_area = w * h
        if not (img_area * 0.01 < contour_area < img_area * 0.9):
            return None
        return (x, y, w, h)

    def _find_inner_lcd_screen_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_lcd_blue = np.array([85, 50, 100])
        upper_lcd_blue = np.array([105, 255, 255])
        mask = cv2.inRange(hsv, lower_lcd_blue, upper_lcd_blue)
        kernel = np.ones((5,5), np.uint8)
        morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        main_contour = max(contours, key=cv2.contourArea)
        min_area = image.shape[0] * image.shape[1] * 0.05
        if cv2.contourArea(main_contour) < min_area:
            return None
        return main_contour

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
        rect = self._order_points(pts)
        (w, h) = dst_size
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (w, h))
        return warped

    def _split_digits(self, image: np.ndarray) -> List[np.ndarray]:
        h, w = image.shape[:2]
        digit_width = w // self.expected_digits
        results = []
        for i in range(self.expected_digits):
            x_start = i * digit_width
            x_end = (i + 1) * digit_width
            margin = int(digit_width * 0.05)
            digit_slice = image[:, x_start + margin : x_end - margin]
            if digit_slice.size == 0: continue
            contours, _ = cv2.findContours(digit_slice.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                final_contour = max(contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(final_contour)
                if w_b > 5 and h_b > 10:
                    results.append(digit_slice[y_b:y_b+h_b, x_b:x_b+w_b])
                else:
                    results.append(digit_slice)
            else:
                 results.append(digit_slice)
        if len(results) == self.expected_digits:
            return results
        else:
            return []

    def _correct_shear(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """ハフ変換で画像のせん断を補正し、補正後の画像と角度を返す"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=5)
        if lines is None: return image, 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))
            if 75 < angle_deg < 105:
                deviation = angle_rad - (np.pi / 2)
                angles.append(deviation)
        if not angles: return image, 0.0
        counts, bin_edges = np.histogram(angles, bins=20, range=(-np.pi/4, np.pi/4))
        dominant_deviation_rad = bin_edges[np.argmax(counts)]
        shear_factor = np.tan(dominant_deviation_rad)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dominant_angle_deg = np.degrees(dominant_deviation_rad)
        return corrected_image, dominant_angle_deg

    def _recognize_7_segment_digit(self, digit_image: np.ndarray) -> Optional[int]:
        if digit_image is None or digit_image.size == 0: return None
        h, w = digit_image.shape[:2]
        if h < 10 or w < 5: return None
        rois = {
            'a': (0.2, 0.0, 0.8, 0.2), 'b': (0.7, 0.1, 1.0, 0.45), 'c': (0.7, 0.55, 1.0, 0.9),
            'd': (0.2, 0.8, 0.8, 1.0), 'e': (0.0, 0.55, 0.3, 0.9), 'f': (0.0, 0.1, 0.3, 0.45),
            'g': (0.2, 0.4, 0.8, 0.6)
        }
        def is_on(segment_roi):
            x1, y1, x2, y2 = segment_roi
            roi_abs = digit_image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            if roi_abs.size == 0: return False
            active_pixels = np.count_nonzero(roi_abs)
            return (active_pixels / roi_abs.size) > 0.1
        try:
            pattern = tuple(is_on(rois[seg]) for seg in ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
            return self.seven_segment_patterns.get(pattern)
        except (IndexError, ValueError):
            return None

    def detect_and_warp_screen(self, image: np.ndarray) -> Optional[np.ndarray]:
        outer_frame_coords = self._find_main_score_frame(image)
        search_image, offset = (image[outer_frame_coords[1]:outer_frame_coords[1]+outer_frame_coords[3], outer_frame_coords[0]:outer_frame_coords[0]+outer_frame_coords[2]], (outer_frame_coords[0], outer_frame_coords[1])) if outer_frame_coords else (image, (0, 0))
        if not outer_frame_coords:
            print("情報: 外側フレームが見つからないため、画像全体からLCDスクリーンを探索します。")
        contour = self._find_inner_lcd_screen_contour(search_image)
        if contour is None: return None
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_abs = np.intp(box + offset)
        (w, h) = rect[1]
        if w < h: (w, h) = (h, w)
        warped_screen = self._four_point_transform(image, box_abs, (int(w), int(h)))
        return warped_screen

    def split_screen_into_regions(self, screen_image: np.ndarray) -> Dict[str, np.ndarray]:
        if screen_image is None or screen_image.size == 0: return {}
        h, w = screen_image.shape[:2]
        total_parts, x1_split, x2_split = 7, int(2 * (w / 7)), int(5 * (w / 7))
        left_img, middle_img, right_img = screen_image[:, 0:x1_split], screen_image[:, x1_split:x2_split], screen_image[:, x2_split:w]
        mid_h = middle_img.shape[0] // 2
        return {'上家': left_img, '対面': middle_img[0:mid_h, :], '自分': middle_img[mid_h:, :], '下家': right_img}
    
    def _is_valid_score(self, score: int) -> bool:
        if not (self.min_score <= score <= self.max_score): return False
        if len(str(score)) != self.expected_digits: return False
        return True

    def _process_player_score(self, region_image: np.ndarray, player: str) -> Optional[int]:
        """完全に補正されたプレイヤー領域の画像からスコアを読み取る"""
        try:
            if region_image.size == 0: return None
            digit_images = self._split_digits(region_image)
            if len(digit_images) != self.expected_digits:
                print(f"警告: {player}の領域で期待される桁数を切り出せませんでした。")
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

    def process_score_image(self, image: np.ndarray) -> Dict[str, int]:
        """画像から全プレイヤーの点数を読み取り（新しいパイプライン）"""
        # 1. パース補正
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None: return {}
        
        # 2. 二値化
        gray = cv2.cvtColor(warped_screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3. せん断補正
        corrected_binary, _ = self._correct_shear(binary)

        # 4. 領域分割
        region_images = self.split_screen_into_regions(corrected_binary)

        # 5. 各領域からスコア読み取り
        scores = {}
        for player, region_image in region_images.items():
            # 他家は上半分をクロップ
            if player != '自分':
                h, w = region_image.shape
                region_image = region_image[0:int(h * 0.7)]

            score = self._process_player_score(region_image, player)
            if score is not None:
                scores[player] = score
        return scores

    def process_uploaded_image(self, uploaded_file) -> Dict[str, int]:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("アップロードされた画像を読み込めませんでした")
        return self.process_score_image(image)

    def get_full_debug_bundle(self, image: np.ndarray) -> Dict[str, Any]:
        debug_bundle = {}
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            debug_bundle['warped_screen'] = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(debug_bundle['warped_screen'], "Not Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return debug_bundle
        debug_bundle['warped_screen'] = warped_screen

        gray = cv2.cvtColor(warped_screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        corrected_binary, angle = self._correct_shear(binary)
        debug_bundle['shear_corrected_screen'] = corrected_binary
        debug_bundle['shear_angles'] = {'screen': angle}

        region_images = self.split_screen_into_regions(corrected_binary)
        debug_bundle['split_regions'] = {}
        deskewed_digits_by_player = {}

        for player, region_image in region_images.items():
            # 他家は上半分をクロップ
            if player != '自分':
                h, w = region_image.shape
                region_image = region_image[0:int(h * 0.7)]

            debug_bundle['split_regions'][player] = region_image
            deskewed_digits_by_player[player] = self._split_digits(region_image)

        debug_bundle['deskewed_digits'] = deskewed_digits_by_player
        return debug_bundle
