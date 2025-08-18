import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List, Any
import config

class ScoreImageProcessor:
    """スリムスコア28Sの点数表示を読み取るクラス"""
    
    def __init__(self):
        # 設定をconfigから読み込む
        self.min_score = config.MIN_SCORE
        self.max_score = config.MAX_SCORE
        self.expected_digits = config.EXPECTED_DIGITS
        self.seven_segment_patterns = config.SEVEN_SEGMENT_PATTERNS

    def _find_main_score_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.HSV_LOWER_WHITE, config.HSV_UPPER_WHITE)

        dilate_kernel = np.ones(config.FRAME_DILATE_KERNEL_SIZE, np.uint8)
        erode_kernel = np.ones(config.FRAME_ERODE_KERNEL_SIZE, np.uint8)

        dilated = cv2.dilate(mask, dilate_kernel, iterations = 2)
        eroded = cv2.erode(dilated, erode_kernel, iterations = 2)

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        img_area = image.shape[0] * image.shape[1]
        contour_area = w * h

        min_area = img_area * config.FRAME_CONTOUR_AREA_MIN_RATIO
        max_area = img_area * config.FRAME_CONTOUR_AREA_MAX_RATIO

        if not (min_area < contour_area < max_area):
            return None
        return (x, y, w, h)

    def _find_inner_lcd_screen_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.HSV_LOWER_LCD_BLUE, config.HSV_UPPER_LCD_BLUE)

        kernel = np.ones(config.LCD_MORPH_CLOSE_KERNEL_SIZE, np.uint8)
        morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.LCD_MORPH_CLOSE_ITERATIONS)

        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        main_contour = max(contours, key=cv2.contourArea)
        min_area = image.shape[0] * image.shape[1] * config.LCD_CONTOUR_AREA_MIN_RATIO
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

    def _split_digits(self, image: np.ndarray, tight_crop: bool = True) -> List[np.ndarray]:
        """Finds the bounding box of all digits, then slices them based on that box."""
        # Find the bounding box of all digit contours combined
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Filter out very small noise contours before finding the main bounding box
        min_contour_area = 5  # A small threshold to filter noise
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if not valid_contours:
            return []

        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Crop the image to the bounding box of all digits
        digit_block = image[y:y+h, x:x+w]

        # Slice the digit block into individual digits
        h_block, w_block = digit_block.shape[:2]
        digit_width = w_block / self.expected_digits  # Use float for more precision
        results = []

        for i in range(self.expected_digits):
            x_start = int(i * digit_width)
            x_end = int((i + 1) * digit_width)

            # Slice the digit from the block
            digit_slice = digit_block[:, x_start:x_end]

            if digit_slice.size == 0:
                continue

            if not tight_crop:
                results.append(digit_slice)
                continue

            # Find the tight bounding box for the digit within the slice
            slice_contours, _ = cv2.findContours(digit_slice.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if slice_contours:
                main_contour = max(slice_contours, key=cv2.contourArea)
                sx, sy, sw, sh = cv2.boundingRect(main_contour)

                # Ensure the detected contour is reasonably sized
                if sw > config.DIGIT_MIN_WIDTH and sh > config.DIGIT_MIN_HEIGHT:
                    results.append(digit_slice[sy:sy+sh, sx:sx+sw])
                else:
                    results.append(digit_slice) # Append the slice if contour is too small
            else:
                results.append(digit_slice) # Append the slice if no contour found

        # Ensure we have the correct number of digits before returning
        if len(results) == self.expected_digits:
            return results
        else:
            # This can happen if some slices are empty or detection fails
            return []

    def _correct_shear_hough(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """ハフ変換で画像のせん断を補正し、補正後の画像と角度を返す"""
        edges = cv2.Canny(image, config.HOUGH_CANNY_THRESHOLD_1, config.HOUGH_CANNY_THRESHOLD_2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=config.HOUGH_THRESHOLD,
                                minLineLength=config.HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=config.HOUGH_MAX_LINE_GAP)
        if lines is None: return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))
            if config.HOUGH_ANGLE_MIN < angle_deg < config.HOUGH_ANGLE_MAX:
                deviation = angle_rad - (np.pi / 2)
                angles.append(deviation)

        if not angles: return image, 0.0

        # ヒストグラムで最頻値のビンの中心を求める
        counts, bin_edges = np.histogram(angles, bins=config.HOUGH_ANGLE_BINS, range=config.HOUGH_ANGLE_RANGE)
        max_index = np.argmax(counts)
        dominant_deviation_rad = (bin_edges[max_index] + bin_edges[max_index+1]) / 2

        # せん断係数を計算
        shear_factor = -np.tan(dominant_deviation_rad)

        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)

        dominant_angle_deg = np.degrees(dominant_deviation_rad)
        return corrected_image, dominant_angle_deg

    def _correct_shear_manual(self, image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, float]:
        """指定された角度でせん断補正を行う"""
        angle_rad = np.radians(angle_deg)
        # 指定された傾斜角度と逆方向に補正をかけるため、tanの結果を反転させる
        shear_factor = -np.tan(angle_rad)

        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)

        return corrected_image, angle_deg

    def _correct_shear_zeros(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """下二桁の'00'を認識してせん断補正を行う"""
        # Get uniformly sized slices for hstack
        digit_images = self._split_digits(image, tight_crop=False)

        if len(digit_images) < self.expected_digits:
            return image, 0.0

        zero_one = digit_images[-2]
        zero_two = digit_images[-1]

        if zero_one.size == 0 or zero_two.size == 0:
            return image, 0.0

        zeros_image = np.hstack((zero_one, zero_two))

        # Cannyの前に二値化してエッジを明確にする
        _, zeros_image_binary = cv2.threshold(zeros_image, 127, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(zeros_image_binary, config.HOUGH_CANNY_THRESHOLD_1, config.HOUGH_CANNY_THRESHOLD_2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=1, minLineLength=5, maxLineGap=10) # '00'用にパラメータを調整(超寛容)
        if lines is None:
            return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))
            if 80 < angle_deg < 100: # 垂直に近い線のみを対象
                deviation = angle_rad - (np.pi / 2)
                angles.append(deviation)

        if not angles:
            return image, 0.0

        counts, bin_edges = np.histogram(angles, bins=config.HOUGH_ANGLE_BINS, range=(-np.pi/9, np.pi/9)) # 20度以内
        if np.sum(counts) == 0:
            return image, 0.0

        max_index = np.argmax(counts)
        dominant_deviation_rad = (bin_edges[max_index] + bin_edges[max_index+1]) / 2

        # 検出された傾きと逆方向に補正するため、tanの結果を反転
        shear_factor = -np.tan(dominant_deviation_rad)

        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)

        dominant_angle_deg = np.degrees(dominant_deviation_rad)
        return corrected_image, dominant_angle_deg

    def _recognize_7_segment_digit(self, digit_image: np.ndarray) -> Optional[int]:
        if digit_image is None or digit_image.size == 0: return None
        h, w = digit_image.shape[:2]
        if h < config.DIGIT_MIN_HEIGHT or w < config.DIGIT_MIN_WIDTH: return None

        rois = config.SEVEN_SEGMENT_ROIS

        def is_on(segment_roi):
            x1, y1, x2, y2 = segment_roi
            roi_abs = digit_image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            if roi_abs.size == 0: return False
            active_pixels = np.count_nonzero(roi_abs)
            return (active_pixels / roi_abs.size) > config.SEGMENT_ACTIVATION_THRESHOLD if roi_abs.size > 0 else False
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
        x1_split = int(w * config.SCREEN_X1_SPLIT_RATIO)
        x2_split = int(w * config.SCREEN_X2_SPLIT_RATIO)

        left_img = screen_image[:, 0:x1_split]
        middle_img = screen_image[:, x1_split:x2_split]
        right_img = screen_image[:, x2_split:w]

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

    def _image_processing_pipeline(self, image: np.ndarray, debug=False,
                                   shear_correction_method: str = 'hough',
                                   manual_shear_angle: float = 0.0) -> Dict[str, Any]:
        """画像処理のメインパイプライン。通常処理とデバッグ処理を統合。"""
        debug_bundle = {}

        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            if debug:
                debug_bundle['warped_screen'] = np.zeros((100, 300, 3), dtype=np.uint8)
                cv2.putText(debug_bundle['warped_screen'], "Not Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return debug_bundle
            else:
                return {}

        if debug:
            debug_bundle['warped_screen'] = warped_screen

        gray = cv2.cvtColor(warped_screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones(config.BINARY_MORPH_OPEN_KERNEL_SIZE, np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=config.BINARY_MORPH_OPEN_ITERATIONS)

        if shear_correction_method == 'hough':
            corrected_binary, angle = self._correct_shear_hough(binary)
        elif shear_correction_method == 'manual':
            corrected_binary, angle = self._correct_shear_manual(binary, manual_shear_angle)
        elif shear_correction_method == 'zeros':
            corrected_binary, angle = self._correct_shear_zeros(binary)
        else:
            corrected_binary, angle = binary, 0.0

        if debug:
            debug_bundle['shear_corrected_screen'] = corrected_binary
            debug_bundle['shear_angles'] = {'screen': angle}

        region_images = self.split_screen_into_regions(corrected_binary)

        if debug:
            deskewed_digits_by_player = {}
            for player, region_image in region_images.items():
                if player != '自分':
                    h, w = region_image.shape
                    region_image = region_image[0:int(h * config.PLAYER_REGION_CROP_RATIO)]
                digits = self._split_digits(region_image)
                deskewed_digits_by_player[player] = digits if digits else []
            debug_bundle['deskewed_digits'] = deskewed_digits_by_player
            return debug_bundle
        else:
            scores = {}
            for player, region_image in region_images.items():
                if player != '自分':
                    h, w = region_image.shape
                    region_image = region_image[0:int(h * config.PLAYER_REGION_CROP_RATIO)]
                score = self._process_player_score(region_image, player)
                if score is not None:
                    scores[player] = score
            return scores

    def process_score_image(self, image: np.ndarray, shear_correction_method: str = 'zeros', manual_shear_angle: float = 0.0) -> Dict[str, int]:
        """画像から全プレイヤーの点数を読み取る"""
        return self._image_processing_pipeline(image, debug=False, shear_correction_method=shear_correction_method, manual_shear_angle=manual_shear_angle)

    def get_full_debug_bundle(self, image: np.ndarray, shear_correction_method: str = 'zeros', manual_shear_angle: float = 0.0) -> Dict[str, Any]:
        """デバッグ用の途中経過画像をすべて取得する"""
        return self._image_processing_pipeline(image, debug=True, shear_correction_method=shear_correction_method, manual_shear_angle=manual_shear_angle)
