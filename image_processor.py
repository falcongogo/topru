import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, Optional, Tuple, List, Any
import config

class ScoreImageProcessor:
    """スリムスコア28Sの点数表示を読み取るクラス"""
    
    def __init__(self):
        self.min_score = config.MIN_SCORE
        self.max_score = config.MAX_SCORE
        self.expected_digits = config.EXPECTED_DIGITS
        self.seven_segment_patterns = config.SEVEN_SEGMENT_PATTERNS

        template_path = 'tests/test_data/template_00.png'
        self.template_00 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template_00 is None:
            print(f"警告: テンプレート画像が見つかりません: {template_path}。アンカーベースの探索は無効になります。")
            self.template_00 = np.zeros((10, 10), dtype=np.uint8)

    def _find_score_by_00_anchor(self, region_image: np.ndarray) -> Optional[np.ndarray]:
        if region_image.size == 0 or self.template_00 is None or self.template_00.shape[0] < 2:
            return None

        template = self.template_00
        h, w = region_image.shape
        th, tw = template.shape

        if h < th or w < tw: return None

        res = cv2.matchTemplate(region_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        threshold = 0.6
        if max_val < threshold:
            return None

        tl = max_loc

        w00 = tw
        score_width_estimate = w00 * (5 / 2)
        score_x_start = (tl[0] + tw) - score_width_estimate

        crop_x_start = max(0, int(score_x_start))
        crop_x_end = tl[0] + tw
        y_padding = int(th * 0.2)
        crop_y_start = max(0, tl[1] - y_padding)
        crop_y_end = min(h, tl[1] + th + y_padding)

        return region_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    def _recognize_score_from_image(self, score_image: np.ndarray) -> Optional[int]:
        if score_image is None or score_image.size == 0: return None
        try:
            padded_image = cv2.copyMakeBorder(score_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(padded_image, config=custom_config)
            cleaned_text = "".join(filter(str.isdigit, text))
            if not cleaned_text: return None
            score = int(cleaned_text)
            if not str(score).endswith("00"): return None
            if self._is_valid_score(score): return score
            return None
        except (pytesseract.TesseractNotFoundError, ValueError, TypeError):
            return None

    def _find_main_score_frame(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.HSV_LOWER_WHITE, config.HSV_UPPER_WHITE)
        dilate_kernel = np.ones(config.FRAME_DILATE_KERNEL_SIZE, np.uint8)
        erode_kernel = np.ones(config.FRAME_ERODE_KERNEL_SIZE, np.uint8)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=2)
        eroded = cv2.erode(dilated, erode_kernel, iterations=2)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        img_area = image.shape[0] * image.shape[1]
        contour_area = w * h
        min_area = img_area * config.FRAME_CONTOUR_AREA_MIN_RATIO
        max_area = img_area * config.FRAME_CONTOUR_AREA_MAX_RATIO
        if not (min_area < contour_area < max_area): return None
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
        if cv2.contourArea(main_contour) < min_area: return None
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
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []
        min_contour_area = 5
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if not valid_contours: return []
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        digit_block = image[y:y+h, x:x+w]
        h_block, w_block = digit_block.shape[:2]
        digit_width = w_block / self.expected_digits
        results = []
        for i in range(self.expected_digits):
            x_start, x_end = int(i * digit_width), int((i + 1) * digit_width)
            digit_slice = digit_block[:, x_start:x_end]
            if digit_slice.size == 0: continue
            if not tight_crop:
                results.append(digit_slice)
                continue
            slice_contours, _ = cv2.findContours(digit_slice.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if slice_contours:
                main_contour = max(slice_contours, key=cv2.contourArea)
                sx, sy, sw, sh = cv2.boundingRect(main_contour)
                if sw > config.DIGIT_MIN_WIDTH and sh > config.DIGIT_MIN_HEIGHT:
                    results.append(digit_slice[sy:sy+sh, sx:sx+sw])
                else:
                    results.append(digit_slice)
            else:
                results.append(digit_slice)
        if len(results) == self.expected_digits: return results
        return []

    def _correct_shear_hough(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        edges = cv2.Canny(image, config.HOUGH_CANNY_THRESHOLD_1, config.HOUGH_CANNY_THRESHOLD_2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=config.HOUGH_THRESHOLD, minLineLength=config.HOUGH_MIN_LINE_LENGTH, maxLineGap=config.HOUGH_MAX_LINE_GAP)
        if lines is None: return image, 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))
            if config.HOUGH_ANGLE_MIN < angle_deg < config.HOUGH_ANGLE_MAX:
                angles.append(angle_rad - (np.pi / 2))
        if not angles: return image, 0.0
        counts, bin_edges = np.histogram(angles, bins=config.HOUGH_ANGLE_BINS, range=config.HOUGH_ANGLE_RANGE)
        max_index = np.argmax(counts)
        dominant_deviation_rad = (bin_edges[max_index] + bin_edges[max_index+1]) / 2
        shear_factor = -np.tan(dominant_deviation_rad)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)
        return corrected_image, np.degrees(dominant_deviation_rad)

    def _correct_shear_manual(self, image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, float]:
        angle_rad = np.radians(angle_deg)
        shear_factor = -np.tan(angle_rad)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)
        return corrected_image, angle_deg

    def _correct_shear_zeros(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        digit_images = self._split_digits(image, tight_crop=False)
        if len(digit_images) < self.expected_digits: return image, 0.0
        zero_one, zero_two = digit_images[-2], digit_images[-1]
        if zero_one.size == 0 or zero_two.size == 0: return image, 0.0
        zeros_image = np.hstack((zero_one, zero_two))
        _, zeros_image_binary = cv2.threshold(zeros_image, 127, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(zeros_image_binary, config.HOUGH_CANNY_THRESHOLD_1, config.HOUGH_CANNY_THRESHOLD_2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=5, maxLineGap=10)
        if lines is None: return image, 0.0
        angles = [np.arctan2(y2 - y1, x2 - x1) - (np.pi / 2) for line in lines for x1, y1, x2, y2 in line if 80 < abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 100]
        if not angles: return image, 0.0
        counts, bin_edges = np.histogram(angles, bins=config.HOUGH_ANGLE_BINS, range=(-np.pi/9, np.pi/9))
        if np.sum(counts) == 0: return image, 0.0
        max_index = np.argmax(counts)
        dominant_deviation_rad = (bin_edges[max_index] + bin_edges[max_index+1]) / 2
        shear_factor = -np.tan(dominant_deviation_rad)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        (h, w) = image.shape[:2]
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, corrected_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)
        return corrected_image, np.degrees(dominant_deviation_rad)

    def _recognize_7_segment_digit(self, digit_image: np.ndarray) -> Optional[int]:
        if digit_image is None or digit_image.size == 0: return None
        h, w = digit_image.shape[:2]
        if h < config.DIGIT_MIN_HEIGHT or w < config.DIGIT_MIN_WIDTH: return None
        rois = config.SEVEN_SEGMENT_ROIS
        def is_on(segment_roi):
            x1, y1, x2, y2 = segment_roi
            roi_abs = digit_image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            if roi_abs.size == 0: return False
            return (np.count_nonzero(roi_abs) / roi_abs.size) > config.SEGMENT_ACTIVATION_THRESHOLD
        try:
            pattern = tuple(is_on(rois[seg]) for seg in ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
            return self.seven_segment_patterns.get(pattern)
        except (IndexError, ValueError): return None

    def detect_and_warp_screen(self, image: np.ndarray) -> Optional[np.ndarray]:
        outer_frame_coords = self._find_main_score_frame(image)
        search_image, offset = (image[outer_frame_coords[1]:outer_frame_coords[1]+outer_frame_coords[3], outer_frame_coords[0]:outer_frame_coords[0]+outer_frame_coords[2]], (outer_frame_coords[0], outer_frame_coords[1])) if outer_frame_coords else (image, (0, 0))
        if not outer_frame_coords: print("情報: 外側フレームが見つからないため、画像全体からLCDスクリーンを探索します。")
        contour = self._find_inner_lcd_screen_contour(search_image)
        if contour is None: return None
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_abs = np.intp(box + offset)
        (w, h) = rect[1]
        if w < h: (w, h) = (h, w)
        return self._four_point_transform(image, box_abs, (int(w), int(h)))

    def split_screen_into_regions(self, screen_image: np.ndarray) -> Dict[str, np.ndarray]:
        if screen_image is None or screen_image.size == 0: return {}
        h, w = screen_image.shape[:2]
        x1_split, x2_split = int(w * config.SCREEN_X1_SPLIT_RATIO), int(w * config.SCREEN_X2_SPLIT_RATIO)
        left_img, middle_img, right_img = screen_image[:, 0:x1_split], screen_image[:, x1_split:x2_split], screen_image[:, x2_split:w]
        mid_h = middle_img.shape[0] // 2
        return {'上家': left_img, '対面': middle_img[0:mid_h, :], '自分': middle_img[mid_h:, :], '下家': right_img}
    
    def _is_valid_score(self, score: int) -> bool:
        return self.min_score <= score <= self.max_score and len(str(score)) == self.expected_digits

    def _process_player_score(self, region_image: np.ndarray, player: str) -> Optional[int]:
        try:
            score_image = self._find_score_by_00_anchor(region_image)
            if score_image is None: return None
            return self._recognize_score_from_image(score_image)
        except Exception as e:
            print(f"OCR読み取りエラー({player}): {e}")
            return None

    def _image_processing_pipeline(self, image: np.ndarray, debug=False, shear_correction_method: str = 'hough', manual_shear_angle: float = 0.0) -> Dict[str, Any]:
        debug_bundle = {}
        warped_screen = self.detect_and_warp_screen(image)
        if warped_screen is None:
            if debug:
                debug_bundle['warped_screen'] = np.zeros((100, 300, 3), dtype=np.uint8)
                cv2.putText(debug_bundle['warped_screen'], "Not Found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return debug_bundle
            return {}
        if debug: debug_bundle['warped_screen'] = warped_screen
        gray = cv2.cvtColor(warped_screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones(config.BINARY_MORPH_OPEN_KERNEL_SIZE, np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=config.BINARY_MORPH_OPEN_ITERATIONS)

        shear_map = {'hough': self._correct_shear_hough, 'manual': self._correct_shear_manual, 'zeros': self._correct_shear_zeros}
        if shear_correction_method in shear_map:
            args = (binary, manual_shear_angle) if shear_correction_method == 'manual' else (binary,)
            corrected_binary, angle = shear_map[shear_correction_method](*args)
        else:
            corrected_binary, angle = binary, 0.0

        if debug:
            debug_bundle['shear_corrected_screen'] = corrected_binary
            debug_bundle['shear_angles'] = {'screen': angle}

        region_images = self.split_screen_into_regions(corrected_binary)
        if debug:
            processed_regions = {}
            for player, region_image in region_images.items():
                processed_region = region_image
                if processed_region.size > 0:
                    h, w = processed_region.shape
                    if player in ['上家', '下家']:
                        start_y = h // 3
                        end_y = h - (h // 3)
                        processed_region = processed_region[start_y:end_y, :]
                    elif player == '対面':
                        start_y = h // 20
                        end_y = h - (h * 4 // 9)
                        start_x = w // 3
                        end_x = w - (w * 1 // 10)
                        processed_region = processed_region[start_y:end_y,start_x:end_x]
                processed_regions[player] = processed_region

            debug_bundle['split_region_images'] = processed_regions

            debug_bundle['anchored_score_regions'] = {}
            for player, region_image in processed_regions.items():
                score_image = self._find_score_by_00_anchor(region_image)
                debug_bundle['anchored_score_regions'][player] = [score_image] if score_image is not None else []

            debug_bundle['deskewed_digits'] = debug_bundle['anchored_score_regions']

            scores = {}
            for player, score_images in debug_bundle['anchored_score_regions'].items():
                if score_images:
                    score = self._recognize_score_from_image(score_images[0])
                    if score is not None:
                        scores[player] = score
            debug_bundle['scores'] = scores

            return debug_bundle
        else:
            scores = {}
            for player, region_image in region_images.items():
                if region_image.size > 0:
                    h, w = region_image.shape
                    if player in ['上家', '下家']:
                        start_y = h // 3
                        end_y = h - (h // 3)
                        region_image = region_image[start_y:end_y, :]
                    elif player == '対面':
                        start_y = h // 20
                        end_y = h - (h * 4 // 9)
                        start_x = w // 3
                        end_x = w - (w * 1 // 10)
                        region_image = region_image[start_y:end_y, start_x:end_x]
                score = self._process_player_score(region_image, player)
                if score is not None: scores[player] = score
            return scores

    def process_score_image(self, image: np.ndarray, shear_correction_method: str = 'zeros', manual_shear_angle: float = 0.0) -> Dict[str, int]:
        return self._image_processing_pipeline(image, debug=False, shear_correction_method=shear_correction_method, manual_shear_angle=manual_shear_angle)

    def get_full_debug_bundle(self, image: np.ndarray, shear_correction_method: str = 'zeros', manual_shear_angle: float = 0.0) -> Dict[str, Any]:
        return self._image_processing_pipeline(image, debug=True, shear_correction_method=shear_correction_method, manual_shear_angle=manual_shear_angle)
