import numpy as np

# --- App Settings ---
PLAYERS = ['自分', '下家', '対面', '上家']
DEFAULT_SCORES = {'自分': 28000, '下家': 35000, '対面': 30000, '上家': 27000}
TOP_DIFF_COLOR_THRESHOLDS = {
    'red': 10000,
    'orange': 5000,
    'green': 0
}

# --- Image Processor Settings ---

# Score validation
MIN_SCORE = 1000
MAX_SCORE = 99999
EXPECTED_DIGITS = 5

# 7-segment digit recognition patterns
SEVEN_SEGMENT_PATTERNS = {
    (True, True, True, True, True, True, False): 0, (False, True, True, False, False, False, False): 1,
    (True, True, False, True, True, False, True): 2, (True, True, True, True, False, False, True): 3,
    (False, True, True, False, False, True, True): 4, (True, False, True, True, False, True, True): 5,
    (True, False, True, True, True, True, True): 6, (True, True, True, False, False, False, False): 7,
    (True, True, True, True, True, True, True): 8, (True, True, True, True, False, True, True): 9,
}

# --- Image Processing Parameters ---

# Frame detection
HSV_LOWER_WHITE = np.array([0, 0, 150])
HSV_UPPER_WHITE = np.array([180, 50, 255])
FRAME_DILATE_KERNEL_SIZE = (5, 5)
FRAME_ERODE_KERNEL_SIZE = (5, 5)
FRAME_CONTOUR_AREA_MIN_RATIO = 0.01
FRAME_CONTOUR_AREA_MAX_RATIO = 0.9

# LCD screen detection
HSV_LOWER_LCD_BLUE = np.array([85, 50, 100])
HSV_UPPER_LCD_BLUE = np.array([105, 255, 255])
LCD_MORPH_CLOSE_KERNEL_SIZE = (5, 5)
LCD_MORPH_CLOSE_ITERATIONS = 3
LCD_CONTOUR_AREA_MIN_RATIO = 0.05

# Digit splitting
DIGIT_MARGIN_RATIO = 0.05
DIGIT_MIN_WIDTH = 5
DIGIT_MIN_HEIGHT = 10
MIN_GAP_WIDTH = 3

# Shear correction (Hough Transform)
HOUGH_CANNY_THRESHOLD_1 = 50
HOUGH_CANNY_THRESHOLD_2 = 150
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 5
HOUGH_ANGLE_MIN = 75  # degrees
HOUGH_ANGLE_MAX = 105 # degrees
HOUGH_ANGLE_BINS = 20
HOUGH_ANGLE_RANGE = (-np.pi/4, np.pi/4)

# 7-segment ROI definitions (x1, y1, x2, y2)
SEVEN_SEGMENT_ROIS = {
    'a': (0.2, 0.0, 0.8, 0.2), 'b': (0.7, 0.1, 1.0, 0.45), 'c': (0.7, 0.55, 1.0, 0.9),
    'd': (0.2, 0.8, 0.8, 1.0), 'e': (0.0, 0.55, 0.3, 0.9), 'f': (0.0, 0.1, 0.3, 0.45),
    'g': (0.2, 0.4, 0.8, 0.6)
}
SEGMENT_ACTIVATION_THRESHOLD = 0.7

# Screen splitting
SCREEN_TOTAL_PARTS = 7
SCREEN_X1_SPLIT_RATIO = 2 / 7
SCREEN_X2_SPLIT_RATIO = 5 / 7
PLAYER_REGION_CROP_RATIO = 0.7 # For non-"self" players

# Image preprocessing
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
BINARY_MORPH_OPEN_KERNEL_SIZE = (3, 3)
BINARY_MORPH_OPEN_ITERATIONS = 1

# --- Game Rule Settings ---
POINTS_PER_KYOTAKU = 1000
POINTS_PER_TSUMIBO_RON = 300
POINTS_PER_TSUMIBO_TSUMO = 100

# --- Points Lookup Settings ---
# from points_lookup.py
CHILD_RON_MANGAN = 8000
PARENT_RON_MANGAN = 12000
PARENT_TSUMO_MANGAN = 4000
CHILD_TSUMO_MANGAN = 2000

# This can be refactored to be data-driven later
HIGH_TIER_HANDS_THRESHOLDS = {
    'ron': {
        'child': [
            (32000, '役満'), (24000, '三倍満'), (16000, '倍満'),
            (12000, '跳満'), (8000, '満貫')
        ],
        'parent': [
            (48000, '役満'), (36000, '三倍満'), (24000, '倍満'),
            (18000, '跳満'), (12000, '満貫')
        ]
    },
    'tsumo': {
        'child': [ # child_pay value
            (8000, '役満'), (6000, '三倍満'), (4000, '倍満'),
            (3000, '跳満'), (2000, '満貫')
        ],
        'parent': [ # per_person_pay value
            (16000, '役満'), (12000, '三倍満'), (8000, '倍満'),
            (6000, '跳満'), (4000, '満貫')
        ]
    }
}
