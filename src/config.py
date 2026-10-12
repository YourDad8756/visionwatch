"""
config.py - All adjustable settings for VisionWatch in one place.

Beginners: if detection feels too sensitive or not sensitive enough,
try changing the numbers in this file instead of editing the main code.
"""

APP_TITLE = "VisionWatch - Smart Desk Monitor"
WINDOW_NAME = "VisionWatch"

# ---------- Frame size ----------
FRAME_WIDTH = 640        # every input frame is resized to fit inside this box
MAX_FRAME_HEIGHT = 540   # (keeps tall images from becoming bigger than a screen)
PANEL_WIDTH = 320        # width of the information panel on the right
PANEL_MIN_HEIGHT = 480   # minimum height of the whole window

# ---------- Image Operations (preprocessing) ----------
BLUR_KERNEL = (5, 5)     # Gaussian blur size (must be odd numbers)
MORPH_KERNEL_SIZE = 5    # size of the structuring element for noise removal

# ---------- Image Segmentation (HSV colour ranges) ----------
# OpenCV Hue goes from 0 to 179 (not 360). Each range is (lower, upper) in H, S, V.
# Red wraps around the hue circle, so it needs two ranges.
COLOR_RANGES = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (130, 255, 255))],
    "yellow": [((20, 110, 100), (35, 255, 255))],
}
COLOR_NAMES = ["red", "green", "blue", "yellow"]

# Colours (B, G, R) used when DRAWING boxes and labels for each object colour
COLOR_BGR = {
    "red": (60, 60, 255),
    "green": (80, 220, 80),
    "blue": (255, 140, 60),
    "yellow": (0, 220, 255),
}

MIN_OBJECT_AREA = 600    # ignore colour blobs smaller than this many pixels

# ---------- Motion detection (background subtraction) ----------
MOTION_HISTORY = 300          # how many frames the background model remembers
MOTION_VAR_THRESHOLD = 40     # higher = less sensitive to small changes
MOTION_MIN_AREA = 1200        # ignore moving blobs smaller than this many pixels
MOTION_WARMUP_FRAMES = 30     # frames used to learn the background first

# ---------- Haar cascade face detection ----------
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5        # higher = fewer false faces, but may miss real ones
FACE_MIN_SIZE = (50, 50)

# ---------- Object tracking ----------
TRACK_MAX_DISTANCE = 80       # max pixels an object may move between frames
TRACK_MAX_MISSING = 15        # frames an object may disappear before it is forgotten
TRACK_TRAIL_LENGTH = 30       # how many past positions are drawn as a trail
