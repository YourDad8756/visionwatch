"""
processing.py - The computer vision techniques used by VisionWatch.

1. Image Operations ........ resize, Gaussian blur, colour conversion, morphology
2. Image Segmentation ...... HSV colour thresholding -> mask -> contours
3. Motion Detection ........ MOG2 background subtraction (+ tracker.py for tracking)
4. Haar Cascade ............ frontal face detection
5. Image Analysis .......... area, centroid and shape of every segmented object
"""

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ======================================================================
# 1. IMAGE OPERATIONS
# ======================================================================
@dataclass
class Preprocessed:
    """The different versions of one frame that later steps need."""
    resized: np.ndarray   # BGR frame at display size
    blurred: np.ndarray   # smoothed BGR frame (less noise)
    hsv: np.ndarray       # HSV version used for colour segmentation
    gray_eq: np.ndarray   # equalised grayscale version used for Haar faces


def resize_to_fit(frame):
    """Resize the frame so it fits inside FRAME_WIDTH x MAX_FRAME_HEIGHT (keeps aspect ratio)."""
    height, width = frame.shape[:2]
    scale = min(config.FRAME_WIDTH / width, config.MAX_FRAME_HEIGHT / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def preprocess(frame):
    """Resize, blur and convert one frame into the colour spaces we need."""
    resized = resize_to_fit(frame)
    blurred = cv2.GaussianBlur(resized, config.BLUR_KERNEL, 0)   # removes small noise
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)               # colour is easier in HSV
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)                             # evens out brightness
    return Preprocessed(resized, blurred, hsv, gray_eq)


def clean_mask(mask):
    """Morphology: 'open' removes tiny white dots, 'close' fills small holes."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# ======================================================================
# 2. IMAGE SEGMENTATION (colour thresholding)
# ======================================================================
def segment_color(hsv, color_name):
    """Return a black/white mask: white pixels are inside the HSV range of `color_name`."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in config.COLOR_RANGES[color_name]:
        part = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, part)
    return clean_mask(mask)


# ======================================================================
# 5. IMAGE ANALYSIS (contours, centroid, area, shape)
# ======================================================================
@dataclass
class DetectedObject:
    contour: np.ndarray
    bbox: tuple        # (x, y, w, h)
    centroid: tuple    # (cx, cy)
    area: float
    shape: str


def classify_shape(contour):
    """Guess a simple shape name from the contour's outline."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "object"
    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / (perimeter * perimeter)   # 1.0 = perfect circle
    if circularity > 0.82:
        return "circle"
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)   # simplify the outline
    corners = len(approx)
    if corners == 3:
        return "triangle"
    if corners == 4:
        _, _, w, h = cv2.boundingRect(approx)
        return "square" if 0.9 <= w / h <= 1.1 else "rectangle"
    return "object"


def find_objects(mask, min_area=config.MIN_OBJECT_AREA):
    """Find the white blobs in a mask and measure them. Biggest object first."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue                                   # too small: probably noise
        moments = cv2.moments(contour)
        cx = int(moments["m10"] / moments["m00"])      # centre of mass
        cy = int(moments["m01"] / moments["m00"])
        objects.append(DetectedObject(
            contour=contour,
            bbox=cv2.boundingRect(contour),
            centroid=(cx, cy),
            area=area,
            shape=classify_shape(contour),
        ))
    objects.sort(key=lambda obj: obj.area, reverse=True)
    return objects


# ======================================================================
# 3. MOTION DETECTION (background subtraction)
# ======================================================================
@dataclass
class MotionResult:
    mask: np.ndarray                       # white = pixels that moved
    boxes: list = field(default_factory=list)   # bounding boxes of big moving blobs
    percent: float = 0.0                   # % of the frame that is moving
    ready: bool = False                    # False while the background is still being learned

    @property
    def detected(self):
        return self.ready and len(self.boxes) > 0


class MotionDetector:
    """Learns the static background and reports anything that moves in front of it."""

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOTION_HISTORY,
            varThreshold=config.MOTION_VAR_THRESHOLD,
            detectShadows=True,          # shadows are marked gray (127) so we can drop them
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.frames_seen = 0

    def apply(self, blurred_frame):
        """Feed one frame; get back a MotionResult."""
        foreground = self.subtractor.apply(blurred_frame)
        self.frames_seen += 1

        # Keep only definite foreground (255) and drop shadows (127).
        _, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)     # remove specks
        mask = cv2.dilate(mask, self.kernel, iterations=2)             # join nearby parts

        ready = self.frames_seen > config.MOTION_WARMUP_FRAMES
        result = MotionResult(mask=mask, ready=ready)
        if not ready:
            return result            # still learning: do not report any motion yet

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= config.MOTION_MIN_AREA:
                result.boxes.append(cv2.boundingRect(contour))
        result.percent = 100.0 * cv2.countNonZero(mask) / mask.size
        return result


# ======================================================================
# 4. HAAR CASCADE FACE DETECTION
# ======================================================================
def load_face_cascade():
    """
    Load the Haar cascade XML file. Looks first in assets/ (copy included in the
    repository), then in OpenCV's own data folder. Returns None if it is not found.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "assets", config.CASCADE_FILENAME),
        os.path.join(cv2.data.haarcascades, config.CASCADE_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    return None


def detect_faces(cascade, gray_eq):
    """Return a list of face boxes (x, y, w, h) found in an equalised gray image."""
    faces = cascade.detectMultiScale(
        gray_eq,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=config.FACE_MIN_SIZE,
    )
    return [tuple(int(v) for v in face) for face in faces]
