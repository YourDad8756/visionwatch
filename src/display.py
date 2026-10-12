"""
display.py - Everything that is DRAWN on screen: boxes, contours, trails,
banners, and the information panel on the right side of the window.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (90, 220, 90)
YELLOW = (0, 215, 255)
RED = (70, 70, 255)
ORANGE = (0, 140, 255)
PANEL_BG = (38, 30, 28)


def put_text(image, text, origin, scale=0.5, color=WHITE, thickness=1):
    """Shortcut for cv2.putText with anti-aliasing."""
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def put_label(image, text, origin, color, scale=0.5):
    """Text with a filled background so it stays readable on any picture."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    y = max(y, h + 4)                                    # keep inside the image
    cv2.rectangle(image, (x, y - h - 4), (x + w + 6, y + baseline - 2), color, -1)
    put_text(image, text, (x + 3, y - 3), scale, (0, 0, 0), 1)


# ----------------------------------------------------------------------
# Overlays drawn on the video frame
# ----------------------------------------------------------------------
def draw_objects(canvas, objects, ids, color_name, tracker):
    """Contours, boxes, labels, centre points and motion trails of segmented objects."""
    color = config.COLOR_BGR[color_name]
    for obj, object_id in zip(objects, ids):
        x, y, w, h = obj.bbox
        cv2.drawContours(canvas, [obj.contour], -1, color, 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), WHITE, 1)
        cv2.circle(canvas, obj.centroid, 4, WHITE, -1)
        put_label(canvas, f"{color_name} #{object_id} {obj.shape}", (x, y - 2), color)

        trail = list(tracker.trails.get(object_id, []))
        if len(trail) > 1:                               # draw the path the object took
            cv2.polylines(canvas, [np.array(trail, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def draw_motion(canvas, boxes):
    """Orange boxes around the moving regions."""
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ORANGE, 2)
        put_label(canvas, "motion", (x, y - 2), ORANGE)


def draw_faces(canvas, faces):
    """Green boxes around detected faces."""
    for x, y, w, h in faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), GREEN, 2)
        put_label(canvas, "face", (x, y - 2), GREEN)


def draw_banner(canvas, text, color):
    """A big status message in the top-left corner of the video."""
    put_label(canvas, text, (8, 28), color, scale=0.7)


def draw_mask_preview(canvas, mask, scale=0.25):
    """Picture-in-picture: a small copy of the colour mask in the bottom-right corner."""
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    h, w = small.shape[:2]
    x0 = canvas.shape[1] - w - 8
    y0 = canvas.shape[0] - h - 8
    canvas[y0:y0 + h, x0:x0 + w] = small
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), WHITE, 1)
    put_text(canvas, "color mask", (x0 + 4, y0 + 14), 0.4, GREEN)


# ----------------------------------------------------------------------
# Information panel
# ----------------------------------------------------------------------
@dataclass
class PanelInfo:
    """All the values the panel shows. Filled in by main.py every frame."""
    status: str = "RUNNING"
    source_label: str = ""
    fps: float = 0.0
    motion_text: str = ""
    motion_color: tuple = GREY
    face_text: str = ""
    face_color: tuple = GREY
    color_text: str = ""
    color_color: tuple = GREY
    object_lines: list = field(default_factory=list)


HELP_LINES = [
    "SPACE  pause / resume",
    "Q/ESC  quit",
    "M F  toggle motion/faces",
    "V      toggle mask preview",
    "S      save screenshot",
]


def build_panel(height, info):
    """Create the right-hand information panel as an image."""
    panel = np.full((height, config.PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    put_text(panel, "VisionWatch", (14, 36), 0.9, WHITE, 2)
    put_text(panel, "Smart Desk Monitor", (14, 58), 0.5, GREY)
    cv2.line(panel, (14, 70), (config.PANEL_WIDTH - 14, 70), GREY, 1)

    status_color = GREEN if info.status == "RUNNING" else YELLOW
    put_text(panel, f"STATUS: {info.status}", (14, 94), 0.55, status_color, 2)
    put_text(panel, f"Source: {info.source_label[:26]}", (14, 116), 0.45, WHITE)
    put_text(panel, f"FPS: {info.fps:.1f}", (14, 136), 0.45, WHITE)

    put_text(panel, "DETECTIONS", (14, 168), 0.5, GREY, 1)
    put_text(panel, info.motion_text, (14, 192), 0.5, info.motion_color)
    put_text(panel, info.face_text, (14, 214), 0.5, info.face_color)
    put_text(panel, info.color_text, (14, 236), 0.5, info.color_color)

    y = 258
    for line in info.object_lines[:3]:
        put_text(panel, line[:38], (22, y), 0.4, WHITE)
        y += 17
    if len(info.object_lines) > 3:
        put_text(panel, f"+ {len(info.object_lines) - 3} more", (22, y), 0.4, GREY)

    y = height - 150
    put_text(panel, "CONTROLS", (14, y), 0.5, GREY, 1)
    for line in HELP_LINES:
        y += 18
        put_text(panel, line, (14, y), 0.42, WHITE)
    return panel


def compose_window(frame, info):
    """Put the video frame and the panel side by side into one image."""
    height = max(frame.shape[0], config.PANEL_MIN_HEIGHT)
    if frame.shape[0] < height:                       # pad short frames with black
        frame = cv2.copyMakeBorder(frame, 0, height - frame.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack([frame, build_panel(height, info)])
