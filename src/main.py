"""
VisionWatch - Smart Desk Monitor
A beginner-friendly computer vision application built with Python and OpenCV.

Techniques: Image Operations, Motion and Object Tracking (background subtraction),
Haar Cascade face detection.

Run examples (from the project folder):
    python src/main.py                                   (live webcam 0)
    python src/main.py --source assets/sample_video.mp4  (recorded video)
    python src/main.py --source assets/sample_shapes.jpg (still image)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

import config
import display
from input_source import SourceError, open_source
from processing import (MotionDetector, MotionResult, PROJECT_ROOT, detect_faces,
                        load_face_cascade, preprocess)
from tracker import CentroidTracker

SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")


@dataclass
class AppState:
    """What the user can switch on/off while the program runs."""
    paused: bool = False
    show_motion: bool = True
    show_faces: bool = True
    show_mask: bool = True


@dataclass
class Context:
    """Objects that must live for the whole run (models, trackers, cache)."""
    cascade: object = None
    motion: MotionDetector = field(default_factory=MotionDetector)
    last_motion: MotionResult = None


# ----------------------------------------------------------------------
# Core: run all enabled techniques on one frame and draw the results
# ----------------------------------------------------------------------
def analyze_frame(frame, state, ctx, source, update_motion=True):
    """Returns (annotated_frame, PanelInfo)."""
    prep = preprocess(frame)                       # Image Operations
    canvas = prep.resized.copy()
    info = display.PanelInfo(status="PAUSED" if state.paused else "RUNNING",
                             source_label=source.label)

    # ---- Motion detection ----
    if source.is_static:
        info.motion_text = "Motion: n/a (still image)"
    elif not state.show_motion:
        info.motion_text = "Motion: OFF"
    else:
        if update_motion or ctx.last_motion is None:
            ctx.last_motion = ctx.motion.apply(prep.blurred)
        motion = ctx.last_motion
        if not motion.ready:
            info.motion_text = "Motion: learning background..."
            info.motion_color = display.YELLOW
        elif motion.detected:
            info.motion_text = f"Motion: DETECTED ({motion.percent:.1f}%)"
            info.motion_color = display.ORANGE
            display.draw_motion(canvas, motion.boxes)
            display.draw_banner(canvas, "MOTION DETECTED", display.ORANGE)
        else:
            info.motion_text = "Motion: none"
            info.motion_color = display.GREEN

    # ---- Haar cascade faces ----
    if not state.show_faces:
        info.face_text = "Faces: OFF"
    elif ctx.cascade is None:
        info.face_text = "Faces: cascade file missing"
        info.face_color = display.RED
    else:
        faces = detect_faces(ctx.cascade, prep.gray_eq)
        display.draw_faces(canvas, faces)
        info.face_text = f"Faces: {len(faces)}"
        info.face_color = display.GREEN if faces else display.GREY

    if state.paused:
        display.draw_banner(canvas, "PAUSED", display.YELLOW)
    return canvas, info


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def save_image(path, image):
    """Save an image. imencode + tofile also works for Windows paths with special characters."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ok, buffer = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if not ok:
        return False
    buffer.tofile(path)
    return True


def handle_key(key, state, ctx, window_image):
    """Change the program state according to a key press. Returns False to quit."""
    if key in (ord("q"), 27):                         # Q or ESC
        return False
    if key == ord(" "):
        state.paused = not state.paused
    elif key == ord("m"):
        state.show_motion = not state.show_motion
        if state.show_motion:
            ctx.motion = MotionDetector()             # relearn the background
            ctx.last_motion = None
    elif key == ord("f"):
        state.show_faces = not state.show_faces
    elif key == ord("v"):
        state.show_mask = not state.show_mask
    elif key == ord("s"):
        name = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        path = os.path.join(SCREENSHOT_DIR, name)
        print(f"Screenshot saved: {path}" if save_image(path, window_image)
              else "Could not save the screenshot.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionWatch - motion detection and face detection.")
    parser.add_argument("--source", default="0",
                        help="webcam number (0, 1, ...), or a path to an image or video. "
                             "Default: 0 (first webcam)")
    parser.add_argument("--loop", action="store_true", help="repeat a video file forever")
    parser.add_argument("--no-faces", action="store_true", help="start with face detection off")
    parser.add_argument("--headless", action="store_true",
                        help="no window: process frames and print results (for testing)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames (mainly for --headless)")
    parser.add_argument("--save-output", default=None,
                        help="save the last window image to this file when finished")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def run(args):
    try:
        source = open_source(args.source)
    except SourceError as error:
        print(f"ERROR: {error}")
        return 1

    state = AppState(show_faces=not args.no_faces)
    ctx = Context(cascade=load_face_cascade())
    if ctx.cascade is None:
        print("WARNING: Haar cascade file not found - face detection is disabled.")
        state.show_faces = False

    max_frames = args.max_frames
    if args.headless and max_frames is None:           # avoid endless headless runs
        max_frames = 1 if source.is_static else (100 if source.kind == "webcam" else None)

    print(f"Input: {source.label} ({source.kind}). Press Q or ESC in the window to quit.")
    frame, window_image = None, None
    frame_count, fps, last_time = 0, 0.0, time.perf_counter()
    exit_code = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # --- get the next frame (not while paused) ---
            if not state.paused or frame is None:
                ok, new_frame = source.read()
                if not ok or new_frame is None:
                    if source.kind == "video" and args.loop and source.rewind():
                        continue
                    if frame_count == 0:
                        print("ERROR: The input did not deliver any frames.")
                        exit_code = 1
                    else:
                        print("End of input.")
                    break
                frame = new_frame
                if not state.paused:
                    frame_count += 1

            # --- process and draw ---
            canvas, info = analyze_frame(frame, state, ctx, source,
                                         update_motion=not state.paused)
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps   # smooth
            last_time = now
            info.fps = fps
            window_image = display.compose_window(canvas, info)

            if args.headless:
                if frame_count % 10 == 0 or frame_count == 1:
                    print(f"frame {frame_count:4d} | {info.motion_text} | {info.face_text}")
            else:
                cv2.imshow(config.WINDOW_NAME, window_image)
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break                              # user closed the window with the X
                if source.is_static:
                    delay = 30
                elif source.kind == "video":
                    spent_ms = (time.perf_counter() - loop_start) * 1000
                    delay = max(1, int(1000 / max(fps, 1.0)) - int(spent_ms))
                else:
                    delay = 1
                key = cv2.waitKey(delay) & 0xFF
                if not handle_key(key, state, ctx, window_image):
                    break

            if max_frames is not None and frame_count >= max_frames:
                break

    finally:
        source.release()
        if args.save_output and window_image is not None:
            save_image(args.save_output, window_image)
        if not args.headless:
            cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
