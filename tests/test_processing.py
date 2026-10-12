"""
Simple automated checks for VisionWatch. Run from the project folder with:
    python -m unittest discover -s tests -v
No extra packages are needed (uses Python's built-in unittest).
"""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from input_source import SourceError, open_source          # noqa: E402
from processing import (MotionDetector, detect_faces, find_objects,   # noqa: E402
                        load_face_cascade, preprocess, segment_color)
from tracker import CentroidTracker                         # noqa: E402


def blank(color=(220, 220, 220)):
    return np.full((480, 640, 3), color, dtype=np.uint8)


class SegmentationTests(unittest.TestCase):
    def test_red_circle_is_found_with_correct_shape(self):
        image = blank()
        cv2.circle(image, (200, 200), 50, (40, 40, 220), -1)      # BGR red
        mask = segment_color(preprocess(image).hsv, "red")
        objects = find_objects(mask)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].shape, "circle")
        self.assertAlmostEqual(objects[0].centroid[0], 200, delta=3)

    def test_other_colors_do_not_trigger(self):
        image = blank()
        cv2.rectangle(image, (100, 100), (200, 200), (200, 90, 30), -1)   # blue
        mask = segment_color(preprocess(image).hsv, "red")
        self.assertEqual(len(find_objects(mask)), 0)

    def test_tiny_blobs_are_ignored(self):
        image = blank()
        cv2.circle(image, (50, 50), 5, (0, 210, 240), -1)         # small yellow dot
        mask = segment_color(preprocess(image).hsv, "yellow")
        self.assertEqual(len(find_objects(mask)), 0)


class TrackerTests(unittest.TestCase):
    def test_object_keeps_its_id_while_moving(self):
        tracker = CentroidTracker()
        first = tracker.update([(100, 100)])
        second = tracker.update([(110, 105)])
        self.assertEqual(first, second)

    def test_new_far_object_gets_new_id(self):
        tracker = CentroidTracker()
        tracker.update([(100, 100)])
        ids = tracker.update([(100, 100), (500, 400)])
        self.assertEqual(sorted(ids), [1, 2])

    def test_lost_object_is_forgotten(self):
        tracker = CentroidTracker(max_missing=2)
        tracker.update([(100, 100)])
        for _ in range(4):
            tracker.update([])
        self.assertEqual(len(tracker.positions), 0)


class MotionTests(unittest.TestCase):
    def test_moving_square_is_detected_after_warmup(self):
        detector = MotionDetector()
        result = None
        for i in range(60):
            frame = blank()
            if i >= 40:                                            # square appears and moves
                x = 100 + (i - 40) * 15
                cv2.rectangle(frame, (x, 200), (x + 80, 280), (30, 30, 30), -1)
            result = detector.apply(frame)
        self.assertTrue(result.ready)
        self.assertTrue(result.detected)

    def test_static_scene_reports_no_motion(self):
        detector = MotionDetector()
        for _ in range(50):
            result = detector.apply(blank())
        self.assertFalse(result.detected)


class FaceTests(unittest.TestCase):
    def test_face_found_in_sample_image(self):
        cascade = load_face_cascade()
        self.assertIsNotNone(cascade, "Haar cascade XML was not found")
        image = cv2.imread(os.path.join(ROOT, "assets", "sample_face.jpg"))
        self.assertIsNotNone(image)
        self.assertGreaterEqual(len(detect_faces(cascade, preprocess(image).gray_eq)), 1)

    def test_no_face_in_plain_image(self):
        cascade = load_face_cascade()
        self.assertEqual(len(detect_faces(cascade, preprocess(blank()).gray_eq)), 0)


class InputTests(unittest.TestCase):
    def test_missing_file_gives_friendly_error(self):
        with self.assertRaises(SourceError):
            open_source("this_file_does_not_exist.jpg")

    def test_unsupported_extension(self):
        path = os.path.join(ROOT, "README.md")
        with self.assertRaises(SourceError):
            open_source(path)

    def test_sample_image_and_video_open(self):
        image_source = open_source(os.path.join(ROOT, "assets", "sample_shapes.jpg"))
        self.assertTrue(image_source.is_static)
        ok, frame = image_source.read()
        self.assertTrue(ok and frame is not None)

        video_source = open_source(os.path.join(ROOT, "assets", "sample_video.mp4"))
        ok, frame = video_source.read()
        video_source.release()
        self.assertTrue(ok and frame is not None)


if __name__ == "__main__":
    unittest.main()
