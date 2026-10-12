"""
generate_samples.py - Creates the small sample files in the assets/ folder:
    assets/sample_shapes.jpg  - still image with four coloured shapes
    assets/sample_video.mp4   - short video with moving coloured objects

You do NOT need to run this: the files are already included in the repository.
Run it only if you want to recreate them:  python tools/generate_samples.py
"""

import os

import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
WIDTH, HEIGHT = 640, 480

# Colours in BGR order
RED, GREEN, BLUE, YELLOW = (40, 40, 220), (50, 180, 50), (200, 90, 30), (0, 210, 240)


def make_background():
    """A plain light-gray 'desk' with a soft vertical gradient and a darker strip."""
    column = np.linspace(225, 195, HEIGHT, dtype=np.uint8).reshape(-1, 1)
    gray = np.repeat(column, WIDTH, axis=1)
    background = cv2.merge([gray, gray, gray])
    cv2.rectangle(background, (0, 400), (WIDTH, HEIGHT), (150, 150, 150), -1)
    return background


def make_still_image():
    image = make_background()
    cv2.circle(image, (130, 130), 60, RED, -1)                                  # red circle
    cv2.rectangle(image, (260, 80), (420, 180), GREEN, -1)                      # green rectangle
    cv2.fillPoly(image, [np.array([(520, 190), (460, 70), (580, 70)])], BLUE)   # blue triangle
    cv2.rectangle(image, (120, 270), (220, 370), YELLOW, -1)                    # yellow square
    cv2.circle(image, (400, 300), 45, (110, 110, 110), -1)                      # gray distractor
    path = os.path.join(ASSETS, "sample_shapes.jpg")
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("Created", path)


def make_video(seconds=8, fps=15):
    path = os.path.join(ASSETS, "sample_video.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not create the video file (mp4v codec not available).")
    background = make_background()
    total = seconds * fps
    for i in range(total):
        frame = background.copy()
        t = i / total
        # Static yellow square: never moves, so it is part of the background.
        cv2.rectangle(frame, (30, 330), (110, 410), YELLOW, -1)
        # Red ball moves left -> right along a gentle wave (starts after the learning period).
        if i > 40:
            x = int(60 + (WIDTH - 120) * (i - 40) / (total - 40))
            y = int(200 + 60 * np.sin(t * 12))
            cv2.circle(frame, (x, y), 36, RED, -1)
        # Blue rectangle moves up and down on the right.
        if i > 55:
            y = int(150 + 120 * abs(np.sin(t * 6)))
            cv2.rectangle(frame, (500, y), (580, y + 60), BLUE, -1)
        writer.write(frame)
    writer.release()
    print("Created", path)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    make_still_image()
    make_video()
