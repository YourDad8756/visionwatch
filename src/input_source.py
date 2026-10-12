"""
input_source.py - Opens and validates the input: a still image, a video file,
or a live webcam. Every problem raises a SourceError with a friendly message.
"""

import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


class SourceError(Exception):
    """Raised when the input cannot be opened (missing file, no webcam, ...)."""


class FrameSource:
    """A small wrapper so the main loop can treat all inputs the same way."""

    def __init__(self, kind, label, capture=None, image=None, fps=30.0):
        self.kind = kind          # "image", "video" or "webcam"
        self.label = label        # text shown in the panel
        self.fps = fps            # playback speed hint (used for video files)
        self._capture = capture
        self._image = image

    @property
    def is_static(self):
        """True for a still image (there is no motion to detect)."""
        return self.kind == "image"

    def read(self):
        """Return (success, frame). A still image returns the same picture each time."""
        if self.kind == "image":
            return True, self._image.copy()
        ok, frame = self._capture.read()
        return ok, frame

    def rewind(self):
        """Jump back to the first frame of a video file (used by --loop)."""
        if self.kind != "video":
            return False
        return self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Free the camera / video file."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _read_image_file(path):
    """Read an image. np.fromfile also works for Windows paths with special characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise SourceError(f"Could not read the file '{path}': {error}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(f"'{path}' is not a valid image (the file may be corrupted).")
    return image


def _open_webcam(index):
    """Try to open webcam number `index`; return a FrameSource."""
    # On Windows the DirectShow backend (CAP_DSHOW) is usually the most reliable one.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = capture.read()          # make sure it can really deliver frames
            if ok:
                return FrameSource("webcam", f"webcam {index}", capture=capture)
        capture.release()
    raise SourceError(
        f"Could not open webcam {index}. Check that a camera is connected, that no other "
        "app (Zoom, Teams, Camera) is using it, and that Windows camera privacy settings "
        "allow desktop apps. You can also try --source 1, or test with "
        "--source assets/sample_video.mp4"
    )


def open_source(spec):
    """
    Open the input described by `spec`:
      "0", "1", ... or "webcam"  -> live webcam
      path to an image           -> still image
      path to a video            -> recorded video
    """
    spec = str(spec).strip().strip('"').strip("'")

    if spec.lower() == "webcam":
        spec = "0"
    if spec.isdigit():
        return _open_webcam(int(spec))

    path = os.path.expanduser(spec)
    if not os.path.isfile(path):
        raise SourceError(
            f"File not found: '{path}'. Check the spelling and the folder. "
            "Run the program from the project folder, or give the full path."
        )

    extension = os.path.splitext(path)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return FrameSource("image", os.path.basename(path), image=_read_image_file(path))

    if extension in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise SourceError(f"OpenCV could not open the video '{path}'.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1:   # 0 or NaN means "unknown"
            fps = 30.0
        return FrameSource("video", os.path.basename(path), capture=capture, fps=fps)

    raise SourceError(
        f"Unsupported file type '{extension}'. Use one of: "
        f"{', '.join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))}"
    )
