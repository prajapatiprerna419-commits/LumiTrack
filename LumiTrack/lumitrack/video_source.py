"""
video_source.py
---------------
A small wrapper around cv2.VideoCapture so the rest of the program does not
care whether frames come from the webcam or from a video file.
"""
import cv2

from .config import CAMERA_INDEXES


class VideoSourceError(Exception):
    """Raised when a camera or video file cannot be opened."""


class VideoSource:
    def __init__(self, capture, name, is_camera):
        self.capture = capture
        self.name = name
        self.is_camera = is_camera
        fps = capture.get(cv2.CAP_PROP_FPS)
        # Webcams often report 0 or nonsense, so fall back to 30.
        self.fps = fps if fps and 1 < fps < 120 else 30.0

    # ---- constructors -------------------------------------------------------
    @classmethod
    def open_camera(cls):
        """Try the configured webcam indexes. CAP_DSHOW is the fastest and most
        reliable backend on Windows; if it fails we fall back to the default."""
        for index in CAMERA_INDEXES:
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ok, _ = cap.read()          # make sure frames really arrive
                    if ok:
                        return cls(cap, f"Webcam {index}", True)
                cap.release()
        raise VideoSourceError(
            "Could not open the webcam.\n\n"
            "Please check that:\n"
            " - no other app (Zoom, Teams, Camera app) is using it,\n"
            " - Windows Settings > Privacy > Camera allows desktop apps."
        )

    @classmethod
    def open_file(cls, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(f"Could not open the video file:\n{path}")
        name = path.replace("\\", "/").split("/")[-1]
        return cls(cap, name, False)

    # ---- operations ---------------------------------------------------------
    @property
    def is_file(self):
        return not self.is_camera

    def read(self):
        """Returns (success, frame)."""
        return self.capture.read()

    def rewind(self):
        """Jump back to the first frame (used for looping video files)."""
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
