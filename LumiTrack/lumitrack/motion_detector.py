"""
motion_detector.py
------------------
Finds the largest moving region in a video frame using only classical image
processing (no machine learning, no trained models).

Pipeline for every frame
~~~~~~~~~~~~~~~~~~~~~~~~
1. Convert BGR frame to GRAYSCALE.
2. Apply GAUSSIAN BLUR to suppress camera noise.
3. FRAME / BACKGROUND DIFFERENCING: |current - background| where the
   background is a running average of previous frames (adaptive reference
   frame).  Moving things differ from the background; static things do not.
4. BINARY THRESHOLDING: difference > threshold  ->  255 (motion), else 0.
5. MORPHOLOGICAL OPENING  (erode then dilate)  -> removes small white specks.
   MORPHOLOGICAL CLOSING  (dilate then erode)  -> fills small black holes.
6. CONTOUR DETECTION on the clean binary mask.
7. Pick the LARGEST contour and compute its CENTROID with image moments:
        cx = M10 / M00 ,  cy = M01 / M00
8. SMOOTH the centroid with an exponential moving average so the light glides
   instead of jumping:   s = a * new + (1 - a) * s_previous
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DEFAULT_THRESHOLD, MIN_AREA_RATIO, SMOOTHING_ALPHA


@dataclass
class MotionResult:
    """Everything the rest of the program needs to know about one frame."""
    found: bool                         # was a valid moving region detected?
    center: tuple                       # smoothed (x, y) - always valid
    raw_center: tuple = None            # un-smoothed centroid (None if not found)
    bbox: tuple = None                  # (x, y, w, h) of the largest region
    area: float = 0.0                   # area of the largest region in pixels
    presence: float = 0.0               # 1.0 = motion now, fades to 0 when idle
    mask: np.ndarray = field(default=None, repr=False)   # cleaned binary mask


class MotionDetector:
    # Learning rates of the running-average background
    WARMUP_FRAMES = 15      # first frames: learn the background quickly
    WARMUP_RATE = 0.5
    NORMAL_RATE = 0.02      # afterwards: adapt slowly to lighting changes
    GLOBAL_CHANGE = 0.60    # if >60 % of the pixels change it is NOT a moving
                            # object but a camera / exposure change -> relearn

    def __init__(self, threshold=DEFAULT_THRESHOLD, min_area_ratio=MIN_AREA_RATIO,
                 smoothing=SMOOTHING_ALPHA):
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.smoothing = smoothing
        # Structuring elements (the "brushes" used by morphology)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        self.reset()

    def reset(self):
        """Forget everything (called when a new video source starts)."""
        self._background = None
        self._frame_count = 0
        self._smoothed = None
        self._presence = 0.0

    # ------------------------------------------------------------------ main
    def process(self, frame):
        h, w = frame.shape[:2]
        if self._smoothed is None:                 # start in the frame centre
            self._smoothed = (w / 2.0, h / 2.0)

        # 1 + 2 : grayscale and Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # First frame: it simply becomes the background.
        if self._background is None:
            self._background = gray.astype(np.float32)
            return self._empty_result(h, w)

        # 3 : absolute difference with the background image
        background_u8 = cv2.convertScaleAbs(self._background)
        diff = cv2.absdiff(gray, background_u8)

        # 4 : binary threshold
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Update the running-average background AFTER differencing.
        rate = self.WARMUP_RATE if self._frame_count < self.WARMUP_FRAMES else self.NORMAL_RATE
        cv2.accumulateWeighted(gray, self._background, rate)
        self._frame_count += 1

        # Ignore the warm-up period (the background is still being learnt).
        if self._frame_count <= self.WARMUP_FRAMES:
            return self._empty_result(h, w)

        # Camera exposure jump / whole scene changed -> relearn background.
        if cv2.countNonZero(binary) > self.GLOBAL_CHANGE * h * w:
            self._background = gray.astype(np.float32)
            return self._empty_result(h, w)

        # 5 : morphological opening then closing
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_kernel)

        # 6 : contours (OpenCV 3 / 4 compatible return value)
        found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]

        # 7 : largest contour -> centroid
        best, best_area = None, 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area:
                best, best_area = contour, area

        if best is None or best_area < self.min_area_ratio * h * w:
            return self._no_motion(binary)

        m = cv2.moments(best)
        if m["m00"] == 0:
            return self._no_motion(binary)
        raw = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        # 8 : exponential smoothing of the coordinates
        a = self.smoothing
        sx, sy = self._smoothed
        self._smoothed = (a * raw[0] + (1 - a) * sx, a * raw[1] + (1 - a) * sy)

        self._presence = min(1.0, self._presence + 0.25)
        return MotionResult(
            found=True, center=self._smoothed, raw_center=raw,
            bbox=cv2.boundingRect(best), area=best_area,
            presence=self._presence, mask=binary,
        )

    # --------------------------------------------------------------- helpers
    def _no_motion(self, mask):
        """No valid moving region: keep the last position, fade the presence."""
        self._presence = max(0.0, self._presence - 0.03)
        return MotionResult(found=False, center=self._smoothed,
                            presence=self._presence, mask=mask)

    def _empty_result(self, h, w):
        return self._no_motion(np.zeros((h, w), dtype=np.uint8))
