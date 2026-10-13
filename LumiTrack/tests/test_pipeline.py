"""
Automatic checks of the image-processing pipeline (no webcam or window needed).
Run from the project folder:   python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lumitrack.lighting import SpotlightRenderer
from lumitrack.motion_detector import MotionDetector


def make_frame(cx, cy, noise_seed=0):
    """Grey noisy background with a bright rectangle centred at (cx, cy)."""
    rng = np.random.default_rng(noise_seed)
    frame = rng.integers(80, 90, (360, 640, 3)).astype(np.uint8)
    cv2.rectangle(frame, (cx - 40, cy - 60), (cx + 40, cy + 60), (230, 230, 230), -1)
    return frame


class MotionDetectorTest(unittest.TestCase):
    def test_tracks_moving_object(self):
        det = MotionDetector()
        for i in range(25):                           # learn the empty background
            det.process(make_frame(-500, -500, i))    # object off-screen
        result = None
        for i in range(40):                           # object moves right
            cx = 150 + i * 8
            result = det.process(make_frame(cx, 180, i))
        self.assertTrue(result.found)
        self.assertLess(abs(result.raw_center[0] - cx), 25)
        self.assertLess(abs(result.raw_center[1] - 180), 25)
        # smoothed centre lags a little behind but must be close
        self.assertLess(abs(result.center[0] - cx), 80)

    def test_no_motion_on_static_scene(self):
        det = MotionDetector()
        result = None
        for i in range(40):
            result = det.process(make_frame(320, 180, 0))   # identical frames
        self.assertFalse(result.found)


class LightingTest(unittest.TestCase):
    def test_spotlight_is_brighter_at_centre(self):
        frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        out, mask = SpotlightRenderer().apply(frame, (320, 180), 100, 80, "warm", 30)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(out[180, 320].mean(), out[10, 10].mean())
        self.assertGreater(mask[180, 320], 0.9)
        self.assertLess(mask[10, 10], 0.05)

    def test_warm_is_redder_cool_is_bluer(self):
        frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        r = SpotlightRenderer()
        warm, _ = r.apply(frame, (320, 180), 100, 60, "warm", 30)
        cool, _ = r.apply(frame, (320, 180), 100, 60, "cool", 30)
        self.assertGreater(warm[180, 320, 2], warm[180, 320, 0])   # R > B
        self.assertGreater(cool[180, 320, 0], cool[180, 320, 2])   # B > R


if __name__ == "__main__":
    unittest.main()
