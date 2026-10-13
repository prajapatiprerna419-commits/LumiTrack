"""
lighting.py
-----------
Creates the virtual spotlight and blends it into the video frame.

Algorithm
~~~~~~~~~
1. RADIAL MASK   : draw a filled white circle (radius R) at the motion centre
                   on a black canvas.
2. GAUSSIAN BLUR : blur that circle so its edge fades smoothly -> a soft
                   spotlight (values from 0.0 = dark to 1.0 = fully lit).
3. COLOUR + BRIGHTNESS : the lit version of the frame is multiplied by a
                   brightness gain and a colour tint (warm / cool / white),
                   plus a little additive glow.
4. DIMMING       : the area outside the spotlight is darkened (ambient level)
                   so the light is clearly visible, like a stage.
5. ALPHA BLENDING: result = dark * (1 - mask) + lit * mask
"""
import cv2
import numpy as np

# Colour multipliers in BGR order (OpenCV uses Blue, Green, Red).
TINTS = {
    "warm":  np.array([0.75, 0.95, 1.25], dtype=np.float32),   # orange / yellow
    "cool":  np.array([1.30, 1.05, 0.75], dtype=np.float32),   # blue / cyan
    "white": np.array([1.00, 1.00, 1.00], dtype=np.float32),
}
# Additive glow colour in BGR (adds a coloured "haze" inside the spotlight).
GLOWS = {
    "warm":  np.array([40, 120, 255], dtype=np.float32),
    "cool":  np.array([255, 170, 40], dtype=np.float32),
    "white": np.array([255, 255, 255], dtype=np.float32),
}


class SpotlightRenderer:
    # The soft mask is built at 1/4 resolution then scaled up.  This is much
    # faster (large Gaussian blurs are expensive) and looks identical because
    # the mask is smooth anyway.
    MASK_SCALE = 0.25

    def build_mask(self, shape, center, radius):
        """Returns a float32 mask (h, w) with values 0..1 - the soft spotlight."""
        h, w = shape[:2]
        sw, sh = max(1, int(w * self.MASK_SCALE)), max(1, int(h * self.MASK_SCALE))
        s = self.MASK_SCALE

        small = np.zeros((sh, sw), dtype=np.float32)
        cx, cy = int(center[0] * s), int(center[1] * s)
        r = max(1, int(radius * s))
        cv2.circle(small, (cx, cy), r, 1.0, thickness=-1)        # 1. radial mask

        k = int(r * 1.2) * 2 + 1                                 # odd kernel size
        small = cv2.GaussianBlur(small, (k, k), 0,
                                 borderType=cv2.BORDER_CONSTANT)  # 2. soft edge
        mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def apply(self, frame, center, radius, intensity, mode, ambient, strength=1.0):
        """
        frame     : BGR uint8 image
        center    : (x, y) of the spotlight
        radius    : spotlight radius in pixels
        intensity : 0..100  (brightness of the light)
        mode      : "warm", "cool" or "white"
        ambient   : 0..100  (how bright the area OUTSIDE the light stays)
        strength  : 0..1    (used to fade the light when nothing moves)
        Returns (processed_frame, mask)
        """
        mask = self.build_mask(frame.shape, center, radius) * float(strength)
        k = intensity / 100.0

        f = frame.astype(np.float32)
        dark = f * (ambient / 100.0)                              # 4. dimmed scene

        gain = 1.0 + 0.8 * k                                      # 3. brightness
        lit = f * gain * TINTS[mode]                              #    colour tint
        lit += GLOWS[mode] * (0.18 * k)                           #    soft glow

        m = mask[:, :, None]                                      # 5. blending
        out = dark * (1.0 - m) + lit * m
        return np.clip(out, 0, 255).astype(np.uint8), mask
