"""
config.py
---------
Central place for constants and the user-adjustable Settings object.

The GUI thread writes to a Settings object and the video thread reads from it,
so the sliders take effect immediately without restarting the video.
"""

APP_NAME = "LumiTrack"
APP_TAGLINE = "Real-Time Motion-Responsive Virtual Lighting"

# ---- Video processing -------------------------------------------------------
PROCESS_WIDTH = 640          # every frame is resized to this width for speed
CAMERA_INDEXES = (0, 1)      # webcam indexes tried in order

# ---- Motion detection defaults ---------------------------------------------
DEFAULT_THRESHOLD = 25       # grey-level difference that counts as "motion"
MIN_AREA_RATIO = 0.003       # blob must cover >= 0.3 % of the frame to count
SMOOTHING_ALPHA = 0.25       # tracking smoothing (0 = frozen, 1 = no smoothing)

# ---- Lighting defaults ------------------------------------------------------
DEFAULT_INTENSITY = 70       # percent
DEFAULT_RADIUS = 110         # pixels (on the 640-px-wide processing frame)
DEFAULT_AMBIENT = 35         # percent brightness kept outside the spotlight

# ---- Colours of the user interface -----------------------------------------
BG = "#0e1016"
PANEL = "#171a23"
PANEL_LIGHT = "#212634"
ACCENT = "#ffb020"
ACCENT_DARK = "#c98500"
COOL = "#4cc9f0"
TEXT = "#e8eaf0"
MUTED = "#8a90a2"
DANGER = "#e5484d"
OK = "#3ecf8e"


class Settings:
    """Values controlled by the sliders / buttons in the GUI."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.intensity = DEFAULT_INTENSITY    # 0..100
        self.radius = DEFAULT_RADIUS          # px
        self.ambient = DEFAULT_AMBIENT        # 0..100
        self.threshold = DEFAULT_THRESHOLD    # 5..80
        self.mode = "warm"                    # "warm" | "cool" | "white"
