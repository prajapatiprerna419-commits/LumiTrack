"""
app.py
------
The LumiTrack desktop window (Tkinter, dark theme).

The window only DISPLAYS frames and handles buttons / sliders.  All heavy
image processing happens in processor.VideoProcessor (a background thread),
and this file polls it every ~15 ms with root.after(), so the UI stays smooth.
"""
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import config as C
from .config import Settings
from .processor import VideoProcessor
from .video_source import VideoSource, VideoSourceError

# Size of the picture panels inside the window
MAIN_W, MAIN_H = 420, 315        # Original + Processed
MASK_W, MASK_H = 280, 210        # Motion mask

# Folder (next to the project) where screenshots / videos are saved
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def fit_to_panel(img, width, height):
    """Resize an image to fit a panel, keeping the aspect ratio (black bars)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def to_photo(bgr):
    """OpenCV BGR array -> Tkinter PhotoImage."""
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def placeholder(width, height, text):
    """A plain dark panel with a text hint (shown while no video is running)."""
    img = np.full((height, width, 3), (30, 26, 23), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, text, ((width - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 144, 138), 1, cv2.LINE_AA)
    return img


def save_image(path, img):
    """cv2.imwrite fails on Windows paths with special characters; this works."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise IOError("Could not encode the image.")
    buf.tofile(path)


class LumiTrackApp:
    def __init__(self, root):
        self.root = root
        self.settings = Settings()
        self.processor = None
        self.last_frame_id = -1
        self.last_data = None
        self.loop_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value=self.settings.mode)
        self.sliders = {}

        root.title(f"{C.APP_NAME} - {C.APP_TAGLINE}")
        root.configure(bg=C.BG)
        style = ttk.Style(root)
        style.theme_use("clam")        # "clam" lets us colour the slider handle
        style.configure("Lumi.Horizontal.TScale", background=C.ACCENT,
                        troughcolor=C.PANEL_LIGHT, bordercolor=C.PANEL,
                        lightcolor=C.ACCENT, darkcolor=C.ACCENT_DARK)
        style.map("Lumi.Horizontal.TScale", background=[("active", "#ffc457")])
        root.geometry("1230x720")
        root.minsize(1180, 700)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_header()
        body = tk.Frame(root, bg=C.BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._build_sidebar(body)
        self._build_display(body)

        self._show_placeholders()
        self._set_status("Idle - start the camera or upload a video", C.MUTED)
        self._update_buttons()
        self.root.after(15, self._poll)

    # =================================================================== UI
    def _build_header(self):
        bar = tk.Frame(self.root, bg=C.BG)
        bar.pack(fill="x", padx=18, pady=(12, 10))
        tk.Label(bar, text="LUMI", font=("Segoe UI", 22, "bold"),
                 fg=C.TEXT, bg=C.BG).pack(side="left")
        tk.Label(bar, text="TRACK", font=("Segoe UI", 22, "bold"),
                 fg=C.ACCENT, bg=C.BG).pack(side="left")
        tk.Label(bar, text="   " + C.APP_TAGLINE, font=("Segoe UI", 10),
                 fg=C.MUTED, bg=C.BG).pack(side="left", pady=(10, 0))
        self.status_label = tk.Label(bar, text="", font=("Segoe UI", 10, "bold"),
                                     bg=C.PANEL, padx=14, pady=6)
        self.status_label.pack(side="right")

    def _section(self, parent, title):
        tk.Label(parent, text=title, font=("Segoe UI", 9, "bold"), fg=C.ACCENT,
                 bg=C.PANEL, anchor="w").pack(fill="x", padx=14, pady=(12, 4))

    def _button(self, parent, text, command, kind="normal"):
        colours = {
            "normal":  (C.PANEL_LIGHT, C.TEXT, "#2c3345"),
            "primary": (C.ACCENT, "#1a1200", "#ffc457"),
            "danger":  (C.PANEL_LIGHT, C.DANGER, "#2c3345"),
        }
        bg, fg, hover = colours[kind]
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      activebackground=hover, activeforeground=fg, relief="flat",
                      bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2",
                      pady=7, disabledforeground="#575d70")
        b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _slider(self, parent, label, key, low, high, unit=""):
        row = tk.Frame(parent, bg=C.PANEL)
        row.pack(fill="x", padx=14, pady=(2, 0))
        tk.Label(row, text=label, font=("Segoe UI", 9), fg=C.TEXT,
                 bg=C.PANEL).pack(side="left")
        value_label = tk.Label(row, font=("Segoe UI", 9, "bold"), fg=C.ACCENT, bg=C.PANEL)
        value_label.pack(side="right")

        def on_change(v):
            setattr(self.settings, key, int(float(v)))
            value_label.config(text=f"{int(float(v))}{unit}")

        scale = ttk.Scale(parent, from_=low, to=high, orient="horizontal",
                          style="Lumi.Horizontal.TScale", command=on_change)
        scale.set(getattr(self.settings, key))
        scale.pack(fill="x", padx=14, pady=(0, 2))
        on_change(scale.get())
        self.sliders[key] = (scale, on_change)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=C.PANEL, width=290)
        side.pack(side="left", fill="y", padx=(0, 12))
        side.pack_propagate(False)

        # ---- source ---------------------------------------------------
        self._section(side, "VIDEO SOURCE")
        row = tk.Frame(side, bg=C.PANEL)
        row.pack(fill="x", padx=14)
        row.columnconfigure((0, 1), weight=1, uniform="a")
        self.btn_start = self._button(row, "Start Camera", self.start_camera, "primary")
        self.btn_stop = self._button(row, "Stop Camera", self.stop_video, "danger")
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.btn_upload = self._button(side, "Upload Video...", self.upload_video)
        self.btn_upload.pack(fill="x", padx=14, pady=(8, 0))
        tk.Checkbutton(side, text="Loop uploaded video", variable=self.loop_var,
                       command=self._loop_changed, bg=C.PANEL, fg=C.MUTED,
                       selectcolor=C.PANEL_LIGHT, activebackground=C.PANEL,
                       activeforeground=C.TEXT, font=("Segoe UI", 9), bd=0,
                       highlightthickness=0).pack(anchor="w", padx=12, pady=(4, 0))

        # ---- light controls -------------------------------------------
        self._section(side, "VIRTUAL LIGHT")
        self._slider(side, "Light intensity", "intensity", 0, 100, " %")
        self._slider(side, "Light radius", "radius", 30, 260, " px")
        self._slider(side, "Ambient (outside light)", "ambient", 0, 100, " %")
        self._slider(side, "Motion threshold", "threshold", 8, 70)

        modes = tk.Frame(side, bg=C.PANEL)
        modes.pack(fill="x", padx=14, pady=(10, 0))
        modes.columnconfigure((0, 1, 2), weight=1, uniform="m")
        for i, (text, value) in enumerate((("Warm", "warm"), ("Cool", "cool"),
                                           ("White", "white"))):
            tk.Radiobutton(modes, text=text, value=value, variable=self.mode_var,
                           command=self._mode_changed, indicatoron=False,
                           bg=C.PANEL_LIGHT, fg=C.TEXT, selectcolor=C.ACCENT_DARK,
                           activebackground="#2c3345", activeforeground=C.TEXT,
                           font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
                           pady=6, cursor="hand2").grid(row=0, column=i, sticky="ew",
                                                        padx=2)

        # ---- output -----------------------------------------------
        self._section(side, "OUTPUT")
        row = tk.Frame(side, bg=C.PANEL)
        row.pack(fill="x", padx=14)
        row.columnconfigure((0, 1), weight=1, uniform="b")
        self.btn_shot = self._button(row, "Screenshot", self.save_screenshot)
        self.btn_rec = self._button(row, "Record Video", self.toggle_recording)
        self.btn_shot.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_rec.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.btn_reset = self._button(side, "Reset Settings", self.reset_settings)
        self.btn_reset.pack(fill="x", padx=14, pady=(8, 0))

    def _panel(self, parent, title, width, height):
        box = tk.Frame(parent, bg=C.PANEL)
        tk.Label(box, text=title, font=("Segoe UI", 9, "bold"), fg=C.ACCENT,
                 bg=C.PANEL, anchor="w").pack(fill="x", padx=10, pady=(8, 4))
        label = tk.Label(box, bg="black", bd=0, highlightthickness=0)
        label.pack(padx=8, pady=(0, 8))
        return box, label

    def _build_display(self, parent):
        area = tk.Frame(parent, bg=C.BG)
        area.pack(side="left", fill="both", expand=True)

        top = tk.Frame(area, bg=C.BG)
        top.pack(fill="x")
        box1, self.lbl_original = self._panel(top, "ORIGINAL VIDEO  (green cross = motion centre)", 0, 0)
        box2, self.lbl_processed = self._panel(top, "PROCESSED VIDEO  (virtual spotlight)", 0, 0)
        box1.pack(side="left", padx=(0, 12))
        box2.pack(side="left")

        bottom = tk.Frame(area, bg=C.BG)
        bottom.pack(fill="x", pady=(12, 0))
        box3, self.lbl_mask = self._panel(bottom, "MOTION MASK  (threshold + morphology)", 0, 0)
        box3.pack(side="left", padx=(0, 12))

        stats = tk.Frame(bottom, bg=C.PANEL)
        stats.pack(side="left", fill="both", expand=True)
        tk.Label(stats, text="LIVE STATISTICS", font=("Segoe UI", 9, "bold"),
                 fg=C.ACCENT, bg=C.PANEL, anchor="w").pack(fill="x", padx=12, pady=(8, 6))
        self.stat_vars = {}
        for name in ("Source", "Resolution", "FPS", "Motion centre", "Motion area",
                     "Light mode", "Recording"):
            row = tk.Frame(stats, bg=C.PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=name, width=14, anchor="w", font=("Segoe UI", 9),
                     fg=C.MUTED, bg=C.PANEL).pack(side="left")
            var = tk.StringVar(value="-")
            tk.Label(row, textvariable=var, anchor="w", font=("Segoe UI", 9, "bold"),
                     fg=C.TEXT, bg=C.PANEL).pack(side="left")
            self.stat_vars[name] = var

    # ================================================================ helpers
    def _show_placeholders(self):
        self._set_photo(self.lbl_original, placeholder(MAIN_W, MAIN_H, "Original video"))
        self._set_photo(self.lbl_processed, placeholder(MAIN_W, MAIN_H, "Processed video"))
        self._set_photo(self.lbl_mask, placeholder(MASK_W, MASK_H, "Motion mask"))

    @staticmethod
    def _set_photo(label, bgr):
        photo = to_photo(bgr)
        label.configure(image=photo)
        label.image = photo            # keep a reference or Tk will delete it

    def _set_status(self, text, colour):
        self.status_label.config(text="  " + text + "  ", fg=colour)

    def _running(self):
        return self.processor is not None and self.processor.is_alive()

    def _update_buttons(self):
        running = self._running()
        recording = running and self.processor.is_recording
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal" if running else "disabled")
        self.btn_upload.config(state="normal")
        self.btn_shot.config(state="normal" if self.last_data is not None else "disabled")
        self.btn_rec.config(state="normal" if running else "disabled",
                            text="Stop Recording" if recording else "Record Video",
                            fg=C.DANGER if recording else C.TEXT)

    # ============================================================== callbacks
    def _mode_changed(self):
        self.settings.mode = self.mode_var.get()

    def _loop_changed(self):
        if self.processor is not None:
            self.processor.loop = self.loop_var.get()

    def reset_settings(self):
        self.settings.reset()
        for key, (scale, callback) in self.sliders.items():
            scale.set(getattr(self.settings, key))
            callback(scale.get())
        self.mode_var.set(self.settings.mode)

    def _begin(self, source):
        """Start processing frames from an opened VideoSource."""
        self.stop_video()
        self.last_frame_id, self.last_data = -1, None
        self.processor = VideoProcessor(source, self.settings, self.loop_var.get())
        self.processor.start()
        self.stat_vars["Source"].set(source.name)
        self._set_status("Running - tracking motion", C.OK)
        self._update_buttons()

    def start_camera(self):
        try:
            source = VideoSource.open_camera()
        except VideoSourceError as exc:
            messagebox.showerror("Camera error", str(exc))
            return
        self._begin(source)

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if not path:
            return
        try:
            source = VideoSource.open_file(path)
        except VideoSourceError as exc:
            messagebox.showerror("Video error", str(exc))
            return
        self._begin(source)

    def stop_video(self):
        """Stop the worker thread; it releases the camera / writer itself."""
        if self.processor is not None:
            self.processor.stop()
            self.processor.join(timeout=3)
            self.processor = None
            self._show_placeholders()
            for var in self.stat_vars.values():
                var.set("-")
            self._set_status("Stopped", C.MUTED)
        self._update_buttons()

    def save_screenshot(self):
        data = self.last_data
        if data is None:
            messagebox.showinfo("Screenshot", "Start the video first.")
            return
        try:
            folder = os.path.join(OUTPUT_DIR, "screenshots")
            os.makedirs(folder, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            processed_path = os.path.join(folder, f"lumitrack_{stamp}.png")
            compare_path = os.path.join(folder, f"lumitrack_{stamp}_comparison.png")
            save_image(processed_path, data.processed)
            save_image(compare_path, np.hstack([data.clean_original, data.processed]))
        except Exception as exc:
            messagebox.showerror("Screenshot error", str(exc))
            return
        self._set_status("Screenshot saved", C.OK)
        messagebox.showinfo("Screenshot saved",
                            f"Saved to:\n{processed_path}\n{compare_path}")

    def toggle_recording(self):
        if not self._running():
            return
        if self.processor.is_recording:
            path = self.processor.record_path
            self.processor.stop_recording()
            self._set_status("Recording saved", C.OK)
            messagebox.showinfo("Recording saved", f"Processed video saved to:\n{path}")
        else:
            folder = os.path.join(OUTPUT_DIR, "videos")
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Recording error", str(exc))
                return
            path = os.path.join(folder, f"lumitrack_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            self.processor.start_recording(path)
            self._set_status("Recording...", C.DANGER)
        self._update_buttons()

    # ============================================================ main loop
    def _poll(self):
        """Called every ~15 ms: show the newest processed frame."""
        try:
            p = self.processor
            if p is not None:
                if p.error:
                    message = p.error
                    p.error = None
                    self.stop_video()
                    messagebox.showerror("LumiTrack", message)
                elif p.record_error:
                    message, p.record_error = p.record_error, None
                    messagebox.showerror("Recording error", message)
                    self._update_buttons()
                elif p.finished and not p.is_alive():
                    self.stop_video()
                    self._set_status("Video finished", C.MUTED)
                else:
                    frame_id, data = p.get_latest()
                    if data is not None and frame_id != self.last_frame_id:
                        self.last_frame_id, self.last_data = frame_id, data
                        self._show(data)
        except Exception as exc:                       # never kill the UI loop
            print("UI update error:", exc)
        self.root.after(15, self._poll)

    def _show(self, d):
        self._set_photo(self.lbl_original, fit_to_panel(d.original, MAIN_W, MAIN_H))
        self._set_photo(self.lbl_processed, fit_to_panel(d.processed, MAIN_W, MAIN_H))
        self._set_photo(self.lbl_mask, fit_to_panel(d.mask, MASK_W, MASK_H))

        v = self.stat_vars
        v["Resolution"].set(f"{d.size[0]} x {d.size[1]}")
        v["FPS"].set(f"{d.fps:.1f}")
        v["Motion centre"].set(f"({d.center[0]:.0f}, {d.center[1]:.0f})"
                               + ("" if d.found else "  (idle)"))
        v["Motion area"].set(f"{d.area:.0f} px" if d.found else "no motion")
        v["Light mode"].set(self.settings.mode.capitalize())
        v["Recording"].set("ON" if self.processor and self.processor.is_recording else "off")
        if self.btn_shot["state"] == "disabled":
            self._update_buttons()

    def on_close(self):
        """Release the camera and the video writer before the window closes."""
        try:
            self.stop_video()
        finally:
            self.root.destroy()
