"""
processor.py
------------
Runs the whole video pipeline in a BACKGROUND THREAD so the Tkinter window
never freezes:

    read frame -> resize -> detect motion -> render spotlight -> publish

The GUI simply asks for the newest result every few milliseconds with
get_latest().  Nothing here touches Tkinter widgets (Tkinter is not
thread-safe), data is exchanged through a lock-protected slot.
"""
import threading
import time
from dataclasses import dataclass

import cv2

from .config import PROCESS_WIDTH
from .lighting import SpotlightRenderer
from .motion_detector import MotionDetector


@dataclass
class FrameData:
    """One processed frame plus statistics, handed to the GUI."""
    original: object      # BGR image with the detection box drawn on it
    clean_original: object  # BGR image without any drawing (for screenshots)
    mask: object          # binary motion mask (grey image)
    processed: object     # BGR image with the virtual lighting
    fps: float
    center: tuple
    area: float
    found: bool
    size: tuple           # (width, height)


class VideoProcessor(threading.Thread):
    def __init__(self, source, settings, loop=False):
        super().__init__(daemon=True)
        self.source = source
        self.settings = settings
        self.loop = loop

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._frame_id = 0

        # status flags read by the GUI
        self.error = None            # text of a fatal error, if any
        self.finished = False        # True when the thread has ended

        # recording
        self._writer = None
        self._record_path = None
        self._pending_record = None
        self._stop_record = False
        self.record_error = None

    # ----------------------------------------------------------- public API
    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        """Returns (frame_id, FrameData or None)."""
        with self._lock:
            return self._frame_id, self._latest

    def start_recording(self, path):
        with self._lock:
            self.record_error = None
            self._pending_record = path

    def stop_recording(self):
        with self._lock:
            self._pending_record = None
            self._stop_record = True

    @property
    def is_recording(self):
        return self._writer is not None or self._pending_record is not None

    @property
    def record_path(self):
        return self._record_path

    # --------------------------------------------------------------- thread
    def run(self):
        detector = MotionDetector()
        renderer = SpotlightRenderer()
        fps, last_time = 0.0, time.time()
        failed_reads = 0

        try:
            while not self._stop_event.is_set():
                tick = time.time()
                ok, frame = self.source.read()

                if not ok or frame is None:
                    failed_reads += 1
                    if self.source.is_file and self.loop and failed_reads < 3:
                        self.source.rewind()          # restart the video
                        detector.reset()
                        continue
                    if self.source.is_camera and failed_reads < 30:
                        time.sleep(0.03)              # camera hiccup - retry
                        continue
                    if self.source.is_camera:
                        self.error = "The camera stopped sending frames."
                    break                             # end of video file
                failed_reads = 0

                # Resize to a fixed width: keeps the processing fast.
                h, w = frame.shape[:2]
                if w != PROCESS_WIDTH:
                    frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)),
                                       interpolation=cv2.INTER_AREA)
                if self.source.is_camera:
                    frame = cv2.flip(frame, 1)        # mirror view feels natural

                # --- motion detection -------------------------------------
                s = self.settings
                detector.threshold = s.threshold
                result = detector.process(frame)

                # --- virtual lighting -------------------------------------
                strength = 0.30 + 0.70 * result.presence
                processed, _ = renderer.apply(
                    frame, result.center, s.radius, s.intensity, s.mode,
                    s.ambient, strength)

                # --- picture for the "Original" panel --------------------
                shown = frame.copy()
                if result.found:
                    x, y, bw, bh = result.bbox
                    cv2.rectangle(shown, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                    cv2.drawMarker(shown, (int(result.center[0]), int(result.center[1])),
                                   (0, 255, 120), cv2.MARKER_CROSS, 22, 2)

                # --- recording ------------------------------------------
                self._handle_recording(processed)

                # --- FPS (smoothed) -----------------------------------
                now = time.time()
                inst = 1.0 / max(now - last_time, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                last_time = now

                data = FrameData(
                    original=shown, clean_original=frame, mask=result.mask,
                    processed=processed, fps=fps, center=result.center,
                    area=result.area, found=result.found,
                    size=(frame.shape[1], frame.shape[0]))
                with self._lock:
                    self._latest = data
                    self._frame_id += 1

                # Video files: play at their natural speed.
                if self.source.is_file:
                    delay = 1.0 / self.source.fps - (time.time() - tick)
                    if delay > 0:
                        time.sleep(delay)

        except Exception as exc:                      # report to the GUI
            self.error = f"Processing error: {exc}"
        finally:
            self._close_writer()
            self.source.release()                     # always free the camera
            self.finished = True

    # ------------------------------------------------------------ recording
    def _handle_recording(self, processed):
        with self._lock:
            pending, stop = self._pending_record, self._stop_record
            self._pending_record, self._stop_record = None, False

        if stop:
            self._close_writer()
        if pending:
            h, w = processed.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            record_fps = 20.0 if self.source.is_camera else self.source.fps
            writer = cv2.VideoWriter(pending, fourcc, record_fps, (w, h))
            if writer.isOpened():
                self._writer, self._record_path = writer, pending
            else:
                writer.release()
                self.record_error = "Could not create the output video file."
        if self._writer is not None:
            self._writer.write(processed)

    def _close_writer(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
