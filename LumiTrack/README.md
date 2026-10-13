# LumiTrack - Real-Time Motion-Responsive Virtual Lighting System

A desktop application (Python + OpenCV + NumPy + Tkinter) that watches a webcam
or video file, detects the **largest moving region** with classical image
processing, and makes a **soft virtual spotlight follow it**.
No AI, no machine learning, no face recognition, no paid APIs.

Subject: Image and Video Processing (IVP) - individual mini-project.

---

## 1. Features

| Requirement | Where |
|---|---|
| Webcam input, Start / Stop Camera | `Start Camera`, `Stop Camera` buttons |
| Upload a video instead of webcam | `Upload Video...` (+ optional loop) |
| Grayscale, differencing, threshold, morphology, contours, centroid, smoothing | `lumitrack/motion_detector.py` |
| Radial mask, Gaussian blur, brightness / colour, blending | `lumitrack/lighting.py` |
| Warm / Cool (and White) light modes | Mode buttons |
| Intensity + radius sliders (plus ambient + motion threshold) | Left panel |
| Original, Processed and Motion-Mask views + live statistics | Main window |
| Screenshot | `Screenshot` -> `outputs/screenshots/` |
| Save processed video | `Record Video` -> `outputs/videos/` |
| Reset settings | `Reset Settings` |
| Non-freezing UI | video runs in a background thread (`processor.py`) |
| Camera released on exit | `on_close()` + `finally:` block in the worker thread |

## 2. Project structure

```
LumiTrack/
|-- main.py                  <- run this
|-- requirements.txt
|-- README.md
|-- make_test_video.py       <- optional: makes a sample clip for "Upload Video"
|-- lumitrack/
|   |-- app.py               <- Tkinter GUI (dark theme)
|   |-- processor.py         <- background video thread (the pipeline)
|   |-- motion_detector.py   <- motion detection + tracking
|   |-- lighting.py          <- spotlight mask + blending
|   |-- video_source.py      <- webcam / file wrapper
|   `-- config.py            <- constants, colours, Settings
|-- tests/test_pipeline.py   <- automatic checks of the algorithms
|-- docs/PROJECT_REPORT.md   <- abstract, objectives, viva Q&A ... (also .docx)
`-- outputs/                 <- screenshots and recordings are saved here
```

## 3. Installation (Windows)

1. Install **Python 3.9 - 3.12** from https://www.python.org/downloads/
   (tick **"Add python.exe to PATH"** in the installer; Tkinter is included).
2. Install **VS Code** and its *Python* extension.
3. Unzip the project, then in VS Code: **File > Open Folder... > LumiTrack**.
4. Open the terminal: **Terminal > New Terminal** and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If PowerShell refuses to activate the venv, run once:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
(You may also skip the venv: just run `pip install -r requirements.txt`.)

To run the algorithm checks:  `python -m unittest discover -s tests -v`
To make a test clip:  `python make_test_video.py`  -> creates `sample_video.mp4`

## 4. How to use

1. **Start Camera** (or **Upload Video...**).
2. Wait ~1 second: the program learns the empty background. **Keep still / step
   out of the frame during this time**, then move - the spotlight follows you.
3. Adjust **intensity, radius, ambient, motion threshold** and **Warm / Cool / White**.
4. **Screenshot** and **Record Video** save into the `outputs` folder.
5. **Stop Camera** or close the window - the camera is released.

Tips: good, even room light and a plain background give the best tracking.
If the light jitters, raise *Motion threshold*; if it does not react, lower it.

## 5. Algorithm (short version)

```
frame -> resize 640 px -> GRAYSCALE -> GAUSSIAN BLUR
      -> | frame - running-average background |      (differencing)
      -> BINARY THRESHOLD
      -> MORPHOLOGICAL OPENING, then CLOSING          (noise removal)
      -> FIND CONTOURS -> take the LARGEST
      -> CENTROID via image moments  cx = M10/M00, cy = M01/M00
      -> EXPONENTIAL SMOOTHING   s = 0.25*new + 0.75*s_old
      -> RADIAL MASK (filled circle) -> GAUSSIAN BLUR (soft edge)
      -> brightness gain x colour tint + glow   (lit image)
      -> dim the rest (ambient)
      -> BLEND:  out = dark*(1-mask) + lit*mask
```
Full details, IVP concepts, viva questions: see `docs/PROJECT_REPORT.md`.

## 6. Honest limitations

* A person who stands completely still slowly becomes part of the background
  (the background adapts at 2 % per frame), so the light stays where they last moved.
* A short "ghost" blob can appear where an object *used to be* (left-over in the
  running-average background).
* Camera auto-exposure / auto-focus or flickering lights can create false motion.
  A big sudden change (>60 % of pixels) is detected and the background is relearnt.
* Only the **largest** moving region is tracked (one spotlight).
* Recorded webcam videos are written at 20 fps; if your camera runs at another
  rate the playback speed of the saved clip may differ slightly.
* Tested on a video file through the real GUI/processing code. Webcam capture
  uses standard `cv2.VideoCapture` (DirectShow on Windows) - test it on your own laptop.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| "Could not open the webcam" | Close Zoom / Teams / Camera app. Windows Settings > Privacy > Camera > allow desktop apps. |
| Black webcam image | Try again, or change `CAMERA_INDEXES` in `lumitrack/config.py` to `(1, 0)`. |
| `No module named cv2` | Activate the venv, then `pip install -r requirements.txt`. |
| Window too big for the screen | Use Windows display scaling 100 % - 125 %. |
| Light flickers on nothing | Increase *Motion threshold*; avoid flickering lamps. |
