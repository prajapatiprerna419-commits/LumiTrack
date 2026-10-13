---
title: "LumiTrack - Real-Time Motion-Responsive Virtual Lighting System Using Image and Video Processing"
subtitle: "Mini-Project Report - Image and Video Processing (IVP)"
---

# 1. Abstract

LumiTrack is a real-time desktop application that applies a virtual spotlight effect to live video. The system captures frames from a webcam (or an uploaded video file) and detects moving objects using purely classical image-processing techniques: grayscale conversion, Gaussian filtering, background differencing, binary thresholding, morphological opening and closing, and contour detection. The centroid of the largest moving region is computed with image moments and smoothed with an exponential moving average to avoid sudden jumps. A radial circular mask is then generated at that position, softened with a Gaussian blur, coloured (warm, cool or white) and alpha-blended with the original frame while the surrounding area is dimmed. The result is a soft light that follows the moving person or object. The application, written in Python with OpenCV, NumPy and Tkinter, shows the original video, the motion mask and the processed video side by side, offers sliders for light intensity and radius, and can save screenshots and processed video. No machine learning, deep learning or face recognition is used.

# 2. Problem Statement

Professional lighting that follows a performer, presenter or subject normally needs expensive motorised lights and a human operator. In video calls, lectures, small stages and creative video editing, there is no simple, low-cost way to draw attention to the moving subject. The problem is to design a software system that, using only an ordinary webcam and traditional image-processing methods, detects where movement occurs in a video stream and automatically produces a smooth, realistic virtual light that follows it in real time, without trained recognition models.

# 3. Objectives

1. Capture live video from a webcam, or read a video file, and display it in a GUI.
2. Detect moving regions with frame/background differencing, thresholding and morphological filtering.
3. Locate the centre of the largest moving region using contours and image moments.
4. Smooth the tracked coordinates so that the light moves gracefully.
5. Generate a soft radial light mask using a circle and Gaussian blur.
6. Blend the light with the video using brightness, colour (warm/cool) and alpha blending.
7. Give the user real-time control (intensity, radius, mode) and output options (screenshot, saved video).
8. Keep the user interface responsive by processing video in a separate thread.
9. Demonstrate the practical use of IVP concepts studied in the course.

# 4. Methodology

The project follows a pipeline design with three layers:

* **Input layer** (`video_source.py`) - opens the webcam (DirectShow backend on Windows) or a video file and delivers frames.
* **Processing layer** (`motion_detector.py`, `lighting.py`, `processor.py`) - a background thread reads each frame, detects motion, renders the spotlight and publishes the result.
* **Presentation layer** (`app.py`) - a Tkinter window polls the newest result every ~15 ms and displays it; buttons and sliders update a shared `Settings` object that the processing thread reads.

**Workflow**

```
Webcam / Video file
        |
        v
 Frame extraction (cv2.VideoCapture.read)
        |
        v
 Resize to 640 px width  (+ mirror for webcam)
        |
        v
 +--------------- Motion detection ----------------+
 | Grayscale -> Gaussian blur -> |frame - background|
 | -> Threshold -> Opening -> Closing -> Contours   |
 | -> Largest contour -> Centroid -> Smoothing      |
 +--------------------------------------------------+
        |   (x, y)
        v
 +--------------- Virtual lighting -----------------+
 | Radial circle mask -> Gaussian blur (soft edge)  |
 | -> Brightness x colour tint + glow -> dim ambient |
 | -> Alpha blend with original frame               |
 +--------------------------------------------------+
        |
        v
 Display (Original | Motion mask | Processed) + optional screenshot / recording
```

# 5. Algorithm

**Input:** video stream. **Output:** video with a spotlight following the largest moving region.

1. Open the video source. Repeat for every frame until stopped:
2. Read frame `F`; resize to width 640; flip horizontally if it comes from a webcam.
3. Convert to grayscale: `G = 0.299 R + 0.587 G + 0.114 B`.
4. Smooth with a 21x21 Gaussian filter to reduce sensor noise.
5. If there is no background yet, set `B = G` and go to step 2.
6. Difference image: `D(x,y) = | G(x,y) - B(x,y) |`.
7. Threshold: `M(x,y) = 255 if D(x,y) > T else 0` (`T` default 25, adjustable).
8. Update the running-average background: `B = (1 - a) B + a G` with `a = 0.5` during the first 15 frames and `a = 0.02` afterwards.
9. If more than 60 % of pixels are white, the camera exposure or the whole scene changed: reset `B = G` and skip the frame.
10. Morphological opening (5x5 ellipse) to remove small specks, then closing (17x17 ellipse) to fill holes.
11. Find external contours; choose the one with the maximum area. If its area is below 0.3 % of the frame, treat it as "no motion".
12. Centroid with moments: `cx = M10 / M00`, `cy = M01 / M00`.
13. Smooth: `S_t = 0.25 * C_t + 0.75 * S_(t-1)`.
14. Presence value: rises by 0.25 on motion (max 1), falls by 0.03 per frame without motion; light strength = `0.30 + 0.70 * presence`, so the light stays visible at the last position but dims when nothing moves.
15. Radial mask: draw a filled circle of radius `R` at `S_t` on a black canvas (at 1/4 resolution for speed).
16. Gaussian blur the circle -> soft mask `A(x,y)` in [0,1]; resize to the frame size.
17. Lit image: `L = F * (1 + 0.8 k) * tint + glow * 0.18 k`, where `k` = intensity/100 and `tint` is the warm (R boosted), cool (B boosted) or white multiplier.
18. Dark image: `Dk = F * ambient/100`.
19. Alpha blend: `Out = Dk * (1 - A) + L * A`.
20. Display; optionally save the frame to a screenshot or video file.

# 6. IVP Concepts Used

| Concept | Where it is used |
|---|---|
| Video frame extraction | `cv2.VideoCapture.read()` in `video_source.py` |
| Image resizing / interpolation | `cv2.resize` with `INTER_AREA` (shrinking) and `INTER_LINEAR` (mask upscaling) |
| Colour space conversion (BGR to grayscale) | `cv2.cvtColor` in `motion_detector.py` |
| Spatial filtering - Gaussian smoothing | Noise reduction before differencing; soft edge of the light mask |
| Frame / background differencing | `cv2.absdiff` against a running-average background (`cv2.accumulateWeighted`) |
| Binary thresholding / segmentation | `cv2.threshold(..., THRESH_BINARY)` |
| Morphological operations | `MORPH_OPEN`, `MORPH_CLOSE` with elliptical structuring elements |
| Contour detection and region analysis | `cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect` |
| Image moments / centroid | `cv2.moments` -> `cx = M10/M00`, `cy = M01/M00` |
| Temporal smoothing (low-pass filtering of a signal) | Exponential moving average of the coordinates |
| Mask generation (radial / circular) | `cv2.circle` on a float canvas |
| Brightness and colour adjustment | Gain and per-channel (B, G, R) multipliers, additive glow |
| Image blending (alpha compositing) | `Out = Dark*(1-A) + Lit*A` |
| Video writing / codecs | `cv2.VideoWriter` with `mp4v` |

# 7. Real-World Applications

* Virtual stage / spotlight for school and college presentations and online lectures.
* Automatic subject emphasis in video calls and live streams.
* Interactive art installations and museum displays that react to visitors.
* Security: highlighting the moving region in surveillance footage.
* Retail displays that spotlight a customer approaching a product.
* Video post-production: cheap virtual "follow-spot" effects.
* Teaching tool for demonstrating motion detection and image blending.

# 8. Limitations

* Only the largest moving region is tracked; several moving people give a single light.
* A completely stationary subject is gradually absorbed into the background, so the light stays at the last position and dims.
* Moving shadows, reflections, flickering lights, waving curtains and camera auto-exposure can cause false detections.
* A short "ghost" may remain where an object was before it moved (side effect of the running-average background).
* The system detects *motion*, not *what* is moving: it cannot distinguish a person from any other moving object (by design - no recognition models).
* Performance depends on the CPU; frames are processed at 640 px width to stay real-time.
* Webcam recordings are saved at a fixed 20 fps, which may differ slightly from the camera's true rate.

# 9. Future Scope

* Track several objects with several lights (connected-component labelling / multi-blob tracking).
* Kalman-filter prediction for smoother, lag-free tracking.
* Shadow suppression using HSV or colour-based differencing.
* Selectable light shapes (ellipse, beam, vignette) and colour pickers.
* Export of the motion trail / heat-map of movement.
* Region-of-interest selection and adjustable tracking speed in the GUI.
* Audio-reactive lighting and virtual-camera output for Zoom / Teams.
* Optional GPU acceleration (OpenCV CUDA / OpenCL).

# 10. Testing Summary

* `tests/test_pipeline.py` checks that (a) a moving box is tracked within 25 px, (b) a static scene gives no detection, (c) the spotlight is brighter at the centre than at the edge, (d) warm mode is red-dominant and cool mode blue-dominant. All tests pass.
* The complete application (GUI, background thread, tracking, spotlight, recording, screenshot, reset, shutdown) was run on a video file; recording produced a valid MP4, and after closing the window the worker thread had ended and the video source was released.
* Live webcam capture uses the standard OpenCV `VideoCapture` interface and should be verified on the target laptop.

# 11. Viva Questions and Answers

**Q1. What is the aim of your project?**
To make a virtual spotlight that automatically follows a moving person or object in live video, using only traditional image-processing techniques.

**Q2. Why did you not use machine learning or face detection?**
The subject (IVP) is about classical techniques, and the goal is to react to *movement*, not to recognise identity. Motion detection is fast, needs no training data, and runs in real time on any laptop CPU.

**Q3. What is frame differencing?**
Subtracting one frame (or a reference background) from the current frame pixel by pixel. Pixels that changed a lot belong to moving objects. `D = |current - reference|`.

**Q4. Which method did you use - frame differencing or background subtraction?**
A running-average background subtraction (an adaptive form of frame differencing): the reference image is a weighted average of past frames updated with `cv2.accumulateWeighted`, so it adapts slowly to lighting changes.

**Q5. Why not subtract only the previous frame?**
Two-frame differencing detects only the edges of an object and fails when it moves slowly. The running average gives a more complete shape.

**Q6. Why convert to grayscale?**
Motion can be found from brightness changes alone; one channel is 3x cheaper to process than three, and thresholding works on a single intensity value.

**Q7. Why do you blur the frame before differencing?**
Gaussian blur averages neighbouring pixels, which removes camera noise that would otherwise appear as tiny false motion.

**Q8. What is thresholding?**
Converting a grey image to a binary one: pixels above a threshold `T` become 255 (motion) and the rest 0. It is the simplest form of image segmentation.

**Q9. What are morphological opening and closing? Why both?**
Opening = erosion then dilation: removes small white noise. Closing = dilation then erosion: fills small black holes inside objects. Together they give a clean, solid blob.

**Q10. What is a structuring element?**
A small shape (here an ellipse) that slides over the image and defines the neighbourhood used by erosion and dilation.

**Q11. How do you find the centre of the moving object?**
Find contours, choose the largest by area, and compute image moments. The centroid is `cx = M10/M00`, `cy = M01/M00`, where `M00` is the area and `M10`, `M01` are the first-order moments.

**Q12. Why do you track only the largest contour?**
It ignores small leftover noise and gives one stable target for the spotlight.

**Q13. How do you stop the light from jumping?**
With exponential smoothing: `S = a*new + (1-a)*S_old` (a = 0.25). It is a low-pass filter on the position signal; smaller `a` = smoother but slower.

**Q14. How is the spotlight created?**
Draw a filled white circle on a black image (radial mask), blur it with a Gaussian filter to get soft edges, then use it as an alpha mask.

**Q15. Why is Gaussian blur used on the mask?**
A hard circle looks artificial. Gaussian blur makes the intensity fall off smoothly from the centre, like real light.

**Q16. Explain the blending formula.**
`Out = Dark*(1-A) + Lit*A`. Where the mask `A` is 1 the pixel comes from the brightened, tinted image; where `A` is 0 it comes from the dimmed image; in between it is a mix (alpha compositing).

**Q17. How do warm and cool lighting work?**
Each colour channel is multiplied by a different factor. OpenCV stores BGR, so warm boosts Red (and lowers Blue), cool boosts Blue (and lowers Red).

**Q18. What is the role of the morphological kernel size?**
A bigger kernel removes bigger noise but may also remove small real motion or merge nearby objects.

**Q19. How did you keep the GUI from freezing?**
Video capture and processing run in a separate `threading.Thread`. The Tkinter main thread only displays the newest frame using `root.after()`. Data is exchanged through a lock-protected variable.

**Q20. Why are frames resized to 640 px?**
Processing cost grows with the number of pixels. Smaller frames keep the pipeline real-time, with no noticeable loss in tracking accuracy.

**Q21. What happens if the camera cannot be opened?**
The program catches the failure and shows an error message with suggestions (close other apps, check Windows camera privacy). The camera is always released in a `finally` block and when the window closes.

**Q22. What if somebody stands still?**
They slowly merge into the background (learning rate 0.02), so no motion is detected; the light stays at the last position and dims. This is a known limitation of motion-based methods.

**Q23. What if the room lighting suddenly changes?**
If more than 60 % of the pixels change at once, the program treats it as a global change and relearns the background instead of reporting a huge "object".

**Q24. What are the limitations of your system?**
One light only, shadows and flicker cause false motion, stationary subjects are lost, and it cannot tell what is moving.

**Q25. How can it be improved?**
Multi-object tracking, Kalman filtering, shadow removal, more light shapes, and a virtual-camera output for video calls.

**Q26. What is the difference between motion detection and object recognition?**
Motion detection only finds *where pixels change*; recognition decides *what* an object is, usually with a trained model. This project does only the former.

**Q27. What is the time complexity?**
Each step (blur, difference, threshold, morphology, mask blending) touches every pixel a constant number of times, so the cost per frame is O(W x H) for a frame of width W and height H.

**Q28. Why is the mask computed at a quarter of the resolution?**
A large Gaussian blur is expensive. Because the mask is smooth anyway, blurring a small version and enlarging it by interpolation looks identical but is much faster.
