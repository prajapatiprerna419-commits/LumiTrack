"""
make_test_video.py  (OPTIONAL helper)
-------------------------------------
Creates 'sample_video.mp4': a plain moving box on a textured background.
Use it to test the "Upload Video" button if you have no other video handy.
This is only a test clip for the uploader - the webcam mode never uses it.
"""
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 30, 12
rng = np.random.default_rng(1)
background = (rng.integers(60, 110, (H, W, 3))).astype(np.uint8)
background = cv2.GaussianBlur(background, (31, 31), 0)

out = cv2.VideoWriter("sample_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for i in range(FPS * SECONDS):
    t = i / FPS
    frame = background.copy()
    cx = int(W / 2 + 220 * np.sin(t * 0.9))
    cy = int(H / 2 + 120 * np.sin(t * 1.7))
    cv2.rectangle(frame, (cx - 40, cy - 70), (cx + 40, cy + 70), (60, 160, 230), -1)
    cv2.circle(frame, (cx, cy - 90), 25, (180, 200, 240), -1)
    out.write(frame)
out.release()
print("Created sample_video.mp4")
