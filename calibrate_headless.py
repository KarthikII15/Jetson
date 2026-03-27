#!/usr/bin/env python3
"""
FRS2 Headless Direction Calibration - Improved
ONE person only, tracks the largest face, smooths Y values
"""
import cv2
import time
import json
import signal
import sys
from datetime import datetime
from collections import deque

RTSP_URL = "rtsp://admin:Mli%40Frs!2026@172.18.3.201:554/h264"
OUTPUT_DIR = "/opt/frs/calibration"
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)
CASCADE = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE)

MODE = sys.argv[1] if len(sys.argv) > 1 else "entry"
print(f"{'='*55}")
print(f"FRS2 Direction Calibration — {MODE.upper()}")
print(f"{'='*55}")
print(f"⚠ ONLY ONE PERSON should be in frame")
print(f"⚠ Walk slowly and steadily past the camera")
print(f"⚠ Press Ctrl+C when done")
print(f"{'='*55}")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ret, frame = cap.read()
h, w = frame.shape[:2]
print(f"✅ Camera: {w}x{h}")
print(f"   Y=0 is TOP of frame, Y={h} is BOTTOM")
print()

y_data = []
y_smooth = deque(maxlen=5)  # smoothing window
start = time.time()
last_detection = 0
no_face_count = 0

def finish(sig=None, frm=None):
    cap.release()
    if len(y_data) < 5:
        print(f"\n⚠ Only {len(y_data)} detections — need at least 5")
        print("  Try again with only ONE person in frame")
        sys.exit(1)

    ys = [d['y'] for d in y_data]
    
    # Use linear regression for robust direction
    n = len(ys)
    xs = list(range(n))
    mean_x = sum(xs)/n
    mean_y = sum(ys)/n
    num = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
    den = sum((xs[i]-mean_x)**2 for i in range(n))
    slope = num/den if den != 0 else 0
    
    delta = ys[-1] - ys[0]
    direction_symbol = "↓ DOWNWARD" if slope > 0 else "↑ UPWARD"
    
    print(f"\n{'='*55}")
    print(f"RESULTS — {MODE.upper()} walkthrough:")
    print(f"  Detections  : {len(y_data)}")
    print(f"  Y start     : {ys[0]}")
    print(f"  Y end       : {ys[-1]}")
    print(f"  Y min/max   : {min(ys)} / {max(ys)}")
    print(f"  Net delta   : {delta:+d} px")
    print(f"  Slope       : {slope:+.2f} px/frame")
    print(f"  Direction   : {direction_symbol}")
    
    if slope > 5:
        print(f"\n  → Person moved DOWNWARD in frame (Y increased)")
        print(f"  → If this was ENTRY: entry_direction = 'increasing'")
    elif slope < -5:
        print(f"\n  → Person moved UPWARD in frame (Y decreased)")
        print(f"  → If this was ENTRY: entry_direction = 'decreasing'")
    else:
        print(f"\n  ⚠ Slope too flat — movement unclear")
        print(f"  → Camera may be overhead, or person didn't move enough")

    fname = f"{OUTPUT_DIR}/{MODE}_{datetime.now().strftime('%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump({'mode': MODE, 'delta': delta, 'slope': slope,
                   'y_data': y_data, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    print(f"\n  Saved: {fname}")
    print(f"{'='*55}")
    sys.exit(0)

signal.signal(signal.SIGINT, finish)

frame_count = 0
print("Waiting for face...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1
    if frame_count % 3 != 0:  # 5 FPS effective
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces — stricter params
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, 
        minSize=(100, 100), maxSize=(600, 600)
    )

    now = time.time() - start

    if len(faces) == 0:
        no_face_count += 1
        if no_face_count % 10 == 0:
            print(f"  t={now:.1f}s  [no face]")
        continue
    
    no_face_count = 0

    if len(faces) > 1:
        print(f"  t={now:.1f}s  ⚠ {len(faces)} faces — only ONE person please")
        continue

    x, y, fw, fh = faces[0]
    cy = y + fh // 2
    
    # Smooth Y
    y_smooth.append(cy)
    cy_smooth = int(sum(y_smooth) / len(y_smooth))
    
    # Skip if too close to last detection (< 0.1s)
    if y_data and now - y_data[-1]['t'] < 0.1:
        continue

    y_data.append({'y': cy_smooth, 'raw_y': cy, 'y_norm': round(cy/h, 3), 't': round(now, 2)})
    
    arrow = '▼' if len(y_data) < 2 or cy_smooth > y_data[-2]['y'] else '▲' if cy_smooth < y_data[-2]['y'] else '─'
    print(f"  t={now:.1f}s  Y={cy_smooth:4d} ({cy_smooth/h:.2f})  {arrow}  [{len(y_data)} pts]")
