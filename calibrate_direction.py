#!/usr/bin/env python3
"""
FRS2 Direction Calibration Tool
Records Y-centroid of detected faces during walkthroughs
to determine entry_direction for your camera.
"""
import cv2
import time
import json
import numpy as np
from datetime import datetime

RTSP_URL = "rtsp://admin:Mli%40Frs!2026@172.18.3.201:554/h264"
OUTPUT_DIR = "/opt/frs/calibration"
SESSION_FILE = f"{OUTPUT_DIR}/session_{datetime.now().strftime('%H%M%S')}.json"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simple face detector using OpenCV Haar cascade for calibration only
face_cascade = cv2.CascadeClassifier(
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
)

print("=" * 60)
print("FRS2 DIRECTION CALIBRATION TOOL")
print("=" * 60)
print(f"Camera: {RTSP_URL}")
print()
print("Instructions:")
print("  Press 'e' → start recording ENTRY walkthrough")
print("  Press 'x' → start recording EXIT walkthrough")  
print("  Press 's' → stop current recording")
print("  Press 'r' → show results and determine direction")
print("  Press 'q' → quit")
print("=" * 60)

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"✅ Camera opened: {w}x{h}")
    print(f"   Y range: 0 (top) → {h} (bottom)")
else:
    print("❌ Cannot read frame")
    exit(1)

sessions = []
current_session = None
recording = False

def draw_overlay(frame, current_session, recording):
    h, w = frame.shape[:2]
    # Draw horizontal thirds
    cv2.line(frame, (0, h//3), (w, h//3), (100,100,100), 1)
    cv2.line(frame, (0, 2*h//3), (w, 2*h//3), (100,100,100), 1)
    cv2.putText(frame, "TOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.putText(frame, "MID", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.putText(frame, "BOT", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    
    if recording and current_session:
        label = f"● REC [{current_session['type'].upper()}] - {len(current_session['y_values'])} pts"
        cv2.putText(frame, label, (w//2-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        cv2.putText(frame, "e=ENTRY  x=EXIT  s=STOP  r=RESULTS  q=QUIT", 
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    
    for (x, y, fw, fh) in faces:
        cx = x + fw // 2
        cy = y + fh // 2
        cy_norm = cy / h
        
        # Draw face box and centroid
        cv2.rectangle(frame, (x,y), (x+fw, y+fh), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
        cv2.putText(frame, f"Y={cy} ({cy_norm:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Horizontal line at face Y
        cv2.line(frame, (0, cy), (w, cy), (0,255,255), 1)
        
        if recording and current_session:
            current_session['y_values'].append({
                'y': cy, 'y_norm': round(cy_norm, 3), 
                't': round(time.time(), 3)
            })
    
    frame = draw_overlay(frame, current_session, recording)
    cv2.imshow('FRS2 Direction Calibration', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('e'):
        current_session = {'type': 'entry', 'y_values': [], 'start': time.time()}
        recording = True
        print("🟢 Recording ENTRY walkthrough... walk past camera now")
    
    elif key == ord('x'):
        current_session = {'type': 'exit', 'y_values': [], 'start': time.time()}
        recording = True
        print("🔴 Recording EXIT walkthrough... walk past camera now")
    
    elif key == ord('s'):
        if recording and current_session and len(current_session['y_values']) > 0:
            sessions.append(current_session)
            print(f"✅ Saved {current_session['type']} session: {len(current_session['y_values'])} points")
            ys = [p['y'] for p in current_session['y_values']]
            print(f"   Y range: {min(ys)} → {max(ys)} (delta: {ys[-1]-ys[0]:+d})")
        recording = False
        current_session = None
    
    elif key == ord('r'):
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        
        entry_deltas = []
        exit_deltas = []
        
        for s in sessions:
            if len(s['y_values']) < 3:
                continue
            ys = [p['y'] for p in s['y_values']]
            delta = ys[-1] - ys[0]
            print(f"  {s['type'].upper():8s}: Y {ys[0]} → {ys[-1]} (delta: {delta:+d}) {'↓' if delta>0 else '↑'}")
            if s['type'] == 'entry':
                entry_deltas.append(delta)
            else:
                exit_deltas.append(delta)
        
        if entry_deltas and exit_deltas:
            avg_entry = sum(entry_deltas) / len(entry_deltas)
            avg_exit = sum(exit_deltas) / len(exit_deltas)
            print(f"\n  Avg ENTRY delta: {avg_entry:+.1f}")
            print(f"  Avg EXIT  delta: {avg_exit:+.1f}")
            
            entry_direction = "increasing" if avg_entry > 0 else "decreasing"
            suggested_threshold = int(min(abs(avg_entry), abs(avg_exit)) * 0.4)
            suggested_threshold = max(20, min(80, suggested_threshold))
            
            print(f"\n✅ RESULT:")
            print(f"   entry_direction = \"{entry_direction}\"")
            print(f"   y_threshold     = {suggested_threshold}")
            print(f"\n   Add to /opt/frs/config.json:")
            print(f'   "entry_direction": "{entry_direction}",')
            print(f'   "y_threshold": {suggested_threshold}')
            
            # Save to file
            result = {
                'entry_direction': entry_direction,
                'y_threshold': suggested_threshold,
                'sessions': sessions,
                'timestamp': datetime.now().isoformat()
            }
            with open(SESSION_FILE, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n   Saved to: {SESSION_FILE}")
        else:
            print("  ⚠ Need at least 1 entry AND 1 exit session")
        print("="*60)
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
