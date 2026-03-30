#!/usr/bin/env python3
"""
direction_test.py — FRS2 X-axis direction detection prototype
Mirrors the C++ sliding-window + line-crossing logic from runner.cpp.

Usage:
    # Run on a video file (uses OpenCV DNN face detector):
    python3 direction_test.py --video /path/to/door.mp4

    # Run on a pre-exported CSV of detections (faster, no GPU needed):
    # CSV format: frame,track_id,x1,y1,x2,y2
    python3 direction_test.py --csv /path/to/detections.csv

    # Tune parameters:
    python3 direction_test.py --video door.mp4 --threshold 30 --window 4 \
        --line_x 960 --mode slope --cooldown 30 --ttl 10
"""

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import cv2
import numpy as np

# ── Config defaults (match config.json) ───────────────────────────────────────
DEFAULT_THRESHOLD   = 30.0   # min px X-movement across window
DEFAULT_WINDOW      = 4      # sliding window frames
DEFAULT_LINE_X      = 960.0  # virtual vertical boundary
DEFAULT_MODE        = "slope" # "slope" or "line_cross"
DEFAULT_COOLDOWN    = 30.0   # seconds per employee+direction
DEFAULT_TTL         = 10.0   # seconds before track expires
DEFAULT_ENTRY_DIR   = "increasing" # X increases = entry (left→right)


# ── Track state ───────────────────────────────────────────────────────────────
@dataclass
class FaceTrack:
    track_id:        str
    x_history:       deque = field(default_factory=deque)
    last_x:          float = -1.0
    last_seen:       float = field(default_factory=time.time)
    direction_fired: bool  = False
    committed_dir:   str   = ""


# ── Direction engine (mirrors runner.cpp) ─────────────────────────────────────
class DirectionEngine:
    def __init__(self, args):
        self.x_threshold  = args.threshold
        self.window_size  = args.window
        self.line_x       = args.line_x
        self.mode         = args.mode
        self.entry_dir    = args.entry_dir
        self.cooldown_sec = args.cooldown
        self.track_ttl    = args.ttl

        self.tracks:    Dict[str, FaceTrack] = {}
        self.dir_cooldown: Dict[str, float]  = {}
        self.events = []

    # ── Track management ──────────────────────────────────────────────────────
    def assign_track(self, detection_id: str, cx: float) -> str:
        """Assign cx to the nearest existing track or create a new one."""
        now = time.time()
        self._purge_stale(now)

        best_id   = None
        best_dist = 150.0
        for tid, trk in self.tracks.items():
            if not trk.x_history:
                continue
            dist = abs(trk.x_history[-1] - cx)
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is not None:
            trk = self.tracks[best_id]
            gap = now - trk.last_seen
            if gap > 5.0 and trk.direction_fired:
                trk.direction_fired = False
                trk.committed_dir   = ""
                trk.x_history.clear()
                trk.last_x = -1.0
                print(f"  [Track] {best_id} reset after {gap:.1f}s gap")
            # Jitter filter: only push when meaningfully different
            if not trk.x_history or abs(cx - trk.x_history[-1]) > 5.0:
                trk.last_x = trk.x_history[-1] if trk.x_history else cx
                trk.x_history.append(cx)
                if len(trk.x_history) > self.window_size:
                    trk.x_history.popleft()
            trk.last_seen = now
            return best_id

        # New track
        tid = detection_id
        trk = FaceTrack(track_id=tid, last_seen=now)
        trk.last_x = cx
        trk.x_history.append(cx)
        self.tracks[tid] = trk
        print(f"  [Track] New track {tid}")
        return tid

    def _purge_stale(self, now: float):
        stale = [tid for tid, trk in self.tracks.items()
                 if now - trk.last_seen > self.track_ttl]
        for tid in stale:
            print(f"  [Track] Purging stale track {tid}")
            del self.tracks[tid]

    # ── Direction computation ─────────────────────────────────────────────────
    def compute_direction_slope(self, x_history: deque) -> str:
        """Sliding-window linear regression (mirrors C++ computeDirection)."""
        if len(x_history) < self.window_size:
            return "unknown"

        net_delta = x_history[-1] - x_history[0]
        total_movement = abs(net_delta)

        # Linear regression slope
        n  = len(x_history)
        xs = list(x_history)
        si, sx, six, sii = 0.0, 0.0, 0.0, 0.0
        for i, xv in enumerate(xs):
            si  += i
            sx  += xv
            six += i * xv
            sii += i * i
        denom = n * sii - si * si
        slope = (n * six - si * sx) / (denom + 1e-6)

        print(f"    slope={slope:.1f} net_delta={net_delta:.1f} "
              f"movement={total_movement:.1f} threshold={self.x_threshold:.0f}")

        if total_movement < self.x_threshold:
            return "stationary"

        increasing = net_delta > 0
        if self.entry_dir == "increasing":
            return "entry" if increasing else "exit"
        else:
            return "exit" if increasing else "entry"

    def compute_direction_line_cross(self, prev_x: float, curr_x: float) -> str:
        """Instant line-crossing detection (mirrors computeDirectionLineCross)."""
        dead = 5.0
        lo = self.line_x - dead
        hi = self.line_x + dead
        if prev_x < lo and curr_x >= hi:
            return "entry" if self.entry_dir == "increasing" else "exit"
        if prev_x > hi and curr_x <= lo:
            return "exit" if self.entry_dir == "increasing" else "entry"
        return "unknown"

    # ── Cooldown ──────────────────────────────────────────────────────────────
    def check_direction_cooldown(self, emp_id: str, direction: str, date: str) -> bool:
        key = f"{emp_id}_{direction}_{date}"
        now = time.time()
        last = self.dir_cooldown.get(key)
        if last is not None and now - last < self.cooldown_sec:
            print(f"    [Cooldown] {key} still in cooldown ({now-last:.0f}s / {self.cooldown_sec:.0f}s)")
            return False
        self.dir_cooldown[key] = now
        return True

    # ── Process one detection ─────────────────────────────────────────────────
    def process(self, frame_no: int, track_id: str, cx: float,
                emp_id: str = "unknown") -> Optional[str]:
        tid = self.assign_track(track_id, cx)
        trk = self.tracks[tid]

        x_str = " ".join(str(int(x)) for x in trk.x_history)
        print(f"  [{frame_no}] {tid} size={len(trk.x_history)} fired={trk.direction_fired} X=[{x_str}]")

        if trk.direction_fired:
            return None

        if self.mode == "line_cross":
            direction = self.compute_direction_line_cross(trk.last_x, cx)
        else:
            direction = self.compute_direction_slope(trk.x_history)

        if direction in ("entry", "exit"):
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self.check_direction_cooldown(emp_id, direction, date):
                trk.direction_fired = True
                trk.committed_dir   = direction
                ts = datetime.now(timezone.utc).isoformat()
                event = {
                    "frame": frame_no,
                    "track_id": tid,
                    "employee_id": emp_id,
                    "direction": direction,
                    "timestamp": ts,
                    "x_history": list(trk.x_history),
                }
                self.events.append(event)
                print(f"\n  🧭 COMMITTED: {emp_id} → {direction} at frame {frame_no} "
                      f"(track: {tid})\n")
                return direction
        return None


# ── OpenCV face detection (DNN, no TRT needed) ────────────────────────────────
def load_face_detector():
    """Load lightweight OpenCV DNN face detector (300x300 SSD)."""
    # Attempt to use the built-in face detector from opencv-contrib
    try:
        net = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades + "../dnn/deploy.prototxt",
            cv2.data.haarcascades + "../dnn/res10_300x300_ssd_iter_140000.caffemodel"
        )
        return ("dnn", net)
    except Exception:
        pass
    # Fallback: Haar cascade (fast, less accurate)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("haar", cascade)


def detect_faces_dnn(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < 0.5:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def detect_faces_haar(cascade, frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    boxes  = []
    for (x, y, w, h) in faces:
        boxes.append((x, y, x + w, y + h))
    return boxes


# ── Run on video file ─────────────────────────────────────────────────────────
def run_video(args):
    engine  = DirectionEngine(args)
    det_type, detector = load_face_detector()
    print(f"[detector] using {det_type}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {args.video}", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = 0
    track_counter = 0

    print(f"[video] fps={fps:.1f}  line_x={args.line_x}  mode={args.mode}")
    print(f"[video] threshold={args.threshold}px  window={args.window}  "
          f"cooldown={args.cooldown}s  ttl={args.ttl}s\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if det_type == "dnn":
            boxes = detect_faces_dnn(detector, frame)
        else:
            boxes = detect_faces_haar(detector, frame)

        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) / 2.0
            tid = f"trk{track_counter}"
            engine.process(frame_no, tid, cx)
            track_counter += 1

            # Annotate frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.line(frame, (int(args.line_x), 0),
                     (int(args.line_x), frame.shape[0]), (0, 100, 255), 1)
            cv2.putText(frame, f"cx={int(cx)}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if args.show:
            cv2.imshow("FRS Direction Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return engine.events


# ── Run on CSV file ───────────────────────────────────────────────────────────
def run_csv(args):
    """
    CSV format: frame, track_id, x1, y1, x2, y2
    (export from a real FRS run via spdlog or OpenCV)
    """
    engine = DirectionEngine(args)
    print(f"[csv] mode={args.mode} threshold={args.threshold}px "
          f"window={args.window} line_x={args.line_x}\n")

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_no = int(row["frame"])
            track_id = row["track_id"]
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            cx = (x1 + x2) / 2.0
            emp_id = row.get("employee_id", "unknown")
            engine.process(frame_no, track_id, cx, emp_id)

    return engine.events


# ── Calibration helper ────────────────────────────────────────────────────────
def print_calibration_summary(events):
    """Print summary to help calibrate x_threshold."""
    if not events:
        print("\n[calibration] No direction events fired.")
        print("  → Increase --threshold or ensure people walk past line_x.")
        return

    print(f"\n{'─'*60}")
    print(f"  CALIBRATION SUMMARY — {len(events)} direction event(s)")
    print(f"{'─'*60}")
    for ev in events:
        hist = ev["x_history"]
        delta = hist[-1] - hist[0] if hist else 0
        print(f"  frame={ev['frame']:>5}  {ev['direction']:>5}  "
              f"emp={ev['employee_id']:<16}  "
              f"Δx={delta:>+6.0f}px  X={[int(x) for x in hist]}")
    print(f"{'─'*60}")
    print(f"  Tip: at your camera FPS, set --threshold to ~70-80% of the")
    print(f"       typical |Δx| across {len(events[0]['x_history'])} frames.")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FRS2 X-axis direction detection prototype")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to video file")
    src.add_argument("--csv",   help="Path to detections CSV (frame,track_id,x1,y1,x2,y2)")

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Min X-movement (px) across window [{DEFAULT_THRESHOLD}]")
    p.add_argument("--window",    type=int,   default=DEFAULT_WINDOW,
                   help=f"Sliding window size [{DEFAULT_WINDOW}]")
    p.add_argument("--line_x",   type=float, default=DEFAULT_LINE_X,
                   help=f"Virtual boundary X position [{DEFAULT_LINE_X}]")
    p.add_argument("--mode",     choices=["slope", "line_cross"], default=DEFAULT_MODE,
                   help=f"Detection mode [{DEFAULT_MODE}]")
    p.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                   help=f"Cooldown seconds per employee+direction [{DEFAULT_COOLDOWN}]")
    p.add_argument("--ttl",      type=float, default=DEFAULT_TTL,
                   help=f"Track TTL seconds [{DEFAULT_TTL}]")
    p.add_argument("--entry_dir", choices=["increasing", "decreasing"],
                   default=DEFAULT_ENTRY_DIR,
                   help=f"Which X direction = entry [{DEFAULT_ENTRY_DIR}]")
    p.add_argument("--show", action="store_true",
                   help="Display annotated video (requires display)")
    return p.parse_args()


def main():
    args   = parse_args()
    events = run_video(args) if args.video else run_csv(args)
    print_calibration_summary(events)


if __name__ == "__main__":
    main()
