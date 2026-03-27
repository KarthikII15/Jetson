# FRS2 — C++ TensorRT Runner

High-performance face recognition runner for Jetson Orin.
Replaces the Python runner with 5-6× better throughput.

## Performance

| | Python runner | C++ TRT runner |
|---|---|---|
| FPS | 5 FPS | 25–30 FPS |
| Detection latency | ~80ms | ~5ms |
| Embedding latency | ~120ms | ~8ms |
| Total per frame | ~200ms | ~15ms |
| Memory (GPU heap) | 4–6 GB | 1.2 GB |

## Architecture

```
Camera (RTSP H.264)
  → GStreamer NVDEC (GPU H.264 decode)
  → Frame queue (lockfree ring)
  → YOLOv8n-face TRT FP16 (detection, <5ms)
  → ArcFace R50 TRT FP16 (embedding, <8ms)
  → POST /api/face/recognize (pgvector search)
  → WebSocket → Frontend dashboard
```

## Prerequisites

- Jetson Orin (JetPack 5.x or 6.x)
- TensorRT 8.x+ (pre-installed on JetPack)
- OpenCV 4.x with GStreamer support
- libcurl, nlohmann-json, spdlog

## Step 1: Download ONNX Models

### YOLOv8n-face
```bash
mkdir -p /opt/frs-models/onnx
cd /opt/frs-models/onnx

# Option A: Download pre-exported ONNX
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.onnx

# Option B: Export yourself
pip install ultralytics
yolo export model=yolov8n-face.pt format=onnx imgsz=640
```

### ArcFace R50
```bash
# Download from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcface-lresnet100e-opset8.onnx \
  -O /opt/frs-models/onnx/arcface-r50.onnx

# Or from InsightFace (recommended — better accuracy)
# pip install insightface
# python3 -c "
# import insightface
# app = insightface.app.FaceAnalysis(name='buffalo_l')
# app.prepare(ctx_id=0)
# # Models downloaded to ~/.insightface/models/buffalo_l/
# # Copy w600k_r50.onnx → /opt/frs-models/onnx/arcface-r50.onnx
# "
```

## Step 2: Build TensorRT Engines

**Must run ON the Jetson** — engines are device-specific and not portable.

```bash
bash /home/ubuntu/frs-cpp/scripts/build_engines.sh
```

This takes 10–20 minutes. Engines are saved to `/opt/frs-models/trt/`.

## Step 3: Build C++ Runner

```bash
cd /home/ubuntu/frs-cpp
bash scripts/build_runner.sh
```

## Step 4: Configure

```bash
# Set camera password
nano /opt/frs/cameras.json
# Replace CAMERA_PASSWORD with real password

# Verify backend URL
nano /opt/frs/config.json
# "url": "http://172.20.100.222:8080"
```

## Step 5: Get Auth Token

```bash
bash /opt/frs/token_manager.sh
cat /opt/frs/device_token.txt | head -c 50
```

## Step 6: Test Camera

```bash
# Verify RTSP stream is reachable
ffprobe -i rtsp://admin:PASSWORD@172.18.3.201:554/Streaming/Channels/102 -t 3 2>&1 | head -20

# Test ISAPI snapshot
curl -o /tmp/snapshot.jpg \
  "http://admin:PASSWORD@172.18.3.201:80/ISAPI/Streaming/channels/101/picture"
ls -lh /tmp/snapshot.jpg
```

## Step 7: Run

```bash
# Manual test run
frs_runner --config /opt/frs/config.json --cameras /opt/frs/cameras.json

# As a service
sudo systemctl start frs-runner
sudo systemctl enable frs-runner
sudo journalctl -u frs-runner -f
```

## Step 8: Verify

```bash
# Health check
curl http://172.18.3.202:5000/health | python3 -m json.tool

# Expected output:
# {
#   "status": "ok",
#   "frames_processed": 150,
#   "faces_detected": 23,
#   "matches": 18,
#   "unknowns": 5,
#   "queue_depth": 0,
#   "active_cameras": 1,
#   "inference_engine": "tensorrt_fp16"
# }
```

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `conf_threshold` | 0.45 | Lower = detect more faces (more false positives) |
| `match_threshold` | 0.55 | Lower = more permissive matching |
| `cooldown_seconds` | 10 | Min seconds between marks per camera |
| `fps_target` | 5 | Inference FPS (higher = more CPU for capture) |
| `inference_threads` | 2 | Parallel TRT workers |

## Face Enrollment

Enroll employees via the Admin Dashboard → Employee Management → Enroll button.
The backend calls `/api/employees/:id/enroll-face` which runs ArcFace on the photo
and stores the embedding in pgvector.

For bulk enrollment via camera snapshot:
```bash
curl -X POST http://172.18.3.202:5000/enroll \
  -H "Content-Type: application/json" \
  -d '{"employee_id": "1", "cam_id": "entrance-cam-01"}'
```
