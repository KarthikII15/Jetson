#!/bin/bash
# J01_build_cpp_runner.sh
# Builds the FRS2 C++ TensorRT runner on Jetson Orin.
# Run this ONCE on the Jetson after copying the FRS-CPP directory.
# Usage: bash J01_build_cpp_runner.sh

set -e
JETSON_IP="172.18.3.202"
BACKEND_IP="172.20.100.222"
CAM_IP="172.18.3.201"
CAM_PASS="Mli@Frs!2026"

echo "=================================================="
echo " FRS2 C++ Runner — Full Build & Setup"
echo " Jetson: $JETSON_IP"
echo " Backend: $BACKEND_IP"
echo " Camera: $CAM_IP"
echo "=================================================="

# ── 1. Fix CUDA PATH ──────────────────────────────────────────────────────────
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

if ! grep -q "cuda/bin" ~/.bashrc; then
  echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
fi

echo ""
echo "[1/7] Checking CUDA and TensorRT..."
nvcc --version 2>/dev/null | grep "release" || echo "  ⚠  nvcc not on path (CUDA libraries still present)"
TRT_HEADER=$(find /usr/include -name "NvInfer.h" 2>/dev/null | head -1)
[ -n "$TRT_HEADER" ] && echo "  ✅ TensorRT headers: $TRT_HEADER" || echo "  ❌ TensorRT headers not found"
TRT_LIB=$(find /usr/lib -name "libnvinfer.so*" 2>/dev/null | head -1)
[ -n "$TRT_LIB" ] && echo "  ✅ TensorRT lib: $TRT_LIB" || echo "  ❌ libnvinfer not found"

# ── 2. Install build dependencies ────────────────────────────────────────────
echo ""
echo "[2/7] Installing build dependencies..."
sudo apt-get update -q 2>&1 | tail -1
sudo apt-get install -y \
  g++ pkg-config ninja-build \
  libopencv-dev \
  libcurl4-openssl-dev \
  nlohmann-json3-dev \
  libspdlog-dev \
  -q 2>&1 | tail -3

echo "  ✅ Build deps installed"

# ── 3. Locate FRS-CPP source ──────────────────────────────────────────────────
echo ""
echo "[3/7] Locating FRS-CPP source..."

# Look in common places
SRC_DIR=""
for d in \
  "$HOME/FRS-CPP" \
  "$HOME/frs-cpp" \
  "/home/ubuntu/FRS-CPP" \
  "/opt/frs-cpp" \
  "/home/motivity/FRS/frs-cpp"; do
  if [ -f "$d/CMakeLists.txt" ]; then
    SRC_DIR="$d"
    break
  fi
done

if [ -z "$SRC_DIR" ]; then
  echo "  ❌ FRS-CPP not found. Copy the directory to ~/FRS-CPP first:"
  echo "     scp -r /path/to/FRS-CPP ubuntu@$JETSON_IP:~/FRS-CPP"
  exit 1
fi

echo "  ✅ Found at: $SRC_DIR"

# ── 4. Write token_manager.cpp (missing from original) ───────────────────────
echo ""
echo "[4/7] Adding missing token_manager.cpp..."

cat > "$SRC_DIR/src/token_manager.cpp" << 'CPPEOF'
// src/token_manager.cpp — stub (token management handled by shell cron)
// The C++ runner simply reads /opt/frs/device_token.txt written by token_manager.sh
// No C++ token fetching needed.
CPPEOF

echo "  ✅ token_manager.cpp stub created"

# Also fix CMakeLists.txt — replace token_manager.cpp with the stub (it's already listed)
# Verify it compiles by checking the include
if ! grep -q "token_manager.cpp" "$SRC_DIR/CMakeLists.txt"; then
  echo "  ℹ  token_manager.cpp not in CMakeLists, adding..."
  sed -i 's/src\/runner.cpp/src\/runner.cpp\n    src\/token_manager.cpp/' "$SRC_DIR/CMakeLists.txt"
fi

# ── 5. Write production config.json ──────────────────────────────────────────
echo ""
echo "[5/7] Writing production configs..."

sudo mkdir -p /opt/frs /opt/frs-models/trt /opt/frs-models/onnx /opt/frs-models/logs
sudo chown -R $USER:$USER /opt/frs /opt/frs-models

# Use DLA0 engine for face detection (more power-efficient)
# Use GPU engine for ArcFace (DLA doesn't support PReLU)
DLA_ENGINE="/opt/frs-models/trt/yolov8n-face-dla0-fp16.engine"
GPU_ENGINE="/opt/frs-models/trt/yolov8n-face-fp16.engine"
DET_ENGINE="$GPU_ENGINE"  # fallback to GPU if DLA not available

if [ -f "$DLA_ENGINE" ]; then
  DET_ENGINE="$DLA_ENGINE"
  echo "  ✅ Using DLA0 detection engine"
else
  echo "  ℹ  DLA engine not found, using GPU engine"
fi

cat > /opt/frs/config.json << JSONEOF
{
  "models": {
    "face_detection": {
      "engine_path": "$DET_ENGINE",
      "onnx_path":   "/opt/frs-models/onnx/yolov8n-face.onnx"
    },
    "face_embedding": {
      "engine_path": "/opt/frs-models/trt/arcface-r50-fp16.engine",
      "onnx_path":   "/opt/frs-models/onnx/arcface-r50.onnx"
    }
  },
  "backend": {
    "url":        "http://$BACKEND_IP:8080",
    "token_path": "/opt/frs/device_token.txt"
  },
  "conf_threshold":    0.45,
  "nms_threshold":     0.45,
  "match_threshold":   0.55,
  "cooldown_seconds":  10,
  "inference_threads": 2,
  "queue_depth":       64
}
JSONEOF

# Write cameras.json with real password
cat > /opt/frs/cameras.json << JSONEOF
{
  "cameras": [
    {
      "id":          "entrance-cam-01",
      "device_code": "entrance-cam-01",
      "enabled":     true,
      "rtsp_url":    "rtsp://admin:$CAM_PASS@$CAM_IP:554/Streaming/Channels/102",
      "rtsp_main":   "rtsp://admin:$CAM_PASS@$CAM_IP:554/Streaming/Channels/101",
      "snapshot_url":"http://admin:$CAM_PASS@$CAM_IP:80/ISAPI/Streaming/channels/101/picture",
      "fps_target":  5,
      "width":       1280,
      "height":      720,
      "hw_decode":   true
    }
  ]
}
JSONEOF

chmod 600 /opt/frs/cameras.json  # protect password
echo "  ✅ /opt/frs/config.json written"
echo "  ✅ /opt/frs/cameras.json written (password protected)"

# ── 6. Build C++ runner ───────────────────────────────────────────────────────
echo ""
echo "[6/7] Building frs_runner..."
BUILD_DIR="$SRC_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect TRT include path
TRT_INC=$(dirname "$TRT_HEADER")
echo "  TRT include: $TRT_INC"

cmake "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja \
  -DTRT_ROOT="$(dirname $TRT_INC)" \
  -DCMAKE_CXX_FLAGS="-I$TRT_INC" \
  2>&1 | grep -E "Found|NOT|ERROR|error" | head -15

echo "  Building with $(nproc) cores..."
ninja -j$(nproc) 2>&1 | tail -5

if [ ! -f "$BUILD_DIR/frs_runner" ]; then
  echo "  ❌ Build failed — check output above"
  exit 1
fi

sudo cp "$BUILD_DIR/frs_runner" /usr/local/bin/frs_runner
sudo chmod +x /usr/local/bin/frs_runner
echo "  ✅ frs_runner installed at /usr/local/bin/frs_runner"

# ── 7. Install systemd service ────────────────────────────────────────────────
echo ""
echo "[7/7] Installing systemd service..."

sudo tee /etc/systemd/system/frs-runner.service > /dev/null << 'SERVICE'
[Unit]
Description=FRS2 TensorRT Face Recognition Runner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/frs
ExecStartPre=/opt/frs/token_manager.sh
ExecStart=/usr/local/bin/frs_runner \
  --config /opt/frs/config.json \
  --cameras /opt/frs/cameras.json \
  --port 5000
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=frs-runner
Environment=CUDA_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
echo "  ✅ frs-runner.service installed"

echo ""
echo "=================================================="
echo " ✅ BUILD COMPLETE"
echo "=================================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Get auth token:    bash /opt/frs/token_manager.sh"
echo "  2. Test camera:       bash ~/J02_test_connections.sh"
echo "  3. Start runner:      sudo systemctl start frs-runner"
echo "  4. Watch logs:        sudo journalctl -u frs-runner -f"
echo "  5. Check health:      curl http://localhost:5000/health | python3 -m json.tool"
echo ""
