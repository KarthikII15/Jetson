#!/bin/bash
# build_runner.sh — Build frs_runner C++ binary on Jetson Orin
# Run this ON the Jetson after copying the frs-cpp directory
# Usage: bash build_runner.sh

set -e

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$SRC_DIR/build"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/opt/frs"
MODELS_DIR="/opt/frs-models"

echo "=================================================="
echo " FRS2 — C++ Runner Build"
echo " Source: $SRC_DIR"
echo " $(date)"
echo "=================================================="
echo ""

# ── Check dependencies ────────────────────────────────────────────────────────
echo "[1/6] Checking dependencies..."

check_dep() {
  if ! dpkg -l "$1" &>/dev/null && ! command -v "$2" &>/dev/null; then
    echo "  Installing $1..."
    sudo apt-get install -y "$1" -q
  else
    echo "  ✅ $1"
  fi
}

sudo apt-get update -q

# Core build tools
check_dep "cmake"        "cmake"
check_dep "g++"          "g++"
check_dep "ninja-build"  "ninja"

# TensorRT (usually pre-installed on Jetson)
if pkg-config --exists tensorrt 2>/dev/null || [ -f /usr/include/NvInfer.h ]; then
  echo "  ✅ TensorRT"
else
  echo "  Installing TensorRT..."
  sudo apt-get install -y tensorrt libnvinfer-dev libnvonnxparsers-dev -q
fi

# OpenCV with GStreamer support
if pkg-config --exists opencv4 2>/dev/null; then
  GST_SUPPORT=$(pkg-config --libs opencv4 | grep -c gstreamer || true)
  if [ "$GST_SUPPORT" -gt 0 ]; then
    echo "  ✅ OpenCV (with GStreamer)"
  else
    echo "  ⚠  OpenCV found but GStreamer support may be missing"
    echo "     For full HW acceleration, build OpenCV from source with -DWITH_GSTREAMER=ON"
  fi
else
  echo "  Installing OpenCV..."
  sudo apt-get install -y libopencv-dev -q
fi

# libcurl
check_dep "libcurl4-openssl-dev" "curl-config"

# nlohmann-json
check_dep "nlohmann-json3-dev" ""

# spdlog
check_dep "libspdlog-dev" ""

echo ""

# ── Configure and build ───────────────────────────────────────────────────────
echo "[2/6] Configuring CMake..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tail -10

echo ""
echo "[3/6] Building ($(nproc) cores)..."
ninja -j$(nproc) 2>&1

echo ""
echo "[4/6] Installing binary..."
sudo cp "$BUILD_DIR/frs_runner" "$INSTALL_DIR/frs_runner"
sudo chmod +x "$INSTALL_DIR/frs_runner"
echo "  ✅ Installed: $INSTALL_DIR/frs_runner"

# ── Setup config directories ──────────────────────────────────────────────────
echo ""
echo "[5/6] Setting up config directories..."
sudo mkdir -p "$CONFIG_DIR" "$MODELS_DIR/onnx" "$MODELS_DIR/trt" "$MODELS_DIR/logs"
sudo chown -R ubuntu:ubuntu "$CONFIG_DIR" "$MODELS_DIR"

# Copy config files if not present
if [ ! -f "$CONFIG_DIR/config.json" ]; then
  cp "$SRC_DIR/config/config.json"  "$CONFIG_DIR/config.json"
  echo "  ✅ Copied config.json to $CONFIG_DIR"
  echo "  ⚠  Edit $CONFIG_DIR/config.json — set backend URL and model paths"
fi

if [ ! -f "$CONFIG_DIR/cameras.json" ]; then
  cp "$SRC_DIR/config/cameras.json" "$CONFIG_DIR/cameras.json"
  echo "  ✅ Copied cameras.json to $CONFIG_DIR"
  echo "  ⚠  Edit $CONFIG_DIR/cameras.json — set CAMERA_PASSWORD"
fi

cp "$SRC_DIR/scripts/token_manager.sh" "$CONFIG_DIR/token_manager.sh"
chmod +x "$CONFIG_DIR/token_manager.sh"

# ── Install systemd service ───────────────────────────────────────────────────
echo ""
echo "[6/6] Installing systemd service..."

sudo tee /etc/systemd/system/frs-runner.service > /dev/null << 'SERVICE'
[Unit]
Description=FRS2 Face Recognition Attendance Runner
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

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
echo "  ✅ Service installed: frs-runner.service"

# ── Setup token cron ──────────────────────────────────────────────────────────
(crontab -l 2>/dev/null | grep -v token_manager; \
 echo "*/25 * * * * /opt/frs/token_manager.sh >> /var/log/frs_token.log 2>&1") \
 | crontab -
echo "  ✅ Token refresh cron set (every 25 min)"

echo ""
echo "=================================================="
echo " ✅ Build complete"
echo "=================================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Set camera password:"
echo "   nano /opt/frs/cameras.json"
echo "   # Change CAMERA_PASSWORD to real password"
echo ""
echo "2. Build TensorRT engines (if not done):"
echo "   bash $SRC_DIR/scripts/build_engines.sh"
echo ""
echo "   Or download pre-built engines if available:"
echo "   # Place .engine files in /opt/frs-models/trt/"
echo ""
echo "3. Get initial auth token:"
echo "   bash /opt/frs/token_manager.sh"
echo ""
echo "4. Test camera connection:"
echo "   ffprobe -i rtsp://admin:PASS@172.18.3.201:554/Streaming/Channels/102 -t 3"
echo ""
echo "5. Start the runner:"
echo "   sudo systemctl start frs-runner"
echo "   sudo systemctl enable frs-runner"
echo "   sudo journalctl -u frs-runner -f"
echo ""
echo "6. Check health:"
echo "   curl http://172.18.3.202:5000/health | python3 -m json.tool"
echo ""
