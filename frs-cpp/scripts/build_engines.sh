#!/bin/bash
# build_engines.sh — Convert ONNX models to TensorRT FP16 engines on Jetson Orin
# Must run ON the Jetson — engines are not portable between GPUs
# Usage: bash build_engines.sh

set -e

ONNX_DIR="/opt/frs-models/onnx"
TRT_DIR="/opt/frs-models/trt"
LOG_DIR="/opt/frs-models/logs"

echo "=================================================="
echo " FRS2 — TensorRT Engine Builder"
echo " Jetson Orin | $(date)"
echo "=================================================="
echo ""

mkdir -p "$ONNX_DIR" "$TRT_DIR" "$LOG_DIR"

# ── Check trtexec is available ────────────────────────────────────────────────
if ! command -v trtexec &>/dev/null; then
  # Try common Jetson locations
  TRTEXEC=""
  for p in /usr/src/tensorrt/bin/trtexec \
            /usr/local/tensorrt/bin/trtexec \
            /usr/bin/trtexec; do
    [ -f "$p" ] && TRTEXEC="$p" && break
  done
  if [ -z "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found. Install TensorRT:"
    echo "  sudo apt install tensorrt"
    exit 1
  fi
  alias trtexec="$TRTEXEC"
  TRTEXEC_CMD="$TRTEXEC"
else
  TRTEXEC_CMD="trtexec"
fi

echo "Using trtexec: $($TRTEXEC_CMD --version 2>&1 | head -1)"
echo ""

# ── Get GPU info ──────────────────────────────────────────────────────────────
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── Helper function ────────────────────────────────────────────────────────────
build_engine() {
  local name="$1"
  local onnx="$2"
  local engine="$3"
  local extra_flags="${4:-}"

  echo "Building: $name"
  echo "  ONNX:   $onnx"
  echo "  Engine: $engine"

  if [ ! -f "$onnx" ]; then
    echo "  ERROR: ONNX file not found: $onnx"
    echo "  Download it first — see README.md"
    return 1
  fi

  if [ -f "$engine" ]; then
    echo "  Engine already exists — skipping (delete to rebuild)"
    return 0
  fi

  echo "  Building FP16 engine (this takes 5-15 minutes)..."
  $TRTEXEC_CMD \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    --fp16 \
    --workspace=4096 \
    --buildOnly \
    $extra_flags \
    2>&1 | tee "$LOG_DIR/${name}.build.log" | grep -E "Building|Timing|Engine|Error|Warning|FP16"

  if [ -f "$engine" ]; then
    SIZE=$(du -sh "$engine" | cut -f1)
    echo "  ✅ Engine built: $engine ($SIZE)"
  else
    echo "  ❌ Engine build failed — check $LOG_DIR/${name}.build.log"
    return 1
  fi
  echo ""
}

# ── Build YOLOv8n-face ────────────────────────────────────────────────────────
# Input:  1×3×640×640
# Output: 1×5×8400  (cx,cy,w,h,conf)
# Source: https://github.com/derronqi/yolov8-face (export to ONNX)
build_engine \
  "yolov8n-face" \
  "$ONNX_DIR/yolov8n-face.onnx" \
  "$TRT_DIR/yolov8n-face-fp16.engine" \
  "--shapes=input:1x3x640x640"

# ── Build ArcFace R50 ─────────────────────────────────────────────────────────
# Input:  1×3×112×112
# Output: 1×512
# Source: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface
build_engine \
  "arcface-r50" \
  "$ONNX_DIR/arcface-r50.onnx" \
  "$TRT_DIR/arcface-r50-fp16.engine" \
  "--shapes=input.1:1x3x112x112"

# ── Benchmark engines ─────────────────────────────────────────────────────────
echo "=================================================="
echo " Benchmarking built engines"
echo "=================================================="
echo ""

if [ -f "$TRT_DIR/yolov8n-face-fp16.engine" ]; then
  echo "YOLOv8n-face FP16 (100 iterations):"
  $TRTEXEC_CMD \
    --loadEngine="$TRT_DIR/yolov8n-face-fp16.engine" \
    --iterations=100 \
    2>&1 | grep -E "mean|median|percentile|Throughput"
  echo ""
fi

if [ -f "$TRT_DIR/arcface-r50-fp16.engine" ]; then
  echo "ArcFace R50 FP16 (100 iterations):"
  $TRTEXEC_CMD \
    --loadEngine="$TRT_DIR/arcface-r50-fp16.engine" \
    --iterations=100 \
    2>&1 | grep -E "mean|median|percentile|Throughput"
  echo ""
fi

echo "=================================================="
echo " Engine build complete"
echo "=================================================="
echo ""
echo "Engines at: $TRT_DIR/"
ls -lh "$TRT_DIR/" 2>/dev/null
echo ""
echo "Next: bash build_runner.sh"
