#!/bin/bash
# =============================================================================
# start_server.sh
# Launches the Ollama daemon with multi-NPU config, then starts the FastAPI
# wrapper (api_server.py).
#
# NPU layout from npu-smi (8x Huawei Ascend 910B3, 64 GB HBM each):
#   NPU 0-7 — all available
#
# Prerequisites:
#   • Ollama built with CANN support (run build_ollama_npu.sh first)
#   • Huawei CANN Toolkit installed (default: /usr/local/Ascend/ascend-toolkit)
#
# To restrict which NPUs are used, edit NPUS below.
# =============================================================================

set -e

# ---------- Config — edit as needed ----------
NPUS="0,1,2,3,4,5,6,7"      # Ascend NPU IDs to expose to Ollama
OLLAMA_PORT=11434            # Ollama daemon port
API_PORT=8000                # FastAPI server port
MODEL_NAME="llama3"          # Must match what you used in setup_model.sh
LOG_DIR="./logs"

# Path to your CANN toolkit installation (adjust if installed elsewhere)
CANN_TOOLKIT_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
# ---------------------------------------------

mkdir -p "$LOG_DIR"

echo "=== Starting Ollama + FastAPI LLM Server (Huawei Ascend NPU) ==="
echo "NPUs            : $NPUS"
echo "Ollama port     : $OLLAMA_PORT"
echo "API port        : $API_PORT"
echo "Model           : $MODEL_NAME"

# ---------- 0. Source CANN environment ----------
if [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.bash" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.bash"
  echo "[INFO] CANN environment sourced from $CANN_TOOLKIT_ROOT"
elif [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.sh" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.sh"
  echo "[INFO] CANN environment sourced from $CANN_TOOLKIT_ROOT"
else
  echo "[WARN] CANN setenv script not found at $CANN_TOOLKIT_ROOT/bin/setenv.bash"
  echo "       Attempting to set library paths manually..."
  export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/acllib/lib64:${LD_LIBRARY_PATH:-}"
fi

# Ensure CANN runtime and ACL libraries are on the library path
export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/acllib/lib64:${LD_LIBRARY_PATH:-}"

# ---------- 1. Kill any previous Ollama instance ----------
pkill -f "ollama serve" 2>/dev/null && echo "[INFO] Stopped previous Ollama process." || true
sleep 1

# ---------- 2. Launch Ollama daemon ----------
echo "[INFO] Starting Ollama daemon..."

# ASCEND_RT_VISIBLE_DEVICES is the Ascend equivalent of CUDA_VISIBLE_DEVICES.
# It accepts a comma-separated list of NPU indices.
export ASCEND_RT_VISIBLE_DEVICES="$NPUS"
export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
export OLLAMA_NUM_PARALLEL=4          # concurrent requests
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1       # flash attention (supported on 910B3)
export OLLAMA_NUM_GPU=99              # offload ALL layers to NPU; set 0 for CPU-only
export OLLAMA_GPU_OVERHEAD=0          # no reserved memory overhead

# Verify NPUs are visible before starting
echo "[INFO] Checking NPU visibility..."
if command -v npu-smi &>/dev/null; then
  npu-smi info 2>/dev/null | grep -E "^\| [0-9]" | \
    awk -v npus="$NPUS" '
      BEGIN { split(npus, a, ","); for (i in a) want[a[i]]=1 }
      { id=$2; if (id in want) print "[NPU "id"] "$4" — Health:"$6 }
    ' || echo "[INFO] npu-smi output parsed (check manually if blank)"
else
  echo "[WARN] npu-smi not found in PATH; skipping NPU visibility check."
fi

nohup ollama serve \
  > "$LOG_DIR/ollama.log" 2>&1 &

OLLAMA_PID=$!
echo "[INFO] Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo -n "[INFO] Waiting for Ollama to be ready "
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$OLLAMA_PORT/api/tags" &>/dev/null; then
    echo " ready!"
    break
  fi
  echo -n "."
  sleep 1
done

# ---------- 3. Pre-load model into NPU HBM ----------
echo "[INFO] Pre-loading model '$MODEL_NAME' into NPU HBM..."
ollama run "$MODEL_NAME" "Hello" > /dev/null 2>&1 || true
echo "[INFO] Model loaded."

# ---------- 4. Launch FastAPI server ----------
echo "[INFO] Starting FastAPI API server on port $API_PORT..."
ASCEND_RT_VISIBLE_DEVICES="$NPUS" \
  MODEL_NAME="$MODEL_NAME" \
  OLLAMA_BASE_URL="http://localhost:$OLLAMA_PORT" \
  python3 api_server.py \
  > "$LOG_DIR/api_server.log" 2>&1 &

API_PID=$!
echo "[INFO] FastAPI PID: $API_PID"

echo ""
echo "=== All services running ==="
echo "  Ollama API  : http://localhost:$OLLAMA_PORT"
echo "  FastAPI     : http://localhost:$API_PORT"
echo "  Docs        : http://localhost:$API_PORT/docs"
echo "  NPU status  : http://localhost:$API_PORT/npu/status"
echo "  Logs        : $LOG_DIR/"
echo ""
echo "To stop everything: pkill -f 'ollama serve'; pkill -f 'api_server.py'"
