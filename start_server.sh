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
#   • Huawei CANN Toolkit installed by sysadmin (auto-detected from npu-smi or $ASCEND_HOME)
#
# To restrict which NPUs are used, edit NPUS below.
# =============================================================================

set -e

# ---------- Config — edit as needed ----------
# All 8 Ascend 910B3 NPUs (each 64 GB HBM).
# To use a subset, e.g. for smaller models: NPUS="0,1,2,3"
NPUS="0,1,2,3,4,5,6,7"

# NUMA topology (from npu-smi bus IDs — pair NPUs on the same PCIe switch):
#   Group A: NPU 0,1  (bus C1/C2)
#   Group B: NPU 2,3  (bus 81/82)
#   Group C: NPU 4,5  (bus 01/02)
#   Group D: NPU 6,7  (bus 41/42)
# For tensor-parallel inference across all groups, use all 8 (default).
# For lower latency on smaller models, use one NUMA group, e.g. NPUS="0,1"

OLLAMA_PORT=11434            # Ollama daemon port
API_PORT=8000                # FastAPI server port
MODEL_NAME="gpt-oss:20b"          # Ollama hub model tag (must match setup_model.sh)
LOG_DIR="./logs"
HBM_PER_NPU_GB=64           # HBM per Ascend 910B3 in GB
# (CANN_TOOLKIT_ROOT is auto-detected below — no need to set it manually)
# ---------------------------------------------

mkdir -p "$LOG_DIR"

# Make sure the CANN-enabled Ollama binary (from build_ollama_npu.sh) is found first
export PATH="$HOME/.local/bin:$HOME/go/bin:$PATH"

echo "=== Starting Ollama + FastAPI LLM Server (Huawei Ascend NPU) ==="
echo "NPUs            : $NPUS"
echo "Ollama port     : $OLLAMA_PORT"
echo "API port        : $API_PORT"
echo "Model           : $MODEL_NAME"

# ---------- 0. Auto-detect and source CANN environment ----------
# Search order:
#   1. $ASCEND_HOME env var (already set by sysadmin profile)
#   2. Derive from npu-smi binary location (most reliable when npu-smi is on PATH)
#   3. Common shared install locations that don't require root access

find_cann_root() {
  # 1. Already sourced by the login profile?
  if [ -n "${ASCEND_HOME:-}" ] && [ -d "${ASCEND_HOME}" ]; then
    echo "${ASCEND_HOME}"; return 0
  fi
  # 2. Derive from npu-smi binary (npu-smi -> ../../ should be CANN root)
  if command -v npu-smi &>/dev/null; then
    local bin_dir
    bin_dir=$(dirname "$(command -v npu-smi)")
    for candidate in "$(dirname "$bin_dir")" "$(dirname "$(dirname "$bin_dir")")"; do
      if [ -f "$candidate/bin/setenv.bash" ] || [ -f "$candidate/bin/setenv.sh" ]; then
        echo "$candidate"; return 0
      fi
    done
  fi
  # 3. Common non-root / shared install paths
  for candidate in \
      "$HOME/Ascend/ascend-toolkit/latest" \
      "$HOME/ascend-toolkit/latest" \
      "/opt/Ascend/ascend-toolkit/latest" \
      "/opt/ascend/ascend-toolkit/latest"; do
    [ -d "$candidate" ] && echo "$candidate" && return 0
  done
  echo ""; return 1
}

CANN_TOOLKIT_ROOT=$(find_cann_root)

if [ -n "$CANN_TOOLKIT_ROOT" ]; then
  echo "[INFO] CANN toolkit found at: $CANN_TOOLKIT_ROOT"
  if [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.bash" ]; then
    # shellcheck disable=SC1090
    source "$CANN_TOOLKIT_ROOT/bin/setenv.bash"
  elif [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.sh" ]; then
    # shellcheck disable=SC1090
    source "$CANN_TOOLKIT_ROOT/bin/setenv.sh"
  fi
  # Comprehensive LD_LIBRARY_PATH for CANN 25.x on 910B3
  export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/acllib/lib64:${CANN_TOOLKIT_ROOT}/atc/lib64:${CANN_TOOLKIT_ROOT}/fwkacllib/lib64:${LD_LIBRARY_PATH:-}"
  # llama.cpp CANN backend needs ASCEND_TOOLKIT_HOME
  export ASCEND_TOOLKIT_HOME="${CANN_TOOLKIT_ROOT}"
  export ASCEND_HOME="${CANN_TOOLKIT_ROOT}"
  echo "[OK] CANN 25.x environment sourced: $CANN_TOOLKIT_ROOT"
else
  echo "[WARN] CANN toolkit not found. If Ollama fails to use the NPU, set ASCEND_HOME:"
  echo "       export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest"
fi

# ---------- 1. Kill any previous Ollama instance ----------
pkill -f "ollama serve" 2>/dev/null && echo "[INFO] Stopped previous Ollama process." || true
sleep 1

# ---------- 2. Launch Ollama daemon ----------
echo "[INFO] Starting Ollama daemon..."

# ── Ascend device visibility ──────────────────────────────────────────────────
# ASCEND_RT_VISIBLE_DEVICES is the Ascend equivalent of CUDA_VISIBLE_DEVICES.
export ASCEND_RT_VISIBLE_DEVICES="$NPUS"

# ── Multi-card collective communication (HCCL) ────────────────────────────────
# Required for tensor-parallel inference across multiple 910B3 cards.
# Without this, HCCL may refuse to run on non-whitelisted topologies.
export HCCL_WHITELIST_DISABLE=1
# Allow HCCL to use all available network interfaces (avoids binding failures)
export HCCL_IF_BASE_PORT=60000
# Improve HCCL performance on PCIe topology (not NVLink)
export HCCL_INTRA_ROCE_ENABLE=0

# ── Ollama daemon settings ────────────────────────────────────────────────────
export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"

# How many requests to process concurrently.
# 8 NPUs × 64 GB = 512 GB HBM — room for parallel contexts on large models.
# Reduce to 1-2 if a single large model (e.g. 70B) saturates all cards.
export OLLAMA_NUM_PARALLEL=4

# Keep a single large model in HBM at all times; raise if serving multiple
# smaller models simultaneously and total memory allows it.
export OLLAMA_MAX_LOADED_MODELS=1

# Flash attention is supported on Ascend 910B3 — significant throughput gain.
export OLLAMA_FLASH_ATTENTION=1

# Offload ALL transformer layers to NPU HBM (99 = "as many as possible").
# The 910B3 has enough HBM for large models; CPU fallback is unnecessary.
export OLLAMA_NUM_GPU=99

# No memory reserved for OS overhead — all HBM goes to the model.
export OLLAMA_GPU_OVERHEAD=0

# Give large models (e.g. 70B) up to 10 minutes to load into 512 GB HBM.
export OLLAMA_LOAD_TIMEOUT=600

# Keep the model resident in HBM indefinitely (never auto-unload between calls).
# Set to a duration like "5m" if you want automatic eviction.
export OLLAMA_KEEP_ALIVE=-1

# Verify NPUs are visible and all healthy before starting
echo "[INFO] Checking NPU visibility (expecting 8× Ascend 910B3)..."
if command -v npu-smi &>/dev/null; then
  npu-smi info 2>/dev/null | grep -E "^\| [0-9]" | \
    awk -v npus="$NPUS" '
      BEGIN { split(npus, a, ","); for (i in a) want[a[i]]=1 }
      {
        id=$2; health=$6; model=$4
        if (id in want) {
          status = (health == "OK") ? "✓" : "✗ FAULT"
          printf "[NPU %s]  %-8s  Health: %s  %s\n", id, model, health, status
        }
      }
    ' || echo "[INFO] npu-smi output parsed (check manually if blank)"
  # Count healthy NPUs
  HEALTHY=$(npu-smi info 2>/dev/null | grep -E "^\| [0-9]" | awk '$6=="OK"' | wc -l)
  echo "[INFO] Healthy NPUs: $HEALTHY / $(echo $NPUS | tr ',' '\n' | wc -l)"
  if [ "$HEALTHY" -lt "$(echo $NPUS | tr ',' '\n' | wc -l)" ]; then
    echo "[WARN] Some NPUs not healthy — check npu-smi info before proceeding."
  fi
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
