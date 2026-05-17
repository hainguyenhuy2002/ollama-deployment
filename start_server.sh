#!/bin/bash
# =============================================================================
# start_server.sh
# Launches the Ollama daemon with multi-GPU config, then starts the FastAPI
# wrapper (api_server.py).
#
# GPU layout from nvidia-smi (7x A100-SXM4-40GB):
#   GPU 0 — currently busy (90% util) → excluded by default
#   GPU 1-6 — idle, ~40 GB each → used for the model
#
# To include GPU 0, change CUDA_VISIBLE_DEVICES below.
# =============================================================================

set -e

# ---------- Config — edit as needed ----------
GPUS="1,2,3,4,5,6"          # Which GPUs to expose to Ollama (skip busy GPU 0)
OLLAMA_PORT=11434            # Ollama daemon port
API_PORT=8000                # FastAPI server port
MODEL_NAME="llama3.3:70b"   # Must match what you used in setup_model.sh
LOG_DIR="./logs"
# ---------------------------------------------

mkdir -p "$LOG_DIR"

echo "=== Starting Ollama + FastAPI LLM Server ==="
echo "GPUs            : $GPUS"
echo "Ollama port     : $OLLAMA_PORT"
echo "API port        : $API_PORT"
echo "Model           : $MODEL_NAME"

# ---------- 1. Kill any previous Ollama instance ----------
pkill -f "ollama serve" 2>/dev/null && echo "[INFO] Stopped previous Ollama process." || true
sleep 1

# ---------- 2. Launch Ollama daemon ----------
echo "[INFO] Starting Ollama daemon..."

export CUDA_VISIBLE_DEVICES="$GPUS"
export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
export OLLAMA_NUM_PARALLEL=4          # concurrent requests
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1       # enable flash attention (A100 supports it)

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

# ---------- 3. Pre-load model into GPU VRAM ----------
echo "[INFO] Pre-loading model '$MODEL_NAME' into VRAM..."
ollama run "$MODEL_NAME" "Hello" > /dev/null 2>&1 || true
echo "[INFO] Model loaded."

# ---------- 4. Launch FastAPI server ----------
echo "[INFO] Starting FastAPI API server on port $API_PORT..."
CUDA_VISIBLE_DEVICES="$GPUS" \
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
echo "  Logs        : $LOG_DIR/"
echo ""
echo "To stop everything: pkill -f 'ollama serve'; pkill -f 'api_server.py'"
