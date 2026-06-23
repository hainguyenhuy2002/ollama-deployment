#!/usr/bin/env bash
# =============================================================================
# start_server.sh
# Launch a local LLM runtime and the FastAPI wrapper.
#
# Backends:
#   LLM_BACKEND=ollama    - standard Ollama daemon
#   LLM_BACKEND=llama_cpp - llama.cpp server, including Ascend/CANN builds
#
# Ascend 910B3 example:
#   LLM_BACKEND=llama_cpp ACCELERATOR=ascend MODEL_PATH=/path/model-q4_0.gguf \
#     bash start_server.sh
# =============================================================================

set -euo pipefail

LLM_BACKEND="${LLM_BACKEND:-llama_cpp}"
ACCELERATOR="${ACCELERATOR:-ascend}"
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8080}"
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"
MODEL_NAME="${MODEL_NAME:-llama3.3-70b-ascend}"
MODEL_PATH="${MODEL_PATH:-/villa/rhh25/models/llama3.3-70b/model-q4_0.gguf}"
LOG_DIR="${LOG_DIR:-./logs}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$LLAMA_CPP_DIR/build}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_BUILD_DIR/bin/llama-server}"
CTX_SIZE="${CTX_SIZE:-4096}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-$(nproc)}"
READY_TIMEOUT="${READY_TIMEOUT:-}"

mkdir -p "$LOG_DIR"

load_ascend_env() {
  set +u
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
  fi
  set -u

  export ASCEND_VISIBLE_DEVICES
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"
}

stop_existing() {
  pkill -f "ollama serve" 2>/dev/null && echo "[INFO] Stopped previous Ollama process." || true
  pkill -f "llama-server" 2>/dev/null && echo "[INFO] Stopped previous llama-server process." || true
  pkill -f "api_server.py" 2>/dev/null && echo "[INFO] Stopped previous FastAPI process." || true
  sleep 1
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local timeout="${3:-120}"
  echo -n "[INFO] Waiting for $name "
  for _ in $(seq 1 "$timeout"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo " ready!"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo ""
  echo "[ERROR] $name did not become ready at $url after ${timeout}s"
  return 1
}

start_ollama() {
  export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

  if [ "$ACCELERATOR" = "cuda" ]; then
    export CUDA_VISIBLE_DEVICES
    export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
  fi

  echo "[INFO] Starting Ollama daemon on :$OLLAMA_PORT"
  nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
  echo "[INFO] Ollama PID: $!"
  wait_for_url "http://localhost:$OLLAMA_PORT/api/tags" "Ollama" "${READY_TIMEOUT:-120}"

  echo "[INFO] Pre-loading Ollama model '$MODEL_NAME'"
  ollama run "$MODEL_NAME" "Hello" >/dev/null 2>&1 || true
}

start_llama_cpp() {
  if [ "$ACCELERATOR" = "ascend" ]; then
    load_ascend_env
  fi

  if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "[ERROR] llama-server not found or not executable: $LLAMA_SERVER_BIN"
    echo "        Run: bash setup_ascend_llamacpp.sh"
    exit 1
  fi

  if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
  fi

  echo "[INFO] Starting llama.cpp server on :$LLAMA_CPP_PORT"
  echo "[INFO] Model: $MODEL_PATH"
  echo "[INFO] Ascend devices: ${ASCEND_VISIBLE_DEVICES:-unset}"

  nohup "$LLAMA_SERVER_BIN" \
    --host 0.0.0.0 \
    --port "$LLAMA_CPP_PORT" \
    --model "$MODEL_PATH" \
    --alias "$MODEL_NAME" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --threads "$THREADS" \
    --n-gpu-layers "${N_GPU_LAYERS:--1}" \
    ${LLAMA_SERVER_EXTRA_ARGS:-} \
    > "$LOG_DIR/llama-server.log" 2>&1 &

  echo "[INFO] llama-server PID: $!"
  wait_for_url "http://localhost:$LLAMA_CPP_PORT/health" "llama.cpp server" "${READY_TIMEOUT:-900}"
}

start_api() {
  echo "[INFO] Starting FastAPI wrapper on :$API_PORT"
  LLM_BACKEND="$LLM_BACKEND" \
    ACCELERATOR="$ACCELERATOR" \
    MODEL_NAME="$MODEL_NAME" \
    API_HOST="$API_HOST" \
    API_PORT="$API_PORT" \
    OLLAMA_BASE_URL="http://localhost:$OLLAMA_PORT" \
    LLAMA_CPP_BASE_URL="http://localhost:$LLAMA_CPP_PORT" \
    python3 api_server.py > "$LOG_DIR/api_server.log" 2>&1 &

  echo "[INFO] FastAPI PID: $!"
  wait_for_url "http://localhost:$API_PORT/health" "FastAPI" "${API_READY_TIMEOUT:-120}"
}

echo "=== Starting LLM Server ==="
echo "Backend      : $LLM_BACKEND"
echo "Accelerator  : $ACCELERATOR"
echo "Model name   : $MODEL_NAME"
echo "Model path   : $MODEL_PATH"
echo "API port     : $API_PORT"
echo "Logs         : $LOG_DIR"

stop_existing

case "$LLM_BACKEND" in
  ollama)
    start_ollama
    ;;
  llama_cpp)
    start_llama_cpp
    ;;
  *)
    echo "[ERROR] Unknown LLM_BACKEND: $LLM_BACKEND"
    exit 1
    ;;
esac

start_api

echo ""
echo "=== All services running ==="
echo "  API health        : http://localhost:$API_PORT/health"
echo "  OpenAI chat       : http://localhost:$API_PORT/v1/chat/completions"
echo "  Accelerator status: http://localhost:$API_PORT/accelerator/status"
echo "  Logs              : $LOG_DIR/"
echo ""
echo "To stop everything: pkill -f 'ollama serve'; pkill -f 'llama-server'; pkill -f 'api_server.py'"
