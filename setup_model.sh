#!/bin/bash
# =============================================================================
# setup_model.sh
# One-time script to pull a model from the Ollama hub.
# Run this ONCE before starting the API server.
#
# For hub models (llama3, llama3.1, mistral, etc.) no local files are needed —
# Ollama downloads and caches the model automatically.
#
# NPU note: ensure you are running the CANN-enabled Ollama binary built by
# build_ollama_npu.sh. The standard pre-built Ollama binary does NOT support
# Huawei Ascend NPUs and will fall back to CPU.
# =============================================================================

set -e

# ---------- Config — edit as needed ----------
MODEL_NAME="llama3"        # Ollama hub model tag (ollama.com/library)
OLLAMA_PORT=11434
# ---------------------------------------------

# ---------- 1. Locate the CANN-enabled Ollama binary ----------
# build_ollama_npu.sh installs to ~/.local/bin — make sure it's on PATH.
export PATH="$HOME/.local/bin:$HOME/go/bin:$PATH"

if ! command -v ollama &>/dev/null; then
  echo "[ERROR] ollama not found. Build it first:"
  echo "  bash build_ollama_npu.sh"
  echo "Then make sure ~/.local/bin is on your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
fi

echo "[OK] ollama found: $(ollama --version)  ($(command -v ollama))"

# ---------- 2. Start Ollama daemon if not already running ----------
if ! curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; then
  echo "[INFO] Ollama daemon not running — starting temporarily..."
  export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
  nohup ollama serve > /tmp/ollama_setup.log 2>&1 &
  OLLAMA_TMP_PID=$!
  echo "[INFO] Waiting for Ollama to become ready..."
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; then
      echo "[OK] Ollama ready."
      break
    fi
    sleep 1
  done
  STOP_OLLAMA=true
else
  echo "[OK] Ollama daemon already running."
  STOP_OLLAMA=false
fi

# ---------- 3. Pull model from Ollama hub ----------
echo ""
echo "=== Pulling '$MODEL_NAME' from Ollama hub ==="
echo "    (This downloads the model weights — may take several minutes)"
ollama pull "$MODEL_NAME"

echo ""
echo "[OK] Model '$MODEL_NAME' ready."
ollama list

# ---------- 4. Stop temporary daemon if we started it ----------
if [ "${STOP_OLLAMA:-false}" = "true" ]; then
  kill "$OLLAMA_TMP_PID" 2>/dev/null || true
  echo "[INFO] Temporary Ollama daemon stopped."
fi

echo ""
echo "=== Setup complete! ==="
echo "Start the full server with:"
echo "  bash start_server.sh"
