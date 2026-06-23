#!/usr/bin/env bash
# =============================================================================
# setup_ascend_llamacpp.sh
# Build llama.cpp with the CANN backend for Ascend 910B/910B3 NPUs.
# =============================================================================

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$LLAMA_CPP_DIR/build}"

load_ascend_env() {
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
  else
    echo "[WARN] Could not find Ascend set_env.sh under /usr/local/Ascend."
  fi
}

echo "=== Building llama.cpp with CANN backend ==="
load_ascend_env

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake is required."
  exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  echo "[INFO] Cloning llama.cpp into $LLAMA_CPP_DIR"
  git clone --depth=1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
  echo "[INFO] Updating existing llama.cpp checkout"
  git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
  -DGGML_CANN=on \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" --target llama-server llama-quantize

echo ""
echo "=== Build complete ==="
echo "llama-server : $BUILD_DIR/bin/llama-server"
echo "llama-quantize: $BUILD_DIR/bin/llama-quantize"
echo ""
echo "Next:"
echo "  MODEL_PATH=/path/to/model-q4_0.gguf bash start_server.sh"
