#!/bin/bash
# =============================================================================
# build_ollama_npu.sh
# Builds Ollama from source with Huawei Ascend CANN backend support.
#
# Why this is needed
# ------------------
# Official Ollama pre-built binaries only support NVIDIA CUDA, AMD ROCm, and
# Apple Metal. To use Huawei Ascend NPUs you must build Ollama yourself so that
# its embedded llama.cpp is compiled with -DGGML_CANN=ON.
#
# Hardware: Huawei Ascend 910B3 (64 GB HBM each)
# CANN version tested: 25.5.0 (npu-smi version shown in npu-smi info)
#
# Prerequisites (install before running this script)
# --------------------------------------------------
#   1. Huawei CANN Toolkit >= 7.0
#      Download: https://www.hiascend.com/software/cann/community
#      Default install path: /usr/local/Ascend/ascend-toolkit/latest
#
#   2. Go >= 1.22
#      sudo apt install golang-go   OR   https://go.dev/dl/
#
#   3. cmake >= 3.24, gcc/g++ >= 10, git, curl
#      sudo apt install cmake build-essential git curl
#
# Usage
# -----
#   bash build_ollama_npu.sh
#   # Then add the install dir to PATH and run start_server.sh
# =============================================================================

set -euo pipefail

# ---------- Config — edit as needed ----------
OLLAMA_REPO="https://github.com/ollama/ollama.git"
OLLAMA_VERSION="main"                          # or pin to a tag, e.g. "v0.3.12"
INSTALL_DIR="$HOME/.local/bin"                 # where to put the built `ollama` binary
BUILD_DIR="$(pwd)/ollama-cann-build"           # temporary build directory

# CANN toolkit root — adjust if you installed elsewhere
CANN_TOOLKIT_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
# ---------------------------------------------

echo "========================================================"
echo " Ollama NPU Build — Huawei Ascend 910B3 (CANN backend)"
echo "========================================================"
echo "Ollama branch/tag : $OLLAMA_VERSION"
echo "Install dir        : $INSTALL_DIR"
echo "Build dir          : $BUILD_DIR"
echo "CANN root          : $CANN_TOOLKIT_ROOT"
echo ""

# ---------- 0. Sanity checks ----------
fail() { echo "[ERROR] $*" >&2; exit 1; }

command -v go    &>/dev/null || fail "Go not found. Install: sudo apt install golang-go"
command -v cmake &>/dev/null || fail "cmake not found. Install: sudo apt install cmake"
command -v git   &>/dev/null || fail "git not found. Install: sudo apt install git"
command -v npu-smi &>/dev/null || fail "npu-smi not found. Is the CANN toolkit installed?"

GO_VER=$(go version | awk '{print $3}' | sed 's/go//')
echo "[OK] Go $GO_VER"
echo "[OK] $(cmake --version | head -1)"
echo "[OK] $(npu-smi --version 2>/dev/null | head -1 || echo 'npu-smi present')"

if [ ! -d "$CANN_TOOLKIT_ROOT" ]; then
  fail "CANN toolkit not found at $CANN_TOOLKIT_ROOT. Edit CANN_TOOLKIT_ROOT in this script."
fi
echo "[OK] CANN toolkit found at $CANN_TOOLKIT_ROOT"

# ---------- 1. Source CANN environment ----------
echo ""
echo "[1/5] Sourcing CANN environment..."
if [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.bash" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.bash"
elif [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.sh" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.sh"
else
  echo "[WARN] setenv script not found; setting library paths manually."
  export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/acllib/lib64:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${CANN_TOOLKIT_ROOT}/lib64:${CANN_TOOLKIT_ROOT}/acllib/lib64:${LD_LIBRARY_PATH:-}"
echo "[OK] CANN env ready"

# ---------- 2. Clone Ollama ----------
echo ""
echo "[2/5] Cloning Ollama ($OLLAMA_VERSION)..."
if [ -d "$BUILD_DIR" ]; then
  echo "      Build dir already exists — pulling latest..."
  git -C "$BUILD_DIR" fetch --quiet origin
  git -C "$BUILD_DIR" checkout "$OLLAMA_VERSION" --quiet
  git -C "$BUILD_DIR" pull --quiet origin "$OLLAMA_VERSION" 2>/dev/null || true
else
  git clone --branch "$OLLAMA_VERSION" --depth=1 "$OLLAMA_REPO" "$BUILD_DIR"
fi
echo "[OK] Ollama source at $BUILD_DIR"

# ---------- 3. Build llama.cpp with CANN backend ----------
echo ""
echo "[3/5] Building llama.cpp with CANN backend (-DGGML_CANN=ON)..."

LLAMA_BUILD_DIR="$BUILD_DIR/llm/build"
mkdir -p "$LLAMA_BUILD_DIR"

LLAMA_SRC=""
for candidate in \
    "$BUILD_DIR/llm/llama.cpp" \
    "$BUILD_DIR/llm/llama" \
    "$BUILD_DIR/llama.cpp"; do
  if [ -d "$candidate" ]; then
    LLAMA_SRC="$candidate"
    break
  fi
done

if [ -z "$LLAMA_SRC" ]; then
  echo "      Running go generate to fetch llama.cpp..."
  cd "$BUILD_DIR"
  go generate ./llm/... 2>&1 | tail -5 || true
  cd - > /dev/null
  for candidate in \
      "$BUILD_DIR/llm/llama.cpp" \
      "$BUILD_DIR/llm/llama"; do
    if [ -d "$candidate" ]; then
      LLAMA_SRC="$candidate"
      break
    fi
  done
fi

[ -n "$LLAMA_SRC" ] || fail "Could not locate llama.cpp source inside $BUILD_DIR"
echo "[OK] llama.cpp source: $LLAMA_SRC"

cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD_DIR" \
    -DGGML_CANN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX="$LLAMA_BUILD_DIR/install"

cmake --build "$LLAMA_BUILD_DIR" \
    --config Release \
    -j"$(nproc)" \
    --target llama-server ggml

echo "[OK] llama.cpp built with CANN support"

# ---------- 4. Build Ollama Go binary ----------
echo ""
echo "[4/5] Building Ollama Go binary..."
cd "$BUILD_DIR"

export OLLAMA_SKIP_CUDA_GENERATE=1
export OLLAMA_SKIP_ROCM_GENERATE=1
export CGO_CFLAGS="-I${LLAMA_SRC}/include -I${LLAMA_SRC}/ggml/include"
export CGO_LDFLAGS="-L${LLAMA_BUILD_DIR} -lggml -lllama -Wl,-rpath,${LLAMA_BUILD_DIR}"

go build -o ollama . 2>&1 | tail -10

cd - > /dev/null
echo "[OK] Ollama binary built: $BUILD_DIR/ollama"

# ---------- 5. Install ----------
echo ""
echo "[5/5] Installing to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
cp "$BUILD_DIR/ollama" "$INSTALL_DIR/ollama"
chmod +x "$INSTALL_DIR/ollama"

echo ""
echo "========================================================"
echo " Build complete!"
echo "========================================================"
echo ""
echo "Binary location : $INSTALL_DIR/ollama"
echo ""
echo "Next steps:"
echo "  1. Add $INSTALL_DIR to your PATH (if not already):"
echo "       export PATH=\"$INSTALL_DIR:\$PATH\""
echo ""
echo "  2. Register your model (one time):"
echo "       bash setup_model.sh"
echo ""
echo "  3. Start the server:"
echo "       bash start_server.sh"
echo ""
echo "  4. Check NPU status:"
echo "       curl http://localhost:8000/npu/status | python3 -m json.tool"
echo ""
echo "Troubleshooting:"
echo "  • If ollama serve exits immediately, check logs/ollama.log"
echo "  • Confirm CANN libs are on LD_LIBRARY_PATH (source CANN setenv.bash first)"
echo "  • Verify NPUs are visible: npu-smi info"
echo "  • Ollama CANN issues tracker: https://github.com/ollama/ollama/issues"
