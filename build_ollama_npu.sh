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
# No sudo required
# ----------------
# Go is installed to ~/go if not already present.
# The Ollama binary is installed to ~/.local/bin.
# cmake, gcc, and git must already be available (ask your sysadmin if missing).
#
# Prerequisites
# -------------
#   • Huawei CANN Toolkit >= 7.0  (needs sysadmin to install system-wide)
#     Default path: /usr/local/Ascend/ascend-toolkit/latest
#   • cmake >= 3.24, gcc/g++ >= 10, git, curl  (system packages — no sudo here)
#   • Go >= 1.22  → auto-installed to ~/go by this script if missing
#
# Usage
# -----
#   bash build_ollama_npu.sh
# =============================================================================

set -euo pipefail

# ---------- Config — edit as needed ----------
OLLAMA_REPO="https://github.com/ollama/ollama.git"
OLLAMA_VERSION="main"                          # or pin to a tag, e.g. "v0.3.12"
INSTALL_DIR="$HOME/.local/bin"                 # where to put the built `ollama` binary
BUILD_DIR="$(pwd)/ollama-cann-build"           # temporary build directory
GO_INSTALL_DIR="$HOME/go"                      # Go toolchain install location (no sudo)
GO_VERSION="1.23.4"                            # Go version to download if missing (>= Ollama's go.mod requirement)

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

# ---------- 0. Helpers ----------
fail() { echo "[ERROR] $*" >&2; exit 1; }

# ---------- 0a. Auto-install Go (no sudo) ----------
install_go_local() {
  local arch
  arch=$(uname -m)
  case "$arch" in
    x86_64)  arch="amd64" ;;
    aarch64) arch="arm64" ;;
    *)       fail "Unsupported architecture: $arch. Download Go manually from https://go.dev/dl/" ;;
  esac

  local tarball="go${GO_VERSION}.linux-${arch}.tar.gz"
  local url="https://go.dev/dl/${tarball}"
  local tmp_dir
  tmp_dir=$(mktemp -d)

  echo "[INFO] Downloading Go ${GO_VERSION} (${arch}) from go.dev..."
  if command -v curl &>/dev/null; then
    curl -fsSL "$url" -o "$tmp_dir/$tarball"
  elif command -v wget &>/dev/null; then
    wget -q "$url" -O "$tmp_dir/$tarball"
  else
    fail "Neither curl nor wget found. Cannot download Go automatically."
  fi

  echo "[INFO] Extracting Go to $GO_INSTALL_DIR ..."
  rm -rf "$GO_INSTALL_DIR"
  mkdir -p "$(dirname "$GO_INSTALL_DIR")"
  tar -xzf "$tmp_dir/$tarball" -C "$(dirname "$GO_INSTALL_DIR")"
  # The tarball unpacks to a directory named "go" — rename if needed
  if [ -d "$(dirname "$GO_INSTALL_DIR")/go" ] && [ "$(dirname "$GO_INSTALL_DIR")/go" != "$GO_INSTALL_DIR" ]; then
    mv "$(dirname "$GO_INSTALL_DIR")/go" "$GO_INSTALL_DIR"
  fi
  rm -rf "$tmp_dir"

  export PATH="$GO_INSTALL_DIR/bin:$PATH"
  echo "[OK] Go installed to $GO_INSTALL_DIR"
  echo "     Add to your shell permanently:"
  echo "       echo 'export PATH=\"$GO_INSTALL_DIR/bin:\$PATH\"' >> ~/.bashrc"
}

# Check Go; auto-install if missing or too old
if ! command -v go &>/dev/null; then
  echo "[WARN] Go not found — installing locally (no sudo required)..."
  install_go_local
elif ! go version | awk '{split($3,v,"go"); split(v[2],n,"."); if(n[1]*100+n[2] < 122) exit 1}' 2>/dev/null; then
  echo "[WARN] Go version is too old (need >= 1.22) — installing newer version locally..."
  install_go_local
fi

# ---------- 0b. Check remaining dependencies ----------
command -v cmake   &>/dev/null || fail "cmake not found. Ask your sysadmin: sudo apt install cmake"
command -v git     &>/dev/null || fail "git not found. Ask your sysadmin: sudo apt install git"
command -v gcc     &>/dev/null || fail "gcc not found. Ask your sysadmin: sudo apt install build-essential"
command -v npu-smi &>/dev/null || fail "npu-smi not found. Is the CANN toolkit installed and on PATH?"

GO_VER=$(go version | awk '{print $3}' | sed 's/go//')
echo "[OK] Go $GO_VER  ($(which go))"
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

# ── 3a. Locate llama.cpp source ────────────────────────────────────────────

# Critical Go env overrides — set before ANY go invocation:
#
#   GOTOOLCHAIN=local   Ollama's go.mod has "toolchain go1.26.x" which normally
#                       triggers an automatic download of that toolchain from
#                       proxy.golang.org.  "local" tells Go: use whatever is
#                       installed, never try to fetch a newer one.
#
#   GOPROXY=direct      Bypass proxy.golang.org entirely; fetch modules straight
#                       from their source (GitHub etc.).  Set to "off" if the
#                       build machine has no outbound internet at all.
#
#   GONOSUMDB=*         Skip the checksum database (also hosted by Google).
#
export GOTOOLCHAIN=local
export GOPROXY=direct
export GONOSUMDB="*"

LLAMA_SRC=""

# Check if llama.cpp is already present (re-run scenario)
for candidate in \
    "$BUILD_DIR/llm/llama.cpp" \
    "$BUILD_DIR/llm/llama" \
    "$BUILD_DIR/llama.cpp"; do
  if [ -d "$candidate" ]; then
    LLAMA_SRC="$candidate"
    break
  fi
done

# Try go generate (now with GOTOOLCHAIN=local, it won't phone home for go1.26)
if [ -z "$LLAMA_SRC" ]; then
  echo "      Running go generate (GOTOOLCHAIN=local — no toolchain download)..."
  cd "$BUILD_DIR"
  GOTOOLCHAIN=local GOPROXY=direct GONOSUMDB="*" \
    go generate ./llm/... 2>&1 | tail -10 || true
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

# Fallback: clone llama.cpp directly from GitHub
# (used when go generate fails due to network restrictions)
if [ -z "$LLAMA_SRC" ]; then
  echo "      go generate did not produce llama.cpp — cloning directly from GitHub..."

  # Try to detect the exact commit Ollama pins for this version
  LLAMA_COMMIT=""
  for search_file in \
      "$BUILD_DIR/llm/generate.go" \
      "$BUILD_DIR/llm/gen.go" \
      "$BUILD_DIR/llm/llm.go"; do
    if [ -f "$search_file" ]; then
      LLAMA_COMMIT=$(grep -oP '(?<=[Cc]ommit\s*=\s*")[0-9a-f]{7,40}' "$search_file" 2>/dev/null | head -1 || true)
      [ -n "$LLAMA_COMMIT" ] && break
    fi
  done

  LLAMA_CLONE_DIR="$BUILD_DIR/llm/llama.cpp"
  mkdir -p "$(dirname "$LLAMA_CLONE_DIR")"

  if [ -n "$LLAMA_COMMIT" ]; then
    echo "      Detected pinned llama.cpp commit: $LLAMA_COMMIT"
    git clone --depth=1 https://github.com/ggerganov/llama.cpp "$LLAMA_CLONE_DIR"
    # Fetch the specific commit (depth=1 may not have it; fall back to full fetch)
    git -C "$LLAMA_CLONE_DIR" fetch --depth=1 origin "$LLAMA_COMMIT" 2>/dev/null \
      || git -C "$LLAMA_CLONE_DIR" fetch origin "$LLAMA_COMMIT"
    git -C "$LLAMA_CLONE_DIR" checkout "$LLAMA_COMMIT"
    echo "[OK] llama.cpp checked out at $LLAMA_COMMIT"
  else
    echo "      Could not detect pinned commit — cloning latest llama.cpp HEAD..."
    echo "      (This may cause minor API mismatches; rebuild if Ollama crashes.)"
    git clone --depth=1 https://github.com/ggerganov/llama.cpp "$LLAMA_CLONE_DIR"
  fi

  LLAMA_SRC="$LLAMA_CLONE_DIR"
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
# Keep network overrides active for go build as well
export GOTOOLCHAIN=local
export GOPROXY=direct
export GONOSUMDB="*"

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
echo "  1. Add the install dirs to your PATH (paste into ~/.bashrc to make permanent):"
echo "       export PATH=\"$GO_INSTALL_DIR/bin:$INSTALL_DIR:\$PATH\""
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
