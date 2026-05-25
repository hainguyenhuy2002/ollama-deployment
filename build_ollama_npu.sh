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
# Hardware  : 8× Huawei Ascend 910B3, 64 GB HBM each (512 GB total)
# CANN ver  : 25.5.0  (confirmed via npu-smi info)
# NUMA topo : 4 NUMA groups, 2 NPUs each
#               NPU 0,1 → PCIe bus C1/C2
#               NPU 2,3 → PCIe bus 81/82
#               NPU 4,5 → PCIe bus 01/02
#               NPU 6,7 → PCIe bus 41/42
#
# No sudo required
# ----------------
# Go is installed to ~/go if not already present.
# The Ollama binary is installed to ~/.local/bin.
# cmake, gcc, and git must already be available (ask your sysadmin if missing).
#
# Prerequisites
# -------------
#   • Huawei CANN Toolkit 25.x  (installed by sysadmin — auto-detected)
#     Set ASCEND_HOME if auto-detection fails:
#       export ASCEND_HOME=/path/to/ascend-toolkit/latest
#   • cmake >= 3.24, gcc/g++ >= 10, git, curl  (system packages)
#   • Go >= 1.22  → auto-installed to ~/go by this script if missing
#
# Usage
# -----
#   bash build_ollama_npu.sh
# =============================================================================

set -euo pipefail

# ---------- Config — edit as needed ----------
OLLAMA_REPO="https://github.com/ollama/ollama.git"
OLLAMA_VERSION="v0.9.0"                        # pin to a stable release tag
INSTALL_DIR="$HOME/.local/bin"                 # where to put the built `ollama` binary
BUILD_DIR="$(pwd)/ollama-cann-build"           # temporary build directory
GO_INSTALL_DIR="$HOME/go"                      # Go toolchain install location (no sudo)
GO_VERSION="1.23.4"                            # Go version to download if missing
NPU_COUNT=8                                    # number of Ascend 910B3 NPUs on this server
HBM_PER_NPU_MB=65536                          # HBM per NPU in MB (64 GB)
# (CANN_TOOLKIT_ROOT is auto-detected below — override via $ASCEND_HOME if needed)
# ---------------------------------------------

# Banner is printed after CANN detection so CANN_TOOLKIT_ROOT is populated
print_banner() {
  echo "========================================================"
  echo " Ollama NPU Build — 8× Ascend 910B3 (CANN 25.5.0)"
  echo "========================================================"
  echo "Ollama version  : $OLLAMA_VERSION"
  echo "Install dir     : $INSTALL_DIR"
  echo "Build dir       : $BUILD_DIR"
  echo "CANN root       : ${CANN_TOOLKIT_ROOT:-<auto-detecting...>}"
  echo "NPU count       : $NPU_COUNT × 910B3 ($(( NPU_COUNT * HBM_PER_NPU_MB / 1024 )) GB HBM total)"
  echo ""
}

# ---------- 0. Helpers ----------
fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
ok()   { echo "[OK]   $*"; }

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

# ---------- 1. Auto-detect and source CANN environment ----------
echo ""
echo "[1/5] Locating CANN toolkit (target: 25.5.0 / Ascend 910B3)..."

find_cann_root() {
  # 1. Explicit override
  if [ -n "${ASCEND_HOME:-}" ] && [ -d "${ASCEND_HOME}" ]; then
    echo "${ASCEND_HOME}"; return 0
  fi
  # 2. Derive from npu-smi binary path (most reliable)
  if command -v npu-smi &>/dev/null; then
    local bin_dir
    bin_dir=$(dirname "$(command -v npu-smi)")
    for candidate in "$(dirname "$bin_dir")" "$(dirname "$(dirname "$bin_dir")")"; do
      if [ -f "$candidate/bin/setenv.bash" ] || [ -f "$candidate/bin/setenv.sh" ]; then
        echo "$candidate"; return 0
      fi
    done
  fi
  # 3. Common CANN 25.x install paths (sysadmin-managed and user-local)
  for candidate in \
      "/usr/local/Ascend/ascend-toolkit/latest" \
      "/usr/local/Ascend/ascend-toolkit/25.5.0" \
      "$HOME/Ascend/ascend-toolkit/latest" \
      "$HOME/Ascend/ascend-toolkit/25.5.0" \
      "$HOME/ascend-toolkit/latest" \
      "/opt/Ascend/ascend-toolkit/latest" \
      "/opt/Ascend/ascend-toolkit/25.5.0" \
      "/opt/ascend/ascend-toolkit/latest"; do
    [ -d "$candidate" ] && echo "$candidate" && return 0
  done
  echo ""; return 1
}

CANN_TOOLKIT_ROOT=$(find_cann_root)

if [ -z "$CANN_TOOLKIT_ROOT" ]; then
  fail "CANN toolkit not found. Set ASCEND_HOME to its root, e.g.:
       export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest"
fi
ok "CANN toolkit: $CANN_TOOLKIT_ROOT"

# Source the CANN environment (sets ASCEND_TOOLKIT_HOME, PATH, LD_LIBRARY_PATH, etc.)
if [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.bash" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.bash"
elif [ -f "$CANN_TOOLKIT_ROOT/bin/setenv.sh" ]; then
  # shellcheck disable=SC1090
  source "$CANN_TOOLKIT_ROOT/bin/setenv.sh"
fi

# llama.cpp's CANN backend looks for ASCEND_TOOLKIT_HOME (not CANN_TOOLKIT_ROOT).
# Export both so every downstream tool finds what it needs.
export ASCEND_TOOLKIT_HOME="${CANN_TOOLKIT_ROOT}"
export ASCEND_HOME="${CANN_TOOLKIT_ROOT}"

# Build a comprehensive LD_LIBRARY_PATH covering CANN 25.x directory layout.
# 910B3 uses the standard acllib paths; atc and fwkacllib may also be needed at runtime.
_CANN_LIBS="${CANN_TOOLKIT_ROOT}/lib64"
_CANN_LIBS+=":${CANN_TOOLKIT_ROOT}/acllib/lib64"
_CANN_LIBS+=":${CANN_TOOLKIT_ROOT}/atc/lib64"
_CANN_LIBS+=":${CANN_TOOLKIT_ROOT}/fwkacllib/lib64"
export LD_LIBRARY_PATH="${_CANN_LIBS}:${LD_LIBRARY_PATH:-}"

# Confirm the CANN version matches what npu-smi reports (25.5.0)
if command -v npu-smi &>/dev/null; then
  DETECTED_CANN_VER=$(npu-smi info 2>/dev/null | grep -oP 'Version:\s*\K[\d.]+' | head -1 || true)
  [ -n "$DETECTED_CANN_VER" ] && info "Detected CANN/npu-smi version: $DETECTED_CANN_VER"
fi
ok "CANN env ready"

# Print banner now that CANN_TOOLKIT_ROOT is resolved
print_banner

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
    -DCMAKE_INSTALL_PREFIX="$LLAMA_BUILD_DIR/install" \
    -DGGML_NATIVE=OFF \
    -DASCEND_TOOLKIT_HOME="${CANN_TOOLKIT_ROOT}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build with all available CPU threads; on a multi-socket server this can be
# large — cap at 32 to avoid OOM during compilation if RAM is limited.
BUILD_JOBS=$(( $(nproc) < 32 ? $(nproc) : 32 ))
info "Building with $BUILD_JOBS parallel jobs (nproc=$(nproc))"

cmake --build "$LLAMA_BUILD_DIR" \
    --config Release \
    -j"$BUILD_JOBS" \
    --target llama-server ggml

ok "llama.cpp built with CANN support (910B3)"

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

# Tag the binary with build metadata for easy identification later
BUILD_TAGS="cann"
go build -tags "$BUILD_TAGS" -o ollama . 2>&1 | tail -10

cd - > /dev/null
ok "Ollama binary built: $BUILD_DIR/ollama"

# ---------- 5. Install ----------
echo ""
echo "[5/6] Installing to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
cp "$BUILD_DIR/ollama" "$INSTALL_DIR/ollama"
chmod +x "$INSTALL_DIR/ollama"

# ---------- 6. Verify binary and NPU visibility ----------
echo ""
echo "[6/6] Verifying installation..."

export PATH="$INSTALL_DIR:$GO_INSTALL_DIR/bin:$PATH"

OLLAMA_VER=$("$INSTALL_DIR/ollama" --version 2>/dev/null || echo "unknown")
ok "ollama binary: $OLLAMA_VER"

# Confirm all 8 NPUs are visible via npu-smi
echo ""
info "NPU status (all 8 Ascend 910B3 cards should show OK):"
npu-smi info 2>/dev/null | grep -E "(NPU|910B3|Health|OK|ERROR)" | head -20 || true

# Quick CANN library sanity check
if ldconfig -p 2>/dev/null | grep -q "libascendcl"; then
  ok "libascendcl found in ldconfig cache"
else
  # Not in ldconfig — check LD_LIBRARY_PATH directly
  for libdir in $(echo "$LD_LIBRARY_PATH" | tr ':' ' '); do
    if [ -f "$libdir/libascendcl.so" ]; then
      ok "libascendcl.so found in $libdir"
      break
    fi
  done
fi

echo ""
echo "========================================================"
echo " Build complete!  (Ascend 910B3 × $NPU_COUNT — CANN 25.5.0)"
echo "========================================================"
echo ""
echo "Binary          : $INSTALL_DIR/ollama"
echo "Total HBM       : $(( NPU_COUNT * HBM_PER_NPU_MB / 1024 )) GB  ($NPU_COUNT × $(( HBM_PER_NPU_MB / 1024 )) GB)"
echo ""
echo "Next steps:"
echo "  1. Add install dirs to PATH (add to ~/.bashrc to make permanent):"
echo "       export PATH=\"$GO_INSTALL_DIR/bin:$INSTALL_DIR:\$PATH\""
echo "       export ASCEND_TOOLKIT_HOME=\"$CANN_TOOLKIT_ROOT\""
echo "       export LD_LIBRARY_PATH=\"$_CANN_LIBS:\$LD_LIBRARY_PATH\""
echo ""
echo "  2. Pull a model (first time only):"
echo "       bash setup_model.sh"
echo ""
echo "  3. Start the server (uses all 8 NPUs by default):"
echo "       bash start_server.sh"
echo ""
echo "  4. Check NPU utilisation:"
echo "       watch -n1 npu-smi info"
echo "       curl http://localhost:8000/npu/status | python3 -m json.tool"
echo ""
echo "Troubleshooting:"
echo "  • Ollama exits immediately  → check logs/ollama.log; ensure CANN env is sourced"
echo "  • CANN libs missing         → source \$ASCEND_TOOLKIT_HOME/bin/setenv.bash"
echo "  • NPU not detected          → check ASCEND_RT_VISIBLE_DEVICES in start_server.sh"
echo "  • Multi-card inference slow → check HCCL_WHITELIST_DISABLE=1 in start_server.sh"
echo "  • Upstream CANN issues      → https://github.com/ollama/ollama/issues (filter: cann)"
