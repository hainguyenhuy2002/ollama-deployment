#!/bin/bash
# =============================================================================
# setup_model.sh
# One-time script to register a local HuggingFace model with Ollama.
# Run this ONCE before starting the API server.
# =============================================================================

set -e

MODEL_PATH="/villa/rhh25/mixtral"
MODEL_NAME="mixtral-local"          # Name Ollama will use
MODELFILE_PATH="./Modelfile"

echo "=== Ollama Local Model Setup ==="
echo "Model path : $MODEL_PATH"
echo "Model name : $MODEL_NAME"

# ---------- 1. Sanity checks ----------
if ! command -v ollama &>/dev/null; then
  echo "[ERROR] ollama not found. Install from https://ollama.com/download"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model directory not found: $MODEL_PATH"
  exit 1
fi

echo "[OK] ollama found: $(ollama --version)"
echo "[OK] Model directory exists"

# ---------- 2. Detect model file type ----------
# Ollama natively supports GGUF; safetensors need conversion first.
GGUF_FILE=$(find "$MODEL_PATH" -name "*.gguf" 2>/dev/null | head -1)

if [ -n "$GGUF_FILE" ]; then
  echo "[OK] GGUF file detected: $GGUF_FILE"
  FROM_LINE="FROM $GGUF_FILE"
else
  # HuggingFace safetensors — point Ollama at the directory directly.
  # Ollama >= 0.1.38 supports importing HF repos via directory path.
  echo "[INFO] No GGUF found; using HuggingFace directory format."
  FROM_LINE="FROM $MODEL_PATH"
fi

# ---------- 3. Write Modelfile ----------
cat > "$MODELFILE_PATH" <<EOF
# Auto-generated Modelfile — edit system prompt as needed.
$FROM_LINE

PARAMETER num_ctx        4096
PARAMETER num_predict    -1
PARAMETER temperature    0.7
PARAMETER top_p          0.9
PARAMETER repeat_penalty 1.1

# Number of GPU layers: -1 = offload everything to GPU
PARAMETER num_gpu        -1

SYSTEM """You are a helpful AI assistant."""
EOF

echo "[OK] Modelfile written to $MODELFILE_PATH"
cat "$MODELFILE_PATH"

# ---------- 4. Register model with Ollama ----------
echo ""
echo "=== Creating model '$MODEL_NAME' in Ollama (this may take a few minutes) ==="
ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"

echo ""
echo "=== Setup complete! ==="
echo "Run 'ollama list' to verify, then start the API server with:"
echo "  bash start_server.sh"
