#!/usr/bin/env bash
# Merge a LoRA adapter, convert it to GGUF, and create an Ollama model.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-finetune}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen2.5-0.5b-medical-lora}"
MERGED_DIR="${MERGED_DIR:-outputs/qwen2.5-0.5b-medical-merged}"
GGUF_PATH="${GGUF_PATH:-outputs/qwen2.5-0.5b-medical-q4_0.gguf}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-0.5b-medical-npu}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"

if [ ! -d "$VENV_DIR" ]; then
  echo "[ERROR] Missing $VENV_DIR. Run finetune/setup_finetune_env.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python finetune/merge_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "${GGUF_PATH%.gguf}-f16.gguf"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "${GGUF_PATH%.gguf}-f16.gguf" "$GGUF_PATH" Q4_0

cat > outputs/Modelfile.finetuned <<EOF
FROM $GGUF_PATH
PARAMETER temperature 0.2
SYSTEM You are a careful clinical pharmacy assistant. Give concise and safe answers.
EOF

ollama create "$OLLAMA_MODEL" -f outputs/Modelfile.finetuned
echo "[INFO] Created Ollama model: $OLLAMA_MODEL"
