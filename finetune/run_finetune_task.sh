#!/usr/bin/env bash
# Run a small real-data LoRA task on Ascend NPU.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-finetune}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-medalpaca/medical_meadow_medical_flashcards}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATA_PATH="${DATA_PATH:-data/medical_flashcards_lora.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-0.5b-medical-lora}"
MAX_STEPS="${MAX_STEPS:-800}"
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"
STOP_SERVING="${STOP_SERVING:-1}"

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
}

mkdir -p logs data outputs
load_ascend_env
export ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

if [ "$STOP_SERVING" = "1" ]; then
  pkill -f "llama-server" 2>/dev/null || true
  pkill -f "api_server.py" 2>/dev/null || true
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Fine-tune venv not found. Creating it first."
  bash finetune/setup_finetune_env.sh
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Preparing dataset: $DATASET_NAME"
python finetune/prepare_dataset.py \
  --dataset "$DATASET_NAME" \
  --split "$DATASET_SPLIT" \
  --output "$DATA_PATH" \
  --max-samples "$MAX_SAMPLES"

echo "[INFO] Starting LoRA fine-tuning on Ascend device(s): $ASCEND_VISIBLE_DEVICES"
npu-smi info | tee "logs/finetune_npu_before.log"

python finetune/train_lora_npu.py \
  --model "$BASE_MODEL" \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS"

npu-smi info | tee "logs/finetune_npu_after.log"
echo "[INFO] Fine-tuning complete. Adapter: $OUTPUT_DIR"
