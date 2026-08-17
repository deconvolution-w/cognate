#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE="${ACCELERATE:-accelerate}"
CONFIG="${1:-${ROOT}/configs/train_4gpu.yaml}"
OUTPUT="${2:-${ROOT}/outputs/train_4gpu}"
RESUME_FROM="${3:-}"

cd "${ROOT}"
mkdir -p "${OUTPUT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ROOT}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

extra_args=()
log_name="training.log"
if [[ -n "${RESUME_FROM}" ]]; then
  extra_args+=(--resume-from "${RESUME_FROM}")
  log_name="training_resume.log"
fi

"${ACCELERATE}" launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port 0 \
  --rdzv_backend static \
  --module src.training \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  "${extra_args[@]}" \
  2>&1 | tee "${OUTPUT}/${log_name}"
