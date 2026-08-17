#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${1:-${ROOT}/configs/train_single.yaml}"
OUTPUT="${2:-${ROOT}/outputs/train_single}"
RESUME_FROM="${3:-}"

cd "${ROOT}"
mkdir -p "${OUTPUT}"
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

visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${visible_devices%%,*}"

"${PYTHON}" -u -m src.training_single \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  "${extra_args[@]}" \
  2>&1 | tee "${OUTPUT}/${log_name}"
