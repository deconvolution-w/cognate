#!/usr/bin/env bash
set -euo pipefail

input_csv="${1:-data/examples/pairs.csv}"
output_csv="${2:-outputs/evaluation/scores.csv}"
model_args=()
if [[ -n "${DEVICE:-}" ]]; then
  model_args+=(--device "${DEVICE}")
fi

python -m src.cli score-batch \
  --input "$input_csv" \
  --out "$output_csv" \
  "${model_args[@]}"
