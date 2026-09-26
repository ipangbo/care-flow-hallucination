#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-datasets}"
OUTPUT_ROOT="${2:-results}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

MODELS=(llava-next-7b qwen3.5-9b internvl3-8b)
DATASETS=(mmiu muirbench mibench mantis mihbench mirb)

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    prefix="${OUTPUT_ROOT}/${model}_${dataset}"
    "${PYTHON}" run_vanilla.py \
      --model "${model}" \
      --dataset-dir "${DATA_ROOT}/${dataset}" \
      --device "${DEVICE}" \
      --output "${prefix}_vanilla.jsonl" \
      --resume

    "${PYTHON}" run_care.py \
      --model "${model}" \
      --dataset-dir "${DATA_ROOT}/${dataset}" \
      --device "${DEVICE}" \
      --baseline-predictions "${prefix}_vanilla.jsonl" \
      --output "${prefix}_care.jsonl" \
      --resume
  done
done
