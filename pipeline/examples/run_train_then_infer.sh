#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${INPUT_TRAIN_COOLER_PATHS:?Set INPUT_TRAIN_COOLER_PATHS to a training metadata TSV}"
: "${COOLER_PATH:?Set COOLER_PATH to the target .mcool file}"
: "${CLEAN_COOLER_PATH:?Set CLEAN_COOLER_PATH to the reference .mcool file}"
: "${STAGE3_CLEAN_COOLER_PATH:?Set STAGE3_CLEAN_COOLER_PATH to the enhanced-map reference .mcool file}"
: "${LABEL:?Set LABEL to the sample name}"
: "${HICFOUNDATION_SIF:?Set HICFOUNDATION_SIF to the HiCFoundation image}"
: "${HICFOUNDATION_INFERENCE:?Set HICFOUNDATION_INFERENCE to inference.py}"
: "${HICFOUNDATION_MODEL:?Set HICFOUNDATION_MODEL to the HiCFoundation checkpoint}"

args=(
  --input_train_cooler_paths "$INPUT_TRAIN_COOLER_PATHS"
  --cooler_path "$COOLER_PATH"
  --clean_cooler_path "$CLEAN_COOLER_PATH"
  --stage3_clean_cooler_path "$STAGE3_CLEAN_COOLER_PATH"
  --label "$LABEL"
  --hicfoundation_sif "$HICFOUNDATION_SIF"
  --hicfoundation_inference "$HICFOUNDATION_INFERENCE"
  --hicfoundation_model "$HICFOUNDATION_MODEL"
)

if [[ -n "${RESOLUTIONS:-}" ]]; then args+=(--resolutions "$RESOLUTIONS"); fi
if [[ -n "${NUM_EPOCH:-}" ]]; then args+=(--num_epoch "$NUM_EPOCH"); fi
if [[ -n "${OUTDIR:-}" ]]; then args+=(--outdir "$OUTDIR"); fi

# shellcheck disable=SC2086
nextflow run "$example_dir/../train_three_stage_inference.nf" "${args[@]}" ${NEXTFLOW_EXTRA_ARGS:-}
