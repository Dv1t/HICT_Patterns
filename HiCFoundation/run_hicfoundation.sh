#!/usr/bin/env bash
# Run HiCFoundation enhancement for one label.

set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: run_hicfoundation.sh LABEL [options]
  --input-coords FILE       Stage 1 breakpoint CSV
  --input-mcool FILE        Input .mcool
  --output-mcool FILE       Enhanced output .mcool
  --genome FILE             Genome/chromosome sizes output
  --input-cool FILE         Temporary 5kb .cool output
  --sif FILE                HiCFoundation Singularity image
EOF
    exit 2
}

[[ $# -ge 1 ]] || usage
LABEL="$1"
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HICFOUNDATION_DIR="${HICFOUNDATION_DIR:-${ROOT_DIR}/HiCFoundation}"
PIPELINE_DIR="${PIPELINE_DIR:-${ROOT_DIR}/pipeline}"
COOLER_ROOT="${COOLER_ROOT:-/mnt/tank/scratch/vdravgelis/ClusterBuffer}"
BUFFER_DIR="${BUFFER_DIR:-$COOLER_ROOT}"
SIF="${SIF:-/mnt/tank/scratch/vdravgelis/HiCFoundation/hicfoundation_image.sif}"

INPUT_MCOOL="${COOLER_ROOT}/${LABEL}/${LABEL}.mcool"
INPUT_COORDS="${PIPELINE_DIR}/${LABEL}_output/results/${LABEL}_detected_breakpoints.csv"
OUTPUT_MCOOL="${COOLER_ROOT}/${LABEL}/${LABEL}_enhanced.mcool"
GENOME="${COOLER_ROOT}/${LABEL}/${LABEL}.genome"
INPUT_COOL="${COOLER_ROOT}/${LABEL}/${LABEL}_5kb.cool"

while (($#)); do
    case "$1" in
        --input-coords) INPUT_COORDS="$2"; shift 2 ;;
        --input-mcool) INPUT_MCOOL="$2"; shift 2 ;;
        --output-mcool) OUTPUT_MCOOL="$2"; shift 2 ;;
        --genome) GENOME="$2"; shift 2 ;;
        --input-cool) INPUT_COOL="$2"; shift 2 ;;
        --sif) SIF="$2"; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage ;;
    esac
done

for path in "$INPUT_MCOOL" "$INPUT_COORDS" "$SIF"; do
    [[ -f "$path" ]] || { printf 'Required file does not exist: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUTPUT_MCOOL")"

CONTAINER_COORDS="/app/pipeline/${LABEL}_output/results/${LABEL}_detected_breakpoints.csv"
EXTRA_BINDS=()
if [[ "$INPUT_COORDS" != "$PIPELINE_DIR/"* ]]; then
    CONTAINER_COORDS="/app/coords/$(basename "$INPUT_COORDS")"
    EXTRA_BINDS+=(--bind "$(dirname "$INPUT_COORDS"):/app/coords")
else
    CONTAINER_COORDS="/app/pipeline/${INPUT_COORDS#"$PIPELINE_DIR/"}"
fi

cooler dump -t chroms "${INPUT_MCOOL}::/resolutions/1000" > "$GENOME"
cooler cp "${INPUT_MCOOL}::/resolutions/5000" "$INPUT_COOL"

singularity exec --nv \
    --bind "${ROOT_DIR}/data:/app/data" \
    --bind "${HICFOUNDATION_DIR}:/app/code" \
    --bind "${BUFFER_DIR}:/app/buffer" \
    --bind "${PIPELINE_DIR}:/app/pipeline" \
    "${EXTRA_BINDS[@]}" \
    "$SIF" bash -c "
        cd /app/code &&
        source /opt/conda/bin/activate &&
        conda activate HiCFoundation &&
        python inference.py \
            --resolution 5000 \
            --input_coords ${CONTAINER_COORDS} \
            --input /app/buffer/${LABEL}/${LABEL}_5kb.cool \
            --batch_size 4 \
            --num_workers 0 \
            --genome_id /app/buffer/${LABEL}/${LABEL}.genome \
            --model_path /app/code/hicfoundation_model/hicfoundation_resolution.pth.tar \
            --task 3 \
            --bound 8000 \
            --input_row_size 224 \
            --input_col_size 224 \
            --output ${LABEL}_output
    "

cooler zoomify -n 16 -r 15000,25000,50000 --balance --balance-args '--nproc 16' \
    -o "$OUTPUT_MCOOL" \
    "${HICFOUNDATION_DIR}/${LABEL}_output/HiCFoundation_enhanced.cool"
