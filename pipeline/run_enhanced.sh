#!/bin/bash

SIF=/mnt/tank/scratch/vdravgelis/HiCFoundation/hicfoundation_image.sif

singularity exec --nv \
    --bind /mnt/tank/scratch/vdravgelis/data:/app/data \
    --bind /mnt/tank/scratch/vdravgelis/HiCFoundation:/app/HiCFoundation \
    --bind /mnt/tank/scratch/vdravgelis/pipeline:/app/code \
    --bind /mnt/tank/scratch/vdravgelis/ClusterBuffer:/app/buffer \
"$SIF" \
bash -c "
    cd /app/code &&
    source /opt/conda/bin/activate &&
    conda activate HiCFoundation &&
    python /app/code/scripts/run_model_enhanced.py /app/buffer/Gor_Chm/Gor_Chm.mcool 10000 /app/code/Gor_Chm_output/training/weight_15000.pt /app/HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar 4 0.9 Gor_Chm_out.csv --coords_csv /app/code/Gor_Chm_output/results/gor_detected_breakpoints.csv
"