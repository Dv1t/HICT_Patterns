# Pipeline examples

These launchers correspond to the two supported ways of running the pipeline
described in the repository README. Run them from any directory after setting
the environment variables shown below.

## 1. Inference with existing weights

`run_existing_weights.sh` runs the original-map inference, HiCFoundation
enhancement, and enhanced-map inference using an existing weights manifest.

```bash
export COOLER_PATH=/data/Siamang_Chm.mcool
export CLEAN_COOLER_PATH=/data/CHM13_15_25_50.mcool
export STAGE3_CLEAN_COOLER_PATH=/data/CHM13_Siamang_Chm_enhanced.mcool
export WEIGHTS_PATHS=/data/Siamang_weights_paths.tsv
export LABEL=Siamang_Chm
export HICFOUNDATION_SIF=/images/hicfoundation.sif
export HICFOUNDATION_INFERENCE=$PWD/HiCFoundation/inference.py
export HICFOUNDATION_MODEL=$PWD/HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar

pipeline/examples/run_existing_weights.sh
```

The weights manifest is tab-separated, with one checkpoint and its training
resolution per line. See `weights_paths.tsv.example`. Set
`STAGE3_WEIGHTS_PATHS` to use a different manifest for Stage 3; otherwise the
Stage 1 manifest is reused.

## 2. Train once, then run all three stages

`run_train_then_infer.sh` trains one model per resolution and reuses those
weights for both inference stages.

```bash
export INPUT_TRAIN_COOLER_PATHS=/data/training_samples.tsv
export COOLER_PATH=/data/Siamang_Chm.mcool
export CLEAN_COOLER_PATH=/data/CHM13_15_25_50.mcool
export STAGE3_CLEAN_COOLER_PATH=/data/CHM13_Siamang_Chm_enhanced.mcool
export LABEL=Siamang_Chm
export NUM_EPOCH=160
export HICFOUNDATION_SIF=/images/hicfoundation.sif
export HICFOUNDATION_INFERENCE=$PWD/HiCFoundation/inference.py
export HICFOUNDATION_MODEL=$PWD/HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar

pipeline/examples/run_train_then_infer.sh
```

The training manifest must contain `label`, `SV .mcool`, `clean .mcool`, and
`answer CSV`, separated by tabs. See `training_samples.tsv.example`.

Both launchers forward `RESOLUTIONS`, `OUTDIR`, and `NEXTFLOW_EXTRA_ARGS` when
set. `RESOLUTIONS` is a comma-separated list such as `15000,25000,50000`.
