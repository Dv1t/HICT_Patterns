#!/usr/bin/env bash

nextflow run ../three_stage_inference.nf \
  --cooler_path Gor_Chm.mcool \
  --clean_cooler_path CHM13_5_15_25_50.mcool \
  --weights_paths weights_paths.tsv \
  --label Gor_test \
  --hicfoundation_model ../../HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar
