$ErrorActionPreference = 'Stop'

$exampleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$existing = Get-Content (Join-Path $exampleDir 'run_existing_weights.sh') -Raw
$training = Get-Content (Join-Path $exampleDir 'run_train_then_infer.sh') -Raw
$weights = Get-Content (Join-Path $exampleDir 'weights_paths.tsv.example') | Where-Object { $_ -and -not $_.StartsWith('#') }
$samples = Get-Content (Join-Path $exampleDir 'training_samples.tsv.example') | Where-Object { $_ -and -not $_.StartsWith('#') }

if ($existing -notmatch 'three_stage_inference\.nf') { throw 'Existing-weights example does not run three_stage_inference.nf' }
if ($training -notmatch 'train_three_stage_inference\.nf') { throw 'Training example does not run train_three_stage_inference.nf' }
foreach ($name in 'COOLER_PATH', 'CLEAN_COOLER_PATH', 'STAGE3_CLEAN_COOLER_PATH', 'HICFOUNDATION_SIF', 'HICFOUNDATION_INFERENCE', 'HICFOUNDATION_MODEL') {
    if ($existing -notmatch [regex]::Escape($name) -or $training -notmatch [regex]::Escape($name)) { throw "Missing shared variable: $name" }
}
if ($existing -notmatch 'WEIGHTS_PATHS' -or $existing -notmatch 'stage3_weights_paths') { throw 'Existing-weights example is missing its weight manifests' }
if ($training -notmatch 'INPUT_TRAIN_COOLER_PATHS' -or $training -notmatch 'NUM_EPOCH') { throw 'Training example is missing training parameters' }
if ($weights.Count -ne 3 -or $samples.Count -ne 2) { throw 'Example metadata templates have unexpected row counts' }
foreach ($row in $weights) { if (($row -split "`t").Count -ne 2) { throw 'Weights template rows must have two tab-separated fields' } }
foreach ($row in $samples) { if (($row -split "`t").Count -ne 4) { throw 'Training template rows must have four tab-separated fields' } }

Write-Output 'Pipeline example smoke tests passed.'
