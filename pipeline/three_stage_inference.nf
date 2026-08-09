/*
 * Three-stage HICT Patterns workflow:
 *
 *   Stage 1: infer breakpoints on the original .mcool
 *   Stage 2: enhance the original .mcool with HiCFoundation, using the
 *            Stage 1 breakpoint CSV as input_coords
 *   Stage 3: run HICT Patterns inference on the enhanced .mcool
 *
 * Example:
 * nextflow run three_stage_inference.nf \
 *   --cooler_path /path/Gor_Chm.mcool \
 *   --clean_cooler_path /path/CHM13_15_25_50.mcool \
 *   --weights_paths /path/Pan_troglodytes_weights_paths.tsv \
 *   --stage3_weights_paths /path/Gor_Chm_weights_paths.tsv \
 *   --label Gor_Chm \
 *   --hicfoundation_inference ../HiCFoundation/inference.py \
 *   --hicfoundation_model ../HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar
 */

def required_params = [
    'cooler_path',
    'clean_cooler_path',
    'label',
    'weights_paths',
    'hicfoundation_inference',
    'hicfoundation_model'
]

required_params.each { p ->
    if (!params."$p")
        error "Missing required parameter: --${p}"
}

def stage3_weights_paths = params.weights_paths
params.stage3_run_model_script = params.stage3_run_model_script ?: 'scripts/run_model_enhanced.py'
params.outdir = params.outdir ?: "${params.label}_three_stage_output"

log.info """
HICT PATTERNS  T H R E E - S T A G E  P I P E L I N E
=======================================================
label                         : ${params.label}
stage 1 cooler                : ${params.cooler_path}
stage 1 clean cooler          : ${params.clean_cooler_path}
stage 1 weights               : ${params.weights_paths}
""".stripIndent().trim()

include { inference_model; inference_model_enhanced } from './modules/run_model.nf'
include { calculate_expected as calculate_expected_stage1 } from './modules/calculate_npy.nf'
include { calculate_expected as calculate_expected_stage3 } from './modules/calculate_npy.nf'
include { enhance_with_hicfoundation } from './modules/enhance.nf'

workflow {
    generate_script = file(params.generate_npy_script)
    stage1_inference_script = file(params.run_model_script)
    stage3_inference_script = file(params.stage3_run_model_script)

    stage1_weights_ch = Channel
        .fromPath(params.weights_paths)
        .splitText()
        .map { line ->
            def (weight_path_str, resolution) = line.trim().split(/\t/)
            tuple(file(weight_path_str), resolution)
        }

    stage1_coolers = Channel.fromList([
        tuple(params.label, file(params.cooler_path), 'test'),
        tuple(params.label, file(params.clean_cooler_path), 'test_clean')
    ])

    stage1_expected = stage1_coolers
        .combine(Channel.fromList(params.resolutions))
        .map { label, mcool, type, resolution ->
            tuple(label, type, mcool, resolution, generate_script)
        }
        | calculate_expected_stage1

    stage1_split = stage1_expected
        .filter { label, type, resolution, npy -> type.startsWith('test') }
        .branch { label, type, resolution, npy ->
            target: type == 'test'
            clean: type == 'test_clean'
        }

    stage1_target_expected = stage1_split.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    stage1_clean_expected = stage1_split.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    stage1_resolutions = params.resolutions.collect { it as int }.sort()
    stage1_weights = stage1_weights_ch
        .toSortedList { a, b -> (a[1] as int) <=> (b[1] as int) }
        .map { sorted_tuples -> sorted_tuples.collect { it[0] } }

    inference_model(
        stage1_weights,
        Channel.of(stage1_resolutions),
        Channel.of(file(params.cooler_path)),
        Channel.of(file(params.clean_cooler_path)),
        stage1_target_expected,
        stage1_clean_expected,
        stage1_inference_script,
        Channel.of(params.label)
    )

    stage2_input = inference_model.result
        .map { input_coords ->
            tuple(params.label, file(params.cooler_path), input_coords)
        }

    stage2 = enhance_with_hicfoundation(
        stage2_input,
        file(params.hicfoundation_inference),
        file(params.hicfoundation_model)
    )

    stage3_coolers = stage2.enhanced
        .map { label, enhanced_mcool, stage1_breakpoints ->
            tuple(label, enhanced_mcool, 'test')
        }
        .mix(Channel.of(tuple(params.label, file(params.clean_cooler_path), 'test_clean')))

    stage3_expected = stage3_coolers
        .combine(Channel.fromList(params.resolutions))
        .map { label, mcool, type, resolution ->
            tuple(label, type, mcool, resolution, generate_script)
        }
        | calculate_expected_stage3

    stage3_split = stage3_expected
        .filter { label, type, resolution, npy -> type.startsWith('test') }
        .branch { label, type, resolution, npy ->
            target: type == 'test'
            clean: type == 'test_clean'
        }

    stage3_target_expected = stage3_split.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    stage3_clean_expected = stage3_split.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    stage3_weights_ch = Channel
        .fromPath(stage3_weights_paths)
        .splitText()
        .map { line ->
            def (weight_path_str, resolution) = line.trim().split(/\t/)
            tuple(file(weight_path_str), resolution)
        }

    stage3_weights = stage3_weights_ch
        .toSortedList { a, b -> (a[1] as int) <=> (b[1] as int) }
        .map { sorted_tuples -> sorted_tuples.collect { it[0] } }

    stage3_input_csv = stage2.enhanced
        .map { label, enhanced_mcool, stage1_breakpoints -> stage1_breakpoints }

    inference_model_enhanced(
        stage3_weights,
        Channel.of(stage1_resolutions),
        stage2.enhanced.map { label, enhanced_mcool, csv -> enhanced_mcool },
        Channel.of(file(params.clean_cooler_path)),
        stage3_target_expected,
        stage3_clean_expected,
        stage3_input_csv,
        stage3_inference_script,
        Channel.of("${params.label}_with_CHM13_enhanced")
    )
}
