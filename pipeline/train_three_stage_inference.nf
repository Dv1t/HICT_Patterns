/*
 * Train once, then run the complete three-stage workflow:
 *
 *   1. Train HICT Patterns models from --input_train_cooler_paths.
 *   2. Run inference on the original .mcool using the trained weights.
 *   3. Enhance that .mcool with HiCFoundation using Stage 2 coordinates.
 *   4. Run inference on the enhanced .mcool using the SAME trained weights.
 *
 * Example:
 * nextflow run train_three_stage_inference.nf \
 *   --input_train_cooler_paths /path/train.tsv \
 *   --cooler_path /path/Gor_Chm.mcool \
 *   --clean_cooler_path /path/CHM13_15_25_50.mcool \
 *   --label Gor_Chm \
 *   --hicfoundation_inference ../HiCFoundation/inference.py \
 *   --hicfoundation_model ../HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar
 *
 * Each line in train.tsv must contain:
 *   training_label<TAB>sv_mcool<TAB>clean_mcool<TAB>answer_csv
 */

def required_params = [
    'cooler_path',
    'clean_cooler_path',
    'input_train_cooler_paths',
    'label',
    'hicfoundation_inference',
    'hicfoundation_model'
]

required_params.each { p ->
    if (!params."$p")
        error "Missing required parameter: --${p}"
}

params.outdir = "${params.label}_trained_three_stage_output"

log.info """
HICT PATTERNS  T R A I N + T H R E E - S T A G E  P I P E L I N E
================================================================
label                         : ${params.label}
training metadata             : ${params.input_train_cooler_paths}
stage 1 cooler                : ${params.cooler_path}
stage 1 clean cooler          : ${params.clean_cooler_path}
resolutions                   : ${params.resolutions}
""".stripIndent().trim()

include { train_model } from './modules/train_model.nf'
include { inference_model; inference_model_enhanced } from './modules/run_model.nf'
include { calculate_expected as calculate_expected_train } from './modules/calculate_npy.nf'
include { calculate_expected as calculate_expected_stage3 } from './modules/calculate_npy.nf'
include { enhance_with_hicfoundation } from './modules/enhance.nf'

process rename_answer {
    tag "$label"
    executor 'local'

    input:
    tuple val(label), path(answer_file)

    output:
    tuple val(label), path("${label}_${answer_file.name}")

    script:
    """
    cp "${answer_file}" "${label}_${answer_file.name}"
    """
}

workflow {
    generate_script = file("${projectDir}/scripts/generate_npy.py")
    stage1_inference_script = file("${projectDir}/scripts/run_model.py")
    stage3_inference_script = file("${projectDir}/scripts/run_model_enhanced.py")
    train_script = file("${projectDir}/scripts/train_model_wm.py")
    models_library = file("${projectDir}/scripts/models.py")

    train_meta_ch = Channel
        .fromPath(params.input_train_cooler_paths)
        .splitText()
        .map { line ->
            def (label, sv_mcool_path, clean_mcool_path, answer) = line.trim().split(/\t/)
            tuple(label, file(sv_mcool_path), file(clean_mcool_path), answer)
        }

    sv_meta_ch = train_meta_ch.map { label, sv_mcool, clean_mcool, answer ->
        tuple(label, sv_mcool, answer)
    }
    clean_meta_ch = train_meta_ch.map { label, sv_mcool, clean_mcool, answer ->
        tuple(label, clean_mcool, answer)
    }

    sv_mcools_ch = sv_meta_ch
        .map { label, mcool, answer -> [label, mcool] }
        .toList()
        .map { rows -> rows.sort { a, b -> a[0] <=> b[0] }.collect { it[1] } }

    clean_mcools_ch = clean_meta_ch
        .map { label, mcool, answer -> [label, mcool] }
        .toList()
        .map { rows -> rows.sort { a, b -> a[0] <=> b[0] }.collect { it[1] } }

    answers_to_rename_ch = train_meta_ch
        .filter { label, sv_mcool, clean_mcool, answer -> answer && answer != '-' }
        .map { label, sv_mcool, clean_mcool, answer -> tuple(label, file(answer)) }

    answers_ch = rename_answer(answers_to_rename_ch)
        .toList()
        .map { rows -> rows.sort { a, b -> a[0] <=> b[0] }.collect { it[1] } }

    train_sv_coolers = train_meta_ch.map { label, sv_mcool, clean_mcool, answer ->
        tuple(label, sv_mcool, 'train_SV')
    }
    train_clean_coolers = train_meta_ch.map { label, sv_mcool, clean_mcool, answer ->
        tuple(label, clean_mcool, 'train_clean')
    }
    test_coolers = Channel.fromList([
        tuple(params.label, file(params.cooler_path), 'test'),
        tuple(params.label, file(params.clean_cooler_path), 'test_clean')
    ])

    all_coolers = train_sv_coolers.concat(train_clean_coolers).concat(test_coolers)

    expected = all_coolers
        .combine(Channel.fromList(params.resolutions))
        .map { label, mcool, type, resolution ->
            tuple(label, type, mcool, resolution, generate_script)
        }
        | calculate_expected_train

    test_expected = expected.filter { label, type, resolution, npy -> type.startsWith('test') }
    test_split = test_expected.branch { label, type, resolution, npy ->
        target: type == 'test'
        clean: type == 'test_clean'
    }
    test_target_expected = test_split.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }
    test_clean_expected = test_split.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    train_expected = expected.filter { label, type, resolution, npy -> !type.startsWith('test') }
    train_split = train_expected.branch { label, type, resolution, npy ->
        SV: type == 'train_SV'
        clean: type == 'train_clean'
    }
    sv_expected_by_res = train_split.SV
        .map { label, type, resolution, npy -> tuple(resolution, label, npy) }
        .groupTuple(by: 0)
        .map { resolution, labels, npys ->
            def sorted_rows = [labels, npys].transpose().sort { it[0] }
            tuple(resolution, sorted_rows.collect { it[1] })
        }
    clean_expected_by_res = train_split.clean
        .map { label, type, resolution, npy -> tuple(resolution, label, npy) }
        .groupTuple(by: 0)
        .map { resolution, labels, npys ->
            def sorted_rows = [labels, npys].transpose().sort { it[0] }
            tuple(resolution, sorted_rows.collect { it[1] })
        }

    per_res = sv_expected_by_res.join(clean_expected_by_res)
    train_results = per_res
        .combine(sv_mcools_ch.toList())
        .combine(clean_mcools_ch.toList())
        .combine(answers_ch.toList())
        .map { resolution, sv_npy, clean_npy, sv_mcools, clean_mcools, answers ->
            tuple(
                sv_mcools,
                clean_mcools,
                sv_npy,
                clean_npy,
                answers,
                params.num_epoch,
                resolution,
                train_script,
                models_library
            )
        }
        | train_model

    // The one training result channel is used by BOTH inference stages.
    trained_weights = train_results.weights_and_res
        .toSortedList { a, b -> (a[1] as int) <=> (b[1] as int) }
        .map { sorted_tuples -> sorted_tuples.collect { it[0] } }

    resolutions = params.resolutions.collect { it as int }.sort()

    // Stage 1: original map, using the trained weights.
    inference_model(
        trained_weights,
        Channel.of(resolutions),
        Channel.of(file(params.cooler_path)),
        Channel.of(file(params.clean_cooler_path)),
        test_target_expected,
        test_clean_expected,
        stage1_inference_script,
        Channel.of(params.label)
    )

    // Stage 2: use Stage 1's CSV as HiCFoundation input_coords.
    stage2_input = inference_model.out.result.map { input_coords ->
        tuple(params.label, file(params.cooler_path), input_coords)
    }
    stage2 = enhance_with_hicfoundation(
        stage2_input,
        file("${file(projectDir).parent}/HiCFoundation/inference.py"),
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

    stage3_split = stage3_expected.branch { label, type, resolution, npy ->
        target: type == 'test'
        clean: type == 'test_clean'
    }
    stage3_target_expected = stage3_split.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }
    stage3_clean_expected = stage3_split.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    // Reuse the exact same trained_weights channel for Stage 3.
    inference_model_enhanced(
        trained_weights,
        Channel.of(resolutions),
        stage2.enhanced.map { label, enhanced_mcool, csv -> enhanced_mcool },
        Channel.of(file(params.clean_cooler_path)),
        stage3_target_expected,
        stage3_clean_expected,
        stage2.enhanced.map { label, enhanced_mcool, csv -> csv },
        stage3_inference_script,
        Channel.of("${params.label}_with_CHM13_enhanced")
    )
}
