def required_params = [
    'cooler_path',
    'clean_cooler_path',
    'label',
    'weights_paths',
    'input_csv'
]

required_params.each { p ->
    if (!params."$p")
        error "Missing required parameter: --${p}"
}

params.outdir = "${params.label}_output"

log.info """
HICT PATTERNS  P I P E L I N E
===========================================
label                    : ${params.label}
cooler_path              : ${params.cooler_path}
resolutions              : ${params.resolutions}
input_csv                : ${params.input_csv}
run_model_script         : ${params.run_model_script}
""".stripIndent().trim()

include { inference_model_enhanced } from './modules/run_model.nf'
include { calculate_expected } from './modules/calculate_npy.nf'

workflow {
    generate_script = file(params.generate_npy_script)
    models_library = file(params.models_library_scripts)
    run_model_script = file(params.run_model_script)

    
    weights_ch = Channel
        .fromPath(params.weights_paths)
        .splitText()
        .map { line ->
            def (weight_path_str, resolution) = line.trim().split(/\t/)
            tuple(file(weight_path_str), resolution)
        }

    test_cooler_ch = Channel.fromList([
        tuple(params.label, file(params.cooler_path), 'test'),
        tuple(params.label, file(params.clean_cooler_path), 'test_clean')
    ])

    /*
     * ===============================
     * Compute expected
     * ===============================
    */

    expected_ch = test_cooler_ch
        .combine(Channel.fromList(params.resolutions))
        .map { label, mcool, type, res ->
            tuple(label, type, mcool, res, generate_script)
        }
        | calculate_expected

    
    // Keep test only

    test_expected_ch = expected_ch
        .filter { label, type, res, npy -> type.startsWith('test') }
    
    // Split target / Clean expected

    target_split_expected = test_expected_ch.branch { label, type, res, npy ->
        target:    type == 'test'
        clean: type == 'test_clean'
    }
    
    sorted_test_target_expected = target_split_expected.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    sorted_test_clean_expected = target_split_expected.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    // Sort resolution from high to low, to process maps consistently

    sorted_resolutions_val = params.resolutions.collect { it as int }.sort()
    sorted_weights_list_ch = weights_ch
        .toSortedList { a, b -> (a[1] as int) <=> (b[1] as int) }
        .map { sorted_tuples -> 
            sorted_tuples.collect { it[0] } 
        }

    inference_model_enhanced (
        sorted_weights_list_ch,
        Channel.of(sorted_resolutions_val),
        Channel.of(file(params.cooler_path)),
        Channel.of(file(params.clean_cooler_path)),
        sorted_test_target_expected,
        sorted_test_clean_expected,
        Channel.of(file(params.input_csv)),
        run_model_script,
        Channel.of(params.label)
    )
}