def required_params = [
    'cooler_path',
    'input_train_cooler_paths',
    'label'
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
input_train_cooler_paths : ${params.input_train_cooler_paths}
resolutions              : ${params.resolutions}
""".stripIndent().trim()


include { inference_model_one_map as inference_model} from './modules/run_model.nf'
include { calculate_expected } from './modules/calculate_npy.nf'
include { train_model_one_map as train_model } from './modules/train_model.nf'

process rename_answer {
    tag "$label"
    // Executes locally and instantly, no container needed
    executor 'local' 

    input:
    tuple val(label), path(answer_file)

    output:
    // Emit the label and the newly named file
    tuple val(label), path("${label}_${answer_file.name}")

    script:
    """
    # Create a copy with the label attached to guarantee a unique name
    cp "${answer_file}" "${label}_${answer_file.name}"
    """
}

workflow {
    generate_script = file(params.generate_npy_script)
    train_script = file(params.train_model_script)
    models_library = file(params.models_library_scripts)
    run_model_script = file(params.run_model_script)
    // Load training metadata

    train_meta_ch = Channel
        .fromPath(params.input_train_cooler_paths)
        .splitText()
        .map { line ->
            def (label, mcool_path_str, answer) = line.trim().split(/\t/)
            tuple(label, file(mcool_path_str), answer)
        }
        

    // Split SV / Clean


    sv_meta_ch    = train_meta_ch.map{label, sv_mcool, a -> tuple(label, sv_mcool, a)}

    // Collect static lists


    sv_mcools_ch = sv_meta_ch
        .map { label, mcool, a -> [label, mcool] }
        .toList()                                    // emits one List of tuples
        .map { rows ->
            rows.sort { a, b -> a[0] <=> b[0] }
                .collect { it[1] }
        }

    // 1. Filter out empty answers and map to a tuple
    answers_to_rename_ch = train_meta_ch
        .filter { l, m, a -> a && a != '-' }
        .map { l, m, a -> tuple(l, file(a)) }

    // 2. Pass through the renamer, gather, sort, and extract the files
    answers_ch = rename_answer(answers_to_rename_ch)
        .toList()
        .map { rows ->
            // Sort alphabetically by label (a[0]) to perfectly match sv_mcools_ch
            rows.sort { a, b -> a[0] <=> b[0] }
                .collect { it[1] } // Extract just the renamed file
        }

    // Build all coolers (train + test)

    sv_coolers_ch = train_meta_ch.map { l, m, a -> tuple(l, m, 'train_SV') }
    test_cooler_ch = Channel.fromList([
        tuple(params.label, file(params.cooler_path), 'test')
    ])

    all_coolers_ch = sv_coolers_ch.concat(test_cooler_ch)

    /*
     * ===============================
     * Compute expected
     * ===============================
    */

    expected_ch = all_coolers_ch
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
    }
    
    sorted_test_target_expected = target_split_expected.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    // Keep train only

    train_expected_ch = expected_ch
        .filter { label, type, res, npy -> type != 'test' }

    // Split SV / Clean expected

    split_expected = train_expected_ch.branch { label, type, res, npy ->
        SV:    type == 'train_SV'
    }

    // Group by resolution

    sv_expected_by_res_ch = split_expected.SV
        .map { label, type, res, npy -> tuple(res, label, npy) }
        .groupTuple(by: 0)
        .map { res, labels, npys ->
            def sorted = [labels, npys].transpose().sort { it[0] }
            tuple(res, sorted.collect { it[1] })
        }

    // Combine everything per resolution

    per_res_ch = sv_expected_by_res_ch

    sv_mcools_val    = sv_mcools_ch.toList()
    answers_val      = answers_ch.toList()

    /*
     * ===============================
     * Train model
     * ===============================
    */

    train_results = per_res_ch
        .combine(sv_mcools_val)
        .combine(answers_val)
        .map { res, sv_npy, sv_mcools, answers ->
            tuple(
                sv_mcools,
                sv_npy,
                answers,
                params.num_epoch,
                res,
                train_script,
                models_library
            )
        }
        | train_model
    
    weights_ch = train_results.weights_and_res

    // Sort resolution from high to low, to process maps consistently

    sorted_resolutions_val = params.resolutions.collect { it as int }.sort()
    sorted_weights_list_ch = weights_ch
        .toSortedList { a, b -> (a[1] as int) <=> (b[1] as int) }
        .map { sorted_tuples -> 
            sorted_tuples.collect { it[0] } 
        }

    inference_model (
        sorted_weights_list_ch,
        Channel.of(sorted_resolutions_val),
        Channel.of(file(params.cooler_path)),
        sorted_test_target_expected,
        run_model_script,
    )
}