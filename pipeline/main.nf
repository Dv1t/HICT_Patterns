params.generate_npy_script = 'scripts/generate_npy.py'
params.train_model_script = 'scripts/train_model.py'
params.models_library_scripts = 'scripts/models.py'
params.run_model_script = 'scripts/run_model.py'


def required_params = [
    'cooler_path',
    'clean_cooler_path',
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


process calculate_expected {
    publishDir "${params.outdir}/expected", mode:'copy'
    container 'vitdrav/calculate_expected:latest'

    input:
    tuple val(label), val(type), path(mcool), val(resolution), path(script)

    output:
    tuple val(label), val(type), val(resolution), path("${label}_${type}_exp_${resolution}.npy")

    script:
    """
    cooltools compute-expected "${mcool}::/resolutions/${resolution}" \
        -o "${label}_${resolution}.tsv"

    python $script ${label}_${resolution}.tsv "${label}_${type}" $resolution
    """
}


process train_model {
    container 'vitdrav/hict_model:latest'
    publishDir "${params.outdir}/training", mode:'copy'

    input:
    tuple path(sv_mcools),
          path(clean_mcools),
          path(sv_expected),
          path(clean_expected),
          path(answers),
          val(num_epoch),
          val(resolution),
          path(train_script),
          path(models_library)

    output:
    tuple path("*.pt"), val(resolution), emit: weights_and_res
    path "*.log", emit: logs

    script:
    """
    python $train_script "$sv_mcools" "$clean_mcools" "$sv_expected" "$clean_expected" "$answers" $num_epoch $resolution > ${resolution}.log
    """
}

process inference_model {
    container 'vitdrav/hict_model:latest'
    publishDir "${params.outdir}/results", mode:'copy'

    input:
        path weights
        val resolutions_array
        path mcool
        path clean_mcool
        path target_expected
        path clean_expected
        path inference_script
    output:
        path "*.csv", emit: result
        path "*.log", emit: log
    script:
    """ 
        python $inference_script $mcool $clean_mcool "$target_expected" "$clean_expected" "$resolutions_array" "$weights" \
        $params.step $params.labels_cutoff detected_breakpoints.csv > inference_logs.log
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
            def (label, mcool_path_str, clean_mcool_path_str, answer) = line.trim().split(/\t/)
            tuple(label, file(mcool_path_str), file(clean_mcool_path_str), answer)
        }

    // Split SV / Clean


    sv_meta_ch    = train_meta_ch.map{label, sv_mcool, clean_mcool, a -> tuple(label, sv_mcool, a)}
    clean_meta_ch = train_meta_ch.map{label, sv_mcool, clean_mcool, a ->  tuple(label, clean_mcool, a)}

    // Collect static lists


    sv_mcools_ch = sv_meta_ch
        .map { label, mcool, a -> [label, mcool] }
        .toList()                                    // emits one List of tuples
        .map { rows ->
            rows.sort { a, b -> a[0] <=> b[0] }
                .collect { it[1] }
        }

    clean_mcools_ch = clean_meta_ch
        .map { label, mcool, a -> [label, mcool] }
        .toList()
        .map { rows ->
            rows.sort { a, b -> a[0] <=> b[0] }
                .collect { it[1] }
        }

    answers_ch = train_meta_ch
        .map { l, m, cm, a -> a }
        .filter { it && it != '-' }
        .map { file(it) }
        .collect()

    // Build all coolers (train + test)

    sv_coolers_ch = train_meta_ch.map { l, m, cm, a -> tuple(l, m, 'train_SV') }
    clean_sv_coolers_ch = sv_coolers_ch.concat(train_meta_ch.map { l, m, cm, a -> tuple(l, cm, 'train_clean') })
    test_cooler_ch = Channel.fromList([
        tuple(params.label, file(params.cooler_path), 'test'),
        tuple(params.label, file(params.clean_cooler_path), 'test_clean')
    ])

    all_coolers_ch = clean_sv_coolers_ch.concat(test_cooler_ch)

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
        clean: type == 'test_clean'
    }
    
    sorted_test_target_expected = target_split_expected.target
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    sorted_test_clean_expected = target_split_expected.clean
        .toSortedList { a, b -> (a[2] as int) <=> (b[2] as int) }
        .map { list -> list.collect { it[3] } }

    // Keep train only

    train_expected_ch = expected_ch
        .filter { label, type, res, npy -> type != 'test' }

    // Split SV / Clean expected

    split_expected = train_expected_ch.branch { label, type, res, npy ->
        SV:    type == 'train_SV'
        clean: type == 'train_clean'
    }

    // Group by resolution

    sv_expected_by_res_ch = split_expected.SV
        .map { label, type, res, npy -> tuple(res, label, npy) }
        .groupTuple(by: 0)
        .map { res, labels, npys ->
            def sorted = [labels, npys].transpose().sort { it[0] }
            tuple(res, sorted.collect { it[1] })
        }

    clean_expected_by_res_ch = split_expected.clean
        .map { label, type, res, npy -> tuple(res, label, npy) }
        .groupTuple(by: 0)
        .map { res, labels, npys ->
            def sorted = [labels, npys].transpose().sort { it[0] }
            tuple(res, sorted.collect { it[1] })
        }

    // Combine everything per resolution

    per_res_ch = sv_expected_by_res_ch
        .join(clean_expected_by_res_ch)

    sv_mcools_val    = sv_mcools_ch.toList()
    clean_mcools_val = clean_mcools_ch.toList()
    answers_val      = answers_ch.toList()

    /*
     * ===============================
     * Train model
     * ===============================
    */

    train_results = per_res_ch
        .combine(sv_mcools_val)
        .combine(clean_mcools_val)
        .combine(answers_val)
        .map { res, sv_npy, clean_npy, sv_mcools, clean_mcools, answers ->
            tuple(
                sv_mcools,
                clean_mcools,
                sv_npy,
                clean_npy,
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
        Channel.of(file(params.clean_cooler_path)),
        sorted_test_target_expected,
        sorted_test_clean_expected,
        run_model_script,
    )
}