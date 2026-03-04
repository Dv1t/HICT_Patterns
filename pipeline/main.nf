params.generate_npy_script = 'scripts/generate_npy.py'
params.train_model_script = 'scripts/train_model.py'
params.models_library_scripts = 'scripts/models.py'


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
    path "*.pt",  emit: weight
    path "*.log", emit: logs

    script:
    """
    python $train_script "$sv_mcools" "$clean_mcools" "$sv_expected" "$clean_expected" "$answers" $num_epoch $resolution > ${resolution}.log
    """
}



workflow {
    generate_script = file(params.generate_npy_script)
    train_script = file(params.train_model_script)
    models_library = file(params.models_library_scripts)

    /*
     * ===============================
     * 1️⃣ Load training metadata
     * ===============================
     */

    train_meta_ch = Channel
        .fromPath(params.input_train_cooler_paths)
        .splitText()
        .map { line ->
            def (label, mcool_path_str, clean_mcool_path_str, answer) = line.trim().split(/\t/)
            tuple(label, file(mcool_path_str), file(clean_mcool_path_str), answer)
        }
    /*
     * ===============================
     * 2️⃣ Split SV / Clean
     * ===============================
     */

    sv_meta_ch    = train_meta_ch.map{label, sv_mcool, clean_mcool, a -> tuple(label, sv_mcool, a)}
    clean_meta_ch = train_meta_ch.map{label, sv_mcool, clean_mcool, a ->  tuple(label, clean_mcool, a)}
    /*
     * ===============================
     * 3️⃣ Collect static lists
     * ===============================
     */

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

    /*
     * ===============================
     * 4️⃣ Build all coolers (train + test)
     * ===============================
     */

    sv_coolers_ch = train_meta_ch.map { l, m, cm, a -> tuple(l, m, 'train_SV') }
    clean_sv_coolers_ch = sv_coolers_ch.concat(train_meta_ch.map { l, m, cm, a -> tuple(l, cm, 'train_clean') })
    test_cooler_ch = Channel.of(
        tuple(params.label, file(params.cooler_path), 'test')
    )

    all_coolers_ch = clean_sv_coolers_ch.concat(test_cooler_ch)

    /*
     * ===============================
     * 5️⃣ Compute expected
     * ===============================
     */

    expected_ch = all_coolers_ch
        .combine(Channel.fromList(params.resolutions))
        .map { label, mcool, type, res ->
            tuple(label, type, mcool, res, generate_script)
        }
        | calculate_expected

    /*
     * ===============================
     * 6️⃣ Keep train only
     * ===============================
     */

    train_expected_ch = expected_ch
        .filter { label, type, res, npy -> type != 'test' }

    /*
     * ===============================
     * 7️⃣ Split SV / Clean expected
     * ===============================
     */

    split_expected = train_expected_ch.branch { label, type, res, npy ->
        SV:    type == 'train_SV'
        clean: type == 'train_clean'
    }

    /*
     * ===============================
     * 8️⃣ Group by resolution
     * ===============================
     */

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

    /*
     * ===============================
     * 9️⃣ Combine everything per resolution
     * ===============================
     */
    per_res_ch = sv_expected_by_res_ch
        .join(clean_expected_by_res_ch)

    sv_mcools_val    = sv_mcools_ch.toList()
    clean_mcools_val = clean_mcools_ch.toList()
    answers_val      = answers_ch.toList()

    per_res_ch
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
}