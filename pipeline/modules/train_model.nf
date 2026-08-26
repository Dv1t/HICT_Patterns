process train_model {
    container 'hictpatterns/hict_model:latest'
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

process train_model_one_map {
    container 'hictpatterns/hict_model:latest'
    publishDir "${params.outdir}/training", mode:'copy'

    input:
    tuple path(sv_mcools),
          path(sv_expected),
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
    python $train_script "$sv_mcools" "$sv_expected" "$answers" $num_epoch $resolution > ${resolution}.log
    """
}