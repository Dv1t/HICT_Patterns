process inference_model {
    container 'hictpatterns/hict_model:latest'
    publishDir "${params.outdir}/results", mode:'copy'

    input:
        path weights
        val resolutions_array
        path mcool
        path clean_mcool
        path target_expected
        path clean_expected
        path inference_script
        val run_label
    output:
        path "*.csv", emit: result
        path "*.log", emit: log
    script:
    """ 
        python $inference_script $mcool $clean_mcool "$target_expected" "$clean_expected" "$resolutions_array" "$weights" \
        $params.step $params.labels_cutoff ${run_label}_detected_breakpoints.csv > inference_logs.log
    """
}

process inference_model_one_map {
    container 'hictpatterns/hict_model:latest'
    publishDir "${params.outdir}/results", mode:'copy'

    input:
        path weights
        val resolutions_array
        path mcool
        path target_expected
        path inference_script
        val run_label
    output:
        path "*.csv", emit: result
        path "*.log", emit: log
    script:
    """ 
        python $inference_script $mcool "$target_expected"  "$resolutions_array" "$weights" \
        $params.step $params.labels_cutoff ${run_label}_detected_breakpoints.csv > inference_logs.log
    """
}

process inference_model_enhanced {
    container 'hictpatterns/hict_model:latest'
    publishDir "${params.outdir}/results", mode:'copy'

    input:
        path weights
        val resolutions_array
        path mcool
        path clean_mcool
        path target_expected
        path clean_expected
        path input_csv
        path inference_script
        val run_label
    output:
        path "*.csv", emit: result
        path "*.log", emit: log
    script:
    """ 
        python $inference_script $mcool $clean_mcool "$target_expected" "$clean_expected" "$resolutions_array" "$weights" \
        $params.step $params.labels_cutoff ${run_label}_detected_breakpoints.csv "$input_csv" > inference_logs.log
    """
}