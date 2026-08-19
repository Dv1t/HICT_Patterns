process prepare_hicfoundation_input {
    tag "${label}"
    container 'hictpatterns/hict_model:latest'

    input:
    tuple val(label), path(input_mcool), path(input_coords)

    output:
    tuple val(label), path("${label}_5kb.cool"), path("${label}.genome"), path(input_coords), emit: prepared

    script:
    """
    cooler dump -t chroms "${input_mcool}::/resolutions/1000" > "${label}.genome"
    cooler cp "${input_mcool}::/resolutions/5000" "${label}_5kb.cool"
    """
}

process run_hicfoundation_inference {
    tag "${label}"
    cpus 1
    accelerator 1

    input:
    tuple val(label), path(input_cool), path(genome), path(input_coords)
    path hicfoundation_inference
    path hicfoundation_model

    output:
    tuple val(label), path("${label}_output/HiCFoundation_enhanced.cool"), path(input_coords), emit: enhanced_cool

    script:
    """
    /opt/conda/bin/conda run -n HiCFoundation python "${hicfoundation_inference}" \\
        --resolution 5000 \\
        --input_coords "${input_coords}" \\
        --input "${input_cool}" \\
        --batch_size 4 \\
        --num_workers 0 \\
        --genome_id "${genome}" \\
        --model_path "${hicfoundation_model}" \\
        --task 3 \\
        --bound 0 \\
        --input_row_size 224 \\
        --input_col_size 224 \\
        --output "${label}_output"
    """
}

process create_enhanced_mcool {
    tag "${label}"
    container 'hictpatterns/hict_model:latest'
    publishDir "${params.outdir}/enhanced", mode: 'copy'

    input:
    tuple val(label), path(enhanced_cool), path(input_coords)

    output:
    tuple val(label), path("${label}_enhanced.mcool"), path("${label}_stage1_breakpoints.csv"), emit: enhanced

    script:
    """
    cooler zoomify -n 8 -r 15000,25000,50000 \\
        --balance --balance-args '--nproc 8' \\
        -o "${label}_enhanced.mcool" \\
        "${enhanced_cool}"

    cp "${input_coords}" "${label}_stage1_breakpoints.csv"
    """
}

workflow enhance_with_hicfoundation {
    take:
    stage2_input
    hicfoundation_inference
    hicfoundation_model

    main:
    prepared = prepare_hicfoundation_input(stage2_input)
    enhanced_cool = run_hicfoundation_inference(
        prepared,
        hicfoundation_inference,
        hicfoundation_model
    )
    final_enhanced = create_enhanced_mcool(enhanced_cool)

    emit:
    enhanced = final_enhanced
}
