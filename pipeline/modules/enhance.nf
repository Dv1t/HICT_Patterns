process enhance_with_hicfoundation {
    tag "${label}"
    container 'hictpatterns/hicfoundation:latest'
    publishDir "${params.outdir}/enhanced", mode: 'copy'
    accelerator 1

    input:
    tuple val(label), path(input_mcool), path(input_coords)
    path hicfoundation_inference
    path hicfoundation_model

    output:
    tuple val(label), path("${label}_enhanced.mcool"), path("${label}_stage1_breakpoints.csv"), emit: enhanced

    script:
    """
    cooler dump -t chroms "${input_mcool}::/resolutions/1000" > "${label}.genome"
    cooler cp "${input_mcool}::/resolutions/5000" "${label}_5kb.cool"

    python "${hicfoundation_inference}" \\
        --resolution 5000 \\
        --input_coords "${input_coords}" \\
        --input "${label}_5kb.cool" \\
        --batch_size 4 \\
        --num_workers 0 \\
        --genome_id "${label}.genome" \\
        --model_path "${hicfoundation_model}" \\
        --task 3 \\
        --bound 8000 \\
        --input_row_size 224 \\
        --input_col_size 224 \\
        --output "${label}_output"

    cooler zoomify -n 16 -r 15000,25000,50000 \\
        --balance --balance-args '--nproc 16' \\
        -o "${label}_enhanced.mcool" \\
        "${label}_output/HiCFoundation_enhanced.cool"

    cp "${input_coords}" "${label}_stage1_breakpoints.csv"
    """
}