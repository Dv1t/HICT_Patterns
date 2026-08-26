
process calculate_expected {
    publishDir "${params.outdir}/expected", mode:'copy'
    container 'hictpatterns/calculate_expected:latest'

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
