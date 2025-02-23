import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd
import sys
from help_functions import get_chromosome_coords
import argparse


def visualize_chromosomewise(file_path, resolution, chr_name, chr_name_second, result_file_name, image_file_name):
    c = cooler.Cooler(f'{file_path}::/resolutions/{resolution}')
    matrix = np.log2(c.matrix(balance=False).fetch(chr_name, chr_name_second)+1)

    results = pd.read_csv(result_file_name)
    bp_1 =  [x//resolution for chr_number, x in get_chromosome_coords(results.bp_1, c.chromsizes, 1) if chr_number==c.chromnames.index(chr_name)]
    bp_2 =  [y//resolution for chr_number, y in get_chromosome_coords(results.bp_2, c.chromsizes, 1) if chr_number==c.chromnames.index(chr_name_second)]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(matrix, cmap='Greens')
    plt.scatter(bp_1, bp_2, s=1, c='#E9967A')
    plt.savefig(f'{image_file_name}.png', dpi=500)


def visualize_all(file_path, resolution, result_file_name, image_size):
    c = cooler.Cooler(f'{file_path}::/resolutions/{resolution}')
    matrix = c.matrix(balance=False)

    results = pd.read_csv(result_file_name)
    results.bp_1_res = results.bp_1//resolution
    results.bp_2_res = results.bp_2//resolution
    pad = image_size // 2
    for _, row in results.iterrows():
        mat = np.log2(matrix[row.bp_1_res-pad:row.bp_1_res+pad, row.bp_2_res-pad:row.bp_2_res+pad]+1)
        plt.imsave(f'{row.bp_1}_{row.bp_2}_{row.label}.png',
            mat, cmap='Greens')


def main(cmdline=None):
    if cmdline is None:
        cmdline=sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='This utilite visualize SVs on Hi-C map')
    parser.add_argument('file_path', type=str, help=
                        "Path to HiC file - .mcool format")
    parser.add_argument("--by_chr",'-CHR', action=argparse.BooleanOptionalAction, help="Provide if want to visualize whole chromosome, otherwise all SVs will be displayed")
    parser.add_argument("--resolution", '-R', required=True, type=int, help="Resolution for Hi-C map visualization")
    parser.add_argument('--file_name','-F', required=True, type=str, help="Path to file with SVs coordinates")
    parser.add_argument('--chr', '-C', required=False, type=str, help="Name of chromosome to display if visualizing whole chromosome")
    parser.add_argument('--chr_second', required=False, type=str, help="Name of second chromosome to display if visualizing interchromosomal SVs")
    parser.add_argument('--output', '-O', required=False, type=str, help="Output file name for whole chromosome visualization, default is 'result'")
    parser.add_argument("--image_size", '-S', required=False, type=int, help="Size of an image if visualizing all SVs, default is 50 (produces images 50x50)")

    parser.set_defaults(output='result')
    parser.set_defaults(chr_second='')

    parser.set_defaults(image_size=50)

    args = parser.parse_args(cmdline)
    file_path = args.file_path
    by_chr = args.by_chr
    resolution = args.resolution
    chr_name = args.chr
    chr_name_second = args.chr_second
    result_file_name = args.file_name
    output_name = args.output
    image_size = args.image_size

    if len(chr_name_second) == 0:
        chr_name_second = chr_name
    
    if by_chr:
        visualize_chromosomewise(file_path, resolution, chr_name, chr_name_second, result_file_name, output_name)
    else:
        visualize_all(file_path, resolution, result_file_name, image_size)


if __name__ == '__main__':
    main()

