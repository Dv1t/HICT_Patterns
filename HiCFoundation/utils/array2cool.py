import sys
import os
import pickle
import numpy as np
import pandas as pd
import cooler


# How many pixels to hand to cooler at a time. This bounds the DataFrame that
# exists in RAM at any moment; it has nothing to do with the total pixel count.
PIXEL_CHUNK = 5_000_000


def array2sparse(array):
    """Convert a numpy array to a scipy sparse coo_matrix."""
    from scipy.sparse import coo_matrix
    row, col = np.where(array)
    data = array[row, col]
    return coo_matrix((data, (row, col)), shape=array.shape)


def _chrom_pair(chrom_name, mode):
    if mode == 0 or mode == 2:
        chrom1, chrom2 = chrom_name.split('_')
    else:
        chrom1 = chrom_name
        chrom2 = chrom_name
    if "chr" not in chrom1:
        chrom1 = "chr" + chrom1
    if "chr" not in chrom2:
        chrom2 = "chr" + chrom2
    return chrom1, chrom2


def _build_bins(data, resolution, mode):
    """Chromosome sizes and per-chromosome bin offsets, in sorted key order."""
    chromsizes = {"name": [], "length": []}
    chromosize_add_dict = {}
    accumulate_index = 0
    for chrom_name in sorted(data.keys()):
        chrom1, chrom2 = _chrom_pair(chrom_name, mode)
        if chrom1 != chrom2:
            continue
        cur_array = data[chrom_name]
        chromsizes['name'].append(chrom1)
        chromsizes['length'].append(resolution * cur_array.shape[0])
        chromosize_add_dict[chrom1] = accumulate_index
        accumulate_index += cur_array.shape[0]
    print("collecting bin dict size", chromsizes)
    chrom_dict = pd.DataFrame.from_dict(chromsizes).set_index("name")['length']
    return cooler.binnify(chrom_dict, resolution), chromosize_add_dict


def _iter_pixels(data, chromosize_add_dict, mode, chunk_pixels):
    """
    Yield pixel DataFrames a chunk at a time.

    The original built three genome-wide python lists via `list(ndarray)`,
    which boxes every element into a separate object: ~118 bytes per pixel
    versus 12 for the equivalent numpy arrays, roughly 10x, held for the whole
    genome at once and then duplicated again by DataFrame.from_dict. Streaming
    chunks keeps only `chunk_pixels` rows materialised at any moment.
    """
    for key in sorted(data.keys()):
        chrom1, chrom2 = _chrom_pair(key, mode)
        if chrom1 not in chromosize_add_dict or chrom2 not in chromosize_add_dict:
            continue
        print("processing", chrom1, chrom2, "...")
        matrix = data[key]
        if mode >= 2:
            matrix = array2sparse(matrix)

        # copies, not in-place +=, so the caller's matrices are left untouched
        # (the original mutated matrix.row/matrix.col of the loaded dict)
        row = matrix.row.astype(np.int64) + chromosize_add_dict[chrom1]
        col = matrix.col.astype(np.int64) + chromosize_add_dict[chrom2]
        val = matrix.data.astype(np.float64, copy=True)

        # cooler expects upper-triangle pixels (bin1_id <= bin2_id)
        swap = row > col
        if swap.any():
            row[swap], col[swap] = col[swap], row[swap]

        data[key] = None          # release this chromosome as we go
        matrix = None

        for start in range(0, row.size, chunk_pixels):
            stop = min(start + chunk_pixels, row.size)
            yield pd.DataFrame({
                "bin1_id": row[start:stop],
                "bin2_id": col[start:stop],
                "count": val[start:stop],
            })
        del row, col, val, swap


def array2cool(input_array_pickle, output_cool, resolution, refer_genome_name, mode,
               chunk_pixels=PIXEL_CHUNK, data=None):
    """
    Convert a dict of per-chromosome matrices to a .cool file.

    input_array_pickle: path to the pickle, OR pass an already-loaded dict as
                        `data` to skip a redundant unpickle of a large file.
    """
    if data is None:
        with open(input_array_pickle, 'rb') as f:
            data = pickle.load(f)

    output_dir = os.path.dirname(output_cool)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    bins, chromosize_add_dict = _build_bins(data, resolution, mode)

    print("creating cool file...")
    # ordered=False lets cooler sort/merge the incoming pixel chunks on disk
    # rather than requiring the whole pixel table sorted in memory first.
    cooler.create_cooler(
        output_cool,
        bins=bins,
        pixels=_iter_pixels(data, chromosize_add_dict, mode, chunk_pixels),
        dtypes={'count': float},
        assembly=refer_genome_name,
        ordered=False,
    )


"""
Usage
```
python3 array2cool.py [input.pkl] [output.cool] [resolution] [refer_genome_name] [mode]
```
The input pickle should be in a pickle file as dict: [chrom1_chrom2]:[array] format
for common mode. Here array should be scipy sparse array.
For intra-chromosome only, the dict format can be [chrom]:[array] in pickle files.
[mode]: 0: all chromosome mode (scipy sparse array); 1: intra-chromosome mode
(scipy sparse array); 2: all chromosome mode (numpy array); 3: intra-chromosome
mode (numpy array).
"""
if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: python3 array2cool.py [input.pkl] [output.cool] [resolution] [refer_genome_name] [mode]')
        sys.exit(1)
    input_array_pickle = os.path.abspath(sys.argv[1])
    output_cool = os.path.abspath(sys.argv[2])
    resolution = int(sys.argv[3])
    refer_genome_name = str(sys.argv[4])
    mode = int(sys.argv[5])
    array2cool(input_array_pickle, output_cool, resolution, refer_genome_name, mode)
