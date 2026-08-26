"""
Merge N per-split HiCFoundation_enhanced.pkl outputs (from running inference.py
separately on N breakpoint-CSV splits produced by split_breakpoints.py) back
into a single combined pkl, ready for the normal pkl2others conversion step.

Why element-wise max is correct here (not a double-counting approximation):
  - The diagonal-scan windows in Inference_Dataset are generated unconditionally,
    independent of --input_coords, so every split's run recomputes the SAME
    diagonal-band predictions from the SAME model on the SAME data -> those
    entries are numerically identical (or differ only by float rounding) across
    all N outputs. max() just picks one of the (matching) values, not a sum.
  - The off-diagonal (breakpoint-driven) entries are disjoint across splits,
    since each split only contains its own subset of breakpoints -> max()
    simply keeps each entry as-is; no entry is present in more than one split
    to be "combined" incorrectly.

Usage:
    python merge_enhanced_outputs.py \
        --inputs Gor_test_output_part1/HiCFoundation_enhanced.pkl \
                 Gor_test_output_part2/HiCFoundation_enhanced.pkl \
        --output Gor_test_output/HiCFoundation_enhanced.pkl

Then run the normal downstream conversion (pkl2others) on --output as usual.
"""
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="Paths to the per-split HiCFoundation_enhanced.pkl files")
    parser.add_argument("--output", required=True,
                         help="Path to write the merged combined pkl")
    args = parser.parse_args()

    merged = {}
    for path in args.inputs:
        with open(path, "rb") as f:
            part_dict = pickle.load(f)
        print(f"Loaded {path}: {len(part_dict)} chromosomes")
        for chrom, mat in part_dict.items():
            mat = mat.tocsr()
            if chrom not in merged:
                merged[chrom] = mat
            else:
                merged[chrom] = merged[chrom].maximum(mat)

    for chrom in merged:
        merged[chrom] = merged[chrom].tocoo()
        print(f"  {chrom}: merged nnz = {merged[chrom].nnz}")

    with open(args.output, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote merged output to {args.output}")


if __name__ == "__main__":
    main()
