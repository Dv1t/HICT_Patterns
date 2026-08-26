"""
Split a breakpoint CSV (columns: chr,x,y - same format used by --input_coords)
into N roughly-balanced parts, so inference.py can be run N times, each with
a smaller off-diagonal breakpoint set, to fit within a RAM budget.

Round-robin assignment (row i -> part i % N) rather than contiguous blocks:
breakpoint CSVs are typically sorted by chromosome/position, so a contiguous
block could dump a whole cluster of nearby breakpoints into a single part,
defeating the point of balancing memory load. Round-robin spreads clustered
regions evenly across all parts.

Usage:
    python split_breakpoints.py --input Gor_test_detected_breakpoints.csv \
        --num_splits 2 --output_prefix Gor_test_detected_breakpoints_part

Produces:
    Gor_test_detected_breakpoints_part1.csv
    Gor_test_detected_breakpoints_part2.csv
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the full breakpoint CSV (chr,x,y,...)")
    parser.add_argument("--num_splits", type=int, default=2, help="Number of parts to split into")
    parser.add_argument("--output_prefix", required=True,
                         help="Output files will be named <prefix>1.csv, <prefix>2.csv, ...")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    before = len(df)

    # dedup first (exact duplicate rows just waste compute/memory in every
    # split identically) - safe no-op if already deduped upstream
    df = df.drop_duplicates(subset=["chr", "x", "y"]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Deduped input breakpoints: {before} -> {after}")

    part_sizes = [0] * args.num_splits
    for split_idx in range(args.num_splits):
        part_df = df.iloc[split_idx::args.num_splits]
        out_path = f"{args.output_prefix}{split_idx + 1}.csv"
        part_df.to_csv(out_path, index=False)
        part_sizes[split_idx] = len(part_df)
        print(f"  {out_path}: {len(part_df)} breakpoints")

    print(f"Total: {after} breakpoints split into {args.num_splits} parts "
          f"(sizes: {part_sizes})")


if __name__ == "__main__":
    main()
