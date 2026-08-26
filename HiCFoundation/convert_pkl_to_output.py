"""
Convert a (merged) HiCFoundation_enhanced.pkl into the final output format,
using the same pkl2others utility main_worker_sv_detection.py already calls
internally. Use this on the pkl produced by merge_enhanced_outputs.py, since
each individual split run already wrote its own (incomplete) converted file
as a side effect of running the full pipeline.

Usage:
    python convert_pkl_to_output.py \
        --input_pkl Gor_test_output/HiCFoundation_enhanced.pkl \
        --output Gor_test_output/HiCFoundation_enhanced.cool \
        --resolution 5000 \
        --genome_id Gor_test.genome
"""
import argparse
import os
from ops.file_format_convert import pkl2others


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True,
                         help="Path to the (merged) HiCFoundation_enhanced.pkl")
    parser.add_argument("--output", required=True,
                         help="Path to write the converted output file, e.g. "
                              "HiCFoundation_enhanced.cool - extension determines format, "
                              "same as main_worker_sv_detection.py's own logic")
    parser.add_argument("--resolution", type=int, required=True,
                         help="Same --resolution value used for the original inference.py runs")
    parser.add_argument("--genome_id", required=True,
                         help="Same --genome_id value used for the original inference.py runs")
    args = parser.parse_args()

    assert os.path.exists(args.input_pkl), f"input pkl does not exist: {args.input_pkl}"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    pkl2others(args.input_pkl, args.output, args.resolution, args.genome_id)

    if os.path.exists(args.output):
        print(f"Conversion finished. Output written to {args.output}")
    else:
        print("Error: file conversion failed - output file was not created.")
        print(f"The merged .pkl is still available at {args.input_pkl} "
              f"if you need to convert it manually.")


if __name__ == "__main__":
    main()
