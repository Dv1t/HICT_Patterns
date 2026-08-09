import argparse
#!/usr/bin/env python3

import pandas as pd
from collections import defaultdict

BIN_SIZE = 50000
WINDOW_BINS = 48

HALF_SIZE = BIN_SIZE * WINDOW_BINS // 2  # 1,200,000


def overlap(a, b):
    return (
        a["x1"] <= b["x2"]
        and a["x2"] >= b["x1"]
        and a["y1"] <= b["y2"]
        and a["y2"] >= b["y1"]
    )


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra

def process(input_path):
    df = pd.read_csv(input_path)

    df["x1"] = df["x"] - HALF_SIZE
    df["x2"] = df["x"] + HALF_SIZE
    df["y1"] = df["y"] - HALF_SIZE
    df["y2"] = df["y"] + HALF_SIZE

    common_results = []
    union_results = []

    for chrom, group in df.groupby("chr"):

        group = group.reset_index(drop=True)

        uf = UnionFind(len(group))

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if overlap(group.iloc[i], group.iloc[j]):
                    uf.union(i, j)

        components = defaultdict(list)

        for i in range(len(group)):
            components[uf.find(i)].append(i)

        for comp_id, indices in enumerate(components.values(), start=1):

            subset = group.iloc[indices]

            # intersection of all rectangles
            ix1 = subset["x1"].max()
            ix2 = subset["x2"].min()
            iy1 = subset["y1"].max()
            iy2 = subset["y2"].min()

            if ix1 <= ix2 and iy1 <= iy2:
                common_results.append({
                    "chr": chrom,
                    "component": comp_id,
                    "x_start": ix1,
                    "x_end": ix2,
                    "y_start": iy1,
                    "y_end": iy2,
                    "n_rectangles": len(subset)
                })

            # overall covered area
            ux1 = subset["x1"].min()
            ux2 = subset["x2"].max()
            uy1 = subset["y1"].min()
            uy2 = subset["y2"].max()

            union_results.append({
                "chr": chrom,
                "component": comp_id,
                "x_start": ux1,
                "x_end": ux2,
                "y_start": uy1,
                "y_end": uy2,
                "n_rectangles": len(subset)
            })

    pd.DataFrame(common_results).to_csv(
        "common_overlapping_area.csv",
        index=False
    )

    pd.DataFrame(union_results).to_csv(
        "all_overlapping_matrices_area.csv",
        index=False
    )

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group overlapping breakpoint detections and write union/intersection rectangles."
    )
    parser.add_argument("input_csv", help="CSV with chr,x,y detection rows.")
    parser.add_argument(
        "-o",
        "--output-csv",
        help="Output CSV path. Defaults to <input stem>_postprocessed.csv next to the input.",
    )
    parser.add_argument("--resolution", type=int, default=50000, help="Bin size in bp. Default: 50000.")
    parser.add_argument("--matrix-bins", type=int, default=48, help="Detection matrix width/height in bins. Default: 48.")
    return parser.parse_args()


def main():
    args = parse_args()
    process(args.input_csv)

if __name__ == "__main__":
    main()
