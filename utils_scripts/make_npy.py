import pandas as pd
import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# cooltools compute-expected data/apes/Chm_Gor_SV_4DN.mcool::/resolutions/1000 -o Chm_Gor_SV_1kb.tsv

for map in ['Shiamang_SV', 'Shiamang_SV_2' ,'Shiamang_Chm', 'CHM13']:
    for res, res_label in zip([15000, 25000, 50000],['15kb', '25kb', '50kb']):
        expected = pd.read_csv(f'/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/{map}_{res_label}.tsv', sep="\t")

        expectedsum = expected.groupby(["diag"]).agg({"n_valid": "sum", "balanced.sum": "sum"})
        expectedsum["balanced.avg"] = expectedsum["balanced.sum"] / expectedsum["n_valid"]

        target_res = res
        v = np.log(expectedsum["balanced.avg"].values)
        v = v[2: np.min(np.argwhere(~np.isfinite(v[2:])))]
        sv0 = lowess(
            v[int(48 / (res / target_res)) :], np.log(np.arange(int(48 / (res / target_res)), len(v)) + 1), frac=0.01
        )[:, 1]
        sv2 = lowess(
            v[int(48 / (res / target_res)) :], np.log(np.arange(int(48 / (res / target_res)), len(v)) + 1), frac=0.1
        )[:, 1]
        sv = np.hstack(
            [
                v[: int(48 / (res / target_res))],
                sv0[: int(48 / (res / target_res))],
                sv2[int(48 / (res / target_res)) :],
            ]
        )
        np.save(arr=sv, file=f"{map}_exp_{res_label}.npy")