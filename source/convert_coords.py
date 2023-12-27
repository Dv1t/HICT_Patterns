import numpy as np
import cooler
import os
import pandas as pd

source_file = 'result.csv'
cooler_file = 'data/dong_vs_gambiae/dong_colluzzii_4DN.mcool'
local_path = os.getcwd()
detected = np.genfromtxt(f"{local_path}/{source_file}", delimiter=",")
c = cooler.Cooler('data/dong_vs_gambiae/dong_colluzzii_4DN.mcool::/resolutions/500000')
results = {'coord_x':[], 'coord_y':[], 'chrom_x':[], 'chrom_y':[]}
for d in detected:
    chr_x_span = 0
    for chr_len, chr_name in zip(c.chromsizes, c.chromnames):
        if d[0]-chr_x_span < chr_len:
            x = d[0]-chr_x_span
            chr_x = chr_name
            break
        else:
            chr_x_span+=chr_len
    chr_y_span = 0
    for chr_len, chr_name in zip(c.chromsizes, c.chromnames):
        if d[1]-chr_y_span < chr_len:
            y = d[1]-chr_y_span
            chr_y = chr_name
            break
        else:
            chr_y_span+=chr_len
    results['coord_x'].append(int(x))
    results['chrom_x'].append(chr_x)
    results['coord_y'].append(int(y))
    results['chrom_y'].append(chr_y)

result_df = pd.DataFrame.from_dict(results)
result_df.to_csv('result_dong_colluzzii.csv')

