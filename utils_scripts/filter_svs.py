import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd


data = pd.read_csv(f"data/apes/chm_sv.csv")

resolution = 15000
data['start_res'] = data['start'] // resolution
data['end_res'] = data['end'] // resolution

ape = cooler.Cooler(f'data/apes/Gor_Chm.mcool::/resolutions/{resolution}')

with open(f'data/apes/filtered_chm_sv_wm.csv', mode='w') as output:
    output.write('chr,label,start,end\n')
    current_chr = ''
    for index, row in data.iterrows():
        chr_name = row.chr
        if current_chr!=chr_name:
            matrix_full_chr = ape.matrix(balance=False).fetch(chr_name)
            current_chr = chr_name
            print(current_chr)
        if row['end'] - row['start'] < 500000:
            continue
        matrix_start = matrix_full_chr[row.start_res-48:row.start_res+48, row.start_res-48:row.start_res+48]
        if np.count_nonzero(np.nan_to_num(matrix_start, posinf=2, neginf=2))/9216 < 0.25: #0.25
            continue
        small_matrix_start = matrix_start[22:26, 22:26]
        if np.count_nonzero(np.nan_to_num(small_matrix_start, posinf=2, neginf=2))/16 < 0.5: #0.5
            continue

        matrix_end = matrix_full_chr[row.end_res-48:row.end_res+48, row.end_res-48:row.end_res+48]
        if np.count_nonzero(np.nan_to_num(matrix_end, posinf=2, neginf=2))/9216 < 0.25: #0.25
            continue
        small_matrix_end = matrix_end[22:26, 22:26]
        if np.count_nonzero(np.nan_to_num(small_matrix_end, posinf=2, neginf=2))/16 < 0.5: #0.5
            continue
        
        #output.write(f"{chr_name},{row.label},{row['start']},{row['start']}\n")
        #output.write(f"{chr_name},{row.label},{row['end']},{row['end']}\n")
        output.write(f"{chr_name},{row.label},{row['start']},{row['end']}\n")

