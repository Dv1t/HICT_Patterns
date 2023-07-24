import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd

resolution = 10000

for prefix_cool, prefix_csv, prefix_file in zip(('_short_', '_short_2_', '_long_', '2_short_', '2_short_2_', '2_long_'),
                  ('_short', '_short_2', '_long', '_short', '_short_2', '_long'),
                  ('', '', '', '2', '2', '2')):
    print(prefix_cool)
    for chr in ("X", 'Chr2', 'Chr3'):
        c = cooler.Cooler(f'data/ZANU/ZANU{prefix_cool}4DN.mcool::/resolutions/{resolution}')
        mat_balanced = c.matrix(balance=False).fetch(chr)
        mat_balanced += 1

        matrix = np.log10(mat_balanced)
        trans = pd.read_csv(f'data/ZANU/transitions{prefix_csv}.csv')
        trans = trans[trans.chr == chr]
        trans['start_res'] = trans.start // resolution
        trans['end_res'] = trans.end // resolution

        max_value = np.max(matrix)
        min_value = np.min(matrix)
        for label, folder in (('+-','plus_minus'), ('-+','minus_plus'), ('++','plus_plus'), ('--','minus_minus')):
            cur_trans = trans[trans.label == label]
            for i, row in cur_trans.iterrows():
                point_start = (row.start_res, row.start_res)
                point_end = (row.end_res, row.end_res)
                point_area_start = matrix[point_start[1]-10:point_start[1]+10, point_start[0]-10:point_start[0]+10]
                point_area_end = matrix[point_end[1]-10:point_end[1]+10, point_end[0]-10:point_end[0]+10]
                plt.imsave(f'datasets/images_center_10/positive/ZANU{prefix_file}_{chr[-1]}_{row.start}_{row.start}.png', point_area_start, cmap='gray', vmax=max_value, vmin=min_value)
                plt.imsave(f'datasets/images_center_10/positive/ZANU{prefix_file}_{chr[-1]}_{row.end}_{row.end}.png', point_area_end, cmap='gray', vmax=max_value, vmin=min_value)


for prefix_cool, prefix_csv, prefix_file in zip(('_short_', '_short_2_', '_medium_', '_long_', '2_short_', '2_short_2_', '2_medium_', '2_long_'),
                  ('_short', '_short_2', '_medium', '_long', '_short', '_short_2', '_medium', '_long'),
                  ('', '', '', '', '2', '2', '2', '2')):
    print(prefix_cool)
    for chr in ("X", 'Chr2', 'Chr3'):
        c = cooler.Cooler(f'data/arab/ARAB{prefix_cool}4DN.mcool::/resolutions/{resolution}')
        mat_balanced = c.matrix(balance=False).fetch(chr)
        mat_balanced += 1

        matrix = np.log10(mat_balanced)
        trans = pd.read_csv(f'data/arab/transitions{prefix_csv}.csv')
        trans = trans[trans.chr == chr]
        trans['start_res'] = trans.start // resolution
        trans['end_res'] = trans.end // resolution

        max_value = np.max(matrix)
        min_value = np.min(matrix)
        for label, folder in (('+-','plus_minus'), ('-+','minus_plus'), ('++','plus_plus'), ('--','minus_minus')):
            cur_trans = trans[trans.label == label]
            for i, row in cur_trans.iterrows():
                point_start = (row.start_res, row.start_res)
                point_end = (row.end_res, row.end_res)
                point_area_start = matrix[point_start[1]-10:point_start[1]+10, point_start[0]-10:point_start[0]+10]
                point_area_end = matrix[point_end[1]-10:point_end[1]+10, point_end[0]-10:point_end[0]+10]
                plt.imsave(f'datasets/images_center_10/positive/ARAB{prefix_file}_{chr[-1]}_{row.start}_{row.start}.png', point_area_start, cmap='gray', vmax=max_value, vmin=min_value)
                plt.imsave(f'datasets/images_center_10/positive/ARAB{prefix_file}_{chr[-1]}_{row.end}_{row.end}.png', point_area_end, cmap='gray', vmax=max_value, vmin=min_value)