import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd
import random


def make_samples(path, clr_map, resolution, centered: bool):
    local_path = 'D:/Study/HICT/HICT_Patterns/'

    def save_sample_by_row(row, label):
        point_uncentered = (row.start_res, row.end_res)
        point_area_uncentered = matrix[point_uncentered[1] - 10:point_uncentered[1] + 10,
                                point_uncentered[0] - 10:point_uncentered[0] + 10]
        plt.imsave(f'{path}/positive/{label}{prefix_file}_{chr[-1]}_{row.start}_{row.end}.png',
                   point_area_uncentered, cmap=clr_map, vmax=max_value, vmin=min_value)

    def make_negative_samples(number, matrix, label):
        was_added = set()
        for _ in range(number):
            i = random.randint(10, matrix.shape[0] - 10)
            while i in was_added:
                i = random.randint(10, matrix.shape[0] - 10)
            was_added.add(i)
            point = (i, i)
            point_area = matrix[point[1] - 10:point[1] + 10, point[0] - 10:point[0] + 10]
            plt.imsave(
                f'{path}/negative/{label}_{i * resolution}_{i * resolution}.png',
                point_area, cmap=clr_map, vmax=max_value, vmin=min_value)

    samples_count = 0
    for prefix_cool, prefix_csv, prefix_file in zip(
            ('_short_', '_short_2_', '_long_', '2_short_', '2_short_2_', '2_long_'),
            ('_short', '_short_2', '_long', '_short', '_short_2', '_long'),
            ('', '', '', '2', '2', '2')):
        print(prefix_cool)
        for chr in ("X", 'Chr2', 'Chr3'):
            c = cooler.Cooler(f'{local_path}data/ZANU/ZANU{prefix_cool}4DN.mcool::/resolutions/{resolution}')
            mat_balanced = c.matrix(balance=False).fetch(chr)
            mat_balanced += 1

            matrix = np.log10(mat_balanced)
            trans = pd.read_csv(f'{local_path}data/ZANU/transitions{prefix_csv}.csv')
            trans = trans[trans.chr == chr]
            trans['start_res'] = trans.start // resolution
            trans['end_res'] = trans.end // resolution

            max_value = np.max(matrix)
            min_value = np.min(matrix)
            for label, folder in (
            ('+-', 'plus_minus'), ('-+', 'minus_plus'), ('++', 'plus_plus'), ('--', 'minus_minus')):
                cur_trans = trans[trans.label == label]
                for i, row in cur_trans.iterrows():
                    if not centered:
                        save_sample_by_row(row, 'ZANU')
                        samples_count += 1
                    else:
                        point_start = (row.start_res, row.start_res)
                        point_end = (row.end_res, row.end_res)
                        point_area_start = matrix[point_start[1] - 10:point_start[1] + 10,
                                           point_start[0] - 10:point_start[0] + 10]
                        point_area_end = matrix[point_end[1] - 10:point_end[1] + 10,
                                         point_end[0] - 10:point_end[0] + 10]
                        plt.imsave(f'{path}/positive/ZANU{prefix_file}_{chr[-1]}_{row.start}_{row.start}.png',
                                   point_area_start, cmap=clr_map, vmax=max_value, vmin=min_value)
                        plt.imsave(f'{path}/positive/ZANU{prefix_file}_{chr[-1]}_{row.end}_{row.end}.png',
                                   point_area_end, cmap=clr_map, vmax=max_value, vmin=min_value)
                        samples_count += 2
    c = cooler.Cooler(f'{local_path}/data/ZANU/ZANU_clean_4DN.mcool::/resolutions/{resolution}')
    for chr in ('X', '2R', '2L', '3R', '3L'):
        matrix = np.log10(c.matrix(balance=False).fetch(chr) + 1)
        number = min(len(matrix) // 10, samples_count)
        make_negative_samples(number, matrix, 'ZANU')
        samples_count -= number
        if samples_count <= 0:
            break

    samples_count = 0
    for prefix_cool, prefix_csv, prefix_file in zip(
            ('_short_', '_short_2_', '_medium_', '_long_', '2_short_', '2_short_2_', '2_medium_', '2_long_'),
            ('_short', '_short_2', '_medium', '_long', '_short', '_short_2', '_medium', '_long'),
            ('', '', '', '', '2', '2', '2', '2')):
        print(prefix_cool)
        for chr in ("X", 'Chr2', 'Chr3'):
            c = cooler.Cooler(f'{local_path}data/arab/ARAB{prefix_cool}4DN.mcool::/resolutions/{resolution}')
            mat_balanced = c.matrix(balance=False).fetch(chr)
            mat_balanced += 1

            matrix = np.log10(mat_balanced)
            trans = pd.read_csv(f'{local_path}data/arab/transitions{prefix_csv}.csv')
            trans = trans[trans.chr == chr]
            trans['start_res'] = trans.start // resolution
            trans['end_res'] = trans.end // resolution

            max_value = np.max(matrix)
            min_value = np.min(matrix)
            for label, folder in (
            ('+-', 'plus_minus'), ('-+', 'minus_plus'), ('++', 'plus_plus'), ('--', 'minus_minus')):
                cur_trans = trans[trans.label == label]
                for i, row in cur_trans.iterrows():
                    if not centered:
                        save_sample_by_row(row, 'ARAB')
                        samples_count += 1
                    else:
                        point_start = (row.start_res, row.start_res)
                        point_end = (row.end_res, row.end_res)
                        point_area_start = matrix[point_start[1] - 10:point_start[1] + 10,
                                           point_start[0] - 10:point_start[0] + 10]
                        point_area_end = matrix[point_end[1] - 10:point_end[1] + 10,
                                         point_end[0] - 10:point_end[0] + 10]
                        plt.imsave(f'{path}/positive/ARAB{prefix_file}_{chr[-1]}_{row.start}_{row.start}.png',
                                   point_area_start, cmap=clr_map, vmax=max_value, vmin=min_value)
                        plt.imsave(f'{path}/positive/ARAB{prefix_file}_{chr[-1]}_{row.end}_{row.end}.png',
                                   point_area_end, cmap=clr_map, vmax=max_value, vmin=min_value)
                        samples_count += 2

    c = cooler.Cooler(f'{local_path}/data/arab/ARAB_clean_4DN.mcool::/resolutions/{resolution}')
    for chr in ('X', '2R', '2L', '3R', '3L'):
        matrix = np.log10(c.matrix(balance=False).fetch(chr) + 1)
        number = min(len(matrix) // 10, samples_count)
        make_negative_samples(number, matrix, 'ARAB')
        samples_count -= number
        if samples_count <= 0:
            break