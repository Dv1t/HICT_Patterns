import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd

data = pd.read_excel("media-2.xlsx", sheet_name='INV.S2',  skiprows=[0])

united_hict = {
    'species':[],
    'chrom':[],
    'bp':[],
}
local_path = '/mnt/tank/scratch/vdravgelis/'
resolution = 10000
species = 'Gor_CHM13'

ape_chm = cooler.Cooler(f'{local_path}data/mcool/Gor_CHM13_4DN.mcool::/resolutions/{resolution}')
chm =  cooler.Cooler(f'{local_path}data/mcool/CHM13_4DN.mcool::/resolutions/{resolution}')

gor = data.loc[(data.Species == 'Jim_GGO')]
gor['astart_res'] = gor['T2T Chm13\n RefStart'] // resolution
gor['bstart_res'] = gor['QueStart'] // resolution
gor['aend_res'] =  gor['T2T Chm13\n RefEnd'] // resolution
gor['bend_res'] = gor['QueEnd'] // resolution
gor['species']=species
gor['QueChrom_tmp'] = gor['QueChrom_tmp'].map(lambda x: x.rstrip('_hap1').rstrip('_hap2'))
gor = gor.rename(columns={'T2T Chm13 RefChrom': 'achr', 'QueChrom_tmp': 'bchr', 'T2T Chm13\n RefStart': 'astart', 'T2T Chm13\n RefEnd':'bstart'})
gor.reset_index(inplace=True)

united_hict = {
    'species':[],
    'chrom':[],
    'bp':[],
}

with open('good_svs_gor_chm.csv', mode='w') as output:
    output.write('chr,label,start,end\n')
    for index, row in gor.iterrows():
        chr_name = row.achr
        if abs(row.astart_res - row.aend_res) < 5:
            continue
        matrix = ape_chm.matrix(balance=False).fetch(chr_name) 
        matrix_clean = chm.matrix(balance=False).fetch(chr_name)
        matrix = np.log10(matrix)-np.log10(matrix_clean)

        matrix = matrix[row.astart_res-48:row.astart_res+48, row.aend_res-48:row.aend_res+48]

        if np.count_nonzero(np.nan_to_num(matrix, posinf=2, neginf=2))/9216 < 0.2: #0.25
            continue
        
        small_matrix = matrix[22:26, 22:26]

        if np.count_nonzero(np.nan_to_num(small_matrix, posinf=2, neginf=2))/16 < 0.4: #0.5
            continue
        output.write(f"{chr_name},sv,{row['astart']},{row['bstart']}\n")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.matshow(matrix, cmap='bwr')
        plt.savefig(f'images_filter_testing/Gor_CHM13_{chr_name}_{row.astart_res}_{row.aend_res}_{np.count_nonzero(np.nan_to_num(matrix, posinf=2, neginf=2))/9216}_{np.count_nonzero(np.nan_to_num(small_matrix, posinf=2, neginf=2))/16}.png')
        plt.close()