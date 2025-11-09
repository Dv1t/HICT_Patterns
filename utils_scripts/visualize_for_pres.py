import torch
from torch import nn
import cooler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas as pd
import warnings
import time
from torchvision.transforms import GaussianBlur
import random
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from cooltools.lib.numutils import adaptive_coarsegrain
import numpy as np
import matplotlib.pyplot as plt
import cooler
import pandas as pd
import os
import sys
from tqdm import tqdm

def get_matrix(mat_bal, mat_raw, normmat250, eps, isCg, image_size, isLog=True):
        if isCg:
            mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat_bal = mat_cg

        mat250 = np.nanmean(np.nanmean(np.reshape(mat_bal, (image_size, 1, image_size, 1)), axis=3), axis=1)
        mat250+=1e-6
        mat250[np.isnan(mat250)] = 0
        if isLog:
            mat_logb = np.log(mat250 + eps) - np.log(normmat250 + eps)
        else:
            mat_logb = np.sqrt(mat250 / normmat250)
        return mat_logb

local_path = '/mnt/tank/scratch/vdravgelis/'

resolution  = 1000000
ape_chm = cooler.Cooler(f'{local_path}data/mcool/Gor_CHM13_4DN.mcool::/resolutions/{resolution}')


chr_name = 'chr1'
image_size = 180
normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy'

normmat_bydist = np.exp(np.load(normmat_path))[:image_size]
normmat = normmat_bydist[np.abs(np.arange(image_size)[:, None] - np.arange(image_size)[None, :])]
normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
eps = np.min(normmat250)

matrix_raw = ape_chm.matrix(balance=False).fetch(chr_name)[0:image_size,0:image_size]
matrix_bal = ape_chm.matrix(balance=False).fetch(chr_name)[0:image_size,0:image_size]

matrix = get_matrix(matrix_bal ,matrix_raw, normmat250, eps, True, image_size)
matrix = np.reshape(matrix, (image_size, image_size))


results = pd.read_csv(f"/mnt/tank/scratch/vdravgelis/training/real_tests/step_1_50kb.csv")
result_res=50000
image_size = 48
bps = np.array(sorted(list(results.bp_1)), dtype=np.int64)#*result_res//resolution
bps = [bp for bp in bps if bp < ape_chm.chromsizes[0]//result_res]
segments = []
segment_start = bps[0]
segment_end = bps[0]
for bp in bps:
    if bp-segment_end > image_size:
        segments.append((segment_start, segment_end))
        segment_start = bp
    segment_end = bp
segments

fig = plt.figure(figsize=(12,12), dpi=300)
ax = fig.add_subplot(111)
im = ax.matshow(matrix, cmap='bwr')
for segment_start, segment_end in segments:
    plt.plot([segment_start//(resolution//result_res), segment_end//(resolution//result_res)], [segment_start//(resolution//result_res), segment_end//(resolution//result_res)], color='black', linewidth=4)
plt.savefig('normat_chr1.png', dpi=300)
plt.close()