import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import cooler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
from collections import OrderedDict

image_size = 12
resolution = 50000

local_path = 'D:/Study/HICT/HICT_Patterns/'
file_path = 'data\MALI_ARAB\Mali_Arab_4DN.mcool'

c = cooler.Cooler(f'{local_path}{file_path}::/resolutions/{resolution}')
skip = 0
for chr in ('X', '2R', '2L'):
    matrix = np.log2(c.matrix(balance=False).fetch(chr))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(matrix, cmap='Greens')
    fig.colorbar(im)
    detected = np.genfromtxt(f"{local_path}result_Mali_Arab.csv", delimiter=",")
    
    for det in detected:
        x = (det[0] - skip)//resolution-image_size//2
        y = (det[1] - skip)//resolution-image_size//2
        if x > 0 and y > 0:
            rect1 = patches.Rectangle((x, y), image_size, image_size, color='orange', fc = 'none', lw = 0.5)
            ax.add_patch(rect1)
    
    plt.savefig(f'{local_path}Mali_Arab_{chr}.png', dpi=1000)
    skip += c.chromsizes[chr]
