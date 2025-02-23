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

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from hict.patterns.help_functions import get_chromosome_coords, get_genome_coords
from hict.patterns.models import DetectModel, ClassificationModel

def get_chromosome_coords(coords_list, chr_sizes, resolution):
    additive_sizes = np.empty_like(chr_sizes, dtype=np.uint64)
    curr_s = 0
    for i, s in enumerate(chr_sizes):
        curr_s += s
        additive_sizes[i] = curr_s
    result = {}
    for coord in coords_list:
        x_i = 0
        while coord*resolution > additive_sizes[x_i]:
            x_i+=1
            if x_i >= len(additive_sizes):
                break
        if x_i >= len(additive_sizes):
            continue
        x_chr = x_i
        if x_i > 0:
            x = (coord*resolution-additive_sizes[x_i-1]) // resolution
        else:
            x = coord
        
        if x_chr in result:
            result[x_chr].append(x)
        else:
            result[x_chr] = [x, ]
        
    result_list = []
    
    for key, value in result.items():
        for v in value:
            result_list.append((key, int(v)))
    return result_list

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

def perform_detection(model, dataloader, round = True, label_cutoff=0.95):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        if round:
            labels = torch.round(output).detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels==1]
            y_list = position[1][labels==1]
        else:
            labels = output.detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels>=label_cutoff]
            y_list = position[1][labels>=label_cutoff]
        if len(x_list) > 0:
            for x, y in zip(x_list.numpy(), y_list.numpy()):
                detected.append((x, y))
    return detected

def save_result_to_csv(local_path, detected, name):
    np.savetxt(f"{local_path}/{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s')


class EvalDatasetDiag(Dataset):
    def __init__(self, cooler_path, clean_cooler_path, resolution, image_size, step, device):
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.resolution = resolution
        self.image_size = image_size
        self.step = step
        self.device = device
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c
        self.clean_cooler = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')        
        all_chr_len = int(np.sum(c.chromsizes.values, dtype=object))
        self.amount_steps = int((all_chr_len//resolution) // (step))
        self.matrixes_by_chr = {}
        for chr in c.chromnames:
            matrix = c.matrix(balance=False).fetch(chr)
            matrix_clean = self.clean_cooler.matrix(balance=False).fetch(chr)
            
            f_matrix = np.log10(matrix+1e-6) - np.log10(matrix_clean+1e-6)
            f_matrix = 2.*(f_matrix - np.min(f_matrix))/np.ptp(f_matrix)-1

            self.matrixes_by_chr[chr] = torch.from_numpy(f_matrix).to(device=device, dtype=torch.float)

    def __len__(self):
        return self.amount_steps
    
    def __get_matrix(self, x, y):
            x ,y = get_chromosome_coords((x, y), self.cooler.chromsizes, self.resolution)
            chr_num = x[0]
            
            return self.matrixes_by_chr[c.chromnames[chr_num]][x[1]:x[1]+self.image_size, y[1]:y[1]+self.image_size]
    
    def __getitem__(self, idx):
        x, y = idx*self.step, idx*self.step
        mat = self.__get_matrix(x, y)
        try:
            tens = mat.reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        except RuntimeError:
            tens = torch.zeros((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        return tens, (x, y)


file_path = f'{local_path}data/mcool/Gor_CHM13_4DN.mcool'
clean_file_path = f'{local_path}data/mcool/CHM13_4DN.mcool'

c = cooler.Cooler(f'{file_path}::/resolutions/50000')
#Stage 1 - 50k, diagonal detection
print('Started Stage 1')
resolution_1 = 50000
image_size_1 = 48
dataset = EvalDatasetDiag(file_path, clean_file_path, resolution=resolution_1, image_size=image_size_1, step=image_size_1//4, device=device)
print('Stage 1 dataset loaded')
model = DetectModel(image_size=image_size_1, num_models=10)
model.to(device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_50k_48_diag.pt', map_location=device))
model.eval()

detected = perform_detection(model, DataLoader(dataset, batch_size=64))
save_result_to_csv(os.getcwd(), detected, 'real_tests/step_12_50kb')


c = cooler.Cooler(f'{file_path}::/resolutions/25000')
#Stage 1 - 50k, diagonal detection
print('Started Stage 1')
resolution_1 = 25000
image_size_1 = 48
dataset = EvalDatasetDiag(file_path, clean_file_path, resolution=resolution_1, image_size=image_size_1, step=image_size_1//4, device=device)
print('Stage 1 dataset loaded')
model = DetectModel(image_size=image_size_1, num_models=10)
model.to(device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_25k_48_diag.pt', map_location=device))
model.eval()

detected = perform_detection(model, DataLoader(dataset, batch_size=64))
save_result_to_csv(os.getcwd(), detected, 'real_tests/step_12_25kb')

c = cooler.Cooler(f'{file_path}::/resolutions/10000')
#Stage 1 - 50k, diagonal detection
print('Started Stage 1')
resolution_1 = 10000
image_size_1 = 48
dataset = EvalDatasetDiag(file_path, clean_file_path, resolution=resolution_1, image_size=image_size_1, step=image_size_1//4, device=device)
print('Stage 1 dataset loaded')
model = DetectModel(image_size=image_size_1, num_models=10)
model.to(device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_10k_48_diag.pt', map_location=device))
model.eval()

detected = perform_detection(model, DataLoader(dataset, batch_size=64))
save_result_to_csv(os.getcwd(), detected, 'real_tests/step_12_10kb')
