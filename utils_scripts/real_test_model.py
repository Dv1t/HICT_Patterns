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
import random
from cooltools.lib.numutils import adaptive_coarsegrain
from ml_collections import config_dict
import argparse
import json

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

def perform_detection(models, dataloader, round = True, label_cutoff=0.95):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for inputs, position, valid_mat in cur_tqdm:

        inputs = inputs.to(device, non_blocking=True)
        sigmoid  = nn.Sigmoid()
        outputs_by_res = [torch.round(sigmoid(model(inputs[:, i]))) for model, i in zip(models,  range(inputs.shape[1]))]
        #class_predictions = [torch.argmax(output, dim=1) for output in outputs_by_res]
        stacked_predictions = torch.stack(outputs_by_res, dim=0)
        majority_vote_predictions, _ = torch.mode(stacked_predictions, dim=0)
        preds = majority_vote_predictions.resize(majority_vote_predictions.shape[0]).to(device, non_blocking=True, dtype=torch.float)
        
        if round:
            labels = torch.round(preds).detach().cpu().numpy().reshape(-1)
            labels = labels*valid_mat.detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels==1]
            y_list = position[1][labels==1]
        else:
            labels = preds.detach().cpu().numpy().reshape(-1)
            labels = labels*valid_mat.detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels>=label_cutoff]
            y_list = position[1][labels>=label_cutoff]
            
        if len(x_list) > 0:
            for x, y, label in zip(x_list.numpy(), y_list.numpy(), labels):
                detected.append((x, y, label))
    return detected

def save_result_to_csv(local_path, detected, name):
    np.savetxt(f"{local_path}/{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s',
        header='x,y,label')


class EvalDatasetDiag(Dataset):
    
    def __init__(self, cooler_path, resolutions, image_size, clean_cooler_path, normmats_path, normmats_clean_path, step=1):
        self.step = step
        min_res = min(resolutions)
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{min_res}')
        chr_sizes = [int(size) for size in c.chromsizes.values if int(size) > image_size*min_res*10]
        all_chr_len = int(np.sum(chr_sizes))
        self.amount_steps = int((all_chr_len//min_res) // (step))

        self.resolutions = resolutions
        self.image_size = image_size
        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}

        for resolution, normmat_path, normmat_clean_path in zip(resolutions, normmats_path, normmats_clean_path):
            normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
            normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
            self.normmat250[str(resolution)] = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
            self.eps[str(resolution)] = np.min(self.normmat250[str(resolution)])

            normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
            normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
            self.normmat250_clean[str(resolution)] = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
            self.eps_clean[str(resolution)] = np.min(self.normmat250_clean[str(resolution)])
            
        self.coolers_list = {}
        self.matrixes_list = {}            
        for resolution in resolutions:
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.coolers_list[str(resolution)] = c
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix__raw_clean = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix__raw_clean)
   
            self.matrixes_list[str(resolution)] = matrixes_by_chr

    def __len__(self):
        return self.amount_steps
    
    def process_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, resolution, coarse_grain=False):
        if coarse_grain:
            mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
        else:
            mat_cg = mat_bal
            mat_cg_clean = mat_bal_clean

        mat = np.log(mat_cg+self.eps[str(resolution)])
        mat[np.isnan(mat)] = 0
        mat-= np.log(self.normmat250[str(resolution)]+self.eps[str(resolution)])

        mat_clean = np.log(mat_cg_clean+self.eps_clean[str(resolution)])
        mat_clean[np.isnan(mat_clean)] = 0
        mat_clean -= np.log(self.normmat250_clean[str(resolution)]+self.eps_clean[str(resolution)])
        return np.array([mat, mat_clean])
    
    def __get_matrix(self, x, y):
            pad = self.image_size//2
            mat_list = []
            valid_mat = True
            for resolution in self.resolutions:
                x ,y = get_chromosome_coords((x, y), self.coolers_list[str(resolution)].chromsizes, resolution)
                chr_num = x[0]
                x = x[1]
                y = y[1]
                c = self.coolers_list[str(resolution)]
                matrix_full = self.matrixes_list[str(resolution)][c.chromnames[chr_num]]
                if x-pad < 0 or y - pad < 0:
                    mv = max(-(x-pad), -(y-pad))
                    x+=mv
                    y+=mv
                if x+pad > matrix_full[0].shape[0] or y + pad > matrix_full[0].shape[0]:
                    x = min(x,  matrix_full[0].shape[0]-pad-1)
                    y = min(y,  matrix_full[0].shape[0]-pad-1)

                mat_b = matrix_full[0][x-pad:x+pad, y-pad:y+pad].todense()
                mat_r = matrix_full[1][x-pad:x+pad, y-pad:y+pad].todense()
                mat_b_clean = matrix_full[2][x-pad:x+pad, y-pad:y+pad].todense()
                mat_r_clean = matrix_full[3][x-pad:x+pad, y-pad:y+pad].todense()

                if np.count_nonzero(np.nan_to_num(mat_r, posinf=2, neginf=2))/2304 < 0.25: #0.25
                    valid_mat = False
                small_matrix_start = mat_r[22:26, 22:26]
                if np.count_nonzero(np.nan_to_num(small_matrix_start, posinf=2, neginf=2))/16 < 0.5: #0.5
                    valid_mat = False

                mat_norm = self.process_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  resolution, True)

                mat = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
                mat_list.append(mat)
            try:
                tens = torch.stack(mat_list, dim=0).to(device=device, dtype=torch.float)
            except RuntimeError:
                tens = torch.zeros((3, 2, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
                print(matrix_full[0].shape)
                print(x)
                print(y)

            return tens, valid_mat
    
    def __getitem__(self, idx):
        x, y = idx*self.step, idx*self.step
        tens, valid_mat = self.__get_matrix(x, y)
        return tens, (x, y), valid_mat



parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='Path to file with config')

args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    cfg_dict = json.load(file)

cfg = config_dict.ConfigDict(cfg_dict)


validate_cooler = cfg.validate_cooler
clean_validate_cooler = cfg.clean_validate_cooler


normmats_path_val= cfg.normmats_path_val
normmats_clean_path_val= cfg.normmats_clean_path_val


validate_dataset = EvalDatasetDiag(
    cooler_path=validate_cooler,
    clean_cooler_path=clean_validate_cooler,
    resolutions=cfg.resolutions,
    image_size=48,
    normmats_path=normmats_path_val,
    normmats_clean_path=normmats_clean_path_val,
    step=cfg.step)
print('Total steps amount:', len(validate_dataset))
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 1e-6
if cfg.loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
if cfg.loss == 'BCELoss':
    criterion = nn.BCELoss()
if cfg.loss == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()

model_15kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_15kb.to(device=device)
model_15kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[0]}', map_location=device))
model_15kb.eval()

model_25kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_25kb.to(device=device)
model_25kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[1]}', map_location=device))
model_25kb.eval()

model_50kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_50kb.to(device=device)
model_50kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[2]}', map_location=device))
model_50kb.eval()

models = [model_15kb, model_25kb, model_50kb]

num_epochs = 1
dataloaders = dict()
dataloaders['validate'] = validate_dataloader

detected = perform_detection(models, validate_dataloader)
save_result_to_csv(os.getcwd(), detected, cfg.result_save_path)
print(f'Completed for {cfg.model_name} with config {args.cfg_path}')
