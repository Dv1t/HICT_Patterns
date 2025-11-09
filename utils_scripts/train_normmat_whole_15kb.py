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
import torch.optim as optim
from ml_collections import config_dict
import argparse
import json

from cooltools.lib.numutils import adaptive_coarsegrain

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from train_methods import train_model
from hict.patterns.models import DetectModel

torch.manual_seed(42)
np.random.seed(42)

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1024
warnings.filterwarnings('ignore')

class TrainDataset(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, normmat_path, normmat_clean_path, validate=False):
        self.resolution = resolution
        self.image_size = image_size
        self.validate = validate

        sv_count = 0
        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}
        normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
        normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = np.min(self.normmat250)

        normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
        normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250_clean = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps_clean = np.min(self.normmat250_clean)
            
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = []
        self.matrixes_list = []
        self.sv_files_list = []
        for trans_csv_path, cooler_path, clean_cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
            sv_count+=sv_file.shape[0]
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrix_clean_raw = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrixes_by_chr[chr] = (matrix_raw, matrix, matrix_clean_raw, matrix_clean)

            self.matrixes_list.append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            for i, row in sv_file.iterrows():
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c.chromsizes)-2)
                x = row['start'] + random.randint(2*image_size*resolution, 15*image_size*resolution)
                y = row['end'] + random.randint(2*image_size*resolution, 15*image_size*resolution)
                neg_sv['chr'].append(c.chromnames[chr_index])
                neg_sv['label'].append('negative')
                neg_sv['start'].append(x)
                neg_sv['end'].append(y)

                indexes['file_index'].append(index)
                indexes['in_index'].append(sv_file.shape[0]+i)
                indexes['is_sv'].append(False)
            sv_file = pd.concat([sv_file, pd.DataFrame(neg_sv)])
            self.sv_files_list.append(sv_file)
        self.num_classes = 2
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]

    def get_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, coarse_grain=False):
            if coarse_grain:
                mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
                mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
            else:
                mat_cg = mat_bal
                mat_cg_clean = mat_bal_clean

            mat_cg = np.log(mat_cg+self.eps)
            mat_cg_clean = np.log(mat_cg_clean+self.eps_clean)
            mat_cg[np.isnan(mat_cg)] = 0
            mat_cg_clean[np.isnan(mat_cg_clean)] = 0
            
            mat_cg = mat_cg - np.log(self.normmat250+self.eps)
            mat_cg_clean = mat_cg_clean - np.log(self.normmat250_clean+self.eps_clean)

            return np.array([mat_cg, mat_cg_clean])
    
    def __getitem__(self, idx):
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        x = (sv_info.start)//self.resolution
        y = (sv_info.end)//self.resolution
        pad = self.image_size//2
        if row.is_sv and not self.validate:
            x += random.randint(-pad//2, pad//2)
            y += random.randint(-pad//2, pad//2)
        if x-pad < 0 or y - pad < 0:
            mv = max(-(x-pad), -(y-pad))
            x+=mv
            y+=mv
        if x+pad > matrix_full[0].get_shape()[0] or y + pad > matrix_full[0].get_shape()[0]:
            x = min(x,  matrix_full[0].get_shape()[0]-pad-1)
            y = min(y,  matrix_full[0].get_shape()[0]-pad-1)

        mat_r = matrix_full[0][x-pad:x+pad, y-pad:y+pad].todense()
        mat_b = matrix_full[1][x-pad:x+pad, y-pad:y+pad].todense()
        mat_r_clean = matrix_full[2][x-pad:x+pad, y-pad:y+pad].todense()
        mat_b_clean = matrix_full[3][x-pad:x+pad, y-pad:y+pad].todense()

        mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  True)
        if np.random.random() > 0.6:
            mat_norm[0] = np.rot90(mat_norm[0], k=2)
            mat_norm[1] = np.rot90(mat_norm[1], k=2)
        mat = torch.from_numpy(mat_norm[0]).to(device=device, dtype=torch.float)
        try:
            tens = mat.reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].get_shape())
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)

        return tens, 1 if row.is_sv else 0

parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='Path to file with config')

args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    cfg_dict = json.load(file)

cfg = config_dict.ConfigDict(cfg_dict)



train_coolers = cfg.train_coolers
clean_train_coolers = cfg.clean_train_coolers
train_csvs =  cfg.train_csvs
test_csvs = cfg.test_csvs

validate_coolers = cfg.validate_coolers
clean_validate_coolers = cfg.clean_validate_coolers
validate_csvs = cfg.validate_csvs

os.makedirs(f'weights_{cfg.model_name[0]}', exist_ok=True)

train_dataset = TrainDataset(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=15000,
    image_size=48,
    normmat_path=cfg.normmats_path_train[0],
    normmat_clean_path=cfg.normmats_clean_path_train[0])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Total bp amount in train:', len(train_dataset))


test_dataset = TrainDataset(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=test_csvs,
    resolution=15000,
    image_size=48,
    normmat_path=cfg.normmats_path_train[0],
    normmat_clean_path=cfg.normmats_clean_path_train[0])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Total bp amount in test:', len(test_dataset))

validate_dataset = TrainDataset(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=15000,
    image_size=48,
    normmat_path=cfg.normmats_path_val[0],
    normmat_clean_path=cfg.normmats_clean_path_val[0],
    validate=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
print('Total bp amount in val:', len(validate_dataset))

model = DetectModel(in_channels=cfg.get('model_in_channels', 2), image_size=48, num_models=10)
model.to(device=device)
learning_rate = cfg.learning_rate
if cfg.loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
if cfg.loss == 'BCELoss':
    criterion = nn.BCELoss()
if cfg.loss == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 160
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

train_model(dataloaders, model, f'log_15kb_{cfg.model_name[0]}', criterion, optimizer, device, num_epochs, phases= ['train', 'test'], model_name=cfg.model_name[0])
torch.save(model.state_dict(), f'weights_{cfg.model_name[0]}/wm_48_15Kbp.pt')
print(f'Trained 15kb {cfg.model_name[0]} with config {args.cfg_path}')