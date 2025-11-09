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

torch.manual_seed(42)
np.random.seed(42)

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

from train_methods import test_model
from hict.patterns.models import DetectModel


class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolutions, image_size, clean_cooler_list, normmats_path, normmats_clean_path, save_images=0, validate=False):
        sv_count = 0
        self.label_to_index = {'++':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               '+-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               '-+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               '--':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'negative':torch.tensor([1.0, 0.0, 0.0, 0.0,  0.0])}
        self.resolutions = resolutions
        self.image_size = image_size
        self.images_to_save = save_images
        self.validate = validate
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
            
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = {}
        self.matrixes_list = {}
        self.sv_files_list = []
        for trans_csv_path, cooler_path, clean_cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
            sv_count+=sv_file.shape[0]
            for resolution in resolutions:
                c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
                if not str(resolution) in self.coolers_list:
                    self.coolers_list[str(resolution)] = []
                self.coolers_list[str(resolution)].append(c)
                c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
                matrixes_by_chr = {}
                for chr in c.chromnames:
                    matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                    matrix__raw_clean = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                    #f_matrix_raw = np.log(matrix+self.eps) - np.log(matrix_clean+self.eps_clean)

                    matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                    matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                    #f_matrix = np.log(matrix_raw+self.eps) - np.log(matrix_clean+self.eps_clean)
                    matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix__raw_clean)

                if not str(resolution) in self.matrixes_list:
                    self.matrixes_list[str(resolution)] = []        
                self.matrixes_list[str(resolution)].append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            for i in range(sv_file.shape[0]):
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c.chromsizes)-2)
                chr_size = np.sum(c.chromsizes[chr_index])
                x = random.randint(image_size//2*max(resolutions), chr_size - image_size//2*max(resolutions)-1)
                neg_sv['chr'].append(c.chromnames[chr_index])
                neg_sv['label'].append('negative')
                neg_sv['start'].append(x)
                neg_sv['end'].append(x)

                indexes['file_index'].append(index)
                indexes['in_index'].append(sv_file.shape[0]+i)
                indexes['is_sv'].append(False)
            sv_file = pd.concat([sv_file, pd.DataFrame(neg_sv)])
            self.sv_files_list.append(sv_file)
        self.num_classes = 2
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]*2

    def get_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, resolution, coarse_grain=False):
            if coarse_grain:
                mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
                mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
            else:
                mat_cg = mat_bal
                mat_cg_clean = mat_bal_clean
            
            if not self.validate:
                choose_mat = random.randint(1, 9)
                if choose_mat < 5:
                    mat_cg = mat_cg * (random.randint(60, 100)/100)
                if 9 > choose_mat > 5:
                    mat_cg_clean =  mat_cg_clean * (random.randint(60, 100)/100)

            mat = np.log(mat_cg+self.eps[str(resolution)])
            mat[np.isnan(mat)] = 0
            mat-= np.log(self.normmat250[str(resolution)]+self.eps[str(resolution)])

            mat_clean = np.log(mat_cg_clean+self.eps_clean[str(resolution)])
            mat_clean[np.isnan(mat_clean)] = 0
            mat_clean -= np.log(self.normmat250_clean[str(resolution)]+self.eps_clean[str(resolution)])
            return np.array([mat, mat_clean])

    def get_chromosme_padding(self, c, target_chr):
        cur_chr = c.chromnames[0]
        i = 0
        padding = 0
        while cur_chr!=target_chr:
            padding+=c.chromsizes[i]
            i+=1
            cur_chr = c.chromnames[i]
        return padding
    
    def __getitem__(self, idx_d):
        idx = idx_d//2
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        mat_list = []
        pad = self.image_size//2
        if row.is_sv and not self.validate:
            shift = random.randint(-pad//2, pad//2)
        else:
            shift = 0
        for resolution in self.resolutions:
            c = self.coolers_list[str(resolution)][row.file_index]
            matrix_full = self.matrixes_list[str(resolution)][row.file_index][sv_info.chr]
            if idx_d % 2 == 0:
                x = (sv_info.start)//resolution
                y = (sv_info.start)//resolution
            else:
                x = (sv_info.end)//resolution
                y = (sv_info.end)//resolution
            #chr_padding = self.get_chromosme_padding(c, sv_info.chr) // resolution

            x += shift
            y += shift
            if x-pad < 0 or y - pad < 0:
                mv = max(-(x-pad), -(y-pad))
                x+=mv
                y+=mv
            if x+pad > matrix_full[0].shape[0] or y + pad > matrix_full[0].shape[0]:
                x = min(x,  matrix_full[0].shape[0]-pad-1)
                y = min(y,  matrix_full[0].shape[0]-pad-1)
            #x+=chr_padding
            #y+=chr_padding
            mat_b = matrix_full[0][x-pad:x+pad, y-pad:y+pad].todense()
            mat_r = matrix_full[1][x-pad:x+pad, y-pad:y+pad].todense()
            mat_b_clean = matrix_full[2][x-pad:x+pad, y-pad:y+pad].todense()
            mat_r_clean = matrix_full[3][x-pad:x+pad, y-pad:y+pad].todense()

            mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  resolution, True)

            mat = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
            mat_list.append(mat)
        try:
            tens = torch.stack(mat_list, dim=0).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].shape)
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)

        return tens, 1 if row.is_sv else 0

'''
train_coolers = [f'{local_path}/data/apes/Gor_SV_4DN.mcool',f'{local_path}/data/apes/Gor_SV_2_4DN.mcool']
clean_train_coolers = [f'{local_path}/data/apes/Gor_4DN.mcool',f'{local_path}/data/apes/Gor_4DN.mcool']
test_csvs = [f'{local_path}/data/apes/test_filtered_gor_sv.csv',f'{local_path}/data/apes/test_filtered_gor_sv_2.csv'] 

validate_coolers = [f'{local_path}/data/apes/Gor_CHM13_15_25_50.mcool',]
clean_validate_coolers = [f'{local_path}/data/apes/CHM13_15_25_50.mcool',]
validate_csvs = [f'{local_path}/data/apes/good_svs_gor_chm_diag.csv',] 

normmats_path=[f'{local_path}/data/apes/Gor_SV_exp_1kb_10kb.npy',
               f'{local_path}/data/apes/Gor_SV_exp_1kb_25kb.npy',
               f'{local_path}/data/apes/Gor_SV_exp_1kb_50kb.npy']

normmats_clean_path=[f'{local_path}/data/apes/Gor_exp_1kb_10kb.npy',
                     f'{local_path}/data/apes/Gor_exp_1kb_25kb.npy',
                     f'{local_path}/data/apes/Gor_exp_1kb_50kb.npy',]

normmats_path_val=[f'{local_path}/data/apes/Gor_CHM13_exp_1kb_10kb.npy',
                   f'{local_path}/data/apes/Gor_CHM13_exp_1kb_25kb.npy',
                   f'{local_path}/data/apes/Gor_CHM13_exp_1kb_50kb.npy']

normmats_clean_path_val=[f'{local_path}/data/apes/CHM13_exp_1kb_15kb.npy',
                         f'{local_path}/data/apes/CHM13_exp_1kb_25kb.npy',
                         f'{local_path}/data/apes/CHM13_exp_1kb_50kb.npy']
'''


#train_coolers = [f'{local_path}data/mcool/Gor_SV_4DN.mcool',f'{local_path}data/mcool/Gor_SV_2_4DN.mcool']
#clean_train_coolers = [f'{local_path}data/mcool/Gor_4DN.mcool',f'{local_path}data/mcool/Gor_4DN.mcool']
#train_csvs = [f'{local_path}data/sv_csv/train_filtered_gor_sv.csv',f'{local_path}data/sv_csv/train_filtered_gor_sv_2.csv'] 
#test_csvs = [f'{local_path}data/sv_csv/test_filtered_gor_sv.csv',f'{local_path}data/sv_csv/test_filtered_gor_sv_2.csv'] 

parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='Path to file with config')

args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    cfg_dict = json.load(file)

cfg = config_dict.ConfigDict(cfg_dict)


validate_coolers = cfg.validate_coolers#[f'{local_path}data/mcool/Siamang_Chm_15_25_50.mcool',]
clean_validate_coolers = cfg.clean_validate_coolers#[f'{local_path}data/mcool/CHM13_15_25_50.mcool',]
validate_csvs = cfg.validate_csvs#[f'{local_path}data/sv_csv/filtered_SSY_CHM.csv',] 


normmats_path_val= cfg.normmats_path_val#['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Siamang_Chm_exp_1kb_15kb.npy',
                                        #'/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Siamang_Chm_exp_1kb_25kb.npy',
                                        #'/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Siamang_Chm_exp_1kb_50kb.npy']

normmats_clean_path_val= cfg.normmats_clean_path_val#['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_15kb.npy',
                                                    #'/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
                                                    #'/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy']
'''
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolutions=[10000, 25000, 50000],
    image_size=48,
    normmats_path=normmats_path,
    normmats_clean_path=normmats_clean_path,
    save_images = 10)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
'''
validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolutions=cfg.resolutions,
    image_size=48,
    normmats_path=normmats_path_val,
    normmats_clean_path=normmats_clean_path_val,
    save_images = 0,
    validate=True)
print('Total bp amount:', len(validate_dataset))
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

learning_rate = 1e-6
if cfg.loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
if cfg.loss == 'BCELoss':
    criterion = nn.BCELoss()
if cfg.loss == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()

model_15kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_15kb.to(device=device)#weights_normmat_cg_48_diag_15Kbp_two_matrices_chm_gor/normmat_cg_48_diag_15Kbp.pt
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
#dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

test_model(dataloaders, models, f'log_{cfg.model_name}', criterion, device, num_epochs, phases= ['validate'])
print(f'Completed for {cfg.model_name} with config {args.cfg_path}')