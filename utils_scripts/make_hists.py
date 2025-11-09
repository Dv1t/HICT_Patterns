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
from tqdm import tqdm
import time
import math
import matplotlib.pyplot as plt
from cooltools.lib.numutils import adaptive_coarsegrain

sample_count = 600

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

torch.manual_seed(42)
np.random.seed(42)

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 600
warnings.filterwarnings('ignore')

class HistGeneratorDataset(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, normmat_path, normmat_clean_path, validate=False):
        sv_count = 0
        self.resolution = resolution
        self.image_size = image_size
        self.validate = validate

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
            for i in range(sv_file.shape[0]):
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c.chromsizes)-2)
                chr_size = np.sum(c.chromsizes[chr_index])
                x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
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
        return sample_count
    
    def get_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, coarse_grain=False):
        if coarse_grain:
            mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
        else:
            mat_cg = mat_bal
            mat_cg_clean = mat_bal_clean
        
        mat = np.log(mat_cg+self.eps)
        mat[np.isnan(mat)] = 0
        mat-= np.log(self.normmat250+self.eps)

        mat_clean = np.log(mat_cg_clean+self.eps_clean)
        mat_clean[np.isnan(mat_clean)] = 0
        mat_clean -= np.log(self.normmat250_clean+self.eps_clean)
        
        return mat, mat_clean
    
    def __getitem__(self, idx_d):
        idx = idx_d//2
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        if idx_d % 2 == 0:
            x = (sv_info.start)//self.resolution
            y = (sv_info.start)//self.resolution
        else:
            x = (sv_info.end)//self.resolution
            y = (sv_info.end)//self.resolution
        pad = self.image_size//2
        if row.is_sv and not self.validate:
            shift = random.randint(-pad//2, pad//2)
            x += shift
            y += shift
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
        

        mat, mat_clean = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean, True)

        mat = 2.*(mat - np.min(mat))/np.ptp(mat)-1
        mat_clean = 2.*(mat_clean - np.min(mat_clean))/np.ptp(mat_clean)-1

        return torch.from_numpy(mat), torch.from_numpy(mat_clean), 1 if row.is_sv else 0

def run_epoch(phase, dataloader, device):
  cur_tqdm = tqdm(dataloader)
  for inputs, inputs_clean, labels in cur_tqdm:
    
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True, dtype=torch.bool)

    positive_sum = torch.sum(torch.sum(torch.sum(inputs[np.argwhere(np.asarray(labels))], dim=1), dim=1), dim=1)
    negative_sum = torch.sum(torch.sum(torch.sum(inputs[np.argwhere(np.asarray(~labels))], dim=1), dim=1), dim=1)

    all_sum = torch.sum(torch.sum(inputs, dim=1), dim=1)

    inputs_clean = inputs_clean.to(device, non_blocking=True)

    positive_sum_clean = torch.sum(torch.sum(torch.sum(inputs_clean[np.argwhere(np.asarray(labels))], dim=1), dim=1), dim=1)
    negative_sum_clean = torch.sum(torch.sum(torch.sum(inputs_clean[np.argwhere(np.asarray(~labels))], dim=1), dim=1), dim=1)

    all_sum_clean = torch.sum(torch.sum(inputs_clean, dim=1), dim=1)

    #assert positive_sum.shape == 48
    #assert negative_sum.shape == 48
    #assert all_sum.shape == 48

    return positive_sum, negative_sum, all_sum, positive_sum_clean, negative_sum_clean, all_sum_clean

def make_hists(dataloaders, log_folder, device, num_epochs=20, phases= ['train', 'test', 'validate']):
    for phase in dataloaders:
        if phase not in phases:
            phases.append(phase)

    saved_epoch_sum = {phase: [] for phase in phases}

    for phase in phases:
        print("--- Cur phase:", phase)
        positive_sum, negative_sum, all_sum, positive_sum_clean, negative_sum_clean, all_sum_clean = run_epoch(phase, dataloaders[phase], device)
        saved_epoch_sum[phase] = [positive_sum.cpu().numpy(), negative_sum.cpu().numpy(), all_sum.cpu().numpy() , positive_sum_clean.cpu().numpy(), negative_sum_clean.cpu().numpy(), all_sum_clean.cpu().numpy()]
    
    os.makedirs(log_folder, exist_ok=True)
    #np.savetxt(f'{log_folder}/clean_hist_train_all_60.csv', saved_epoch_sum["train"][2], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_test_all_600.csv', saved_epoch_sum["test"][2], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_val_all_600.csv', saved_epoch_sum["validate"][2], delimiter =',')

    #np.savetxt(f'{log_folder}/clean_hist_train_negative_600.csv', saved_epoch_sum["train"][1], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_test_negative_600.csv', saved_epoch_sum["test"][1], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_val_negative_600.csv', saved_epoch_sum["validate"][1], delimiter =',')

    #np.savetxt(f'{log_folder}/clean_hist_train_positive_600.csv', saved_epoch_sum["train"][0], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_test_positive_600.csv', saved_epoch_sum["test"][0], delimiter =',')
    #np.savetxt(f'{log_folder}/clean_hist_val_positive_600.csv', saved_epoch_sum["validate"][0], delimiter =',')

    fig, axs = plt.subplots(1, 3, figsize=(40,20))

    axs[0].set_title(f'Sums of {len(saved_epoch_sum["train"][2])} 48*48 matrices.')
    axs[0].hist(saved_epoch_sum['train'][2], bins=30, alpha=0.6,label='train_chimer')
    axs[0].hist(saved_epoch_sum['test'][2], bins=30, alpha=0.6,label='test_chimer')
    axs[0].hist(saved_epoch_sum['validate'][2], bins=30, alpha=0.6,label='validate_chimer')
    axs[0].hist(saved_epoch_sum['train'][5], bins=30, alpha=0.6,label='train_clean')
    axs[0].hist(saved_epoch_sum['test'][5], bins=30, alpha=0.6,label='test_clean')
    axs[0].hist(saved_epoch_sum['validate'][5], bins=30, alpha=0.6,label='validate_clean')
    axs[0].legend(loc="upper left")

    axs[1].set_title(f'Sums of {len(saved_epoch_sum["train"][1])} negative 48*48 matrices.')
    axs[1].hist(saved_epoch_sum['train'][1], bins=30, alpha=0.6,label='train_chimer')
    axs[1].hist(saved_epoch_sum['test'][1], bins=30, alpha=0.6,label='test_chimer')
    axs[1].hist(saved_epoch_sum['validate'][1], bins=30, alpha=0.6,label='validate_chimer')
    axs[1].hist(saved_epoch_sum['train'][4], bins=30, alpha=0.6,label='train_clean')
    axs[1].hist(saved_epoch_sum['test'][4], bins=30, alpha=0.6,label='test_clean')
    axs[1].hist(saved_epoch_sum['validate'][4], bins=30, alpha=0.6,label='validate_clean')
    axs[1].legend(loc="upper left")

    axs[2].set_title(f'Sums of {len(saved_epoch_sum["train"][0])} positive 48*48 matrices.')
    axs[2].hist(saved_epoch_sum['train'][0], bins=30, alpha=0.6,label='train_chimer')
    axs[2].hist(saved_epoch_sum['test'][0], bins=30, alpha=0.6,label='test_chimer')
    axs[2].hist(saved_epoch_sum['validate'][0], bins=30, alpha=0.6,label='validate_chimer')
    axs[2].hist(saved_epoch_sum['train'][3], bins=30, alpha=0.6,label='train_clean')
    axs[2].hist(saved_epoch_sum['test'][3], bins=30, alpha=0.6,label='test_clean')
    axs[2].hist(saved_epoch_sum['validate'][3], bins=30, alpha=0.6,label='validate_clean')
    axs[2].legend(loc="upper left")

    plt.savefig(f'{log_folder}/combined_hist_train_test_val_normmat_norm.png')


train_coolers = [f'{local_path}data/mcool/Gor_SV_4DN.mcool',f'{local_path}data/mcool/Gor_SV_2_4DN.mcool']
clean_train_coolers = [f'{local_path}data/mcool/Gor_4DN.mcool',f'{local_path}data/mcool/Gor_4DN.mcool']
train_csvs = [f'{local_path}data/sv_csv/train_filtered_gor_sv.csv',f'{local_path}data/sv_csv/train_filtered_gor_sv_2.csv'] 
test_csvs = [f'{local_path}data/sv_csv/test_filtered_gor_sv.csv',f'{local_path}data/sv_csv/test_filtered_gor_sv_2.csv'] 

validate_coolers = [f'{local_path}data/mcool/Gor_CHM13_4DN.mcool',]
clean_validate_coolers = [f'{local_path}data/mcool/CHM13_4DN.mcool',]
validate_csvs = [f'{local_path}data/sv_csv/good_svs_gor_chm_diag.csv',] 

os.makedirs('weights_normmat_cg_48_diag_25Kbp_single', exist_ok=True)

train_dataset = HistGeneratorDataset(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = HistGeneratorDataset(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=test_csvs,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = HistGeneratorDataset(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
    validate=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 1
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

make_hists(dataloaders, 'train_test_validate_sum_hists_600', device, num_epochs, phases= ['train', 'test'])