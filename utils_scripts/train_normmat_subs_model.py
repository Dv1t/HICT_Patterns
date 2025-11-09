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

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from train_methods import train_model
from hict.patterns.models import DetectModel


local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, normmat_path, normmat_clean_path, save_images=0):
        sv_count = 0
        self.label_to_index = {'++':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               '+-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               '-+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               '--':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'negative':torch.tensor([1.0, 0.0, 0.0, 0.0,  0.0])}
        self.resolution = resolution
        self.image_size = image_size
        self.images_to_save = save_images

        normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
        normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = np.min(normmat250)

        normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
        normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        normmat250_clean = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps_clean = np.min(normmat250_clean)
        self.normmat = np.log(normmat250+self.eps)-np.log(normmat250_clean+self.eps_clean)
        print('Loaded clean cooler')
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = []
        self.matrixes_list = []
        self.sv_files_list = []
        for trans_csv_path, cooler_path, clean_cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
            sv_count+=sv_file.shape[0]
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            matrixes_by_chr = {}
            for chr in c.chromnames:
                #matrix = c.matrix(balance=False).fetch(chr)
                #matrix_clean = c_clean.matrix(balance=False).fetch(chr)
                #f_matrix_raw = np.log(matrix+self.eps) - np.log(matrix_clean+self.eps_clean)

                matrix_raw = c.matrix(balance=True).fetch(chr)
                matrix_clean = c_clean.matrix(balance=True).fetch(chr)
                f_matrix = np.log(matrix_raw+self.eps) - np.log(matrix_clean+self.eps_clean)
                matrixes_by_chr[chr] = f_matrix
                    
            self.matrixes_list.append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            sv_areas = set()
            for x in sv_file.start:
                for delta in range(self.image_size//2):
                    sv_areas.add(int(x)+delta)
                    sv_areas.add(int(x)-delta)
            for x in sv_file.end:
                for delta in range(self.image_size//2):
                    sv_areas.add(int(x)+delta)
                    sv_areas.add(int(x)-delta)
            for i in range(sv_file.shape[0]):
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c.chromsizes)-2)
                chr_size = np.sum(c.chromsizes[chr_index])
                x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
                while x in sv_areas:
                    x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
                y = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
                while y in sv_areas:
                    y = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
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

    def get_matrix(self, mat_bal):
            #mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat250 = np.nanmean(np.nanmean(np.reshape(mat_bal, (self.image_size, 1, self.image_size, 1)), axis=3), axis=1)
            mat_logb = mat250 - self.normmat
            mat_logb[np.isnan(mat_logb)] = 0
            return mat_logb

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
        if row.is_sv:
            shift = random.randint(-pad//2, pad//2)
            x += shift
            y += shift
        if x-pad < 0 or y - pad < 0:
            mv = max(-(x-pad), -(y-pad))
            x+=mv
            y+=mv
        if x+pad > matrix_full[0].shape[0] or y + pad > matrix_full[0].shape[0]:
            x = min(x,  matrix_full[0].shape[0]-pad-1)
            y = min(y,  matrix_full[0].shape[0]-pad-1)
        #mat_r = matrix_full[0][x-pad:x+pad, y-pad:y+pad]
        mat_b = matrix_full[x-pad:x+pad, y-pad:y+pad]
        #mat_clean_b = matrix_full[2][x-pad:x+pad, y-pad:y+pad]
        #mat_clean_r = matrix_full[3][x-pad:x+pad, y-pad:y+pad]

        mat_norm = self.get_matrix(mat_b)
        #mat_norm_clean = self.get_matrix(mat_clean_b, mat_clean_r, self.normmat250_clean, self.eps_clean, True, True)

        #mat = torch.from_numpy(mat_norm-mat_norm_clean).to(device=device, dtype=torch.float)
        #mat_norm = 2.*(mat_norm - np.min(mat_norm))/np.ptp(mat_norm)-1
        if row.is_sv and self.images_to_save > 0:
            os.makedirs('saved_matrices_normmat_subs', exist_ok=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.matshow(mat_norm, cmap='bwr')
            fig.colorbar(im)
            plt.savefig(f'saved_matrices_normmat_subs/after_normmat_{self.resolution//1000}_{self.images_to_save}_{sv_info.chr}_{x}_{y}_{shift}.png')
            plt.close()
            self.images_to_save-=1
        mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)
        #mat = torch.nan_to_num(mat)
        try:
            tens = mat.reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].shape)
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)

        return tens, 1 if row.is_sv else 0


train_coolers = [f'{local_path}data/mcool/Gor_SV_4DN.mcool',f'{local_path}data/mcool/Gor_SV_2_4DN.mcool']
clean_train_coolers = [f'{local_path}data/mcool/Gor_4DN.mcool',f'{local_path}data/mcool/Gor_4DN.mcool']
train_csvs = [f'{local_path}data/sv_csv/train_filtered_gor_sv.csv',f'{local_path}data/sv_csv/train_filtered_gor_sv_2.csv'] 
test_csvs = [f'{local_path}data/sv_csv/test_filtered_gor_sv.csv',f'{local_path}data/sv_csv/test_filtered_gor_sv_2.csv'] 

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy',
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy',
    save_images = 10)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 80
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader

train_model(dataloaders, model, 'log_normmat_subs_25', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_normmat_subs/torch_ensemble_25k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy',
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy',
    save_images = 10)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 80
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader

train_model(dataloaders, model, 'log_normmat_subs_50', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_normmat_subs/torch_ensemble_50k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_10kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_10kb.npy',
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_10kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_10kb.npy',
    save_images = 10)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 80
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader

train_model(dataloaders, model, 'log_normmat_subs_10', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])
torch.save(model.state_dict(), f'weights_normmat_subs/torch_ensemble_10k_48_diag.pt')