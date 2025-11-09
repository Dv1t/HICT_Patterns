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


local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

from collections import OrderedDict


class DetectBlock(nn.Module):
    def __init__(self, in_channels):
            super(DetectBlock, self).__init__()
            layers = nn.Sequential(
                #image_size x image_size x 1
                nn.Conv2d(in_channels, in_channels*3,  kernel_size = 3, padding=1),
                #image_size x image_size x 3
                nn.BatchNorm2d(in_channels*3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels*3, in_channels*8,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 8
                nn.BatchNorm2d(in_channels*8),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels*8, 32,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 32
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #(image_size/4 x image_size/4 x 32
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                #(image_size/4 x image_size/4 x 64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #((image_size/8 x image_size/8 x 64
            )
            self.add_module('seq_layer', layers)

    def forward(self, x):
        output = self.seq_layer(x)
        return output

class DetectAssembleBlock(nn.ModuleDict):
    def __init__(self, in_channels, num_models):
        super(DetectAssembleBlock, self).__init__()
        for i in range(num_models):
            block = DetectBlock(in_channels)
            self.add_module('mini_block%d' % (i + 1), block)

    def forward(self, x):
        features = []
        for name, layer in self.items():
            output = layer(x)
            features.append(output)
        return torch.cat(features, 1)

class DetectModel(nn.Module):
    def __init__(self, in_channels=1, image_size=40, num_models=10):
        super(DetectModel, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([]))
        self.features.add_module('super_block', DetectAssembleBlock(in_channels, num_models))

        num_features = ((image_size//8)**2) * 64 * num_models
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.features(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out

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
            self.coolers_list.append(c)
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix_raw = c.matrix(balance=False)
                matrix__raw_clean = c_clean.matrix(balance=False)
                #f_matrix_raw = np.log(matrix+self.eps) - np.log(matrix_clean+self.eps_clean)

                matrix = c.matrix(balance=True)
                matrix_clean = c_clean.matrix(balance=True)
                #f_matrix = np.log(matrix_raw+self.eps) - np.log(matrix_clean+self.eps_clean)
                matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix__raw_clean)
                    
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
        return self.indexes.shape[0]*2

    def get_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, normmat, normmat_clean, coarse_grain=False):
            if coarse_grain:
                mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
                mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
            else:
                mat_cg = mat_bal
                mat_cg_clean = mat_bal_clean
            mat = np.log(mat_cg+self.eps) - np.log(mat_cg_clean+self.eps_clean)
            mat250 = np.nanmean(np.nanmean(np.reshape(mat, (self.image_size, 1, self.image_size, 1)), axis=3), axis=1)
            normmat250  = np.log(normmat+self.eps)-np.log(normmat_clean+self.eps_clean)
            mat_logb = mat250 - normmat250
            mask = np.isnan(mat_logb).astype(int)
            mat_logb[np.isnan(mat_logb)] = 0
            return np.array([mat_logb, mask])

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
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        if idx_d % 2 == 0:
            x = (sv_info.start)//self.resolution
            y = (sv_info.start)//self.resolution
        else:
            x = (sv_info.end)//self.resolution
            y = (sv_info.end)//self.resolution
        chr_padding = self.get_chromosme_padding(c, sv_info.chr) // self.resolution
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
        x+=chr_padding
        y+=chr_padding
        mat_b = matrix_full[0][x-pad:x+pad, y-pad:y+pad]
        mat_r = matrix_full[1][x-pad:x+pad, y-pad:y+pad]
        mat_b_clean = matrix_full[2][x-pad:x+pad, y-pad:y+pad]
        mat_r_clean = matrix_full[3][x-pad:x+pad, y-pad:y+pad]

        mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  self.normmat250, self.normmat250_clean)
        #mat_norm_clean = self.get_matrix(mat_clean_b, mat_clean_r, self.normmat250_clean, self.eps_clean, True, True)

        #mat = torch.from_numpy(mat_norm-mat_norm_clean).to(device=device, dtype=torch.float)
        #mat_norm = 2.*(mat_norm - np.min(mat_norm))/np.ptp(mat_norm)-1
        if row.is_sv and self.images_to_save > 0:
            os.makedirs('saved_matrices_normmat_subs', exist_ok=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.matshow(mat_norm[0], cmap='bwr')
            fig.colorbar(im)
            plt.savefig(f'saved_matrices_normmat_subs/after_normmat_{self.resolution//1000}_{self.images_to_save}_{sv_info.chr}_{x}_{y}_{shift}.png')
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.matshow(np.log(mat_r), cmap='Greens')
            fig.colorbar(im)
            plt.savefig(f'saved_matrices_normmat_subs/raw_mat_{self.resolution//1000}_{self.images_to_save}_{sv_info.chr}_{x}_{y}_{shift}.png')
            plt.close()
            self.images_to_save-=1
        mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)

        #mat = torch.nan_to_num(mat)
        try:
            tens = mat.reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
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

validate_coolers = [f'{local_path}data/mcool/Gor_CHM13_4DN.mcool',]
clean_validate_coolers = [f'{local_path}data/mcool/CHM13_4DN.mcool',]
validate_csvs = [f'{local_path}data/sv_csv/good_svs_gor_chm.csv',] 

os.makedirs('weights_normmat_subs_with_mask', exist_ok=True)

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

validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_10kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_10kb.npy',
    save_images = 10)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 160
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

train_model(dataloaders, model, 'log_normmat_subs_masked_10', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])
torch.save(model.state_dict(), f'weights_normmat_subs_with_mask/torch_ensemble_10k_48_diag.pt')


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

validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
    save_images = 10)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 160
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

train_model(dataloaders, model, 'log_normmat_subs_masked_25', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_normmat_subs_with_mask/torch_ensemble_25k_48_diag.pt')


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

validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=50000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy',
    save_images = 10)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 160
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

train_model(dataloaders, model, 'log_normmat_subs_masked_50', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_normmat_subs_with_mask/torch_ensemble_50k_48_diag.pt')