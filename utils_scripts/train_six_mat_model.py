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

torch.manual_seed(42)
np.random.seed(42)

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

from collections import OrderedDict


class DetectBlock(nn.Module):
    def __init__(self, in_channels, image_size, batch_size):
            super(DetectBlock, self).__init__()
            self.batch_size = batch_size
            conv_layers = nn.Sequential(
                #image_size x image_size x 6
                nn.Conv2d(in_channels, in_channels*2,  kernel_size = 3, padding=1),
                #image_size x image_size x 12
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels*2, in_channels*3,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 18
                nn.BatchNorm2d(in_channels*3),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels*3, in_channels*4,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 24
                nn.BatchNorm2d(in_channels*4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #(image_size/4 x image_size/4 x 24
                nn.Conv2d(in_channels*4, in_channels*6, kernel_size=3, padding=1),
                #(image_size/4 x image_size/4 x 36
                nn.BatchNorm2d(in_channels*6),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),
                #((image_size/8 x image_size/8 x 36
            )
            #linear_input_features_size = ((image_size//8)**2) * (in_channels*6)
            #linear_layers = nn.Sequential(
            #    nn.Linear(linear_input_features_size, linear_input_features_size//2),
            #    nn.Dropout(0.3),
            #    nn.ReLU(),
            #    nn.Linear(linear_input_features_size//2, linear_input_features_size//4),
            #    nn.Dropout(0.2),
            #    nn.ReLU(),
            #    nn.Linear(linear_input_features_size//4,  linear_input_features_size//8),
            #    nn.Dropout(0.1),
            #    nn.ReLU()
            #)
            self.add_module('conv_layer', conv_layers)
            #self.add_module('linear_layer', linear_layers)


    def forward(self, x):
        output_conv = self.conv_layer(x)
        #output = self.linear_layer(torch.flatten(output_conv, 1))
        return output_conv

class DetectAssembleBlock(nn.ModuleDict):
    def __init__(self, in_channels, image_size, batch_size, num_models):
        super(DetectAssembleBlock, self).__init__()
        for i in range(num_models):
            block = DetectBlock(in_channels, image_size, batch_size)
            self.add_module('mini_block%d' % (i + 1), block)

    def forward(self, x):
        features = []
        for name, layer in self.items():
            output = layer(x)
            features.append(output)
        return torch.cat(features, 1)

class DetectModel(nn.Module):
    def __init__(self, in_channels=1, image_size=40, num_models=10, batch_size=512):
        super(DetectModel, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([]))
        self.features.add_module('super_block', DetectAssembleBlock(in_channels, image_size, batch_size, num_models))

        num_features = (((image_size//8)**2) * (in_channels*6)) * num_models
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(2048, 1024),
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

                    matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                    matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
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

            mat = np.log(mat_cg+self.eps[str(resolution)])
            mat[np.isnan(mat)] = 0
            mat-= np.log(self.normmat250[str(resolution)]+self.eps[str(resolution)])

            mat_clean = np.log(mat_cg_clean+self.eps_clean[str(resolution)])
            mat_clean[np.isnan(mat_clean)] = 0
            mat_clean -= np.log(self.normmat250_clean[str(resolution)]+self.eps_clean[str(resolution)])
            return np.array([mat, mat_clean])
    
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

            x += shift
            y += shift
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

            mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  resolution, True)

            mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)
            mat_list.append(mat)
        try:
            tens = torch.cat(mat_list).reshape((6, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].shape)
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)

        return tens, 1 if row.is_sv else 0


train_coolers = [f'{local_path}data/mcool/Siamang_SV_15_25_50.mcool',f'{local_path}data/mcool/Gor_SV_2_15_25_50.mcool']
clean_train_coolers = [f'{local_path}data/mcool/Gor_15_25_50.mcool',f'{local_path}data/mcool/Gor_15_25_50.mcool']
train_csvs = [f'{local_path}data/sv_csv/train_filtered_gor_sv.csv',f'{local_path}data/sv_csv/train_filtered_gor_sv_2.csv'] 
test_csvs = [f'{local_path}data/sv_csv/test_filtered_gor_sv.csv',f'{local_path}data/sv_csv/test_filtered_gor_sv_2.csv'] 

validate_coolers = [f'{local_path}data/mcool/Gor_CHM13_15_25_50.mcool',]
clean_validate_coolers = [f'{local_path}data/mcool/CHM13_15_25_50.mcool',]
validate_csvs = [f'{local_path}data/sv_csv/good_svs_gor_chm.csv',] 

normmats_path=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_15kb.npy',
               '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
               '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy']

normmats_clean_path=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_15kb.npy',
                     '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy',
                     '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy',]

normmats_path_val=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_15kb.npy',
                    '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_25kb.npy',
                    '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_50kb.npy']

normmats_clean_path_val=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_15kb.npy',
                         '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
                         '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy']


model_name = 'six_mat_normmat_cg_larger_model_15_25_50'
os.makedirs(f'weights_{model_name}', exist_ok=True)

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolutions=[15000, 25000, 50000],
    image_size=48,
    normmats_path=normmats_path,
    normmats_clean_path=normmats_clean_path,
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolutions=[15000, 25000, 50000],
    image_size=48,
    normmats_path=normmats_path,
    normmats_clean_path=normmats_clean_path,
    save_images = 10)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolutions=[15000, 25000, 50000],
    image_size=48,
    normmats_path=normmats_path_val,
    normmats_clean_path=normmats_clean_path_val,
    save_images = 10,
    validate=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=6, image_size=48, num_models=20, batch_size=batch_size)
model.to(device=device)
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 160
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['test'] = test_dataloader
dataloaders['validate'] = validate_dataloader

train_model(dataloaders, model, f'log_{model_name}', criterion, optimizer, device, num_epochs, phases= ['train', 'test'], model_name=model_name)
torch.save(model.state_dict(), f'weights_{model_name}/{model_name}.pt')

print(f'Trained {model_name}')