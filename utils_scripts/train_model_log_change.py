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
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from train_methods import train_model
from hict.patterns.models import DetectModel


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
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, save_images=0):
        sv_count = 0
        self.label_to_index = {'++':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               '+-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               '-+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               '--':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'negative':torch.tensor([1.0, 0.0, 0.0, 0.0,  0.0])}
        self.resolution = resolution
        self.image_size = image_size
        self.images_to_save = save_images
        self.eps = 1e-6

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
                matrix = c.matrix(balance=False).fetch(chr)
                matrix_clean = c_clean.matrix(balance=False).fetch(chr)
                f_matrix_raw = np.log(matrix+self.eps) - np.log(matrix_clean+self.eps)

                matrixes_by_chr[chr] = (f_matrix_raw)
                    
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
                neg_sv['end'].append(y)

                indexes['file_index'].append(index)
                indexes['in_index'].append(sv_file.shape[0]+i)
                indexes['is_sv'].append(False)
            sv_file = pd.concat([sv_file, pd.DataFrame(neg_sv)])
            self.sv_files_list.append(sv_file)
        self.num_classes = 2
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]*2

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
        if x+pad > matrix_full.shape[0] or y + pad > matrix_full.shape[0]:
            x = min(x,  matrix_full[0].shape[0]-pad-1)
            y = min(y,  matrix_full[0].shape[0]-pad-1)
        mat_norm = matrix_full[x-pad:x+pad, y-pad:y+pad]

        if row.is_sv and self.images_to_save > 0:
            os.makedirs('saved_matrices_log_change', exist_ok=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.matshow(mat_norm, cmap='bwr')
            fig.colorbar(im)
            plt.savefig(f'saved_matrices_log_change/after_log_change_{self.resolution//1000}_{self.images_to_save}_{sv_info.chr}_{x}_{y}_{shift}.png')
            plt.close()
            self.images_to_save-=1
        mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)

        tens = mat.reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)

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
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=25000,
    image_size=48,
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

train_model(dataloaders, model, 'log_log_change_25', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_log_change/torch_ensemble_25k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
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

train_model(dataloaders, model, 'log_log_change_50', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])

torch.save(model.state_dict(), f'weights_log_change/torch_ensemble_50k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
    save_images = 10)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=test_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
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

train_model(dataloaders, model, 'log_log_change_10', criterion, optimizer, device, num_epochs, phases= ['train', 'test'])
torch.save(model.state_dict(), f'weights_log_change/torch_ensemble_10k_48_diag.pt')