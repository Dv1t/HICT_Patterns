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
batch_size = 512
warnings.filterwarnings('ignore')

class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, normmat_path, normmat_clean_path, validate=False):
        sv_count = 0
        self.label_to_index = {'++':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               '+-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               '-+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               '--':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'negative':torch.tensor([1.0, 0.0, 0.0, 0.0,  0.0])}
        self.resolution = resolution
        self.image_size = image_size
        self.validate = validate
        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}

        normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
        normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = 1e-6

        normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
        normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250_clean = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps_clean = 1e-6
            
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
                matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrixes_by_chr[chr] = (matrix, matrix_clean)

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

    def get_matrix(self, mat_bal, mat_bal_clean):
            mat_bal = np.log(mat_bal+self.eps)
            mat_bal_clean = np.log(mat_bal_clean+self.eps_clean)
            mat_bal[np.isnan(mat_bal)] = 0
            mat_bal_clean[np.isnan(mat_bal_clean)] = 0
            
            #mat = mat_bal - mat_bal_clean
            return  mat_bal
    
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

        mat_b = matrix_full[0][x-pad:x+pad, y-pad:y+pad].todense()
        mat_b_clean = matrix_full[1][x-pad:x+pad, y-pad:y+pad].todense()

        mat_norm = self.get_matrix(mat_b, mat_b_clean)

        mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)
        try:
            tens = mat.reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].get_shape())
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

normmats_path=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_10kb.npy',
               '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
               '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy']

normmats_clean_path=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_10kb.npy',
                     '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_25kb.npy',
                     '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy',]

normmats_path_val=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_10kb.npy',
                    '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_25kb.npy',
                    '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_50kb.npy']

normmats_clean_path_val=['/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_10kb.npy',
                         '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
                         '/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy']




os.makedirs('weights_balance_single_10kb', exist_ok=True)

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=test_csvs,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_exp_1kb_50kb.npy')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy',
    validate=True)
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

train_model(dataloaders, model, 'log_balance_single_10kb', criterion, optimizer, device, num_epochs, phases= ['train', 'test'], model_name='balance_single_10kb')
torch.save(model.state_dict(), f'weights_balance_single_10kb/balance_single_10kb.pt')

print('Trained balance_single_10kb')