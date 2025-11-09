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
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, validate=False):
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
        self.eps = 1e-9
        self.normmat250_clean = {}
        self.eps_clean = {}
            
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

    def __getitem__(self, idx_d):
        idx = idx_d//2
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        mat_list = []
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr][0]
        matrix_full_clean = self.matrixes_list[row.file_index][sv_info.chr][1]

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
        if x+pad > matrix_full.get_shape()[0] or y + pad > matrix_full.get_shape()[0]:
            x = min(x,  matrix_full.get_shape()[0]-pad-1)
            y = min(y,  matrix_full.get_shape()[0]-pad-1)

        mat_numpy = np.log(matrix_full[x-pad:x+pad, y-pad:y+pad].todense()+self.eps)
        mat_numpy_clean = np.log(matrix_full_clean[x-pad:x+pad, y-pad:y+pad].todense()+self.eps)
        mat_numpy[np.isnan(mat_numpy)] = 0
        mat_numpy_clean[np.isnan(mat_numpy_clean)] = 0
        #mat_numpy_log_change = mat_numpy - mat_numpy_clean
        mat = torch.from_numpy(np.array((mat_numpy, mat_numpy_clean))).to(device=device, dtype=torch.float)
        try:
            tens = mat.reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full.get_shape())
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


validate_dataset = TrainDatasetDiagonal(
    cooler_path_list=validate_coolers,
    trans_csv_path_list=validate_csvs,
    clean_cooler_list=clean_validate_coolers,
    resolution=25000,
    image_size=48,
    validate=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_balance_two_matrices/torch_ensemble_balance_48_diag.pt', map_location=device))
model.eval()
learning_rate = 1e-6
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
dataloaders = dict()
dataloaders['test'] = validate_dataloader

train_model(dataloaders, model, 'log_balance_two_matrices', criterion, optimizer, device, num_epochs, phases= ['test'], model_name='balance_two_matrices_test')
print('Trained balance_two_matrices')