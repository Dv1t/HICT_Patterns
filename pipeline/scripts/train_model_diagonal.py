import torch
from torch import nn
import cooler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
import random
import torch.optim as optim
import argparse
import time
import math
from cooltools.lib.numutils import adaptive_coarsegrain

from models import DetectModel

torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1024
warnings.filterwarnings('ignore')

class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, normmat_paths, normmat_clean_paths):
        self.resolution = resolution
        self.image_size = image_size

        self.normat_list = []
        self.eps_list = []
        for normmat_path in normmat_paths:
            normmat_bydist = np.exp(np.load(normmat_path))[:image_size*4]
            normmat = normmat_bydist[np.abs(np.arange(image_size*4)[:, None] - np.arange(image_size*4)[None, :])]
            normmat250 = np.reshape(normmat, (image_size, 4, image_size, 4)).mean(axis=1).mean(axis=2)
            eps = np.min(normmat250)
            if eps <= 0:
                eps=np.finfo(np.double).tiny
            self.normat_list.append(normmat250)
            self.eps_list.append(eps)

        self.normat_clean_list = []
        self.eps_clean_list = []
        for normmat_clean_path in normmat_clean_paths:
            normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*4])
            normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*4)[:, None] - np.arange(image_size*4)[None, :])]
            normmat250_clean = np.reshape(normmat_clean, (image_size, 4, image_size, 4)).mean(axis=1).mean(axis=2)
            eps_clean = np.min(normmat250_clean)
            if eps_clean <= 0:
                eps_clean=np.finfo(np.double).tiny
            self.normat_clean_list.append(normmat250_clean)
            self.eps_clean_list.append(eps_clean)
            
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = []
        self.matrixes_list = []
        self.sv_files_list = []
        for trans_csv_path, cooler_path, clean_cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
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
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]*2


    def get_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, eps, normmat250, eps_clean, normmat250_clean, coarse_grain=False):
        if coarse_grain:
            mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
        else:
            mat_cg = mat_bal
            mat_cg_clean = mat_bal_clean

        mat_cg = np.log(mat_cg+eps)
        mat_cg_clean = np.log(mat_cg_clean+eps_clean)
        mat_cg[np.isnan(mat_cg)] = 0
        mat_cg_clean[np.isnan(mat_cg_clean)] = 0
        
        mat_cg = mat_cg - np.log(normmat250+eps)
        mat_cg_clean = mat_cg_clean - np.log(normmat250_clean+eps_clean)

        return np.array([mat_cg, mat_cg_clean])
    
    def __getitem__(self, idx_d):
        idx = idx_d//2
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        eps = self.eps_list[row.file_index]
        normmat250 = self.normat_list[row.file_index]
        eps_clean = self.eps_clean_list[row.file_index]
        normmat250_clean = self.normat_clean_list[row.file_index]
        
        if idx_d % 2 == 0:
            x = (sv_info.start)//self.resolution
            y = (sv_info.start)//self.resolution
        else:
            x = (sv_info.end)//self.resolution
            y = (sv_info.end)//self.resolution
        pad = self.image_size//2
        to_mask = False
        if row.is_sv:
            shift = random.randint(-pad//2, pad//2)
            to_mask = np.random.random() >= 0.8
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

        mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean, eps, normmat250, eps_clean, normmat250_clean,  True)
        if to_mask:
            if shift > 0:
                mat_norm[0][0:pad+shift-1,0:pad+shift-1] = 0
                mat_norm[1][0:pad+shift-1,0:pad+shift-1] = 0
            else:
                mat_norm[0][pad+shift+1:,pad+shift+1:] = 0
                mat_norm[1][pad+shift+1:,pad+shift+1:] = 0
        if np.random.random() > 0.6:
            mat_norm = np.rot90(mat_norm, k=2)
        tens = torch.from_numpy(mat_norm.copy()).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)

        return tens, 1 if row.is_sv else 0

def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    y_test = []
    y_pred = []
    all_elems_count = 0
    cur_tqdm = tqdm(dataloader)
    for inputs, labels in cur_tqdm:
        bz = inputs.shape[0]
        all_elems_count += bz
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.float)
        outputs = model(inputs)
        outputs = outputs.resize(outputs.shape[0])
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sigmoid  = nn.Sigmoid()
        preds = torch.round(sigmoid(outputs))
        y_test.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        running_loss += loss.item() * bz
        corrects_cnt = torch.sum(preds == labels.detach())
        running_corrects += corrects_cnt
        show_dict = {'Loss': f'{loss.item():.6f}',
                    'Corrects': f'{corrects_cnt.item()}/{bz}',
                    'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
        cur_tqdm.set_postfix(show_dict)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    tp = np.sum((y_test == y_pred) & (y_pred==1))
    tn = np.sum((y_test == y_pred) & (y_pred==0))
    fp = np.sum((y_test != y_pred) & (y_pred==1))
    fn = np.sum((y_test != y_pred) & (y_pred==0))
    print(f'Metrics are:')
    print('tp', tp)
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print("Calculating metrics...")
    epoch_loss = running_loss / all_elems_count
    epoch_acc = running_corrects.float().item() / all_elems_count
    return epoch_loss, epoch_acc

def train_model(dataloader, model, criterion, optimizer, device, num_epochs=20):
    print(f"Training model with params:")
    print(f"Optim: {optimizer}")
    print(f"Criterion: {criterion}")

    saved_epoch_losses = []
    saved_epoch_accuracies = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        print("=" * 100)
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)


        epoch_loss, epoch_acc = run_epoch(model, dataloader, criterion, optimizer, device)

        saved_epoch_losses.append(epoch_loss)
        saved_epoch_accuracies.append(epoch_acc)
        print(f'loss: {epoch_loss:.6f}, '
              f'acc: {epoch_acc:.6f}')
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print("-" * 10)
        print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

    print("*** Training Completed ***")

    return saved_epoch_losses, saved_epoch_accuracies

parser = argparse.ArgumentParser()
parser.add_argument('sv_mcools', type=str, help='List of paths to mcools with SV to train')
parser.add_argument('clean_mcools', type=str, help='List of paths to mcools without SV to highlight train mcools')
parser.add_argument('sv_expected', type=str, help='List of normmat paths for mcools without SV')
parser.add_argument('clean_expected', type=str, help='List of normmat paths for mcools with SV')
parser.add_argument('answers', type=str, help='List of files with coordinates of SV in mcools')
parser.add_argument('num_epoch', type=int, help='Number of epochs to train')
parser.add_argument('resolution', type=int, help='Resolution of HiC map to train for')

args = parser.parse_args()


train_coolers = args.sv_mcools.split(' ')
clean_train_coolers = args.clean_mcools.split(' ')
train_csvs =  args.answers.split(' ')
normmats = args.sv_expected.split(' ')
normmats_clean = args.clean_expected.split(' ')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    clean_cooler_list=clean_train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=int(args.resolution),
    image_size=48,
    normmat_paths=normmats,
    normmat_clean_paths=normmats_clean)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Total bp amount in train:', len(train_dataset))


model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
learning_rate =  1e-06
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = int(args.num_epoch)

train_model(train_dataloader, model, criterion, optimizer, device, num_epochs)

torch.save(model.state_dict(), f'weight_{str(args.resolution)}.pt')