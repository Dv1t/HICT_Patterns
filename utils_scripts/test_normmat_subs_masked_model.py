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
import random
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from cooltools.lib.numutils import adaptive_coarsegrain
from matplotlib import colors

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

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
                matrix_raw = c.matrix(balance=False).fetch(chr)
                matrix__raw_clean = c_clean.matrix(balance=False).fetch(chr)
                #f_matrix_raw = np.log(matrix+self.eps) - np.log(matrix_clean+self.eps_clean)

                matrix = c.matrix(balance=True).fetch(chr)
                matrix_clean = c_clean.matrix(balance=True).fetch(chr)
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
            shift = 0#random.randint(-pad//2, pad//2)
            x += shift
            y += shift
        if x-pad < 0 or y - pad < 0:
            mv = max(-(x-pad), -(y-pad))
            x+=mv
            y+=mv
        if x+pad > matrix_full[0].shape[0] or y + pad > matrix_full[0].shape[0]:
            x = min(x,  matrix_full[0].shape[0]-pad-1)
            y = min(y,  matrix_full[0].shape[0]-pad-1)

        mat_b = matrix_full[0][x-pad:x+pad, y-pad:y+pad]
        mat_r = matrix_full[1][x-pad:x+pad, y-pad:y+pad]
        mat_b_clean = matrix_full[2][x-pad:x+pad, y-pad:y+pad]
        mat_r_clean = matrix_full[3][x-pad:x+pad, y-pad:y+pad]

        mat_norm = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean,  self.normmat250, self.normmat250_clean)
        mat_norm_cg = self.get_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean, self.normmat250, self.normmat250_clean, coarse_grain=True)

        #mat_norm_clean = self.get_matrix(mat_clean_b, mat_clean_r, self.normmat250_clean, self.eps_clean, True, True)

        #mat = torch.from_numpy(mat_norm-mat_norm_clean).to(device=device, dtype=torch.float)
        #mat_norm = 2.*(mat_norm - np.min(mat_norm))/np.ptp(mat_norm)-1
        if row.is_sv and self.images_to_save > 0:
            #1 столбец - raw log + eps
            #2 столбец - balanced mat (no SV) log + eps
            #3 столбец - normmat 
            #4 столбец - coarse grain
            #1 строка - mat (no SV) 
            #2 строка - mat (sv)
            #3 строка - subtraction
            ### в красно-синих матрица - 0 белый
            os.makedirs('saved_matrices_normmat_subs', exist_ok=True)
            fig, axs = plt.subplots(3, 4, figsize=(25,25))

            ###1 строка - mat (no sv) 
            raw_log_clean = axs[0,0].matshow(np.log(mat_r_clean+self.eps_clean), cmap='Greens')
            axs[0,0].set_title('No SV, raw log + eps')
            fig.colorbar(raw_log_clean)

            balance_log_clean = axs[0,1].matshow(np.log(mat_b_clean+self.eps_clean), cmap='Greens')
            axs[0,1].set_title('No SV, balanced log + eps')
            fig.colorbar(balance_log_clean)

            normmat_clean_mat = np.log(mat_b_clean+self.eps_clean)-np.log(self.normmat250_clean+self.eps_clean)
            divnorm_normmat_clean=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(normmat_clean_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(normmat_clean_mat)))
            normmat_clean = axs[0,2].matshow(normmat_clean_mat, cmap='bwr', norm=divnorm_normmat_clean)   
            axs[0,2].set_title('No SV, normmat')         
            fig.colorbar(normmat_clean)

            normmat_cg_clean_mat = np.log(adaptive_coarsegrain(mat_b_clean, mat_r_clean)+self.eps_clean)-np.log(self.normmat250_clean+self.eps_clean)
            divnorm_normmat_cg_clean=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(normmat_cg_clean_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(normmat_cg_clean_mat)))
            normmat_cg_clean = axs[0,3].matshow(normmat_cg_clean_mat, cmap='bwr', norm=divnorm_normmat_cg_clean)
            axs[0,3].set_title('No SV, normmat + coarse grain')
            fig.colorbar(normmat_cg_clean)

            ###2 строка - mat (sv) 
            raw_log_sv = axs[1,0].matshow(np.log(mat_r+self.eps), cmap='Greens')
            axs[1,0].set_title('SV, raw log + eps')
            fig.colorbar(raw_log_sv)

            balance_log_sv = axs[1,1].matshow(np.log(mat_b+self.eps), cmap='Greens')
            axs[1,1].set_title('SV, balanced log + eps')
            fig.colorbar(balance_log_sv)

            normat_sv_mat = np.log(mat_b+self.eps)-np.log(self.normmat250+self.eps)
            divnorm_normat_sv=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(normat_sv_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(normat_sv_mat)))
            normmat_sv = axs[1,2].matshow(normat_sv_mat, cmap='bwr', norm=divnorm_normat_sv)
            axs[1,2].set_title('SV, normmat') 
            fig.colorbar(normmat_sv)

            normmat_cg_sv_mat =np.log(adaptive_coarsegrain(mat_b, mat_r)+self.eps)-np.log(self.normmat250+self.eps)
            divnorm_normat_sv_cg=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(normmat_cg_sv_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(normmat_cg_sv_mat)))
            normmat_cg_sv = axs[1,3].matshow(normmat_cg_sv_mat, cmap='bwr',norm=divnorm_normat_sv_cg)
            axs[1,3].set_title('SV, normmat + coarse grain')
            fig.colorbar(normmat_cg_sv)

            ###3 строка - subtraction
            raw_subs_mat = np.log(mat_r+self.eps)-np.log(mat_r_clean+self.eps)
            divnorm_raw_subs=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(raw_subs_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(raw_subs_mat)))
            raw_subs = axs[2,0].matshow(raw_subs_mat, cmap='bwr', norm=divnorm_raw_subs)
            axs[2,0].set_title('subs, raw log change + eps')
            fig.colorbar(raw_subs)
            
            balance_subs_mat = np.log(mat_b+self.eps)-np.log(mat_b_clean+self.eps)
            divnorm_balance_subs=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(balance_subs_mat)), vcenter=0., vmax=max(0.00001,np.nanmax(balance_subs_mat)))
            balance_subs = axs[2,1].matshow(balance_subs_mat, cmap='bwr', norm=divnorm_balance_subs)
            axs[2,1].set_title('subs, balanced log change + eps')
            fig.colorbar(balance_subs)

            divnorm_normmat_subs=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(mat_norm[0])), vcenter=0., vmax=max(0.00001,np.nanmax(mat_norm[0])))
            normmat_subs = axs[2,2].matshow(mat_norm[0], cmap='bwr', norm=divnorm_normmat_subs)
            axs[2,2].set_title('subs, normmat')
            fig.colorbar(normmat_subs)

            divnorm_normmat_subs_cg=colors.TwoSlopeNorm(vmin=min(-0.00001, np.nanmin(mat_norm_cg[0])), vcenter=0., vmax=max(0.00001,np.nanmax(mat_norm_cg[0])))
            normmat_subs_cg = axs[2,3].matshow(mat_norm_cg[0], cmap='bwr', norm=divnorm_normmat_subs_cg)
            axs[2,3].set_title('subs, normmat + coarse grain')
            fig.colorbar(normmat_subs_cg)

            plt.savefig(f'saved_matrices_normmat_subs/test_after_normmat_{self.resolution//1000}_{self.images_to_save}_{sv_info.chr}_{x}_{y}.png')
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


def run_epoch(model, phase, dataloader):
  if phase == 'train':
      model.train()
  else:
      model.eval()

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
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.round(outputs)
    y_test.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    running_loss += loss.item() * bz
    corrects_cnt = torch.sum(preds == labels.detach())
    running_corrects += corrects_cnt
    show_dict = {'Loss': f'{loss.item():.6f}',
                'Corrects': f'{corrects_cnt.item()}/{bz}',
                'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
    cur_tqdm.set_postfix(show_dict)

  conf_matrix = 0#'metrics.confusion_matrix(y_test, y_pred)'
  y_test = np.array(y_test)
  y_pred = np.array(y_pred)
  tp = np.sum((y_test == y_pred) & (y_pred==1))
  tn = np.sum((y_test == y_pred) & (y_pred==0))
  fp = np.sum((y_test != y_pred) & (y_pred==1))
  fn = np.sum((y_test != y_pred) & (y_pred==0))
  print('tp', tp)
  print('tn', tn)
  print('fp', fp)
  print('fn', fn)

  print("Calculating metrics...")
  f05_macro = 0#metrics.fbeta_score(y_test, y_pred, average="macro", beta=0.5)
  f1_macro = 0#metrics.f1_score(y_test, y_pred, average="macro")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  print(f'{running_corrects.float().item()}/{all_elems_count}')
  return epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix

def test_epoch(model, dataloader):
    with torch.inference_mode():
      return run_epoch(model,'test', dataloader)

def train_epoch(model, dataloader):
    return run_epoch(model, 'train', dataloader)




log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

def train_model(dataloaders, model, num_epochs=20, phases= ['test']):
  print(f"Training model with params:")
  print(f"Optim: {optimizer}")
  print(f"Criterion: {criterion}")

  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}
  saved_epoch_f1_macros = {phase: [] for phase in phases}

  for epoch in range(1, num_epochs + 1):
      start_time = time.time()

      print("=" * 100)
      print(f'Epoch {epoch}/{num_epochs}')
      print('-' * 10)

      for phase in phases:
          print("--- Cur phase:", phase)
          epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix = \
              train_epoch(model, dataloaders[phase]) if phase == 'train' \
                  else test_epoch(model, dataloaders[phase])
          saved_epoch_losses[phase].append(epoch_loss)
          saved_epoch_accuracies[phase].append(epoch_acc)
          saved_epoch_f1_macros[phase].append(f1_macro)
          print(f'{phase} loss: {epoch_loss:.6f}, '
                f'acc: {epoch_acc:.6f}, '
                f'f05_macro: {f05_macro:.6f}, '
                f'f1_macro: {f1_macro:.6f}')
          print("Confusion matrix:")
          print(conf_matrix)

      end_time = time.time()
      epoch_time = end_time - start_time
      print("-" * 10)
      print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies, saved_epoch_f1_macros


train_coolers = [f'{local_path}data/mcool/Gor_CHM13_4DN.mcool',]
clean_train_coolers = [f'{local_path}data/mcool/CHM13_4DN.mcool',]
train_csvs = [f'{local_path}data/sv_csv/good_svs_gor_chm.csv',] 

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_10kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_10kb.npy',
    save_images = 20)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_normmat_subs_with_mask/torch_ensemble_10k_48_diag.pt', map_location=device))
model.eval()


learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 10k testing')
train_model(dataloaders, model, num_epochs)
print('Ended 10k testing')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_50kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_50kb.npy',
    save_images = 20)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_normmat_subs_with_mask/torch_ensemble_50k_48_diag.pt', map_location=device))
model.eval()

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 50k testing')
train_model(dataloaders, model, num_epochs)
print('Ended 50k testing')

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_CHM13_exp_1kb_25kb.npy',
    normmat_clean_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/CHM13_exp_1kb_25kb.npy',
    save_images = 20)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=2, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_normmat_subs_with_mask/torch_ensemble_25k_48_diag.pt', map_location=device))
model.eval()

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 25k testing')
train_model(dataloaders, model, num_epochs)
print('Ended 25k testing')
