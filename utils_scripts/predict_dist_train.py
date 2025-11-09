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
from collections import OrderedDict

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from hict.patterns.help_functions import get_chromosome_coords, get_genome_coords
from hict.patterns.models import DetectAssembleBlock


local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, normmat_path, detection=True, blur=True):
        sv_count = 0
        self.label_to_index = {'++':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               '+-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               '-+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               '--':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'negative':torch.tensor([1.0, 0.0, 0.0, 0.0,  0.0])}
        self.resolution = resolution
        self.image_size = image_size
        self.detection = detection
        if blur:
            self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.use_blur = blur
        normmat_bydist = np.exp(np.load(normmat_path))[:image_size]
        normmat = normmat_bydist[np.abs(np.arange(image_size)[:, None] - np.arange(image_size)[None, :])]
        self.normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = np.min(self.normmat250)
            
        print('Loaded clean cooler')
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = []
        self.matrixes_list = []
        self.sv_files_list = []
        for trans_csv_path, cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
            sv_count+=sv_file.shape[0]
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix = c.matrix(balance=True).fetch(chr)
                matrix_raw = c.matrix(balance=False).fetch(chr)
                matrixes_by_chr[chr] = (matrix, matrix_raw)
                    
            self.matrixes_list.append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            for i, sv in sv_file.iterrows():
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c.chromsizes)-1)
                chr_size = np.sum(c.chromsizes[chr_index])
                x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
                y =  random.randint(image_size*resolution, 10*image_size*resolution)
                neg_sv['chr'].append(c.chromnames[chr_index])
                neg_sv['label'].append('negative')
                neg_sv['start'].append(x)
                neg_sv['end'].append(x+y)

                indexes['file_index'].append(index)
                indexes['in_index'].append(sv_file.shape[0]+i)
                indexes['is_sv'].append(False)
            sv_file = pd.concat([sv_file, pd.DataFrame(neg_sv)])
            self.sv_files_list.append(sv_file)
        if detection:
            self.num_classes = 2
        else:     
            self.num_classes = len(self.sv_files_list[0].label.unique()) + 1
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]*2

    def get_matrix(self, mat_bal, mat_raw, normmat250, eps, isCg, isLog=True):
            if isCg:
                mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
                mat_bal = mat_cg

            mat250 = np.nanmean(np.nanmean(np.reshape(mat_bal, (self.image_size, 1, self.image_size, 1)), axis=3), axis=1)
            mat250[np.isnan(mat250)] = 0
            if isLog:
                mat_logb = np.log(mat250 + eps) - np.log(normmat250 + eps)

            else:
                mat_logb = np.sqrt(mat250 / normmat250)
            return mat_logb

    def __getitem__(self, idx_d):
        idx = idx_d//2
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        if idx_d % 2 == 0:
            x = (sv_info.start)//self.resolution
        else:
            x = (sv_info.end)//self.resolution
        pad = self.image_size//2
        pad_x = 56
        if row.is_sv:
            pad_x = random.randint(-pad, pad)
            x += pad_x
        if x-pad < 0:
            mv = -(x-pad)
            x+=mv
        if x+pad > matrix_full[0].shape[0]:
            x = min(x,  matrix_full[0].shape[0]-pad-1)
        mat_b = matrix_full[0][x-pad:x+pad, x-pad:x+pad]
        mat_r = matrix_full[1][x-pad:x+pad, x-pad:x+pad]
        mat_norm = self.get_matrix(mat_b, mat_r, self.normmat250, self.eps, True, True)
        mat = torch.from_numpy(mat_norm).to(device=device, dtype=torch.float)
        try:
            tens = mat.reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full[0].shape)
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)
        if self.use_blur:
            tens = self.blur(tens)
        if self.detection:
            return tens, 1 if row.is_sv else 0
        else:
            return tens, (pad_x+pad)//12 if row.is_sv else 5


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
    labels = labels.to(device, non_blocking=True, dtype=torch.long)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.argmax(outputs, dim=1)
    y_test.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    running_loss += loss.item() * bz
    corrects_cnt = torch.sum(preds == torch.argmax(labels.detach(), dim=0))
    running_corrects += corrects_cnt
    show_dict = {'Loss': f'{loss.item():.6f}',
                'Corrects': f'{corrects_cnt.item()}/{bz}',
                'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
    cur_tqdm.set_postfix(show_dict)

  print("Calculating metrics...")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  return epoch_loss, epoch_acc

def test_epoch(model, dataloader):
    with torch.inference_mode():
      return run_epoch(model,'test', dataloader)

def train_epoch(model, dataloader):
    return run_epoch(model, 'train', dataloader)


log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

def train_model(dataloaders, model, num_epochs=20):
  print(f"Training model with params:")
  print(f"Optim: {optimizer}")
  print(f"Criterion: {criterion}")

  phases = ['train']
  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}

  for epoch in range(1, num_epochs + 1):
      start_time = time.time()

      print("=" * 100)
      print(f'Epoch {epoch}/{num_epochs}')
      print('-' * 10)

      for phase in phases:
          print("--- Cur phase:", phase)
          epoch_loss, epoch_acc = \
              train_epoch(model, dataloaders[phase]) if phase == 'train' \
                  else test_epoch(model, dataloaders[phase])
          saved_epoch_losses[phase].append(epoch_loss)
          saved_epoch_accuracies[phase].append(epoch_acc)
          print(f'{phase} loss: {epoch_loss:.6f}, '
                f'acc: {epoch_acc:.6f}')

      if epoch == num_epochs:
        plt.title(f'Losses during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_losses['train'], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel(criterion.__class__.__name__)
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/loss_graph_epoch{epoch + 1}.png')
        plt.show()
        plt.close('all')

        plt.title(f'Accuracies during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_accuracies['train'], label='Train Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/acc_graph_epoch{epoch + 1}.png')
        plt.show()
        plt.close('all')

      end_time = time.time()
      epoch_time = end_time - start_time
      print("-" * 10)
      print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies

class ClassificationModel(nn.Module):
    def __init__(self, in_channels=1, image_size=40, num_models=10, num_classes=3):
        super(ClassificationModel, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([]))
        self.features.add_module('super_block', DetectAssembleBlock(in_channels, num_models))

        num_features = ((image_size//8)**2) * 64 * num_models
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        
    def forward(self, x):
        features = self.features(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out

train_coolers = [f'{local_path}data/mcool/Gor_SV_4DN.mcool',]
clean_train_coolers = [f'{local_path}data/mcool/Gor_4DN.mcool',]
train_csvs = [f'{local_path}data/sv_csv/mGorGor.pri_sv.csv',] 

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=25000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_25kb.npy',
    detection=False,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = ClassificationModel(in_channels=1, image_size=48, num_models=10, num_classes=6)
model.to(device=device)
learning_rate = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_normmat_dist/torch_ensemble_25k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=50000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_50kb.npy',
    detection=False,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = ClassificationModel(in_channels=1, image_size=48, num_models=10, num_classes=6)
model.to(device=device)
learning_rate = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_normmat_dist/torch_ensemble_50k_48_diag.pt')


train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    resolution=10000,
    image_size=48,
    normmat_path='/mnt/tank/scratch/vdravgelis/ClusterBuffer/apes_expected/Gor_SV_exp_1kb_10kb.npy',
    detection=False,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = ClassificationModel(in_channels=1, image_size=48, num_models=10, num_classes=6)
model.to(device=device)
learning_rate = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_normmat_dist/torch_ensemble_10k_48_diag.pt')