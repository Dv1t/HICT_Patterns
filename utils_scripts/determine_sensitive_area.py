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

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from hict.patterns.help_functions import get_chromosome_coords, get_genome_coords
from hict.patterns.models import DetectModel, ClassificationModel


local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

class TrainDatasetDiagonal(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, detection=True, blur=True):
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
                    
        print('Loaded clean cooler')
        indexes = {'file_index':[], 'in_index':[], 'is_sv':[]}
        self.coolers_list = []
        self.matrixes_list = []
        self.sv_files_list = []
        for trans_csv_path, cooler_path, clean_cooler_path, index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            
            sv_count+=(sv_file.shape[0]*image_size)
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix = c.matrix(balance=False).fetch(chr)
                matrix_clean = c_clean.matrix(balance=False).fetch(chr)
                
                f_matrix = np.log10(matrix+1e-6) - np.log10(matrix_clean+1e-6)
                f_matrix = 2.*(f_matrix - np.min(f_matrix))/np.ptp(f_matrix)-1

                matrixes_by_chr[chr] = torch.from_numpy(f_matrix).to(device=device, dtype=torch.float)
                    
            self.matrixes_list.append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            for i, sv in sv_file.iterrows():
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
            sv_file = pd.concat([sv_file, pd.DataFrame(neg_sv)])
            self.sv_files_list.append(sv_file)
        if detection:
            self.num_classes = 2
        else:     
            self.num_classes = len(self.sv_files_list[0].label.unique()) + 1
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]*2

    def __getitem__(self, idx_d):
        idx = idx_d//self.image_size
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
            shift = 24 - int(idx_d % self.image_size)
            x += shift
            y += shift
        else:
            assert 1 == 0
        if x-pad < 0 or y - pad < 0:
            mv = max(-(x-pad), -(y-pad))
            x+=mv
            y+=mv
        if x+pad > matrix_full.shape[0] or y + pad > matrix_full.shape[0]:
            x = min(x,  matrix_full.shape[0]-pad-1)
            y = min(y,  matrix_full.shape[0]-pad-1)

        mat = matrix_full[int(x-pad):int(x+pad), int(y-pad):int(y+pad)]
        #mat = torch.nan_to_num(mat)
        try:
            tens = mat.reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        except RuntimeError:
            print(matrix_full.shape)
            print(x)
            print(y)
            print(sv_info.chr.iloc[0])
            print(row.is_sv)
        if self.use_blur:
            tens = self.blur(tens)
        if self.detection:
            return tens, 1 if row.is_sv else 0, shift
        else:
            return tens, self.label_to_index[sv_info.label]


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
  true_shifts_epoch = []
  for inputs, labels, shifts in cur_tqdm:
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
    true_shifts = shifts[torch.tensor(preds, dtype=bool)]
    true_shifts_epoch.extend(true_shifts.detach().cpu().numpy())
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
  return epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix, true_shifts_epoch

def test_epoch(model, dataloader):
    with torch.inference_mode():
      return run_epoch(model,'test', dataloader)

def train_epoch(model, dataloader):
    return run_epoch(model, 'train', dataloader)




log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

def train_model(dataloaders, model, label,num_epochs=20, phases= ['test']):
  print(f"Training model with params:")
  print(f"Optim: {optimizer}")
  print(f"Criterion: {criterion}")

  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}
  saved_epoch_f1_macros = {phase: [] for phase in phases}
  all_true_shifts = []
  for epoch in range(1, num_epochs + 1):
      start_time = time.time()

      print("=" * 100)
      print(f'Epoch {epoch}/{num_epochs}')
      print('-' * 10)

      for phase in phases:
          print("--- Cur phase:", phase)
          epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix, true_shifts = \
              train_epoch(model, dataloaders[phase]) if phase == 'train' \
                  else test_epoch(model, dataloaders[phase])
          all_true_shifts.extend(true_shifts)
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
  all_true_shifts = np.array(all_true_shifts)
  np.savetxt(f"{local_path}/tp_shifts_{label}.csv",
    all_true_shifts,
    delimiter =",",
    fmt ='% s')
  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies, saved_epoch_f1_macros


train_coolers = [f'{local_path}data/mcool/Gor_CHM13_4DN.mcool',]
clean_train_coolers = [f'{local_path}data/mcool/CHM13_4DN.mcool',]
train_csvs = [f'{local_path}data/sv_csv/good_svs_gor_chm_50kb_all.csv',] 

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=10000,
    image_size=48,
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_10k_48_diag.pt', map_location=device))
model.eval()


learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 10k testing')
train_model(dataloaders, model, '10k', num_epochs)
print('Ended 10k testing')

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=50000,
    image_size=48,
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_50k_48_diag.pt', map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()
num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 50k testing')
train_model(dataloaders, model, '50k', num_epochs)
print('Ended 50k testing')

train_dataset = TrainDatasetDiagonal(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler_list=clean_train_coolers,
    resolution=25000,
    image_size=48,
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_updated_normalization/torch_ensemble_25k_48_diag.pt', map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()
num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 25k testing')
train_model(dataloaders, model, '25k', num_epochs)
print('Ended 25k testing')
