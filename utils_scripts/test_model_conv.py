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
            
            sv_count+=sv_file.shape[0]
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
                
                chr_index = random.randint(0, len(c_clean.chromsizes)-2) # -2 for evade choosing chrM
                chr_size = np.sum(c_clean.chromsizes[chr_index])
                try:
                    x = random.randint((image_size//2)*resolution, chr_size - ((image_size//2)*resolution)-1)
                except ValueError:
                    print((image_size//2)*resolution, chr_size - ((image_size//2)*resolution)-1)
                    print(resolution, image_size, chr_size)
                    assert 1 == 0
                y =  random.randint(image_size*resolution, 10*image_size*resolution)
                neg_sv['chr'].append(c_clean.chromnames[chr_index])
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
            x += random.randint(-pad//2, pad//2)
            y += random.randint(-pad//2, pad//2)        
        if x-pad < 0 or y - pad < 0:
            mv = max(-(x-pad), -(y-pad))
            x+=mv
            y+=mv
        if x+pad > matrix_full.shape[0] or y + pad > matrix_full.shape[0]:
            x = min(x,  matrix_full.shape[0]-pad-1)
            y = min(y,  matrix_full.shape[0]-pad-1)

        mat = matrix_full[x-pad:x+pad, y-pad:y+pad]
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
            return tens, 1 if row.is_sv else 0
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
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class DetectModel(nn.Module):
    def __init__(self, in_channels=1, image_size=40, num_models=10):
        super(DetectModel, self).__init__()
        
        layers = nn.Sequential(
                #image_size x image_size x 1
                nn.Conv2d(in_channels, 3*num_models,  kernel_size = 3, padding=1),
                #image_size x image_size x 3
                nn.BatchNorm2d(3*num_models),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(3*num_models, 8*num_models,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 8
                nn.BatchNorm2d(8*num_models),
                nn.ReLU(inplace=True),
                nn.Conv2d(8*num_models, 32*num_models,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 32
                nn.BatchNorm2d(32*num_models),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #(image_size/4 x image_size/4 x 32
                nn.Conv2d(32*num_models, 64*num_models, kernel_size=3, padding=1),
                #(image_size/4 x image_size/4 x 64
                nn.BatchNorm2d(64*num_models),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                #((image_size/8 x image_size/8 x 64
            )
        self.add_module('features', layers)


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

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_single_conv/torch_ensemble_10k_48_diag.pt', map_location=device))
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
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_single_conv/torch_ensemble_50k_48_diag.pt', map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()
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
    detection=True,
    blur=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DetectModel(in_channels=1, image_size=48, num_models=10)
model.to(device=device)
model.load_state_dict(torch.load(f'weights_single_conv/torch_ensemble_25k_48_diag.pt', map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()
num_epochs = 1
dataloaders = dict()
dataloaders['test'] = train_dataloader
print('Started 25k testing')
train_model(dataloaders, model, num_epochs)
print('Ended 25k testing')
