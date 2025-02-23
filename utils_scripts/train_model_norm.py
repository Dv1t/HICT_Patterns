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
            
            sv_count+=sv_file.shape[0]
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix = c.matrix(balance=True).fetch(chr)
                matrix_clean = c_clean.matrix(balance=True).fetch(chr)
                matrix+=1e-6
                matrix_clean+=1e-6
                matrix[np.isnan(matrix)] = 0
                matrix_clean[np.isnan(matrix_clean)] = 0
                f_matrix = np.log10(matrix+1e-6) - np.log10(matrix_clean+1e-6)
                f_matrix = 2.*(f_matrix - np.min(f_matrix))/np.ptp(f_matrix)-1
                matrixes_by_chr[chr] = f_matrix#torch.from_numpy(f_matrix).to(device=device, dtype=torch.float)
                    
            self.matrixes_list.append(matrixes_by_chr)
            neg_sv = {'chr':[], 'label':[], 'start':[], 'end':[]}
            for i, sv in sv_file.iterrows():
                indexes['file_index'].append(index)
                indexes['in_index'].append(i)
                indexes['is_sv'].append(True)
                
                chr_index = random.randint(0, len(c_clean.chromsizes)-1)
                chr_size = np.sum(c_clean.chromsizes[chr_index])
                x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
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

    def get_matrix(cooler):
        expected = pd.read_csv('', sep="\t")
        expectedsum = expected.groupby(["diag"]).agg({"n_valid": "sum", "balanced.sum": "sum"})
        expectedsum["balanced.avg"] = expectedsum["balanced.sum"] / expectedsum["n_valid"]

        from statsmodels.nonparametric.smoothers_lowess import lowess

        v = np.log(expectedsum["balanced.avg"].values)
        v = v[: np.min(np.argwhere(~np.isfinite(v)))]
        sv0 = lowess(
            v[int(400 / (res / 4000)) :], np.log(np.arange(int(400 / (res / 4000)), len(v)) + 1), frac=0.01
        )[:, 1]
        sv2 = lowess(
            v[int(400 / (res / 4000)) :], np.log(np.arange(int(400 / (res / 4000)), len(v)) + 1), frac=0.1
        )[:, 1]
        sv = np.hstack(
            [
                v[: int(400 / (res / 4000))],
                sv0[: int(10000 / (res / 4000))],
                sv2[int(10000 / (res / 4000)) :],
            ]
        )

        sv_mono = np.minimum.accumulate(sv)

        np.save(arr=sv, file=sys.argv[1] + ".npy")
        np.save(arr=sv_mono, file=sys.argv[1] + ".mono.npy")

        expectedtrans = pd.read_csv(sys.argv[1] + ".trans", sep="\t")

        np.save(
            arr=np.log(np.sum(expectedtrans["balanced.sum"]) / np.sum(expectedtrans["n_valid"])),
            file=sys.argv[1] + ".trans.npy",
        )

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
        mat = torch.from_numpy(mat).to(device=device, dtype=torch.float)
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

  print("Calculating metrics...")
  f05_macro = 0#metrics.fbeta_score(y_test, y_pred, average="macro", beta=0.5)
  f1_macro = 0#metrics.f1_score(y_test, y_pred, average="macro")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  return epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix

def test_epoch(model, dataloader):
    with torch.inference_mode():
      return run_epoch(model,'test', dataloader)

def train_epoch(model, dataloader):
    return run_epoch(model, 'train', dataloader)




log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

def train_model(dataloaders, model, num_epochs=20, phases= ['train']):
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

  return saved_epoch_losses, saved_epoch_accuracies, saved_epoch_f1_macros



def calculate_diag_means(matrix: np.ndarray, res = 'exp/obs') -> np.ndarray:
        result = np.zeros_like(matrix, dtype='float64')
        expected = np.zeros_like(matrix, dtype='float64')
        assert (
            matrix.shape[0] == matrix.shape[1]
        ), "Matrix must be square"

        n = matrix.shape[0]
        if torch.cuda.is_available():
            expected = sum(
            (
                torch.diag(
                    torch.full((n-abs(i), ), torch.nanmean(torch.tensor(matrix.diagonal(offset=i), device='cuda', dtype=torch.float))), diagonal=i
                ) for i in range(1-n, n)
            )
        )
        else:
            diagonals_sum = []
            matrix

            expected = sum(
                (
                    np.diag(
                        [np.nanmean(matrix.diagonal(offset=i))] * (n-abs(i)), k=i
                    ) for i in range(1-n, 0)
                )
            )

        if res == 'exp/obs':
            return expected/matrix

        if res == 'exp':
            return expected

        if res == 'exp-obs':
            return expected - matrix

        if res == 'obs-exp':
            return matrix - expected

        if res == 'obs/exp':
            return matrix/expected

        return result

class TrainDatasetPatches(Dataset):
    def __init__(self, cooler_path_list, trans_csv_path_list, resolution, image_size, clean_cooler_list, detection=True, blur=True):
        sv_count = 0
        self.label_to_index = {'inversion':torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
                               'translocation-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               'translocation+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               'translocation':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
                               'translocation_reversed-':torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                               'translocation_reversed+':torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
                               'translocation_reversed':torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
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
        for trans_csv_path, cooler_path, clean_cooler_path,index in tqdm(zip(trans_csv_path_list, cooler_path_list, clean_cooler_list, range(len(trans_csv_path_list)))):
            sv_file = pd.read_csv(trans_csv_path)
            if not self.detection:
                sv_file = sv_file[sv_file.label!='+-']
            sv_count+=sv_file.shape[0]
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.coolers_list.append(c)
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')

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
                
                chr_index = random.randint(0, len(c_clean.chromsizes)-1)
                chr_size = np.sum(c_clean.chromsizes[chr_index])
                x = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
                #y = random.randint(image_size//2*resolution, chr_size - image_size//2*resolution-1)
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
            self.num_classes = 5
        self.indexes = pd.DataFrame(indexes)

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        row = self.indexes.iloc[idx]
        sv_info = self.sv_files_list[row.file_index].iloc[row.in_index]
        c = self.coolers_list[row.file_index]
        matrix_full = self.matrixes_list[row.file_index][sv_info.chr]
        x = (sv_info.start)//self.resolution
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
        #mat =torch.nan_to_num(mat)
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


train_coolers = [f'{local_path}data/mcool/Gor_SV_4DN.mcool',]
clean_train_coolers = [f'{local_path}data/mcool/Gor_4DN.mcool',]
train_csvs = [f'{local_path}data/sv_csv/mGorGor.pri_sv.csv',] 

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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_small_m_norm/torch_ensemble_25k_48_diag.pt')


'''
train_dataset = TrainDatasetPatches(
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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_classic/torch_ensemble_25k_48_patch.pt')
'''


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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_small_m_norm/torch_ensemble_50k_48_diag.pt')


'''
train_dataset = TrainDatasetPatches(
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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_log_change/torch_ensemble_50k_48_patch.pt')
'''


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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_small_m_norm/torch_ensemble_10k_48_diag.pt')


'''
train_dataset = TrainDatasetPatches(
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
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_log_change/torch_ensemble_10k_48_patch.pt')
'''



exit()
#################################
# Train classification model
#################################

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
    print(outputs)
    loss = criterion(outputs, labels)
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.argmax(outputs, dim=1)
    y_test.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    running_loss += loss.item() * bz
    corrects_cnt = torch.sum(preds == torch.argmax(labels.detach(), dim=1))
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

train_dataset = TrainDatasetPatches(
    cooler_path_list=train_coolers,
    trans_csv_path_list=train_csvs,
    clean_cooler=f'{local_path}data/mcool/RHINO_4DN.mcool',
    resolution=10000,
    image_size=24,
    detection=False,
    use_means=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = ClassificationModel(in_channels=1, image_size=24, num_models=10, num_classes=5)
model.to(device=device)
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
dataloaders = dict()
dataloaders['train'] = train_dataloader

train_model(dataloaders, model, num_epochs)

torch.save(model.state_dict(), f'weights_norm/torch_ensemble_10k_24_classify.pt')