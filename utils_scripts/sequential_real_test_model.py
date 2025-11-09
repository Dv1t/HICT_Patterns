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
from cooltools.lib.numutils import adaptive_coarsegrain
from ml_collections import config_dict
import argparse
import json

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from hict.patterns.help_functions import get_chromosome_coords, get_genome_coords
from hict.patterns.models import DetectModel

def get_chromosome_coords(coords_list, chr_sizes, resolution):
    additive_sizes = np.empty_like(chr_sizes, dtype=np.uint64)
    curr_s = 0
    for i, s in enumerate(chr_sizes):
        curr_s += s
        additive_sizes[i] = curr_s
    result = {}
    for coord in coords_list:
        x_i = 0
        while coord*resolution > additive_sizes[x_i]:
            x_i+=1
            if x_i >= len(additive_sizes):
                break
        if x_i >= len(additive_sizes):
            continue
        x_chr = x_i
        if x_i > 0:
            x = (coord*resolution-additive_sizes[x_i-1]) // resolution
        else:
            x = coord
        
        if x_chr in result:
            result[x_chr].append(x)
        else:
            result[x_chr] = [x, ]
        
    result_list = []
    
    for key, value in result.items():
        for v in value:
            result_list.append((key, int(v)))
    return result_list

def get_genome_coords(coords_list, chr, chr_names, chr_sizes, resolution):
    additive_sizes = {}
    curr_s = 0
    for i, s in zip(chr_names, chr_sizes):
        additive_sizes[i] = curr_s
        curr_s += s
    result = []
    for coord in coords_list:
        x, y = coord
        pad_x = additive_sizes[chr]
        x_new = x*resolution+pad_x
        pad_y = additive_sizes[chr]
        y_new = y*resolution+pad_y
        result.append((x_new // resolution, y_new // resolution))
    
    return result

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

def perform_detection(model, dataloader, round = True, label_cutoff=0.95):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for inputs, position in cur_tqdm:        
        inputs = inputs.to(device, non_blocking=True)
        sigmoid  = nn.Sigmoid()
        outputs = torch.round(sigmoid(model(inputs)))
        #class_predictions = [torch.argmax(output, dim=1) for output in outputs_by_res]
        preds = outputs.resize(outputs.shape[0]).to(device, non_blocking=True, dtype=torch.float)
        
        if round:
            labels = torch.round(preds).detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels==1]
            y_list = position[1][labels==1]
        else:
            labels = preds.detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels>=label_cutoff]
            y_list = position[1][labels>=label_cutoff]
            
        if len(x_list) > 0:
            for x, y, label in zip(x_list.numpy(), y_list.numpy(), labels):
                detected.append((x, y))
    return detected

def save_result_to_csv(local_path, detected, name):
    np.savetxt(f"{local_path}/{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s',
        header='x,y,label')


class EvalDatasetDiag(Dataset):
    
    def __init__(self, cooler_path, resolution, image_size, clean_cooler_path, normmat_path, normmat_clean_path, step=1):
        self.step = step
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        chr_sizes = [int(size) for size in c.chromsizes.values if int(size) > image_size*resolution*10]
        all_chr_len = int(np.sum(chr_sizes))
        self.amount_steps = int((all_chr_len//resolution) // (step))

        self.resolution = resolution
        self.image_size = image_size
        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}

        normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
        normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = np.min(self.normmat250)

        normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
        normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250_clean = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps_clean = np.min(self.normmat250_clean)
            
        self.matrixes_list = {}            
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c
        c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
        matrixes_by_chr = {}
        for chr in c.chromnames:
            matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
            matrix__raw_clean = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
            matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
            matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
            matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix__raw_clean)
   
            self.matrixes_list = matrixes_by_chr

    def __len__(self):
        return self.amount_steps
    
    def process_matrix(self, mat_raw, mat_bal, mat_raw_clean, mat_bal_clean, coarse_grain=False):
        if coarse_grain:
            mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
            mat_cg_clean = adaptive_coarsegrain(mat_bal_clean, mat_raw_clean)
        else:
            mat_cg = mat_bal
            mat_cg_clean = mat_bal_clean

        mat = np.log(mat_cg+self.eps)
        mat[np.isnan(mat)] = 0
        mat-= np.log(self.normmat250+self.eps)

        mat_clean = np.log(mat_cg_clean+self.eps_clean)
        mat_clean[np.isnan(mat_clean)] = 0
        mat_clean -= np.log(self.normmat250_clean+self.eps_clean)
        return np.array([mat, mat_clean])
    
    def __get_matrix(self, x, y):
            pad = self.image_size//2
            mat_list = []
            x ,y = get_chromosome_coords((x, y), self.cooler.chromsizes, self.resolution)
            chr_num = x[0]
            x = x[1]
            y = y[1]
            c = self.cooler
            matrix_full = self.matrixes_list[c.chromnames[chr_num]]
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

            mat_norm = self.process_matrix(mat_r, mat_b, mat_r_clean, mat_b_clean, True)

            tens = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)

            return tens
    
    def __getitem__(self, idx):
        x, y = idx*self.step, idx*self.step
        tens = self.__get_matrix(x, y)
        return tens, (x, y)

class DiagPatchesDataset(Dataset):
    def __init__(self, matrices_by_res, resolution, input_resolution, normmat250, normmat250_clean, image_size, patches, device):
        self.matrices_by_res = matrices_by_res
        self.image_size = image_size
        self.normmat250 = normmat250
        self.normmat250_clean = normmat250_clean
        self.eps = np.min(self.normmat250)
        self.eps_clean = np.min(self.normmat250_clean)
        self.device = device
        self.patches = patches
        self.resolution = resolution
        self.input_resolution = input_resolution
    def __len__(self):
        return len(self.patches)

    def __get_matrix(self, patch, res_step, matrix_raw, matrix, matrix_raw_clean, matrix_clean):
        x = int(np.floor(patch * res_step ))
        pad = self.image_size//2
        mat_r = matrix_raw[x-pad:x+pad, x-pad:x+pad].todense()
        mat_b = matrix[x-pad:x+pad, x-pad:x+pad].todense()
        mat_b_clean = matrix_clean[x-pad:x+pad, x-pad:x+pad].todense()
        mat_r_clean = matrix_raw_clean[x-pad:x+pad, x-pad:x+pad].todense()

        if mat_r.shape[0] < self.image_size or mat_r.shape[1] < self.image_size or mat_r_clean.shape[1] < self.image_size or mat_b_clean.shape[1] < self.image_size:
            print('small matrix', mat_r.shape[0], x)
            return np.array([np.zeros((self.image_size, self.image_size)), np.zeros((self.image_size, self.image_size))])
    
        mat_cg = adaptive_coarsegrain(mat_b, mat_r)
        mat_cg_clean = adaptive_coarsegrain(mat_b_clean, mat_r_clean)

        mat = np.log(mat_cg+self.eps)
        mat[np.isnan(mat)] = 0
        mat-= np.log(self.normmat250+self.eps)

        mat_clean = np.log(mat_cg_clean+self.eps_clean)
        mat_clean[np.isnan(mat_clean)] = 0
        mat_clean -= np.log(self.normmat250_clean+self.eps_clean)
        return np.array([mat, mat_clean])

    def __getitem__(self, idx):
        matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrices_by_res
        mat_norm = self.__get_matrix(self.patches[idx], self.input_resolution//self.resolution, 
                                    matrix_raw, matrix, matrix_raw_clean, matrix_clean)
        tens = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        return tens, torch.tensor(self.patches[idx], device=self.device)


class DiagMatrixPredictor():
    def __get_chromosome_coords(self, coord, chr_sizes, resolution):
        additive_sizes = np.empty_like(chr_sizes, dtype=np.uint64)
        curr_s = 0
        for i, s in enumerate(chr_sizes):
            curr_s += s
            additive_sizes[i] = curr_s
        
        x_i = 0
        while coord*resolution > additive_sizes[x_i]:
            x_i+=1
            if x_i >= len(additive_sizes):
                break

        if x_i >= len(additive_sizes):
            exit(1)

        x_chr = x_i
        if x_i > 0:
            x = (coord*resolution-additive_sizes[x_i-1]) // resolution
        else:
            x = coord

        return x_chr, int(x)

    def __perform_detection(self, dataloader, round = True, label_cutoff=0.95):
        detected = []
        cur_tqdm = tqdm(dataloader)
        for inputs, position in cur_tqdm:        
            inputs = inputs.to(device, non_blocking=True)
            sigmoid  = nn.Sigmoid()
            outputs = torch.round(sigmoid(self.model(inputs)))
            #class_predictions = [torch.argmax(output, dim=1) for output in outputs_by_res]
            preds = outputs.resize(outputs.shape[0]).to(device, non_blocking=True, dtype=torch.float)
            
            if round:
                labels = torch.round(preds).detach().cpu().numpy().reshape(-1)
                x_list = position[labels==1]
                y_list = position[labels==1]
            else:
                labels = preds.detach().cpu().numpy().reshape(-1)
                x_list = position[labels>=label_cutoff]
                y_list = position[labels>=label_cutoff]
                
            if len(x_list) > 0:
                for x, y, label in zip(x_list.numpy(), y_list.numpy(), labels):
                    detected.append((x, y))
        return detected
    
    def __make_segments(self, bps, input_resolution):
        print(bps)
        #bps = [bp for bp in bps if bp < ape_chm.chromsizes[0]//result_res]
        segments = []
        segment_start = bps[0]
        segment_end = bps[0]
        for bp in bps:
            if bp-segment_end > self.image_size:
                segments.append((segment_start, segment_end))
                segment_start = bp
            segment_end = bp

        segments_by_chr = {}
        for s in segments:
            chr, x = self.__get_chromosome_coords(s[0], self.cooler.chromsizes, input_resolution)
            chr, y = self.__get_chromosome_coords(s[1], self.cooler.chromsizes, input_resolution)

            if chr in segments_by_chr:
                segments_by_chr[self.cooler.chromnames[chr]].append((x, y))
            else:
                segments_by_chr[self.cooler.chromnames[chr]] = [(x, y), ]
        return segments_by_chr
    
    def segment_to_patches(segments, step):
        patches = []
        for segment in segments:
            x_coords = np.arange(segment[0], segment[1]+1, step=step)
            patches.extend(x_coords)
        return patches
    
    def __init__(self, cooler_path, clean_cooler_path , resolution, input_resolution, image_size, step, normmat_path, normmat_clean_path, coords_list, device, result_name, model ,batch_size = 512):
        self.resolution = resolution
        self.image_size = image_size
        self.step = step
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.result_name = result_name
        self.input_resolution = input_resolution

        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c
        c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')

        normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
        normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250 = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps = np.min(self.normmat250)

        normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
        normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
        self.normmat250_clean = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
        self.eps_clean = np.min(self.normmat250_clean)

        self.coords_list = coords_list
        
        self.segments = self.__make_segments(coords_list, input_resolution)
        self.patches = {}
        for chr, s in self.segments.items():
            patches = DiagMatrixPredictor.segment_to_patches(s, self.step)
            self.patches[chr] = patches

            
        self.matrixes_list = {}            
        matrixes_by_chr = {}
        for chr in c.chromnames:
            matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
            matrix_raw_clean = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
            matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
            matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
            matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix_raw_clean)
        self.matrixes_list = matrixes_by_chr


    def _predict(self, patches, current_chr):
        matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrixes_list[current_chr]
        matrices_by_res = (matrix, matrix_raw, matrix_clean, matrix_raw_clean)
        dataset = DiagPatchesDataset(matrices_by_res, self.resolution, self.input_resolution, self.normmat250, self.normmat250_clean, self.image_size, patches, self.device)
        dl = DataLoader(dataset, batch_size=self.batch_size)
        detected = self.__perform_detection(dl, round=True)
        return detected
    
    def save_result_to_csv(local_path, detected, name):
        with open(f"{local_path}/{name}.csv", "w", newline="") as f:
            f.write('chr,x,y\n')
            for chr, coords in detected.items():
                for coord_pair in coords:
                    f.write(f'{chr},{coord_pair[0]},{coord_pair[1]}\n')
    
    def run(self):
        detected_by_chr = {}
        for current_chr, patches in self.patches.items():
            result = self._predict(patches, current_chr)
            if len(result) > 0:
                if current_chr not in detected_by_chr:
                    detected_by_chr[current_chr] = result
                else:
                    detected_by_chr[current_chr].extend(result)
        
        detected = []

        for chr, item in detected_by_chr.items():
            detected.extend(get_genome_coords(item, chr, self.cooler.chromnames, self.cooler.chromsizes, self.resolution))

        DiagMatrixPredictor.save_result_to_csv(os.getcwd(), detected_by_chr, self.result_name)


        return detected


parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='Path to file with config')

args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    cfg_dict = json.load(file)

cfg = config_dict.ConfigDict(cfg_dict)


validate_cooler = cfg.validate_cooler
clean_validate_cooler = cfg.clean_validate_cooler


normmats_path_val= cfg.normmats_path_val
normmats_clean_path_val= cfg.normmats_clean_path_val


validate_dataset = EvalDatasetDiag(
    cooler_path=validate_cooler,
    clean_cooler_path=clean_validate_cooler,
    resolution=cfg.resolutions[2],
    image_size=48,
    normmat_path=normmats_path_val[2],
    normmat_clean_path=normmats_clean_path_val[2],
    step=cfg.step)
print('Total steps amount:', len(validate_dataset))
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 1e-6
if cfg.loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
if cfg.loss == 'BCELoss':
    criterion = nn.BCELoss()
if cfg.loss == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()

model_15kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_15kb.to(device=device)
model_15kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[0]}', map_location=device))
model_15kb.eval()

model_25kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_25kb.to(device=device)
model_25kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[1]}', map_location=device))
model_25kb.eval()

model_50kb = DetectModel(in_channels=2, image_size=48, num_models=10)
model_50kb.to(device=device)
model_50kb.load_state_dict(torch.load(f'{local_path}training/{cfg.model_path[2]}', map_location=device))
model_50kb.eval()

#models = [model_15kb, model_25kb, model_50kb]

num_epochs = 1
dataloaders = dict()
dataloaders['validate'] = validate_dataloader

detected_50kb = perform_detection(model_50kb, validate_dataloader)

diag_results = np.array(sorted(list([i[0] for i in detected_50kb])), dtype=np.int64)

predictor = DiagMatrixPredictor(validate_cooler, clean_validate_cooler, cfg.resolutions[1], cfg.resolutions[2], 48, cfg.step, normmats_path_val[1], normmats_clean_path_val[1], diag_results, device, f'real_tests/step_{cfg.step}_whole_matrix_25kb', model_25kb)
results_25kb = predictor.run()
diag_results_25kb = np.array(sorted(list([i[0] for i in results_25kb])), dtype=np.int64)

predictor = DiagMatrixPredictor(validate_cooler, clean_validate_cooler, cfg.resolutions[0], cfg.resolutions[1], 48, cfg.step, normmats_path_val[0], normmats_clean_path_val[0], diag_results_25kb, device, f'real_tests/step_{cfg.step}_whole_matrix_15kb', model_15kb)
results_15kb = predictor.run()

save_result_to_csv(os.getcwd(), results_15kb, cfg.result_save_path)
print(f'Completed for {cfg.model_name} with config {args.cfg_path}')
