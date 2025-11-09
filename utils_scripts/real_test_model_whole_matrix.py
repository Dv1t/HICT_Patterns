import torch
import pandas as pd
import cooler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import sys
import warnings
from cooltools.lib.numutils import adaptive_coarsegrain
from ml_collections import config_dict
import argparse
import json
from torch import nn

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from hict.patterns.models import DetectModel

local_path = '/mnt/tank/scratch/vdravgelis/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
warnings.filterwarnings('ignore')

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

class PatchesDataset(Dataset):
    def __init__(self, matrices_by_res, resolutions, input_resolution, normmat250, normmat250_clean, image_size, patches, device):
        self.matrices_by_res = matrices_by_res
        self.image_size = image_size
        self.normmat250 = normmat250
        self.normmat250_clean = normmat250_clean
        self.eps = np.min(self.normmat250)
        self.eps_clean = np.min(self.normmat250_clean)
        self.device = device
        self.patches = patches
        self.resolutions = resolutions
        self.input_resolution = input_resolution
    def __len__(self):
        return len(self.patches)

    def __get_matrix(self, patch, res_step, matrix_raw, matrix, matrix_raw_clean, matrix_clean):
        x = int(np.floor(patch[0] * res_step ))
        y = int(np.floor(patch[1] * res_step ))
        pad = self.image_size//2
        mat_r = matrix_raw[x-pad:x+pad, y-pad:y+pad]
        mat_b = matrix[x-pad:x+pad, y-pad:y+pad]
        mat_b_clean = matrix_clean[x-pad:x+pad, y-pad:y+pad]
        mat_r_clean = matrix_raw_clean[x-pad:x+pad, y-pad:y+pad]

        if mat_r.shape[0] < self.image_size or mat_r.shape[1] < self.image_size or mat_r_clean.shape[1] < self.image_size or mat_b_clean.shape[1] < self.image_size:
            print('small matrix', x, y)
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
        mat_list = []
        for resolution in self.resolutions:
            matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrices_by_res[str(resolution)]
            mat_norm = self.__get_matrix(self.patches[idx], self.input_resolution//resolution, 
                                        matrix_raw, matrix, matrix_raw_clean, matrix_clean)
            mat = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
            mat_list.append(mat)
        tens = torch.stack(mat_list, dim=0).to(device=device, dtype=torch.float)
        return tens, torch.tensor(self.patches[idx], device=self.device)


class WholeMatrixPredictor():
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
            outputs_by_res = [torch.round(sigmoid(model(inputs[:, i]))) for model, i in zip(self.models,  range(inputs.shape[1]))]
            #class_predictions = [torch.argmax(output, dim=1) for output in outputs_by_res]
            stacked_predictions = torch.stack(outputs_by_res, dim=0)
            majority_vote_predictions, _ = torch.mode(stacked_predictions, dim=0)
            preds = majority_vote_predictions.resize(majority_vote_predictions.shape[0]).to(device, non_blocking=True, dtype=torch.float)
            
            if round:
                labels = torch.round(preds).detach().cpu().numpy().reshape(-1)
                x_list = position[0]#[labels==1]
                y_list = position[1]#[labels==1]
            else:
                labels = preds.detach().cpu().numpy().reshape(-1)
                x_list = position[0][labels>=label_cutoff]
                y_list = position[1][labels>=label_cutoff]
                
            if len(x_list) > 0:
                for x, y, label in zip(x_list.numpy(), y_list.numpy(), labels):
                    detected.append((x, y, label))
        return detected
    
    def __make_segments(self, coords_list, input_resolution):
        print('here 7')
        c = self.coolers_list[str(input_resolution)]
        bps = np.array(sorted(list(coords_list)), dtype=np.int64)#*result_res//resolution
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
            chr, x = self.__get_chromosome_coords(s[0], c.chromsizes, input_resolution)
            chr, y = self.__get_chromosome_coords(s[1], c.chromsizes, input_resolution)
            print('here 9')
            if chr in segments_by_chr:
                segments_by_chr[chr].append((x, y))
                print('here 12')
            else:
                segments_by_chr[chr] = [(x, y), ]
                print('here 11')
            print('here 10')
        print('here 8')
        return segments_by_chr
    def segment_to_patches(segment_1, segment_2, step):
        x_coords = np.arange(segment_2[0], segment_2[1]+1, step=step)
        y_coords = np.arange(segment_1[0], segment_1[1]+1, step=step)
        return np.array(np.meshgrid(y_coords, x_coords)).T.reshape(-1, 2)
    
    def generate_index_pairs(n):
        pairs = []
        for i in range(n + 1):
            pairs.append((i, i))
        for d in range(1, n + 1):
            for i in range(n - d + 1):
                j = i + d
                pairs.append((i, j))
        return np.array(pairs)
    
    def __init__(self, cooler_path, clean_cooler_path , resolutions, input_resolution, image_size, step, normmats_path, normmats_clean_path, coords_list, device, result_name, models ,batch_size = 512):
        print('here 6')
        self.resolutions = resolutions
        self.image_size = image_size
        self.step = step
        self.device = device
        self.models = models
        self.batch_size = batch_size
        self.result_name = result_name
        self.input_resolution = input_resolution

        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}

        for resolution, normmat_path, normmat_clean_path in zip(resolutions, normmats_path, normmats_clean_path):
            normmat_bydist = np.exp(np.load(normmat_path))[:image_size*1]
            normmat = normmat_bydist[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
            self.normmat250[str(resolution)] = np.reshape(normmat, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
            self.eps[str(resolution)] = np.min(self.normmat250[str(resolution)])

            normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*1])
            normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*1)[:, None] - np.arange(image_size*1)[None, :])]
            self.normmat250_clean[str(resolution)] = np.reshape(normmat_clean, (image_size, 1, image_size, 1)).mean(axis=1).mean(axis=2)
            self.eps_clean[str(resolution)] = np.min(self.normmat250_clean[str(resolution)])
        print('here 5')
        self.coolers_list = {}
        self.matrixes_list = {}            
        for resolution in resolutions:
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.coolers_list[str(resolution)] = c
            c_clean = cooler.Cooler(f'{clean_cooler_path}::/resolutions/{resolution}')
            matrixes_by_chr = {}
            for chr in c.chromnames:
                matrix_raw = c.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix_raw_clean = c_clean.matrix(balance=False, sparse=True).fetch(chr).tocsr()
                matrix = c.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrix_clean = c_clean.matrix(balance=True, sparse=True).fetch(chr).tocsr()
                matrixes_by_chr[chr] = (matrix, matrix_raw, matrix_clean, matrix_raw_clean)
            self.matrixes_list[str(resolution)] = matrixes_by_chr

        
        self.coords_list = coords_list
        
        self.segments = self.__make_segments(coords_list, input_resolution)
        print('here 4')

    def _predict(self, patches, current_chr):
        print('here 3')
        matrices_by_res = {}
        for resolution in self.resolutions:
            matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrixes_list[str(resolution)][current_chr]
            matrices_by_res[str(resolution)] = (matrix, matrix_raw, matrix_clean, matrix_raw_clean)
        dataset = PatchesDataset(matrices_by_res, self.resolutions, self.input_resolution, self.normmat250, self.normmat250_clean, self.image_size, patches, self.device)
        dl = DataLoader(dataset, batch_size=self.batch_size)
        detected = self.__perform_detection(dl, round=True)
        return detected
    
    def save_result_to_csv(local_path, detected, name):
        with open(f"{local_path}/{name}.csv", "w", newline="") as f:
            f.write('chr,x,y\n')
            for chr, coords in detected.items():
                for coord_pair in coords:
                    f.write(f'{chr},{coord_pair[0]},{coord_pair[1]},{coord_pair[2]}\n')
    
    def run(self):
        print('here')
        detected_by_chr = {}
        for current_chr in self.segments.keys():
            current_segments = self.segments[current_chr]
            n = len(current_segments)
            segment_matrix = WholeMatrixPredictor.generate_index_pairs(n-1)
            i = 0
            detected_indexes = np.zeros(n, dtype=bool)
            while sum(detected_indexes) < len(current_segments) and i < n:
                index_pair = segment_matrix[i]
                if detected_indexes[index_pair[0]] or detected_indexes[index_pair[1]]:
                    i+=1
                    continue
                patches = WholeMatrixPredictor.segment_to_patches(current_segments[index_pair[0]], current_segments[index_pair[1]], self.step)
                result = self._predict(patches, current_chr)
                if len(result) > 0:
                    detected_indexes[index_pair[0]] = True
                    detected_indexes[index_pair[1]] = True
                    if current_chr not in detected_by_chr:
                        detected_by_chr[current_chr] = result
                    else:
                        detected_by_chr[current_chr].extend(result)
                i+=1
        detected = []
        print('here 2')

        c = self.coolers_list[str(self.resolutions[2])]
        for chr, item in detected_by_chr.items():
            detected.extend(get_genome_coords(item, chr, c.chromnames,c.chromsizes, self.resolutions[2]))

        WholeMatrixPredictor.save_result_to_csv(os.getcwd(), detected_by_chr, self.result_name)
        return detected
    
parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='Path to file with config')

args = parser.parse_args()

with open(args.cfg_path, 'r') as file:
    cfg_dict = json.load(file)

cfg = config_dict.ConfigDict(cfg_dict)

image_size = 48
step = cfg.step

validate_cooler = cfg.validate_cooler
clean_validate_cooler = cfg.clean_validate_cooler


normmats_path_val= cfg.normmats_path_val
normmats_clean_path_val= cfg.normmats_clean_path_val

results = pd.read_csv(cfg.results)
results = results[results.label == 1]

diag_results = np.array(sorted(list(results.x)), dtype=np.int64)

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

models = [model_15kb, model_25kb, model_50kb]


predictor = WholeMatrixPredictor(validate_cooler, clean_validate_cooler, cfg.resolutions, cfg.input_resolution, image_size, step, normmats_path_val, normmats_clean_path_val, diag_results, device, 'real_tests/step_1_whole_matrix_50kb', models)
results_wm = predictor.run()

def save_result_to_csv(local_path, detected, name):
    np.savetxt(f"{local_path}/{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s',
        header='x,y,label')
save_result_to_csv(os.getcwd(), results_wm, cfg.result_save_path)
print(f'Completed for {cfg.model_name} with config {args.cfg_path}')

