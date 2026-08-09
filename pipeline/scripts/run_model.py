import torch
import cooler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import warnings
from cooltools.lib.numutils import adaptive_coarsegrain
import argparse
from torch import nn

from models import DetectModel

torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1024
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
    def __init__(self, matrices_by_res, resolutions, normmat250, normmat250_clean, eps, eps_clean, image_size, horizontals, device):
        self.matrices_by_res = matrices_by_res
        self.image_size = image_size
        self.normmat250 = normmat250
        self.normmat250_clean = normmat250_clean
        self.eps = eps
        self.eps_clean = eps_clean
        self.device = device
        self.horizontals = horizontals
        self.resolutions = resolutions
    def __len__(self):
        return self.horizontals.shape[1]

    def __get_matrix(self, x_native_res, y_native_res, resolution, matrix_raw, matrix, matrix_raw_clean, matrix_clean):
        x = int(np.floor(x_native_res // resolution ))
        y = int(np.floor(y_native_res // resolution ))
        pad = self.image_size//2
        mat_r = matrix_raw[x-pad:x+pad, y-pad:y+pad].todense()
        mat_b = matrix[x-pad:x+pad, y-pad:y+pad].todense()
        mat_b_clean = matrix_clean[x-pad:x+pad, y-pad:y+pad].todense()
        mat_r_clean = matrix_raw_clean[x-pad:x+pad, y-pad:y+pad].todense()

        if mat_r.shape[0] < self.image_size or mat_r.shape[1] < self.image_size:
            return np.array([np.zeros((self.image_size, self.image_size)), np.zeros((self.image_size, self.image_size))]), 0
        valid_mat = 1
        if np.count_nonzero(np.nan_to_num(mat_r, posinf=2, neginf=2))/2304 < 0.25: #0.25
            valid_mat = 0
        small_matrix_start = mat_r[22:26, 22:26]
        if np.count_nonzero(np.nan_to_num(small_matrix_start, posinf=2, neginf=2))/16 < 0.5: #0.5
            valid_mat = 0

        mat_cg = adaptive_coarsegrain(mat_b, mat_r)
        mat_cg_clean = adaptive_coarsegrain(mat_b_clean, mat_r_clean)

        mat = np.log(mat_cg+self.eps[str(resolution)])
        mat[np.isnan(mat)] = 0
        mat-= np.log(self.normmat250[str(resolution)]+self.eps[str(resolution)])

        mat_clean = np.log(mat_cg_clean+self.eps_clean[str(resolution)])
        mat_clean[np.isnan(mat_clean)] = 0
        mat_clean -= np.log(self.normmat250_clean[str(resolution)]+self.eps_clean[str(resolution)])
        return np.array([mat, mat_clean]), valid_mat

    def __getitem__(self, idx):
        mat_list = []
        valid_mat_all = 1
        for resolution in self.resolutions:
            matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrices_by_res[str(resolution)]
            mat_norm, valid_mat = self.__get_matrix(self.horizontals[0][idx], self.horizontals[1][idx], resolution, 
                                        matrix_raw, matrix, matrix_raw_clean, matrix_clean)
            valid_mat_all*=valid_mat
            mat = torch.from_numpy(mat_norm).reshape((2, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
            mat_list.append(mat)
        tens = torch.stack(mat_list, dim=0).to(device=device, dtype=torch.float)
        return tens, self.horizontals[0][idx], self.horizontals[1][idx], 1

class WholeMatrixPredictor():
    def __perform_detection(self, dataloader, round = True, label_cutoff=0.95):
        detected = []
        
        for inputs, position_x, position_y, valid_mat in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            sigmoid  = nn.Sigmoid()
            if round:
                outputs_by_res = [torch.round(sigmoid(model(inputs[:, i]))) for model, i in zip(self.models,  range(inputs.shape[1]))]
            else:
                outputs_by_res = [torch.tensor(sigmoid(model(inputs[:, i])) >=label_cutoff, dtype=int) for model, i in zip(models,  range(inputs.shape[1]))]

            stacked_predictions = torch.stack(outputs_by_res, dim=0)
            majority_vote_predictions, _ = torch.mode(stacked_predictions, dim=0)
            preds = majority_vote_predictions.resize(majority_vote_predictions.shape[0]).to(device, non_blocking=True, dtype=torch.float)
            labels = torch.round(preds).detach().cpu().numpy().reshape(-1)
            labels = labels*valid_mat.detach().cpu().numpy().reshape(-1)
            x_list = position_x[labels==1]
            y_list = position_y[labels==1]
                
            if len(x_list) > 0:
                for x, y in zip(x_list.numpy(), y_list.numpy()):
                    detected.append((x, y))
        return detected
    
    def __make_horizontals(self):
        c = self.coolers_list[str(self.resolutions[0])]
        horizontals_by_chr = {}
        for chr_name, x_input, y_input in zip(self.input_results['chr'],self.input_results['x'], self.input_results['y']):
            x = x_input
            if x // max(self.resolutions) < self.image_size:
                continue
            horizontals_x = np.arange(x, c.chromsizes[chr_name] - max(self.resolutions)*self.image_size, max(self.resolutions)*self.step)
            horizontals_y = np.ones_like(horizontals_x)*x
            horizontals = np.stack([horizontals_x, horizontals_y])
            if chr_name in horizontals_by_chr:
                a = horizontals_by_chr[chr_name]
                b = horizontals
                horizontals_by_chr[chr_name] = np.stack([np.concatenate([a[0], b[0]], axis=0), np.concatenate([a[1], b[1]], axis=0)])
            else:
                horizontals_by_chr[chr_name] = horizontals
        return horizontals_by_chr

    def segment_to_patches(self, segment_1, segment_2, step):
        x_coords = np.arange(segment_2[0], segment_2[1]+1, step=step*max(self.resolutions))
        y_coords = np.arange(segment_1[0], segment_1[1]+1, step=step*max(self.resolutions))
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
    
    def __init__(self, cooler_path, clean_cooler_path , resolutions, image_size, step, normmats_path, normmats_clean_path, device, result_name, models, labels_cutoff, batch_size = 512):
        self.resolutions = resolutions
        self.image_size = image_size
        self.step = step
        self.device = device
        self.models = models
        self.batch_size = batch_size
        self.result_name = result_name
        self.labels_cutoff = labels_cutoff

        self.normmat250 = {}
        self.eps = {}
        self.normmat250_clean = {}
        self.eps_clean = {}

        for resolution, normmat_path, normmat_clean_path in zip(resolutions, normmats_path, normmats_clean_path):
            normmat_bydist = np.exp(np.load(normmat_path))[:image_size*4]
            normmat = normmat_bydist[np.abs(np.arange(image_size*4)[:, None] - np.arange(image_size*4)[None, :])]
            self.normmat250[str(resolution)] = np.reshape(normmat, (image_size, 4, image_size, 4)).mean(axis=1).mean(axis=2)
            self.eps[str(resolution)] = np.min(self.normmat250[str(resolution)])

            normmat_bydist_clean = np.exp(np.load(normmat_clean_path)[:image_size*4])
            normmat_clean = normmat_bydist_clean[np.abs(np.arange(image_size*4)[:, None] - np.arange(image_size*4)[None, :])]
            self.normmat250_clean[str(resolution)] = np.reshape(normmat_clean, (image_size, 4, image_size, 4)).mean(axis=1).mean(axis=2)
            self.eps_clean[str(resolution)] = np.min(self.normmat250_clean[str(resolution)])
        self.coolers_list = {}
        self.matrixes_list = {}
        for resolution in resolutions:
            c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
            self.chromsizes = c.chromsizes
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
        
        self.input_results = {}
        self.input_results['chr'] = []
        self.input_results['x'] = []
        self.input_results['y'] = []
        for chr_name, chr_size in self.chromsizes.items():
            for x in range(0, chr_size, 15000*12):
                self.input_results['chr'].append(chr_name)
                self.input_results['x'].append(x)
                self.input_results['y'].append(x)
        
        self.horizontals = self.__make_horizontals()

    def _predict(self, horizontals, current_chr):
        matrices_by_res = {}
        for resolution in self.resolutions:
            matrix, matrix_raw, matrix_clean, matrix_raw_clean = self.matrixes_list[str(resolution)][current_chr]
            matrices_by_res[str(resolution)] = (matrix, matrix_raw, matrix_clean, matrix_raw_clean)
        dataset = PatchesDataset(matrices_by_res, self.resolutions, self.normmat250, self.normmat250_clean, self.eps, self.eps_clean, self.image_size, horizontals, self.device)
        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
        detected = self.__perform_detection(dl, round=False, label_cutoff=self.labels_cutoff)
        return detected
    
    def save_result_to_csv(save_path, detected):
        with open(save_path, "w", newline="") as f:
            f.write('chr,x,y\n')
            for chr, coords in detected.items():
                for coord_pair in coords:
                    f.write(f'{chr},{coord_pair[0]},{coord_pair[1]}\n')
    
    def run(self):
        detected_by_chr = {}
        for current_chr in self.horizontals.keys():
            print('current_chr', current_chr)
            current_horrizontals = self.horizontals[current_chr]
            result = self._predict(current_horrizontals, current_chr)
            if current_chr not in detected_by_chr:
                detected_by_chr[current_chr] = result
            else:
                detected_by_chr[current_chr].extend(result)            
        detected = []

        c = self.coolers_list[str(self.resolutions[2])]
        for chr, item in detected_by_chr.items():
            detected.extend(get_genome_coords(item, chr, c.chromnames,c.chromsizes, self.resolutions[2]))

        WholeMatrixPredictor.save_result_to_csv(self.result_name, detected_by_chr)
        return detected
    
parser = argparse.ArgumentParser()
parser.add_argument('target_mcool', type=str, help='Paths to mcool with SV to search')
parser.add_argument('clean_mcool', type=str, help='Paths to mcool without SV to highlight SVs')
parser.add_argument('target_expected', type=str, help='List of normmat paths for mcool with SV')
parser.add_argument('clean_expected', type=str, help='List of normmat paths for mcool without SV')
parser.add_argument('resolutions', type=str, help='List of resolutions of HiC map to search SVs in')
parser.add_argument('weights', type=str, help='List of paths to weights of the models for each resolution')
parser.add_argument('step', type=int, help='Step of sliding window')
parser.add_argument('labels_cutoff', type=float, help='Threshold for models predictions to be considered true')
parser.add_argument('result_save_path', type=str, help='Path to save results with name of the file')

args = parser.parse_args()

image_size = 48
step = int(args.step)

validate_cooler = args.target_mcool
clean_validate_cooler = args.clean_mcool

normmats_path_val= args.target_expected.split(' ')
normmats_clean_path_val= args.clean_expected.split(' ')

weights = args.weights.split(' ')
resolutions = [int(r) for r in args.resolutions.removesuffix(']').removeprefix('[').split(', ')]

models = []
for weight in weights: 
    model = DetectModel(in_channels=2, image_size=48, num_models=10)
    model.to(device=device)
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    models.append(model)

predictor = WholeMatrixPredictor(validate_cooler, clean_validate_cooler, resolutions, image_size, step, normmats_path_val, normmats_clean_path_val, device, args.result_save_path, models, float(args.labels_cutoff))
results_wm = predictor.run()