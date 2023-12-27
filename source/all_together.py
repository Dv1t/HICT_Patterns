import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import cooler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
from collections import OrderedDict
from torchvision.transforms import GaussianBlur

local_path = 'D:/Study/HICT/HICT_Patterns/'
batch_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DetectBlock(nn.Module):
    def __init__(self, in_channels):
            super(DetectBlock, self).__init__()
            layers = nn.Sequential(
                #image_size x image_size x 1
                nn.Conv2d(in_channels, 3,  kernel_size = 3, padding=1),
                #image_size x image_size x 3
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(3, 8,  kernel_size = 3, padding=1),
                #image_size/2 x image_size/2 x 8
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 32,  kernel_size = 3, padding=1),
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
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out


class EvalDatasetDiag(Dataset):
    def __init__(self, cooler_path, resolution, image_size, step):
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.resolution = resolution
        self.image_size = image_size
        self.step = step
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c        
        all_chr_len = int(np.sum(c.chromsizes.values))
        self.amount_steps = int((all_chr_len//resolution) // (step))


    def __len__(self):
        return self.amount_steps

    def __getitem__(self, idx):
        x, y = idx*self.step, idx*self.step
        mat = self.cooler.matrix(balance=False)[x:x+self.image_size, y:y+self.image_size]
        mat = np.log10(mat)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, (x, y)

class PatchesDiagDataset(Dataset):
    def __init__(self, patches_coords_list, image_size_old, image_size_new, res_old, res_new, step):
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        res_coef = res_old//res_new
        mini_patches_coords_list = []
        for patch, coord in patches_coords_list:
            corner_coord = coord*res_coef
            for i in range(0, patch.shape[0]-image_size_new, step):
                mini_patch = patch[i:i+image_size_new, i:i+image_size_new]
                mini_coord = corner_coord + i
                mini_patches_coords_list.append((mini_patch, mini_coord))
        self.patches_coords_list = mini_patches_coords_list
        self.image_size = image_size_new
    def __len__(self):
        return len(self.patches_coords_list)
    def __getitem__(self, idx):
        mat = np.nan_to_num(self.patches_coords_list[idx][0], neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, (self.patches_coords_list[idx][1], self.patches_coords_list[idx][1])

    
class PatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords):
        self.resolution = resolution
        self.image_size = image_size
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.cooler = c
        #patches_list = []
        coords_list = []
        pad = image_size//2
        for x in tqdm(coords):
            for y in coords:
                if x < y or  abs(x-y)<image_size: 
                    continue
                if x-pad < 0 or y - pad < 0:
                    mv = max(-(x-pad), -(y-pad))
                    x+=mv
                    y+=mv
                if x + pad > c.shape[0] or y + pad > c.shape[1]:
                    mv = max(-c.shape[0]-(x+pad), -c.shape[0]-(y+pad))
                    x-=mv
                    y-=mv
                coords_list.append((x, y))
        self.coords_list = coords_list

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        x, y = self.coords_list[idx]
        mat = np.log10(self.cooler[x-int(self.image_size//2):x+int(self.image_size//2), y-int(self.image_size//2):y+int(self.image_size//2)])
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, self.coords_list[idx]


class ClarifyDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords, patch_size, patch_resolution, step=1):
        self.resolution = resolution
        self.image_size = image_size
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.cooler = c
        patches_list = []
        coords_list = []
        pad = image_size//2
        res_coef = patch_resolution//resolution
        coords_set = set()
        for x, y in tqdm(coords):
            for i in range(0, patch_size, step):
                for j in range(0, patch_size, step):
                    x_i = int((x-(patch_size//2) + i)*res_coef)
                    y_j = int((y-(patch_size//2) + j)*res_coef)
                    if x_i-pad < 0 or y_j - pad < 0 or x_i + pad > c.shape[0] or y_j + pad > c.shape[1]:
                        continue
                    if (x_i, y_j) in coords_set:
                        continue
                    coords_set.add((x_i, y_j))
                    mat = c[x_i-pad:x_i+pad, y_j-pad:y_j+pad]
                    assert mat.shape[0]==image_size
                    assert mat.shape[1]==image_size

                    mat = np.log10(mat)
                    mat = np.nan_to_num(mat, neginf=0, posinf=0)
                    
                    patches_list.append(mat)
                    coords_list.append((x_i, y_j))
        self.patches_list = patches_list
        self.coords_list = coords_list

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        mat = self.patches_list[idx]
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, self.coords_list[idx]


class ArbitraryPatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, patches_borders_list):
        self.resolution = resolution
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.patches_list = []
        self.coords_list = []
        self.group_indexes_list = []
        for patch_borders in patches_borders_list:
            big_patch = c[patch_borders[0][0]:patch_borders[0][0]+patch_borders[1][0], patch_borders[0][1]:patch_borders[0][1]+patch_borders[1][1]]
            self.patches_list.append(big_patch)
            self.coords_list.append((patch_borders[0][0], patch_borders[0][1]))
    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        mat = self.patches_list[idx]
        return mat, self.coords_list[idx]

class ClearPatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords):
        self.resolution = resolution
        self.image_size = image_size
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.cooler = c
        self.coords_list = coords

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        x, y = self.coords_list[idx]
        mat = np.log2(self.cooler[x-int(self.image_size//2):x+int(self.image_size//2), y-int(self.image_size//2):y+int(self.image_size//2)])
        return mat, self.coords_list[idx]


def save_detection_visualisation(detected, name):
    np.savetxt(f"{local_path}{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s')

def perform_detection(model, dataloader):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        labels = torch.round(output).detach().cpu().numpy().reshape(-1)
        x_list = position[0][labels==1]
        y_list = position[1][labels==1]
        if len(x_list) > 0:
            for x, y in zip(x_list.numpy(), y_list.numpy()):
                detected.append((x, y))
    return detected

file_path = 'data/dong_vs_gambiae/dong_colluzzii_4DN.mcool'

#Stage 1 - 50k, diagonal detection
print('Started Stage 1')
resolution_1 = 50000
image_size_1 = 48
dataset = EvalDatasetDiag(f'{local_path}{file_path}', resolution=resolution_1, image_size=image_size_1, step=image_size_1//2)
print('Stage 1 dataset loaded')
model = DetectModel(image_size=image_size_1, num_models=3)
model.to(device)
model.load_state_dict(torch.load(f'{local_path}artifacts/torch_ensemble_50k_48_diag.pt', map_location=device))
model.eval()

detected = perform_detection(model, DataLoader(dataset, batch_size=64))
save_detection_visualisation(detected, 'stage1')

#Stage 2 - 10k, diagonal detection
print('Started Stage 2')
resolution_2 = 10000
image_size_2 = 48

matrices_det = []
c = cooler.Cooler(f'{local_path}{file_path}::/resolutions/{resolution_2}').matrix(balance=False)
for d in detected:
    mat = np.log10(c[d[0]:min(d[0]+int(image_size_1*5), c.shape[0]), d[1]:min(d[1]+int(image_size_1*5), c.shape[1])])
    matrices_det.append((mat, d[0]))

model = DetectModel(image_size=image_size_2)
model.to(device)
model.load_state_dict(torch.load(f'{local_path}artifacts/torch_ensemble_10k_48_diag.pt', map_location=device))
model.eval()

dataset = PatchesDiagDataset(matrices_det, image_size_1, image_size_2, resolution_1, resolution_2, 12)
print('Stage 2 dataset loaded')
detected_2 = perform_detection(model, DataLoader(dataset, batch_size=batch_size))
save_detection_visualisation(detected_2, 'stage2')

#Stage 3 - 10k, whole map detection
print('Started Stage 3')
coords_set = set()
detected_2.sort()
last_d = -(image_size_2+1)
for d in detected_2:
    if d[0]-image_size_2//2 > last_d:
        coords_set.add(d[0]+(image_size_2//2))
        last_d = d[0]
image_size_3 = 48
resolution_3 = 10000
dataset = PatchesDataset(f'{local_path}{file_path}', resolution_3, image_size_3, coords_set)
print('Stage 3 dataset loaded')

model = DetectModel(image_size=image_size_3)
model.to(device)
model.load_state_dict(torch.load(f'{local_path}artifacts/torch_ensemble_10k_48_patch.pt', map_location=device))
model.eval()
detected_3 = perform_detection(model, DataLoader(dataset, batch_size=batch_size))
save_detection_visualisation(detected_3, 'stage3')

#Stage 4 - 5k, improve accuracy
print('Started Stage 4')
image_size_4 = 48
resolution_4 = 5000
dataset = ClarifyDataset(f'{local_path}{file_path}', resolution_4, image_size_4, detected_3, image_size_3, resolution_3)
print('Stage 4 dataset loaded')
model = DetectModel(image_size=image_size_4)
model.to(device)
model.load_state_dict(torch.load(f'{local_path}artifacts/torch_ensemble_5k_48_clr.pt', map_location=device))
model.eval()

detected_4 = perform_detection(model, DataLoader(dataset, batch_size=batch_size))
save_detection_visualisation(detected_4, 'stage4')


#Stage 4.5 - 5k, unite intersected detection boxes
print('Started Stage 4.5')

detected_4 = detected_4[np.argsort(detected_4.sum(axis=1))]
intersec_dist = image_size_4*np.sqrt(2)
def dist(d1, d2):
    return np.sqrt(((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2))

i = 0
groups = []
while i < len(detected_4):
    for j in range(i, len(detected_4)):
        
        if j == (len(detected_4)-1) or dist(detected_4[j], detected_4[j+1]) > intersec_dist:
            break
    center = (detected_4[i][0]//2 + detected_4[j][0]//2, detected_4[i][1]//2 + detected_4[j][1]//2)
    left_up = int(min(detected_4[i][0], detected_4[j][0])-image_size_4//2), int(min(detected_4[i][1], detected_4[j][1])-image_size_4//2)
    width = int(abs(detected_4[i][0] - detected_4[j][0]) + image_size_4)
    height = int(abs(detected_4[i][1] - detected_4[j][1]) + image_size_4)
    i=j+1
    groups.append((left_up, (width, height)))

#Stage 5 - 5k, find exact SVs location
print('Started Stage 5')
image_size_5 = 24
dataset = ArbitraryPatchesDataset(f'{local_path}{file_path}', resolution_4, groups)
detected_5 = []
dataloader = DataLoader(dataset, batch_size=1)
for data, position in tqdm(dataloader):
    center = np.unravel_index(data[0].cpu().numpy().argmax(), data[0].shape)
    dot = ((position[0].item()+center[0], position[1].item()+center[1]))
    detected_5.append(dot)

save_detection_visualisation(detected_5, 'stage5')

#Stage 6 - 1k, try to find location more precisely
print('Started Stage 6')
resolution_6 = 1000
image_size_6 = image_size_5*(resolution_4//resolution_6)
dataset = ClearPatchesDataset(f'{local_path}{file_path}', resolution_6, image_size_6, detected_5)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
detected_6 = []
res_step = resolution_4 // resolution_6
for data, position in tqdm(dataloader):
    patch = data[0].cpu().numpy()
    max_value = np.nanmax(patch)
    center_value = patch[image_size_6//2, image_size_6//2]
    if max_value-center_value > max_value//4:
        center = np.unravel_index(patch.argmax(), patch.shape)
        dot = ((position[0].item()*res_step+center[0], position[1].item()*res_step+center[1]))
        detected_6.append(dot)
    else:
        detected_6.append((position[0].item()*res_step, position[1].item()*res_step))

save_detection_visualisation(detected_6, 'stage6')




