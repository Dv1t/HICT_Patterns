import torch
import cooler
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from help_functions import calculate_diag_means


class EvalDatasetDiag(Dataset):
    def __init__(self, cooler_path, resolution, image_size, step, device):
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.resolution = resolution
        self.image_size = image_size
        self.step = step
        self.device = device
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c        
        all_chr_len = int(np.sum(c.chromsizes.values, dtype=object))
        self.amount_steps = int((all_chr_len//resolution) // (step))


    def __len__(self):
        return self.amount_steps

    def __getitem__(self, idx):
        x, y = idx*self.step, idx*self.step
        try:
            mat = self.cooler.matrix(balance=False)[x:x+self.image_size, y:y+self.image_size]
        except ValueError:
            mat_size = sum(self.cooler.chromsizes)//self.resolution
            if x+self.image_size > mat_size or y + self.image_size > mat_size:
                mv = max(x+self.image_size-mat_size, y+self.image_size-mat_size)
                x-=mv
                y-=mv
            mat = self.cooler.matrix(balance=False)[x:x+self.image_size, y:y+self.image_size]
        mat = np.log10(mat)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, (x, y)

class PatchesDiagDataset(Dataset):
    def __init__(self, patches_coords_list, image_size_old, image_size_new, res_old, res_new, step, device):
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.device = device
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
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, (self.patches_coords_list[idx][1], self.patches_coords_list[idx][1])

class PatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords_list, device, use_means = False):
        self.resolution = resolution
        self.image_size = image_size
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.device = device
        self.use_means = use_means
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c
        self.coords_list = coords_list
        self.current_chr = coords_list[0][0]
        if use_means:
            self.matrix = calculate_diag_means(c.matrix(balance=False).fetch(c.chromnames[self.current_chr[0]], c.chromnames[self.current_chr[1]]))
        else:
            self.matrix = c.matrix(balance=False).fetch(c.chromnames[self.current_chr[0]], c.chromnames[self.current_chr[1]])
    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        chr_name, x_y = self.coords_list[idx]
        x = x_y[0]
        y = x_y[1]
        if chr_name!=self.current_chr:
            self.current_chr = chr_name
            if self.current_chr[0] == self.current_chr[1]:
                if self.use_means:
                    self.matrix = calculate_diag_means(self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]]))
                else:
                    self.matrix = self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]])
            else: 
                self.matrix = self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]])

        mat = np.log10(self.matrix[int(x)-int(self.image_size//2):int(x)+int(self.image_size//2), int(y)-int(self.image_size//2):int(y)+int(self.image_size//2)])
        if mat.shape[0] < self.image_size or mat.shape[1] < self.image_size:
            return torch.zeros((1, self.image_size, self.image_size), device=self.device), torch.tensor((chr_name[0], chr_name[1], x, y), device=self.device)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)

        return tens, torch.tensor((chr_name[0], chr_name[1], x, y), device=self.device)


class ClarifyDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords, patch_size, patch_resolution, device, use_means = True, step=1):
        self.resolution = resolution
        self.image_size = image_size
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.device = device
        self.use_means = use_means
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}')
        self.cooler = c

        patches_list = []
        coords_list = []
        pad = image_size//2
        res_coef = patch_resolution//resolution
        self.current_chr = (coords[0][0], coords[0][1])
        if use_means:
            self.matrix = calculate_diag_means(c.matrix(balance=False).fetch(c.chromnames[self.current_chr[0]], c.chromnames[self.current_chr[1]]))
        else:
            self.matrix = c.matrix(balance=False).fetch(c.chromnames[self.current_chr[0]], c.chromnames[self.current_chr[1]])
        coords_set = set()
        step = patch_size//((patch_size*patch_resolution)//(image_size*resolution))
        for chr_x, chr_y, x, y in tqdm(coords):
            for i in range(0, patch_size, step):
                for j in range(0, patch_size, step):
                    x_i = int((x-(patch_size//2) + i)*res_coef)
                    y_j = int((y-(patch_size//2) + j)*res_coef)
                    if x_i-pad < 0 or y_j - pad < 0 or x_i + pad > c.chromsizes[chr_x]//resolution or y_j + pad > c.chromsizes[chr_y]//resolution:
                        continue
                    if (x_i, y_j) in coords_set:
                        continue
                    coords_set.add((chr_x, chr_y, x_i, y_j))
                    #mat = c[x_i-pad:x_i+pad, y_j-pad:y_j+pad]

                    #mat = np.log10(mat)
                    #mat = np.nan_to_num(mat, neginf=0, posinf=0)
                    
                    #patches_list.append(mat)
                    coords_list.append((chr_x, chr_y, x_i, y_j))
        #self.patches_list = patches_list
        self.coords_list = coords_list

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        chr_x, chr_y, x_i, y_j = self.coords_list[idx]
        if (chr_x, chr_y)!=self.current_chr:
            self.current_chr = (chr_x, chr_y)
            if chr_x == chr_y:
                if self.use_means:
                    self.matrix = calculate_diag_means(self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]]))
                else:
                    self.matrix = self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]])
            else:
                self.matrix = self.cooler.matrix(balance=False).fetch(self.cooler.chromnames[self.current_chr[0]], self.cooler.chromnames[self.current_chr[1]])
        pad = self.image_size//2
        mat = self.matrix[x_i-pad:x_i+pad, y_j-pad:y_j+pad]
        mat = np.log10(mat)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, torch.tensor((chr_x, chr_y, x_i, y_j), device=self.device)


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

class BigPatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, patches_borders_list, device):
        self.resolution = resolution
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.image_size = image_size
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.device = device
        self.patches_list = []
        self.coords_list = []
        self.group_indexes_list = []
        for patch_borders in patches_borders_list:
            big_patch = c[patch_borders[0][0]:patch_borders[0][0]+patch_borders[1][0], patch_borders[0][1]:patch_borders[0][1]+patch_borders[1][1]]
            self.patches_list.append(big_patch)
            for x in range(patch_borders[1][0]-image_size+1):
                for y in range(patch_borders[1][1]-image_size+1):
                    self.coords_list.append((len(self.patches_list)-1,(x, y) ,(patch_borders[0][0], patch_borders[0][1])))
    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        patch_idx, x_y, coords = self.coords_list[idx]
        x, y = x_y
        big_mat = self.patches_list[patch_idx]
        mat = big_mat[x:x+ self.image_size, y:y+self.image_size]
        mat = np.log10(mat)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, (coords[0]+self.image_size//2+x, coords[1]+self.image_size//2+y)

class ClearPatchesDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords, res_step):
        self.resolution = resolution
        self.image_size = image_size
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.cooler = c
        self.res_step = res_step
        self.coords_list = []
        for coord in coords:
            if coord[0]* self.res_step > image_size//2 and coord[1]* self.res_step > image_size//2:
                self.coords_list.append(coord)
    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        x, y = self.coords_list[idx]
        x = int(x * self.res_step)
        y = int(y * self.res_step)
        mat = np.log2(self.cooler[x-int(self.image_size//2):x+int(self.image_size//2), y-int(self.image_size//2):y+int(self.image_size//2)])
        return mat, (x, y)

class ClassificationDataset(Dataset):
    def __init__(self, cooler_path, resolution, image_size, coords, device):
        self.resolution = resolution
        self.image_size = image_size
        c = cooler.Cooler(f'{cooler_path}::/resolutions/{resolution}').matrix(balance=False)
        self.cooler = c
        self.blur = GaussianBlur(kernel_size=3, sigma=1)
        self.device = device
        self.coords_list = []
        for coord in coords:
            if coord[0] > image_size//2 and coord[1] > image_size//2:
                self.coords_list.append(coord)

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        x, y = self.coords_list[idx]
        try:
            mat = np.log2(self.cooler[int(x)-int(self.image_size//2):int(x)+int(self.image_size//2), int(y)-int(self.image_size//2):int(y)+int(self.image_size//2)])
        except ValueError:
            print(x, y, self.image_size)
        mat = np.nan_to_num(mat, neginf=0, posinf=0)
        tens = torch.from_numpy(mat).reshape((1, self.image_size, self.image_size)).to(device=self.device, dtype=torch.float)
        tens = self.blur(tens)
        return tens, self.coords_list[idx]