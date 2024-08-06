import numpy as np
import torch


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

def get_chromosome_coords_double(coords_list, chr_sizes, resolution):
    additive_sizes = np.empty_like(chr_sizes, dtype=np.uint64)
    curr_s = 0
    for i, s in enumerate(chr_sizes):
        curr_s += s
        additive_sizes[i] = curr_s
    result = {}
    for coord_x, coord_y in coords_list:
        x_i = 0
        while coord_x*resolution > additive_sizes[x_i]:
            x_i+=1
            if x_i >= len(additive_sizes):
                break
        if x_i >= len(additive_sizes):
            continue
        x_chr = x_i
        if x_i > 0:
            x = (coord_x*resolution-additive_sizes[x_i-1]) // resolution
        else:
            x = coord_x

        y_i = 0
        while coord_y*resolution > additive_sizes[y_i]:
            y_i+=1
            if y_i >= len(additive_sizes):
                break
        if y_i >= len(additive_sizes):
            continue
        y_chr = y_i
        if y_i > 0:
            y = (coord_y*resolution-additive_sizes[y_i-1]) // resolution
        else:
            y = coord_y

        
        if (x_chr, y_chr) in result:
            result[(x_chr, y_chr)].append((x, y))
        else:
            result[(x_chr, y_chr)] = [(x, y), ]
        
    result_list = []
    
    for key, value in result.items():
        for v in value:
            result_list.append((key[0],key[1], int(v[0]), int(v[1])))
    return result_list


def get_genome_coords(coords_list, chr_names, chr_sizes, resolution):
    additive_sizes = {}
    curr_s = 0
    for i, s in zip(chr_names, chr_sizes):
        additive_sizes[i] = curr_s
        curr_s += s
    result = []
    for coord in coords_list:
        chr_x, chr_y, x, y = coord
        pad_x = additive_sizes[chr_names[int(chr_x)]]
        x_new = x*resolution+pad_x
        pad_y = additive_sizes[chr_names[int(chr_y)]]
        y_new = y*resolution+pad_y
        result.append((x_new // resolution, y_new // resolution))
    
    return result

def get_genome_coords_single(coords_list, chr_names, chr_sizes, resolution):
    additive_sizes = {}
    curr_s = 0
    for i, s in zip(chr_names, chr_sizes):
        additive_sizes[i] = curr_s
        curr_s += s
    result = []
    for coord in coords_list:
        chr_x, x = coord
        pad_x = additive_sizes[str(chr_x)]
        x_new = x*resolution+pad_x
        result.append((x_new // resolution))
    
    return result


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
        ).numpy(force=True)
        else:
            expected = sum(
                (
                    np.diag(
                        [np.nanmean(matrix.diagonal(offset=i))] * (n-abs(i)), k=i
                    ) for i in range(1-n, n)
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