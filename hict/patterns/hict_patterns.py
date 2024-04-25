import torch
import cooler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import pathlib
import h5py
import sys
import pandas as pd
import warnings
import time
from help_functions import get_chromosome_coords, get_genome_coords
from models import DetectModel, ClassificationModel
from datasets import *

def __save_result_to_csv(local_path, detected, name):
    np.savetxt(f"{local_path}/{name}.csv",
        detected,
        delimiter =",",
        fmt ='% s')

def __perform_detection_chroms(model, dataloader, round = True, label_cutoff=0.95):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        if round:
            labels = torch.round(output).detach().cpu().numpy().reshape(-1)
            position_pos = position[labels==1]
        else:
            labels = output.detach().cpu().numpy().reshape(-1)
            position_pos = position[labels>=label_cutoff]
        if len(position_pos) > 0:
            for chr_x, chr_y, x, y in position_pos.cpu().numpy():
                detected.append((chr_x, chr_y, x, y))
    return detected

def __perform_detection_chroms_inverted(model, dataloader):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        labels = torch.round(output).detach().cpu().numpy().reshape(-1)
        position_pos = position[labels==0]
        if len(position_pos) > 0:
            for chr_x, chr_y, x, y in position_pos.cpu().numpy():
                detected.append((chr_x, chr_y, x, y))
    return detected

def __perform_detection(model, dataloader, round = True, label_cutoff=0.95):
    detected = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        if round:
            labels = torch.round(output).detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels==1]
            y_list = position[1][labels==1]
        else:
            labels = output.detach().cpu().numpy().reshape(-1)
            x_list = position[0][labels>=label_cutoff]
            y_list = position[1][labels>=label_cutoff]
        if len(x_list) > 0:
            for x, y in zip(x_list.numpy(), y_list.numpy()):
                detected.append((x, y))
    return detected

__index_to_label = ['negative', '--', '-+', '+-', '++']

def __perform_classification(model, dataloader):
    classified = []
    cur_tqdm = tqdm(dataloader)
    for data, position in cur_tqdm:
        output = model(data)
        labels = torch.argmax(output, dim=1).reshape(-1)
        for x_y, label in zip(position.numpy(), labels):
            #if __index_to_label[label] != 'negative':
            classified.append((x_y[0], x_y[1], __index_to_label[label]))
    return classified

def __group_patches(detected, image_size):
    detected = np.array(detected)
    detected = detected[np.argsort(detected.sum(axis=1))]
    intersec_dist = image_size*np.sqrt(2)

    def dist(d1, d2):
        return np.sqrt(((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2))

    i = 0
    groups = []
    while i < len(detected):
        for j in range(i, len(detected)):
            
            if j == (len(detected)-1) or dist(detected[j], detected[j+1]) > intersec_dist:
                break
        left_up = int(min(detected[i][0], detected[j][0])-image_size//2), int(min(detected[i][1], detected[j][1])-image_size//2)
        width = int(abs(detected[i][0] - detected[j][0]) + image_size)
        height = int(abs(detected[i][1] - detected[j][1]) + image_size)
        i=j+1
        groups.append((left_up, (width, height)))
    return groups


resolutions_list = (100000, 50000, 10000, 5000, 1000)
    
def predict(file_path, search_in_1k, batch_size, device):
    local_path = os.getcwd()
    
    #Stage 1 - 50k, diagonal detection
    print('Started Stage 1')
    resolution_1 = resolutions_list[1]
    image_size_1 = 48
    dataset = EvalDatasetDiag(file_path, resolution=resolution_1, image_size=image_size_1, step=image_size_1//2, device=device)
    print('Stage 1 dataset loaded')
    model = DetectModel(image_size=image_size_1, num_models=3)
    model.to(device)
    model.load_state_dict(torch.load(f'{local_path}/weights/torch_ensemble_50k_48_diag.pt', map_location=device))
    model.eval()

    detected = __perform_detection(model, DataLoader(dataset, batch_size=64))
    __save_result_to_csv(local_path, detected, 'stage1')

    #Stage 2 - 10k, diagonal detection
    print('Started Stage 2')
    resolution_2 = resolutions_list[2]
    image_size_2 = 48

    matrices_det = []
    c = cooler.Cooler(f'{file_path}::/resolutions/{resolution_2}').matrix(balance=False)
    for d in detected:
        mat = np.log10(c[d[0]:min(d[0]+int(image_size_1*5), c.shape[0]), d[1]:min(d[1]+int(image_size_1*5), c.shape[1])])
        matrices_det.append((mat, d[0]))

    model = DetectModel(image_size=image_size_2)
    model.to(device)
    model.load_state_dict(torch.load(f'{local_path}/weights/torch_ensemble_10k_48_diag.pt', map_location=device))
    model.eval()

    dataset = PatchesDiagDataset(matrices_det, image_size_1, image_size_2, resolution_1, resolution_2, 12, device=device)
    print('Stage 2 dataset loaded')
    detected_2 = __perform_detection(model, DataLoader(dataset, batch_size=batch_size))
    __save_result_to_csv(local_path, detected_2, 'stage2')

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
    resolution_3 = resolutions_list[1]
    coords_list = []
    pad = image_size_3//2
    c = cooler.Cooler(f'{file_path}::/resolutions/{resolution_3}')
    chr_coords = get_chromosome_coords(coords_set, c.chromnames, c.chromsizes, resolution_3)
    for chr_x, x in chr_coords:
        for chr_y, y in chr_coords:
            if x < y or  (abs(x-y)<image_size_3 and chr_x == chr_y) : 
                continue
            coords_list.append(((chr_x, chr_y), (x, y)))
    coords_list = sorted(coords_list, key=lambda x: x[0])
    dataset = PatchesDataset(file_path, resolution_3, image_size_3, coords_list, device=device, use_means=True)
    print('Stage 3 dataset loaded')
    
    model = DetectModel(image_size=image_size_3)
    model.to(device)
    model.load_state_dict(torch.load(f'{local_path}/weights/torch_ensemble_50k_48_patch.pt', map_location=device))
    model.eval()
    
    detected_3 = __perform_detection_chroms(model, DataLoader(dataset, batch_size=batch_size))
    #detected_3 = get_genome_coords(detected_3, c.chromnames, c.chromsizes, resolution_3)

    __save_result_to_csv(local_path, detected_3, 'stage3')
 
    print('Started Stage 4')
    image_size_4 = 48
    resolution_4 = resolutions_list[3]

    dataset = ClarifyDataset(file_path, resolution_4, image_size_4, detected_3, image_size_3, resolution_3, device=device, use_means=False)
    print('Stage 4 dataset loaded')
    model = DetectModel(in_channels=1, image_size=image_size_4)
    model.to(device)
    model.load_state_dict(torch.load(f'{local_path}/weights_old/torch_ensemble_5k_48_clr.pt', map_location=device))
    model.eval()

    detected_4 = __perform_detection_chroms(model, DataLoader(dataset, batch_size=batch_size))
    detected_4 = get_genome_coords(detected_4, c.chromnames, c.chromsizes, resolution_4)
    __save_result_to_csv(local_path, detected_4, 'stage4')

    resolution_6 = resolutions_list[2]
    image_size_5 = 24
    if len(detected_4) > 0:
        #Stage 4.5 - 5k, unite intersected detection boxes
        print('Started Stage 4.5')
        groups = __group_patches(detected_4, image_size_4)
        '''
        #Stage 5 - 5k, find exact SVs location
        print('Started Stage 5')
        resolution_5 = resolutions_list[3]
        dataset = BigPatchesDataset(file_path, resolution_5, image_size_5, groups, device)
        model = DetectModel(image_size=image_size_5)
        model.to(device)
        model.load_state_dict(torch.load(f'{local_path}/weights/torch_ensemble_10k_48_patch.pt', map_location=device))
        model.eval()
        detected_5 = __perform_detection(model, DataLoader(dataset, batch_size=batch_size), round=False, label_cutoff=0.99)
        __save_result_to_csv(local_path, detected_5, 'stage5')
        '''
        
        #Stage 5 - 5k, find exact SVs location
        print('Started Stage 5')

        resolution_5 = resolutions_list[3]

        dataset = ArbitraryPatchesDataset(file_path, resolution_5, groups)
        detected_5 = []
        dataloader = DataLoader(dataset, batch_size=1)
        for data, position in tqdm(dataloader):
            center = np.unravel_index(data[0].cpu().numpy().argmax(), data[0].shape)
            dot = ((position[0].item()+center[0], position[1].item()+center[1]))
            detected_5.append(dot)
        __save_result_to_csv(local_path, detected_5, 'stage5')
        detected_for_cls = np.array(detected_5) 
    else:
        detected_for_cls = np.array(detected_3)


    image_size_6 = 24
    dataset = ClassificationDataset(file_path, resolution_6, image_size_6, detected_for_cls // (resolution_6//resolution_4), device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ClassificationModel(in_channels=1, image_size=image_size_6, num_models=10, num_classes=5)
    model.to(device)
    model.load_state_dict(torch.load(f'{local_path}/weights/torch_ensemble_10k_24_classify.pt', map_location=device))
    model.eval()
    classified = __perform_classification(model, DataLoader(dataset, batch_size=batch_size))
    __save_result_to_csv(local_path, classified, 'classification')

    result_file_name = 'stage5.csv'
    last_resolution = resolution_5
    if search_in_1k:
        #Stage 6 - 1k, try to find location more precisely
        print('Started Stage 6')
        resolution_7 = resolutions_list[4]
        image_size_7 = image_size_5*(last_resolution//resolution_7)
        res_step = last_resolution // resolution_7
        dataset = ClearPatchesDataset(file_path, resolution_7, image_size_7, detected_for_cls, res_step)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        detected_6 = []
        for data, position in tqdm(dataloader):
            patch = data[0].cpu().numpy()
            max_value = np.nanmax(patch)
            center_value = patch[image_size_7//2, image_size_7//2]
            if max_value-center_value > max_value//4:
                center = np.unravel_index(patch.argmax(), patch.shape)
                dot = ((position[0].item()+center[0]-image_size_7//2, position[1].item()+center[1]-image_size_7//2))
                detected_6.append(dot)
            else:
                detected_6.append((position[0].item(), position[1].item()))

        __save_result_to_csv(local_path, detected_6, 'stage6')
        result_file_name = 'stage6.csv'
        last_resolution = resolution_7

    last_detections = np.genfromtxt(f"{local_path}/{result_file_name}", delimiter=",")
    results = []
    if len(last_detections) == 2 and not isinstance(last_detections[0], np.ndarray):
        last_detections = [(last_detections[0], last_detections[1]), ]
    for d, cls in zip(last_detections, classified):
        #if cls[2] != 'negative':
        results.append((int(d[0]*last_resolution), int(d[1]*last_resolution), cls[2]))
    
    pd.DataFrame(np.array(results), columns=['bp_1', 'bp_2', 'label']).to_csv(f'{local_path}/result.csv', index=False)

    print('Detection finished! Results are in result.csv')


def main(cmdline=None):
    if cmdline is None:
        cmdline=sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='This utilite find coordinates of structural variations breakpoints in Hi-C data')
    parser.add_argument('file_path', type=str, help=
                        "Path to HiC file - .mcool format, should have 50Kb, 10Kb, 5Kb resolitions and 1kb resolution if --search_in_1k option used")
    parser.add_argument("--search_in_1k", action=argparse.BooleanOptionalAction, help="Provide if want to perform detection on 1Kb resolution")
    parser.add_argument('-B', '--batch_size', required=False, type=int, help="Size of data batch processed simultaneously, larger size reduces time of work but requires more RAM and VRAM")
    parser.add_argument('--device', required=False, type=str, help="Which device use for CNN inference. Possible options are: GPU, CPU and auto. auto option will use GPU if it's possible")
    parser.set_defaults(search_in_1k=False)
    parser.set_defaults(batch_size=512)
    parser.set_defaults(device='AUTO')

    args = parser.parse_args(cmdline)
    file_path = args.file_path
    search_in_1k = args.search_in_1k
    batch_size = args.batch_size

    if not '.mcool' in file_path:
        raise ValueError("input file should be .mcool")
    
    if not os.path.isabs(file_path):
            file_path = pathlib.Path(os.getcwd()) / pathlib.Path(file_path)
    device_arg = args.device.lower()
    if not device_arg in ('auto', 'gpu', 'cpu'):
        raise ValueError("--device argument should be GPU, CPU or auto")
    
    if device_arg=='auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device_arg =='gpu':
        device = 'cuda:0'
    if device_arg =='cpu':
        device = 'cpu'



    resolutions = []
    with h5py.File(file_path, "r") as f:
        for value in f['resolutions'].values():
            resolutions.append(value.name[13:])
    oblig_res = ('50000', '10000', '5000', '1000') if search_in_1k else ('50000', '10000', '5000')
    if not set(oblig_res).issubset(set(resolutions)):
        raise ValueError(".mcool file should contain 50Kb, 10Kb, 5Kb resolitions and 1kb resolution if --search_in_1k option used")
    
    warnings.filterwarnings("ignore")

    start = time.time()
    predict(file_path, search_in_1k, batch_size, device)
    print('Executed in {0:0.1f} seconds'.format(time.time() - start))


if __name__ == '__main__':
    main()