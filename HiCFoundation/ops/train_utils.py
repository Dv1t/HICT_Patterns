import torch
import numpy as np
import torch.nn as nn

def to_device(data, device):
    if data is not None:
        new_data = data.to(device,non_blocking=True)
    else:
        new_data = None
    return new_data

def list_to_device(data_list, device):
    new_data_list = []
    for data in data_list:
        data = to_device(data, device)
        if data is not None:
            data = data.float()
        new_data_list.append(data)
    return new_data_list

def to_value(data):
    if isinstance(data, torch.Tensor):
        return data.item()
    else:
        return data

def create_image(samples):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_mean = torch.tensor(imagenet_mean,device=samples.device)
    imagenet_std = torch.tensor(imagenet_std,device=samples.device)
    new_samples = torch.einsum("bchw,c->bchw",samples,imagenet_std)
    new_samples = torch.clip((new_samples+ imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) * 255, 0, 255)
    return new_samples

def torch_to_nparray(data):
    #https://github.com/pytorch/pytorch/blob/main/torch/utils/tensorboard/summary.py
    #image take n,c,h,w,
    """
    'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32). Out-of-range values
        will be clipped.

    """
    data = data.cpu().numpy()
   #data = data.transpose(0,2,3,1)
    data=np.array(data,dtype=np.uint8)
    return data

def convert_gray_rgbimage(samples):
    """
    input: B,H,W
    """
    #add dimension in 1st dim
    if len(samples.shape)==3:
        samples = samples.unsqueeze(1)
    samples = torch.clip(samples, 0, 1)
    red_channel = torch.ones(samples.shape,device=samples.device)
    gb_channel = 1-samples
    new_samples=torch.cat([red_channel,gb_channel,gb_channel],dim=1)*255
    return new_samples