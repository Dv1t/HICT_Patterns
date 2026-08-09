import os
import numpy as np
import torch
import torch.utils.data
import random
from collections import defaultdict
from scipy.sparse import coo_matrix
import pickle 
from ops.sparse_ops import array_to_coo
from ops.io_utils import load_pickle
def validate_input_size(input_matrix, window_height, window_width):
    """
    Validate the input size is larger than the window size
    Args:
        input_matrix: the input matrix
        window_height: the height of the window
        window_width: the width of the window
    """
    if isinstance(input_matrix, coo_matrix):
        input_matrix = input_matrix.toarray()
    input_height, input_width = input_matrix.shape
    if input_height==window_height and input_width==window_width:
        return True
    return False 

def to_tensor(x):
    """
    Convert the input to tensor
    Args:
        x: the input data
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif x is None:
        x = None
    #if already tensor, do nothing
    elif isinstance(x, torch.Tensor):
        pass
    #if float, convert to tensor
    elif isinstance(x, float):
        x = torch.tensor(x)
    elif isinstance(x, int):
        x = torch.tensor(x)
    return x

def list_to_tensor(x):
    """
    Convert the list to tensor
    Args:
        x: the input list
    """
    y=[]
    for i in x:
        y.append(to_tensor(i))
    return y
class Finetune_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_list,   
                transform=None,
                window_height= 224,
                window_width = 224):
        """
        Args:
            data_list: list of data directories
            transform: the transformation to apply to the data
            window_height: the height of the window
            window_width: the width of the window
        """
        self.data_list = data_list
        self.transform = transform
        self.window_height = window_height
        self.window_width = window_width
        self.train_dict=defaultdict(list)
        self.train_list=[]
        for data_index, data_dir in enumerate(data_list):
            cur_dir = data_dir
            dataset_name = os.path.basename(cur_dir)
            listfiles = os.listdir(cur_dir)
            for file_index,file in enumerate(listfiles):
                cur_path = os.path.join(cur_dir, file)
                if file.endswith('.pkl'):
                    if file_index==0:
                        #verify the input pkl file includes the input key
                        data= load_pickle(cur_path)
                        data_keys = list(data.keys())
                        if 'input' not in data:
                            print("The input key is not included in the pkl file. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            continue
                        #check other keys include in the dict
                        target_exist=False
                        for key in data_keys:
                            if "target" in key:
                                target_exist=True
                                break
                        if not target_exist:
                            print("The target key is not included in the pkl file. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            continue
                        #validate the input size
                        input_matrix = data['input']
                        if not validate_input_size(input_matrix, window_height, window_width):
                            print("The input size is not matched with the window size. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            print("The input size is {}".format(input_matrix.shape))
                            print("The specified window size is {} x {}".format(window_height, window_width))
                            print("Please adjust --input_row_size and --input_col_size to match your input.")
                            continue
                    self.train_dict[dataset_name].append(cur_path)
                    self.train_list.append(cur_path)
                else:
                    print("The file {} is not a .pkl file.".format(file),"It is skipped.")
                    continue    
        print("The number of samples used in the dataset is {}".format(len(self.train_list)))
        #you can either select the train_list or train_dict to do training based on your exprience
    def __len__(self):
        return len(self.train_list)
    
    def convert_rgb(self,data_log,max_value):
        if len(data_log.shape)==2:
            data_log = data_log[np.newaxis,:]
        data_red = np.ones(data_log.shape)
        data_log1 = (max_value-data_log)/max_value
        data_rgb = np.concatenate([data_red,data_log1,data_log1],axis=0,dtype=np.float32)#transform only accept channel last case
        data_rgb = data_rgb.transpose(1,2,0)
        return data_rgb
    
    def __getitem__(self, idx):
        train_file = self.train_list[idx]
        data = load_pickle(train_file)
        input_matrix = data['input']
        if isinstance(input_matrix, coo_matrix):
            input_matrix = input_matrix.toarray()
            #make sure you save the down-diagonal regions if you use the coo_matrix
            #to support off-diagonal submatrix, we did not any automatic symmetrical conversion for your input array.
        input_matrix = np.nan_to_num(input_matrix)
        input_matrix = input_matrix.astype(np.float32)
        input_matrix = np.log10(input_matrix+1)
        max_value = np.max(input_matrix)
        input_matrix = self.convert_rgb(input_matrix,max_value)
        if self.transform:
            input_matrix = self.transform(input_matrix)
        if "input_count" in data:
            total_count = data['input_count']
        else:
            total_count = None #indiates not passing the total count
        
        if "2d_target" in data:
            target_matrix = data['2d_target']
            if isinstance(target_matrix, coo_matrix):
                target_matrix = target_matrix.toarray()
            target_matrix = np.nan_to_num(target_matrix)
            target_matrix = target_matrix.astype(np.float32)
        else:
            target_matrix = None

        if "embed_target" in data:
            embed_target = data['embed_target']
            if isinstance(embed_target, coo_matrix):
                embed_target = embed_target.toarray()
            embed_target = np.nan_to_num(embed_target)
            embed_target = embed_target.astype(np.float32)
        else:
            embed_target = None
        
        if "1d_target" in data:
            target_vector = data['1d_target']
            target_vector = np.nan_to_num(target_vector)
            target_vector = target_vector.astype(np.float32)
        else:
            target_vector = None

        return list_to_tensor([input_matrix, total_count, target_matrix, embed_target, target_vector])
        




        
