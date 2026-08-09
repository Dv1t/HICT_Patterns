import numpy as np
import torch
import torch.utils.data
from scipy.sparse import coo_matrix
import pickle 

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,
                 input_coords,
                transform=None,
                stride=20,
                window_height= 224,
                window_width = 224,
                max_cutoff=None):
        """
        #data_path: the path of the input data
        #transform: the transform applied to the input data
        #stride: the stride of the sliding window
        #window_height: the height of the sliding window
        #window_width: the width of the sliding window
        #max_cutoff: the maximum number of valid pixels in a window
        """
        self.data_path = data_path
        self.transform = transform
        self.stride = stride
        self.window_height = window_height
        self.window_width = window_width
        self.data = pickle.load(open(data_path,'rb'))
        self.max_cutoff = max_cutoff
        self.total_count = 0
        self.input_index = []
        self.dataset_shape = {}
        new_data = {}
        half_window_width = self.window_width//2
        half_window_height = self.window_height//2
        #revise the data to make it to be symmetrical

        for chrom in self.data:
            hic_data = self.data[chrom]
            if chrom not in input_coords:
                print(f"INFO: chromosome {chrom} not in input_coords, skip")
                continue
            coords = input_coords[chrom]
            #if smaller than half window height, skip
            if hic_data.shape[0]<half_window_height:
                continue
            self.total_count += np.sum(hic_data.data)   
            if hic_data.shape[0]==hic_data.shape[1]:
                combine_row = np.concatenate([hic_data.row,hic_data.col])
                combine_col = np.concatenate([hic_data.col,hic_data.row])
                combine_data = np.concatenate([hic_data.data,hic_data.data])
                hic_data.row = combine_row
                hic_data.col = combine_col
                hic_data.data = combine_data #triu part
                #divide to half for the diagonal region
                select_index= (hic_data.row==hic_data.col)
                hic_data.data[select_index] = hic_data.data[select_index]/2
            
            input_row_size= max(hic_data.shape[0],self.window_height) #do padding if necessary
            input_col_size= max(hic_data.shape[1],self.window_width)

            final_hic_data= coo_matrix((hic_data.data,(hic_data.row,hic_data.col)),
                                       shape=(input_row_size,input_col_size))

            new_data[chrom] = final_hic_data
            self.dataset_shape[chrom] = final_hic_data.shape
            row_size = final_hic_data.shape[0]
            col_size = final_hic_data.shape[1]
            
            for (x, y) in coords:
                row_start = max(0,x-half_window_height)
                col_start = max(0,y-half_window_width)
                row_end = min(row_start+self.window_height,row_size)
                col_end = min(col_start+self.window_width,col_size)
                middle_col_point = (col_start+col_end)//2
                self.input_index.append((chrom,row_start,col_start,row_end,col_end,middle_col_point))

            row_iter_list = list(range(0,row_size-self.window_height,stride))+[row_size-self.window_height]+[row_size-self.window_height-stride]
            col_iter_list = list(range(0,col_size-self.window_width,stride))+[col_size-self.window_width]+[col_size-self.window_width-stride]
            for i in row_iter_list:
                for j in col_iter_list:
                    
                    if abs(i-j)>100: #skip the windows that are too far away from the diagonal
                        continue
                    i = max(0,i)
                    j = max(0,j)
                    row_max_bound = min(i+self.window_height,row_size)
                    col_max_bound = min(j+self.window_width,col_size)
                    middle_col_point = (j+col_max_bound)//2
                    self.input_index.append((chrom,i,j,row_max_bound,col_max_bound,middle_col_point))

        self.data = new_data
        print("Total reads of input hic: ",self.total_count)
        print("Total number of input windows: ",len(self.input_index))
    def __len__(self):
        return len(self.input_index)
    
    def convert_rgb(self,data_log,max_value):
        data_red = np.ones(data_log.shape)
        data_log1 = (max_value-data_log)/max_value
        data_rgb = np.concatenate([data_red,data_log1,data_log1],axis=0,dtype=np.float32)
        data_rgb = data_rgb.transpose(1,2,0)
        return data_rgb
    
    def __getitem__(self, idx):
        current_index = self.input_index[idx]
        chrom,row_start,col_start,row_end,col_end,col_middle_point = current_index
        row_record_start = row_start
        col_record_start = col_start #this is specifically kept for embedding infer, which returns the center loc as final location for recording.
        current_array = self.data[chrom]

        submat = np.zeros([1,self.window_height,self.window_width])

        #it is a scipy sparse coo matrix
        select_index1 = (current_array.row>=row_start) & (current_array.row<row_end)
        select_index2 = (current_array.col>=col_start) & (current_array.col<col_end)

        final_row = current_array.row[select_index1&select_index2]
        final_col = current_array.col[select_index1&select_index2]
        final_data = current_array.data[select_index1&select_index2]
        try:
            final_array = coo_matrix((final_data, (final_row-row_start, final_col-col_start)), 
                                    shape = (row_end-row_start,col_end-col_start),dtype=np.float32)
            final_array = final_array.toarray()
        except:
            print("Error: the selected region is empty, please check the input coordinates and the input data")
            print("chrom: ",chrom)
            print("row_start: ",row_start)
            print("col_start: ",col_start)
            print("row_end: ",row_end)
            print("col_end: ",col_end)
            print('final_row: ',final_row)
            print('final_col: ',final_col)
            exit(1)

        if row_start==col_start:
            np.fill_diagonal(final_array,0)

        submat[0,0:row_end-row_start,0:col_end-col_start] = final_array
        input = np.nan_to_num(submat)
        if self.max_cutoff is not None:
            input = np.minimum(input,self.max_cutoff)
            max_value = self.max_cutoff
        else:
            max_value = np.max(input)
        input = np.log10(input+1)
        max_value = np.log10(max_value+1)
        input = self.convert_rgb(input,max_value)
        if self.transform is not None:
            input = self.transform(input)
        return input,self.total_count,[chrom,row_record_start,col_record_start]