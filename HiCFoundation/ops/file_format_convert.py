from utils.hic2array import hic2array
from utils.cool2array import cool2array_intra
from utils.array2hic import array2hic
from utils.array2cool import array2cool
import os 
import numpy as np
from ops.sparse_ops import array_to_coo
from scipy.sparse import coo_matrix
from collections import defaultdict
def write_pkl(return_dict,output_pkl_path):
    import pickle
    with open(output_pkl_path,'wb') as f:
        pickle.dump(return_dict,f)
    print("finish writing to:",output_pkl_path)
def load_pkl(input_pkl):
    import pickle
    with open(input_pkl,'rb') as f:
        return_dict = pickle.load(f)
    return return_dict

def read_text(input_file,config_resolution):
    #records should be readID chr1 pos1 chr2 pos2
    #read line by line to get the sparse matrix
    final_dict=defaultdict(list)
    with open(input_file,'r') as f:
        for line in f:
            line = line.strip().split()
            try:
                chr1 = line[1]
                chr2 = line[3]
                pos1 = int(line[2])//config_resolution
                pos2 = int(line[4])//config_resolution
                final_dict[(chr1,chr2)].append((pos1,pos2))
            except:
                print("*"*40)
                print("Skip line in records:",line)
                print("The line should be in format of [readID chr1 pos1 chr2 pos2]")
                print("*"*40)
    return final_dict

def countlist2coo(input_dict):
    final_dict={}
    for key in input_dict:
        row=[]
        col=[]
        data=[]
        for item in input_dict[key]:
            row.append(item[0])
            col.append(item[1])
            data.append(1)
        max_size = max(max(row),max(col))+1
        cur_array = coo_matrix((data,(row,col)),shape=(max_size,max_size))
        #sum duplicates
        cur_array.sum_duplicates()
        final_dict[key]=cur_array
    return final_dict
def convert_to_pkl(input_file, output_dir,config_resolution):
    output_pkl = os.path.join(output_dir, "input.pkl")
    #if it is a .hic file
    if input_file.endswith('.hic'):
        #convert to .pkl format, only keep intra-chromosome regions
        hic2array(input_file,output_pkl=output_pkl,
                  resolution=config_resolution,normalization="NONE",
                  tondarray=2)
    elif input_file.endswith('.cool'):
        return_dict=cool2array_intra(input_file,normalize=False,
                         tondarray=False,binsize=config_resolution)
        write_pkl(return_dict,output_pkl)
    elif input_file.endswith('.pkl'):
        #load pickle to sanity check
        return_dict = load_pkl(input_file)
        final_dict = {}
        #check if it is dict
        if not isinstance(return_dict,dict):
            raise ValueError("Input pkl file should be a dictionary")
        else:
            for key in return_dict:
                if isinstance(return_dict[key],np.ndarray):
                    final_dict[key] = array_to_coo(return_dict[key])
                elif isinstance(return_dict[key],coo_matrix):
                    final_dict[key] = return_dict[key]
                else:
                    raise ValueError("The value of the dictionary in .pkl should be either numpy array or coo_matrix")
        write_pkl(final_dict,output_pkl)
    elif input_file.endswith('.txt') or input_file.endswith('.pairs'):
        #convert to .pkl format
        initial_dict = read_text(input_file,config_resolution)
        #filter intra-chromosome regions
        final_dict = {}
        for key in initial_dict:
            if key[0] == key[1]:
                final_dict[key[0]] = initial_dict[key]
        #then change it to coo_matrix array
        return_dict = countlist2coo(final_dict)
        write_pkl(return_dict,output_pkl)
    elif input_file.endswith('.npy'):
        #load numpy array
        input_array = np.load(input_file)
        #convert to coo_matrix
        final_array = array_to_coo(input_array)
        #save to pkl
        final_dict={"chr_tmp":final_array}
        write_pkl(final_dict,output_pkl)
    else:
        print("Unsupported file format ",input_file)
        print("Supported file format: .hic/.cool/.pkl/.txt/.npy")
        raise ValueError("Unsupported file format")

    return output_pkl

def pkl2others(input_pkl, output_file,config_resolution,genome_id):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir,exist_ok=True)
    data=load_pkl(input_pkl)
    if output_file.endswith('.txt') or output_file.endswith('.pairs'):
        #write to simple txt
        # [chr1, pos1, chr2, pos2, count]
        with open(output_file,'w') as file:
            file.write("#readID\tchr1\tpos1\tchr2\tpos2\tcount\n")
            for chrom in data:
                if type(data[chrom]) == np.ndarray:
                    for i in range(data[chrom].shape[0]):
                        for j in range(data[chrom].shape[1]):
                            if data[chrom][i,j]>0:
                                file.write(f".\t{chrom}\t{i*config_resolution}\t{chrom}\t{j*config_resolution}\t{data[chrom][i,j]}\n")
                else:
                    for i in range(data[chrom].nnz):
                        row = data[chrom].row[i]
                        col = data[chrom].col[i]
                        count = data[chrom].data[i]
                        file.write(f".\t{chrom}\t{row*config_resolution}\t{chrom}\t{col*config_resolution}\t{count}\n")
    elif output_file.endswith('.npy'):
        if len(data)>1:
            print("Warning: multiple chromosomes detected, please check the output in .pkl format:",input_pkl)
            print("The format is dict in format of [chr]:[scipy.sparse.coo_matrix]")
            return
        current_array = data[list(data.keys())[0]]
        if isinstance(current_array,coo_matrix):
            current_array = current_array.toarray()
        np.save(output_file,current_array)
    elif output_file.endswith('.hic'):
        #https://github.com/aidenlab/juicer/wiki/Pre
        cur_py_path = os.path.abspath(__file__)
        cur_py_dir = os.path.dirname(cur_py_path)
        code_repo_dir=os.path.dirname(cur_py_dir)
        juicer_tools= os.path.join(code_repo_dir,"utils","juicer_tools.jar")
        array2hic(juicer_tools,input_pkl,output_file,config_resolution,genome_id,1)
    elif output_file.endswith('.cool'):
        
        array2cool(input_pkl,output_file,config_resolution,genome_id,1)
    elif output_file.endswith('.pkl'):
        output_file = input_pkl
    else:
        print("Unsupported file format ",output_file)
        output_file=input_pkl
    print("Final output is saved in ",output_file)
    return output_file