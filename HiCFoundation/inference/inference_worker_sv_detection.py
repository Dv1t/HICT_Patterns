import numpy as np
import torch
import torch.nn as nn
from ops.Logger import MetricLogger
from scipy.sparse import coo_matrix

def inference_worker(model,data_loader,log_dir=None,args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Inference: '
    print_freq = args.print_freq
    print("number of iterations: ",len(data_loader))
    dataset_shape_dict = data_loader.dataset.dataset_shape
    #resolution enhancement: accumulate dense running mean/count directly (float32),
    #instead of caching one COO triplet per (heavily overlapping) window in RAM.
    #this is the same low-memory pattern already used for task 4 / task 5 below.
    output_dict={}
    for chrom in dataset_shape_dict:
        current_shape = dataset_shape_dict[chrom]
        mean_array = np.zeros(current_shape, dtype=np.float32)
        count_array = np.zeros(current_shape, dtype=np.float32)
        output_dict[chrom] = {"mean":mean_array,"count":count_array}

    cutoff= 1000
    cutoff = torch.tensor(cutoff).float().cuda()
    log_cutoff = torch.log10(cutoff+1).cuda()
    
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input,total_count,indexes = data
        input = input.cuda()
        total_count = total_count.cuda()
        total_count = total_count.float()
        #match input dtype to the model's parameter dtype (fp32 or fp16)
        model_dtype = next(model.parameters()).dtype
        input = input.to(model_dtype)
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input,total_count) 
            output = output.float()   #upcast back to fp32 before any log10/pow post-processing below

        #resolution enhancement
        output = output*log_cutoff
        output = torch.pow(10,output)-1
        output = torch.clamp(output,min=0)

        output = output.detach().cpu().numpy()
        input = input.detach().cpu().numpy()
        chrs, row_starts, col_starts = indexes
        for i in range(len(output)):
            chr = chrs[i]
            row_start = row_starts[i]
            col_start = col_starts[i]
            row_start = int(row_start)
            col_start = int(col_start)
            row_start = max(0,row_start)
            col_start = max(0,col_start)
            current_shape = dataset_shape_dict[chr]
            row_end = min(row_start+args.input_row_size,current_shape[0])
            col_end = min(col_start+args.input_col_size,current_shape[1])
            current_input = input[i]

            if np.isnan(np.sum(current_input)):
                print("empty matrix:",chr,row_start,col_start)
                continue
            cur_output = output[i]

            #resolution enhancement: accumulate directly into the dense running
            #mean/count arrays (in place) rather than caching a COO triplet per window.
            cur_output = cur_output[:row_end-row_start,:col_end-col_start].astype(np.float32,copy=False)
            output_dict[chr]['mean'][row_start:row_end, col_start:col_end] += cur_output
            output_dict[chr]['count'][row_start:row_end, col_start:col_end] += 1
            

    #resolution enhancement: each chromosome's mean/count are already dense
    #running accumulators (see above), so we only need one pass per chromosome
    #to average overlapping windows and threshold - no giant concatenate/sum_duplicates.
    final_dict={}
    for chrom in list(output_dict.keys()):
        mean_array = output_dict[chrom]['mean']
        count_array = output_dict[chrom]['count']
        count_array = np.maximum(count_array,1)
        mean_array /= count_array          #in-place, avoids a second full-size copy
        #symmetrize (cheap: single dense op, done once per chromosome)
        mean_array = (mean_array + mean_array.T)/2
        #remove very small prediction to save time / memory before sparsifying
        mean_array[mean_array<=0.01] = 0
        prediction_sym = coo_matrix(np.triu(mean_array))
        prediction_sym.eliminate_zeros()
        print("finish summarize %s prediction"%chrom,prediction_sym.nnz)
        final_dict[chrom] = prediction_sym
        #free this chromosome's dense buffers immediately, don't wait for GC
        del output_dict[chrom]
    return final_dict