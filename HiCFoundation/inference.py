import os
import timm
assert timm.__version__ == "0.3.2" # version check for timm
from ops.argparser import  argparser_infer
from ops.file_format_convert import convert_to_pkl
import socket
from inference.main_worker_sv_detection import main_worker

def main(args):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)
    #format processing, convert different formats to .pkl format for further processing
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir,exist_ok=True)
    input_file = os.path.abspath(args.input)
    config_resolution = args.resolution
    input_pkl=convert_to_pkl(input_file, output_dir,config_resolution)
    
    main_worker(args, input_pkl)


if __name__ == '__main__':
    print("HiCFoundation inference started!")
    parser = argparser_infer()
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #print mode based on --task
    if args.task==1:
        print("Reproducibility analysis")
    elif args.task==2:
        print("Loop calling")
    elif args.task==3:
        print("Resolution enhancement")
    elif args.task==4:
        print("Epigenomic assay prediction")
    elif args.task==5:
        print("scHi-C enhancement")
    elif args.task==6:
        print("Hi-C embedding generation")
        embed_depth = args.embed_depth
        if embed_depth>8:
            print("Error: embed_depth is larger than 8, that is beyond decoder depth. Please set embed_depth<=8")
            print("0 indicates the encoder output, k indicates the k-th decoder layer's output")
            exit(1)
    else:
        print("Unknown task specified ",args.task)
        print("Please specify the task using --task with 1,2,3,4,5,6")
        exit(1)
    #check the specied input size, must be a multiple of args.patch_size
    if args.input_row_size%args.patch_size!=0 or args.input_col_size%args.patch_size!=0:
        print("args configuration error: input_row_size and input_col_size must be a multiple of patch_size")
        exit(1)
    #output the args in a beautiful format
    main(args)

