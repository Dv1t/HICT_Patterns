import torch
import os
import torch.nn as nn

from utils.hic_coverage import calculate_coverage
from data_processing.inference_dataset_sv_detection import Inference_Dataset
from ops.io_utils import write_pickle
from ops.file_format_convert import pkl2others
from inference.inference_worker_sv_detection import inference_worker
import torchvision.transforms as transforms
import model.Vision_Transformer_count as Vision_Transformer
from model.Finetune_Model_Head import Finetune_Model_Head
from ops.quantization import quantize_model
import pandas as pd

def configure_dataset(args,input_pkl):
    resolution = int(args.resolution)
    transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #judge if it is a very deep sequencing data, if it is, set max_cutoff to None
    coverage_perresolution = calculate_coverage(input_pkl)/resolution
    if coverage_perresolution>1:
        max_cutoff = None
    else:
        max_cutoff = 100

    stride = args.stride
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    input_coords = {}
    input_coords_df = pd.read_csv(args.input_coords)
    total_rows = 0
    for chrom in input_coords_df['chr'].unique():
        input_coords[str(chrom)] = []
        for x, y in input_coords_df[input_coords_df['chr'] == chrom][['x', 'y']].values:
            input_coords[str(chrom)].append((x//resolution, y//resolution))
        total_rows += len(input_coords[str(chrom)])

    #dedup: input_coords are already resolution-binned, so exact duplicate
    #(x,y) pairs are common in breakpoint call sets (e.g. multiple supporting
    #read-pairs/callers hitting the same bin). Each duplicate would otherwise
    #trigger its own redundant model forward pass AND its own redundant
    #entry in the off-diagonal sparse accumulator in inference_worker, so
    #dedup here saves both compute and memory before any windows are built.
    total_unique = 0
    for chrom in input_coords:
        before = len(input_coords[chrom])
        input_coords[chrom] = list(set(input_coords[chrom]))
        after = len(input_coords[chrom])
        total_unique += after
        if before != after:
            print(f"{chrom}: deduped breakpoints {before} -> {after}")
    print(f"Total breakpoints: {total_rows} -> {total_unique} after dedup")

    dataset = Inference_Dataset(data_path=input_pkl,
                                input_coords=input_coords,
                                transform=transform_input,
                                stride=stride,
                                window_height= input_row_size,
                                window_width = input_col_size,
                                max_cutoff=max_cutoff)
    sample_batch_size = args.batch_size
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    return data_loader_test

def main_worker(args, input_pkl):
    resolution = args.resolution
    #check model_path exists
    model_path = os.path.abspath(args.model_path)
    assert os.path.exists(model_path), "model_path does not exist"
    output_dir = os.path.abspath(args.output)
    dataloader = configure_dataset(args, input_pkl)

    patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
    vit_backbone = Vision_Transformer.__dict__[args.model](img_size=(args.input_row_size,args.input_col_size))

    model = Finetune_Model_Head(vit_backbone, task=args.task,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=patch_wise_size)
    
    #load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if "model" in checkpoint:
        checkpoint_model = checkpoint["model"]
    elif "state_dict" in checkpoint:
        checkpoint_model = checkpoint["state_dict"]
    else:
        checkpoint_model = checkpoint
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Loading fine-tuned task-specific model message:",msg)
    
    model = model.cuda()
    if getattr(args, "quantize", "none") != "none":
        # keep the small task-specific output heads in full precision;
        # the bulk of the compute is in vit_backbone/decoder_blocks anyway
        model = quantize_model(
            model,
            mode=args.quantize,
            skip_modules=["decoder_map", "map_block", "decoder_pred", "map_blocks"],
        )
        print(f"Model quantized with mode={args.quantize}")

    # DataParallel + quantized (torchao) tensor subclasses is unreliable,
    # so only shard across GPUs when running full precision and on more than one GPU.
    if getattr(args, "quantize", "none") == "none" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=None)

    return_dict= inference_worker(model,dataloader,
                                  log_dir=output_dir,
                                  args=args)

    #convert to hic format as final output
    output_pkl = os.path.join(output_dir,"HiCFoundation_enhanced.pkl")
    #revise the return dict key if it has "_", make to one chromosome
    for key in list(return_dict.keys()):
        if "_" in key:
            key_list = key.split("_")
            return_dict[key_list[0]] = return_dict[key]
            del return_dict[key]
    write_pickle(return_dict,output_pkl)
    input_file = os.path.abspath(args.input)
    extention_name = input_file.split('.')[-1]
    output_file = os.path.join(output_dir,"HiCFoundation_enhanced."+extention_name)
    pkl2others(output_pkl, output_file,resolution,args.genome_id)
    if not os.path.exists(output_file):
        print("Error: file conversion failed.")
        print("Resolution enhancement finished!")
        print("The final output is saved in .pkl format, please convert it to other formats manually.")
        print("The .pkl file is saved to ",output_pkl)

    print("Enjoy your HiCFoundation results!")