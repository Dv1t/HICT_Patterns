import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import datetime
import json
import numpy as np
import torchvision.transforms as transforms

from ops.distribute_utils import init_distributed_mode,get_world_size,get_rank,is_main_process
from data_processing.finetune_dataset import Finetune_Dataset
from data_processing.collate_fn import collate_fn
from model.pos_embed import interpolate_pos_embed_inputsize
from ops.Logger import print_important_info,print_warning_info
from ops.io_utils import write_log
import model.lr_decay as lrd
from model.NativeScaler import NativeScalerWithGradNormCount as NativeScaler
from model.model_utils import load_model,save_checkpoint,save_model2path
from finetune.train_epoch import train_epoch
from finetune.val_epoch import val_epoch

def parse_text(config_file, data_dir):
    train_list=[]
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            line = line.replace('\n', '')
            if len(line) == 0:
                continue
            current_path = os.path.join(data_dir, line)
            if not os.path.exists(current_path):
                print("The sub-directory {} does not exist in the data directory".format(current_path))
                print("Please check the sub-directory name in the {} file".format(config_file))
                continue
            train_list.append(current_path)
    return train_list
def configure_data_loader(args):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_dir=os.path.abspath(args.data_path)
    train_config= os.path.abspath(args.train_config)
    train_list = parse_text(train_config, data_dir)
    val_config= os.path.abspath(args.valid_config)
    val_list = parse_text(val_config, data_dir)
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    dataset_train = Finetune_Dataset(train_list,transform=transform_train,
                                     window_height=input_row_size,window_width=input_col_size)
    dataset_val = Finetune_Dataset(val_list,transform=transform_train,
                                        window_height=input_row_size,window_width=input_col_size)
    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
        global_rank = -1
    sample_batch_size = args.batch_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=sample_batch_size, sampler=sampler_train, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=sample_batch_size, sampler=sampler_val, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )
    return data_loader_train, data_loader_val
def config_writer(output_dir,tensorboard_log):
    tensorboard_dir = os.path.join(output_dir,'tensorboard')
    os.makedirs(tensorboard_dir,exist_ok=True)
    if tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(tensorboard_dir)
    else:
        log_writer = None
    return log_writer
def main_worker(gpu, ngpus_per_node,args):
    if ngpus_per_node>1:
        init_distributed_mode(gpu,ngpus_per_node,args)
    else:
        args.distributed=False
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    #configure distributed setting
    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
    else:
        global_rank = -1
        num_tasks = 1
    output_dir = os.path.abspath(args.output)
    if global_rank==0:
        os.makedirs(output_dir,exist_ok=True)
        log_writer =config_writer(output_dir,args.tensorboard)
    elif args.distributed:
        log_writer = None
    else:
        os.makedirs(output_dir,exist_ok=True)
        log_writer = config_writer(output_dir,args.tensorboard)
        

    cudnn.benchmark = True
    device = torch.device(args.device)

    # Data loading code
    data_loader_train, data_loader_val = configure_data_loader(args)
    print("Data loader is configured!")

    import model.Vision_Transformer_count as Vision_Transformer
    #should be a dyanmic input model
    patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
    vit_backbone = Vision_Transformer.__dict__[args.model](img_size=(args.input_row_size,args.input_col_size))

    pretrain_path = os.path.abspath(args.pretrain)
    if os.path.exists(pretrain_path):
        print("Loading pre-trained model from {}".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = vit_backbone.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        #this can apply to most scenarios but not our condition
        patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
        interpolate_pos_embed_inputsize(vit_backbone, checkpoint_model,input_size=patch_wise_size,
                                            use_decoder=False)
        # load pre-trained model
        msg = vit_backbone.load_state_dict(checkpoint_model, strict=False)
        print_important_info("Loading pre-trained encoder message: %s"%str(msg))
    else:
        print_warning_info("You did not load a pre-trained model. The model will be trained from scratch.")
        

    #task 0 indicates this is under fine-tuning setting
    from model.Finetune_Model_Head import Finetune_Model_Head
    model = Finetune_Model_Head(vit_backbone, task=0,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=patch_wise_size)
    if os.path.exists(pretrain_path) and os.path.isfile(pretrain_path) and os.path.getsize() > 1000:
        print("Loading pre-trained model from {} for decoder".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        interpolate_pos_embed_inputsize(model, checkpoint['model'],
                                        input_size=patch_wise_size,use_decoder=True)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print_important_info("Loading pre-trained decoder message: %s"%str(msg))
    model.to(device)
    model_without_ddp = model
    #freeze encoder or not
    #finetune 1: only fine-tune the model's decoder; 2: fine-tune the whole model;
    if args.finetune==1:
        for _, p in model_without_ddp.vit_backbone.named_parameters():
            p.requires_grad = False
        print_important_info("Only fine-tune the model's decoder")

    if args.distributed:
        #not necessary for current setting, since all param with grad
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp =model
    #print out model information
    print("Model is configured!")

    #configure batch size, learning rate, weight decay
    eff_batch_size = args.batch_size * args.accum_iter*get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    print("Learning rate: %.6f"%args.lr)
    print("Accumulative grad iteration: %d"%args.accum_iter)
    print("Effective batch size: %d"%eff_batch_size)
    print("Base learning rate: %.6f"%args.blr)
    
    #configure optimizer
    
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd_decoder(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer is configured!")

    loss_scaler = NativeScaler()
    resume_path=os.path.abspath(args.resume)
    load_model(resume_path,args, model_without_ddp, optimizer, loss_scaler)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir,exist_ok=True)
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(log_dir,exist_ok=True)

    epochs = int(args.epochs)
    start_epoch = int(args.start_epoch)
    start_time = time.time()
    print("Start training from epoch %d"%start_epoch," to epoch %d"%epochs)
    save_freq = args.save_freq
    best_loss = 1e9
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_epoch(model, data_loader_train, optimizer, 
                                  loss_scaler, epoch, device,
                                  log_writer=log_writer, args=args)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if is_main_process():
            write_log(log_dir,"train",log_stats)
        val_stats = val_epoch(model, data_loader_val, device, epoch,
                              log_writer=log_writer, args=args)
        log_stats_val = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,}
        if is_main_process():
            write_log(log_dir,"val",log_stats_val)
        if epoch%save_freq==0 or epoch==epochs-1:
            #output_dir, args,epoch, model_without_ddp, optimizer, loss_scaler
            save_checkpoint(model_dir, args,epoch, model_without_ddp, optimizer, loss_scaler)
        
        val_loss = val_stats['loss']
        if val_loss < best_loss:
            best_loss = val_loss
            #model_path,args,epoch, model_without_ddp, optimizer, loss_scaler
            model_path = os.path.join(model_dir, 'model_best.pth.tar')
            save_model2path(model_path,args,epoch, model_without_ddp, optimizer, loss_scaler)
    total_time = time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Fine-tuning of HiCFoundation is finished!")
    print("The model is saved in {}".format(model_dir))
    print("The log is saved in {}".format(log_dir))