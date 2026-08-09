
import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn.functional as F
import time

from ops.Logger import MetricLogger,SmoothedValue
import model.lr_sched as lr_sched
from finetune.loss import configure_loss
from finetune.train_epoch import list_to_device, to_value, \
                    create_image, torch_to_nparray, convert_gray_rgbimage

def val_epoch(model, data_loader_val, device, epoch,
                              log_writer=None, args=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header="Val Epoch: [{}]".format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    criterion = configure_loss(args)
    num_iter = len(data_loader_val)
    print("number of iterations: ",num_iter)
    for data_iter_step, val_data in enumerate(metric_logger.log_every(data_loader_val, print_freq, header)):
        input_matrix, total_count, target_matrix, embed_target, target_vector = list_to_device(val_data,device=device)
        with torch.no_grad():
            output_embedding, output_2d, output_1d = model(input_matrix, total_count)
        if embed_target is not None:
            embedding_loss = criterion(output_embedding, embed_target)
        else:
            embedding_loss = 0
        if target_matrix is not None:
            #flatten 2d matrix
            output_2d_flatten = torch.flatten(output_2d, start_dim=1,end_dim=-1)
            target_matrix_flatten = torch.flatten(target_matrix, start_dim=1,end_dim=-1)
            output_2d_loss = criterion(output_2d_flatten, target_matrix_flatten)
        else:
            output_2d_loss = 0
        if target_vector is not None:
            output_1d_loss = criterion(output_1d, target_vector)
        else:
            output_1d_loss = 0
        loss = embedding_loss + output_2d_loss + output_1d_loss
        metric_logger.update(loss=to_value(loss))
        metric_logger.update(embedding_loss=to_value(embedding_loss))
        metric_logger.update(output_2d_loss=to_value(output_2d_loss))
        metric_logger.update(output_1d_loss=to_value(output_1d_loss))
        torch.cuda.synchronize() 
        if log_writer is not None and ((data_iter_step + 1) % accum_iter == 0 or data_iter_step==0):
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_val) + epoch) * 1000)
            log_writer.add_scalars('Loss/loss', {'val_loss': to_value(loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/embedding_loss', {'val_loss': to_value(embedding_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/output_2d_loss', {'val_loss': to_value(output_2d_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/output_1d_loss', {'val_loss': to_value(output_1d_loss)}, epoch_1000x)
            if ((data_iter_step+1)//accum_iter)%50==0 or data_iter_step==0:
                #add visualization for your output and input
                new_samples = create_image(input_matrix)
                select_num = min(8,len(new_samples))
                sample_image = torch_to_nparray(new_samples.clone().detach()[:select_num])
                log_writer.add_images('Input_%s'%"val", sample_image, epoch_1000x)
                output_2d_image = convert_gray_rgbimage(output_2d.clone().detach()[:select_num])
                output_2d_image = torch_to_nparray(output_2d_image)
                log_writer.add_images('Output_2d_%s'%"val", output_2d_image, epoch_1000x)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

