import torch
from ops.distribute_utils import is_main_process
from pathlib import Path
import os

def load_model(resume_path,args,model_without_ddp, optimizer, loss_scaler):
    """
    Load the model from the checkpoint
    Args:
        resume_path: the path to the checkpoint
        model_without_ddp: the model
        optimizer: the optimizer
        loss_scaler: the loss scaler
    """
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if resume_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume_path, map_location='cpu')
        msg=model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("model resume message:{}".format(msg))
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

    

def save_checkpoint(output_dir, args,epoch, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)

def save_model2path(model_path,args,epoch, model_without_ddp, optimizer, loss_scaler):
    to_save={
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': args,
    }
    save_on_master(to_save, model_path)

    