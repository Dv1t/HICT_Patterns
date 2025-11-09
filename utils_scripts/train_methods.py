import torch
from tqdm import tqdm
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

def run_epoch(model, phase, dataloader, criterion, optimizer, device):
  if phase == 'train':
      model.train()
  else:
      model.eval()

  running_loss = 0.0
  running_corrects = 0
  y_test = []
  y_pred = []
  all_elems_count = 0
  cur_tqdm = tqdm(dataloader)
  for inputs, labels in cur_tqdm:
    bz = inputs.shape[0]
    all_elems_count += bz
    
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True, dtype=torch.float)

    outputs = model(inputs)
    outputs = outputs.resize(outputs.shape[0])
    loss = criterion(outputs, labels)
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    sigmoid  = nn.Sigmoid()
    preds = torch.round(sigmoid(outputs))
    y_test.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())

    running_loss += loss.item() * bz
    corrects_cnt = torch.sum(preds == labels.detach())
    running_corrects += corrects_cnt
    show_dict = {'Loss': f'{loss.item():.6f}',
                'Corrects': f'{corrects_cnt.item()}/{bz}',
                'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
    cur_tqdm.set_postfix(show_dict)

  y_test = np.array(y_test)
  y_pred = np.array(y_pred)
  tp = np.sum((y_test == y_pred) & (y_pred==1))
  tn = np.sum((y_test == y_pred) & (y_pred==0))
  fp = np.sum((y_test != y_pred) & (y_pred==1))
  fn = np.sum((y_test != y_pred) & (y_pred==0))
  print(f'For {phase} metrics are:')
  print('tp', tp)
  print('tn', tn)
  print('fp', fp)
  print('fn', fn)
  print("Calculating metrics...")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  return epoch_loss, epoch_acc

def test_epoch(model, dataloader, phase_name, criterion, optimizer, device):
    with torch.inference_mode():
      return run_epoch(model, phase_name, dataloader, criterion, optimizer, device)

def train_epoch(model, dataloader, criterion, optimizer, device):
    return run_epoch(model, 'train', dataloader, criterion, optimizer, device)

def train_model(dataloaders, model, log_folder, criterion, optimizer, device, num_epochs=20, phases= ['train'], model_name='six_mat'):
  print(f"Training model with params:")
  print(f"Optim: {optimizer}")
  print(f"Criterion: {criterion}")

  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}

  for epoch in range(1, num_epochs + 1):
    start_time = time.time()

    print("=" * 100)
    print(f'Epoch {epoch}/{num_epochs}')
    print('-' * 10)

    for phase in phases:
        print("--- Cur phase:", phase)
        if phase == 'train':
          epoch_loss, epoch_acc = train_epoch(model, dataloaders[phase], criterion, optimizer, device)
        elif phase == 'test':
          epoch_loss, epoch_acc = test_epoch(model, dataloaders[phase], phase, criterion, optimizer, device)
        elif epoch % 10 == 0:
          epoch_loss, epoch_acc = test_epoch(model, dataloaders[phase], phase, criterion, optimizer, device)
        else:
           continue
        saved_epoch_losses[phase].append(epoch_loss)
        saved_epoch_accuracies[phase].append(epoch_acc)
        print(f'{phase} loss: {epoch_loss:.6f}, '
                f'acc: {epoch_acc:.6f}')
    os.makedirs(log_folder, exist_ok=True)
    if epoch % 10 == 0:
       torch.save(model.state_dict(), f'weights_{model_name}/torch_ensemble_{model_name}_48_diag_{epoch}.pt')
    if epoch == num_epochs:
        plt.title(f'Losses during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_losses['train'], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel(criterion.__class__.__name__)
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/loss_graph_epoch{epoch + 1}.png')
        plt.show()
        plt.close('all')

        plt.title(f'Accuracies during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_accuracies['train'], label='Train Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/acc_graph_epoch{epoch + 1}.png')
        plt.show()
        plt.close('all')

        plt.title(f'Losses during testing. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_losses['test'], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel(criterion.__class__.__name__)
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/loss_graph_epoch{epoch + 1}_test.png')
        plt.show()
        plt.close('all')

        plt.title(f'Accuracies during testing. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_accuracies['test'], label='Test Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/acc_graph_epoch{epoch + 1}_test.png')
        plt.show()
        plt.close('all')

        if 'validate' in phases:
          plt.title(f'Losses during validation. Epoch {epoch}/{num_epochs}.')
          plt.plot(range(1, len(saved_epoch_losses['validate']) + 1), saved_epoch_losses['validate'], label='Loss')
          plt.xlabel('Epochs')
          plt.ylabel(criterion.__class__.__name__)
          plt.legend(loc="upper left")
          plt.savefig(f'{log_folder}/loss_graph_epoch{epoch + 1}_validation.png')
          plt.show()
          plt.close('all')

          plt.title(f'Accuracies during validation. Epoch {epoch}/{num_epochs}.')
          plt.plot(range(1, len(saved_epoch_accuracies['validate']) + 1), saved_epoch_accuracies['validate'], label='Acc')
          plt.xlabel('Epochs')
          plt.ylabel('Accuracy')
          plt.legend(loc="upper left")
          plt.savefig(f'{log_folder}/acc_graph_epoch{epoch + 1}_validation.png')
          plt.show()
          plt.close('all')
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print("-" * 10)
    print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies

def run_test_epoch(models, phase, dataloader, criterion, device, log_folder):
  running_loss = 0.0
  running_corrects = 0
  y_test = []
  y_pred = []
  all_elems_count = 0
  cur_tqdm = tqdm(dataloader)
  all_inputs = []
  for inputs, labels in cur_tqdm:
    bz = inputs.shape[0]
    all_elems_count += bz
    
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True, dtype=torch.float)
    sigmoid  = nn.Sigmoid()
    outputs_by_res = [torch.round(sigmoid(model(inputs[:, i]))) for model, i in zip(models,  range(inputs.shape[1]))]
    #class_predictions = [torch.argmax(output, dim=1) for output in outputs_by_res]
    stacked_predictions = torch.stack(outputs_by_res, dim=0)
    majority_vote_predictions, _ = torch.mode(stacked_predictions, dim=0)
    preds = majority_vote_predictions.resize(majority_vote_predictions.shape[0]).to(device, non_blocking=True, dtype=torch.float)

    y_test.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    all_inputs.extend(inputs.detach().cpu().numpy())

    corrects_cnt = torch.sum(preds == labels.detach())
    running_corrects += corrects_cnt
    show_dict = {
                'Corrects': f'{corrects_cnt.item()}/{bz}',
                'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
    cur_tqdm.set_postfix(show_dict)

  y_test = np.array(y_test)
  y_pred = np.array(y_pred)

  tp = np.sum((y_test == y_pred) & (y_pred==1))
  tn = np.sum((y_test == y_pred) & (y_pred==0))
  fp = np.sum((y_test != y_pred) & (y_pred==1))
  fn = np.sum((y_test != y_pred) & (y_pred==0))
  all_inputs = np.array(all_inputs)
  fp_inputs = all_inputs[(y_test != y_pred) & (y_pred==1)]
  fn_inputs = all_inputs[(y_test != y_pred) & (y_pred==0)]

  os.makedirs(log_folder, exist_ok=True)
  for i, fp_input in enumerate(fp_inputs):
        fig, axs = plt.subplots(figsize=(12,12), ncols=2)
        axs[0] = fig.add_subplot(111)
        im = axs[0].matshow(fp_input[1][0], cmap='bwr')
        axs[1] = fig.add_subplot(111)
        im = axs[1].matshow(fp_input[1][1], cmap='bwr')
        plt.savefig(f'{log_folder}/fp_{i}.png')
        plt.close()
  for i, fn_input in enumerate(fn_inputs):
        fig, axs = plt.subplots(figsize=(12,12), ncols=2)
        axs[0] = fig.add_subplot(111)
        im = axs[0].matshow(fn_input[1][0], cmap='bwr')
        axs[1] = fig.add_subplot(111)
        im = axs[1].matshow(fn_input[1][1], cmap='bwr')
        plt.savefig(f'{log_folder}/fn_{i}.png')
        plt.close()

  print(f'For {phase} metrics are:')
  print('tp', tp)
  print('tn', tn)
  print('fp', fp)
  print('fn', fn)
  print("Calculating metrics...")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  return epoch_loss, epoch_acc

def test_model(dataloaders, model, log_folder, criterion, device, num_epochs=1, phases= ['test']):
  print(f"Training model with params:")
  print(f"Criterion: {criterion}")

  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}

  for epoch in range(1, num_epochs + 1):
    start_time = time.time()

    print("=" * 100)
    print(f'Epoch {epoch}/{num_epochs}')
    print('-' * 10)

    for phase in phases:
        print("--- Cur phase:", phase)
        epoch_loss, epoch_acc = run_test_epoch(model, phase, dataloaders[phase], criterion, device, log_folder)

        saved_epoch_losses[phase].append(epoch_loss)
        saved_epoch_accuracies[phase].append(epoch_acc)
        print(f'{phase} loss: {epoch_loss:.6f}, '
                f'acc: {epoch_acc:.6f}')
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print("-" * 10)
    print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies