import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchaudio
from clearml import Task

task = Task.init(project_name='ESC-50', task_name = 'esc-50-train')

def create_data_loader(train_data, valid_data, cfg):
    train_dataloader = DataLoader(train_data, batch_size = cfg.BATCH_SIZE, num_workers = cfg.NUM_WORKERS, pin_memory = cfg.PIN_MEMORY, shuffle = True)
    valid_dataloader = DataLoader(valid_data, batch_size = cfg.BATCH_SIZE, num_workers = cfg.NUM_WORKERS, pin_memory = cfg.PIN_MEMORY, shuffle = True)
    return train_dataloader , valid_dataloader


def train_fn(model, data_loader, loss_fn, optimiser, scaler, device):
    for input, target in tqdm(data_loader):
        input, target  = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    print(f"Train_loss: {loss.item()}")

def check_accuracy(model, loader, loss_fn, device):
    model.eval()
    valid_loss = 0.0
    for data, target in loader:
        # move tensors to GPU if CUDA is available
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)
    valid_loss = valid_loss/len(loader.sampler)
    model.train()
    print(valid_loss)

def get_mel(cfg):
    # get mel spectrograms
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= cfg.SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    return mel_spectrogram
