import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchaudio


def create_data_loader(train_data, valid_data, cfg):
    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
    )
    return train_dataloader, valid_dataloader


def train_fn(model, data_loader, loss_fn, optimiser, scaler, device):
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    print(f"Train_loss: {loss.item()}")


# def check_accuracy(model, loader, loss_fn, device):
#     model.eval()
#     batch_losses = []
#     trace = []
#     trace_y = []
#     valid_losses = []
#     for i, data in enumerate(tqdm(loader)):
#         x, y = data
#         x = x.to(device, dtype=torch.float32)
#         y = y.to(device, dtype=torch.long)
#         with torch.cuda.amp.autocast():
#           model_y = model(x)
#           loss = loss_fn(model_y, y)
#           trace.append(y.cpu().detach().numpy())
#           trace_y.append(model_y.cpu().detach().numpy())
#           batch_losses.append(loss.item())
#     valid_losses.append(batch_losses)
#     trace = np.concatenate(trace)
#     trace_y = np.concatenate(trace_y)
#     accuracy = np.mean(trace_y.argmax(axis=1) == trace)
#     model.train()
#     print(accuracy)


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
        valid_loss += loss.item() * data.size(0)
    valid_loss = valid_loss / len(loader.sampler)
    print(valid_loss)


def get_mel(cfg):
    # get mel spectrograms
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    return mel_spectrogram
