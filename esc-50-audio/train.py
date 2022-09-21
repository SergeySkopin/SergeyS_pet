from venv import create
import torch
from torch import nn
from torch.utils.data import random_split
from preprocess import ESC50
from model import resnet_model
from utils import create_data_loader, train_fn, check_accuracy, get_mel
from tqdm import tqdm
from omegaconf import DictConfig
import os
import hydra

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    model, train_loader, valid_loader, loss_fn, optimiser, scaler, device, epochs
):
    # check_accuracy(model, valid_loader, device)
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_fn(model, train_loader, loss_fn, optimiser, scaler, device)
        check_accuracy(model, valid_loader, loss_fn, device)


@hydra.main(config_path="configs", config_name="config")
def run_train(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    ann_file = os.path.join(orig_cwd, "ESC-50-master/meta/esc50.csv")
    audio_dir = os.path.join(orig_cwd, "ESC-50-master/audio")
    esc_data = ESC50(
        ann_file, audio_dir, get_mel(cfg), cfg.SAMPLE_RATE, cfg.NUM_SAMPLES, DEVICE
    )
    # create train dataloader and model
    train_dataset, valid_dataset = torch.utils.data.random_split(
        esc_data, [cfg.TRAIN_SIZE, cfg.VALID_SIZE]
    )
    print(f"There are {len(train_dataset)} samples in the dataset.")
    train_dataloader, valid_dataloader = create_data_loader(
        train_dataset, valid_dataset, cfg
    )
    cnn = resnet_model(50).to(DEVICE)
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=cfg.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    # train model
    train(
        cnn,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimiser,
        scaler,
        DEVICE,
        cfg.EPOCHS,
    )
    # save model
    torch.save(cnn.state_dict(), "train.pth")
    print("Saving model")


if __name__ == "__main__":
    run_train()
