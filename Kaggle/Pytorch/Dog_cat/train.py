import os
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CatDog
from efficientnet_pytorch import EfficientNet


def check_accuracy(
    loader, model, loss_fn, input_shape=None, toggle_eval=True, print_accuracy=True
):
    if toggle_eval:
        model.eval()
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0

    y_preds = []
    y_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if input_shape:
                x = x.reshape(x.shape[0], *input_shape)
            scores = model(x)
            predictions = torch.sigmoid(scores) > 0.5
            y_preds.append(
                torch.clip(torch.sigmoid(scores), 0.005, 0.995).cpu().numpy()
            )
            y_true.append(y.cpu().numpy())
            num_correct += (predictions.squeeze(1) == y).sum()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples

    if toggle_eval:
        model.train()

    if print_accuracy:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(log_loss(np.concatenate(y_true, axis=0), np.concatenate(y_preds, axis=0)))

    return accuracy


def save_vectors(model, loader, output_size=(1, 1), file="train_file"):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"data_features/X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"data_features/y_{file}.npy", np.concatenate(labels, axis=0))
    model.train()


def train(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.to(config.DEVICE).unsqueeze(1).float()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


def main():
    model = EfficientNet.from_pretrained("efficientnet")
    model._fc = nn.Linear(2560, 1)
    train_dataset = CatDog(root="data/train/", transform=config.basic_transform)
    test_dataset = CatDog(root="data/test/", transform=config.basic_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = model.to(config.DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, loss_fn, optimizer, scaler)
        check_accuracy(train_loader, model, loss_fn)

    save_vectors(model, train_loader, output_size=(1, 1), file="train_file")
    save_vectors(model, test_loader, output_size=(1, 1), file="test_file")


if __name__ == "__main__":
    main()
