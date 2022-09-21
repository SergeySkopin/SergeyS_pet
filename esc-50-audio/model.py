import torch.nn as nn
from torchvision.models import resnet18


def resnet_model(output_classes):
    # Возьмем претренненую модель resnet, изменим первый сверточный слой и полносвязный слой под наши условия
    resnet_model = resnet18(pretrained=True)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), bias=False)
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, output_classes),
    )
    return resnet_model
