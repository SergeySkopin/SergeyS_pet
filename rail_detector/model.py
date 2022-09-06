import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        need_activation=True,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # ?????
        self.activation = need_activation

    def forward(self, x):
        fd = self.conv(x)
        fd = self.batch_norm(fd)
        if self.activation:
            fd = self.relu(fd)
        return fd


class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBlock(in_channel=3, out_channel=64, kernel_size=3, stride=2),
            ConvBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBlock(in_channel=64, out_channel=64, kernel_size=3, stride=2),
            ConvBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1),
            ConvBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBlock(in_chan=64, out_chan=128, kernel_size=3, stride=2),
            ConvBlock(in_chan=128, out_chan=128, kernel_size=3, stride=1),
            ConvBlock(in_chan=128, out_chan=128, kernel_size=3, stride=1),
        )

    def forward(self, x):
        fd = self.S1(x)
        fd = self.S2(fd)
        fd = self.S3(fd)

        return fd


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBlock(in_channel=3, out_channel=16, kernel_size=3, stride=2)
        self.left_side = nn.Sequential(
            ConvBlock(in_channel=16, out_channel=8, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channel=8, out_channel=16, kernel_size=3, stride=2),
        )
        self.right_side = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_connect = ConvBlock(32, 16)

    def forward(self, x):
        fd = self.conv(x)
        fd_left = self.left_side(fd)
        fd_right = self.right(fd)
        fd = torch.cat([fd_left, fd_right], dim=1)
        fd = self.conv_connect(fd)
        return fd


class ContextEmbeddingBlock(nn.Module):
    def __init__(self):
        super(ContextEmbeddingBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(128)
        self.conv_gap = ConvBlock(
            in_channel=128, out_channel=128, kernel_size=1, stride=1, padding=0
        )
        self.conv_last = ConvBlock(
            in_channel=128, out_channel=128, kernel_size=3, stride=1, activation=False
        )

    def forward(self, x):
        fd = torch.mean(x, dim=(2, 3), keepdim=True)
        fd = self.bn(fd)
        fd = self.conv_gap(fd)
        fd = fd + x
        fd = self.conv_last(fd)
        return fd
