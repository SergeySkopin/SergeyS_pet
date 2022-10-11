from os import fdatasync
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
            kernel_size = kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.activation = need_activation

    def forward(self,x):
        fd = self.conv(x)
        fd = self.batch_norm(fd)
        if self.activation:
            fd = self.relu(fd)
        return fd

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch,self).__init__()
        self.S1 = nn.Sequential(
            ConvBlock(in_channel=3,out_channel=64, kernel_size=3, stride=2),
            ConvBlock(in_channel=64,out_channel=64, kernel_size=3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBlock(in_channel=64,out_channel=64, kernel_size=3, stride=2),
            ConvBlock(in_channel=64,out_channel=64, kernel_size=3, stride=1),
            ConvBlock(in_channel=64,out_channel=64, kernel_size=3, stride=1),
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
        super(StemBlock,self).__init__()
        self.conv = ConvBlock(in_channel=3,out_channel=16, kernel_size=3, stride=2)
        self.left_side = nn.Sequential(
            ConvBlock(in_channel=16,out_channel=8, kernel_size=1,stride=1,padding=0),
            ConvBlock(in_channel=8, out_channel=16, kernel_size=3, stride=2),
        )
        self.right_side = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_connect = ConvBlock(32,16)

    def forward(self,x):
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
        self.conv_gap = ConvBlock(in_channel=128, out_channel=128, kernel_size=1, stride=1, padding=0)
        self.conv_last = ConvBlock(in_channel=128, out_channel=128, kernel_size=3, stride=1, need_activation=False)

    def forward(self, x):
        fd = torch.mean(x, dim=(2, 3), keepdim=True)
        fd = self.bn(fd)
        fd = self.conv_gap(fd)
        fd = fd + x
        fd = self.conv_last(fd)
        return fd

class GELayerS1(nn.Module):
    def __init__(self,in_channel,out_channel,exp_ratio = 6):
        super(GELayerS1, self).__init__()
        b_channel = in_channel * exp_ratio
        self.conv1 = ConvBlock(
            in_channel = in_channel,
            out_channel = out_channel,
            kernel_size = 3,
            stride = 1
        )
        self.depthwiseconv = ConvBlock(
            in_channel = in_channel,
            out_channel = b_channel,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            groups = in_channel,
            bias = False,
            need_activation = False
        )
        self.conv2 = ConvBlock(
            in_channel = b_channel,
            out_channel = out_channel,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
            need_activation = False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fd = self.conv1(x)
        fd = self.depthwiseconv(fd)
        fd = self.conv2(fd)
        fd = fd + x
        fd = self.relu(fd)
        return fd

class GELayerS2(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=6):
        super(GELayerS2, self).__init__()
        b_channel = in_channel * exp_ratio
        self.conv1 = ConvBlock(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1)
        self.depthwiseconv1 = ConvBlock(
            in_channel=in_channel,
            out_channel=b_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channel,
            bias=False,
            need_activation=False,
        )
        self.depthwiseconv2 = ConvBlock(
            in_channel=b_channel,
            out_channel=b_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=b_channel,
            bias=False,
            need_activation=False,
        )
        self.conv2 = ConvBlock(
            in_channel=b_channel,
            out_channel=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            need_activation=False,
        )
        self.shortcut = nn.Sequential(
            ConvBlock(
                in_channel=in_channel,
                out_channel=in_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channel,
                bias=False,
                need_activation=False,
            ),
            ConvBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                need_activation=False,
            ),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fd = self.conv1(x)
        fd = self.depthwiseconv1(fd)
        fd = self.depthwiseconv2(fd)
        fd = self.conv2(fd)
        shortcut = self.shortcut(x)
        fd = fd + shortcut
        fd = self.relu(fd)
        return fd

class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = ContextEmbeddingBlock()

    def forward(self, x):
        fd2 = self.S1S2(x)
        fd3 = self.S3(fd2)
        fd4 = self.S4(fd3)
        fd5_4 = self.S5_4(fd4)
        fd5_5 = self.S5_5(fd5_4)
        return fd2, fd3, fd4, fd5_4, fd5_5

class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self):
        super(BilateralGuidedAggregationLayer, self).__init__()

        # Detail Branch
        self.left1 = nn.Sequential(
            ConvBlock(
                in_channel=128,
                out_channel=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
                need_activation=False,
            ),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            ConvBlock(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                need_activation=False,
            ),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

        # Semantic Branch
        self.right1 = ConvBlock(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            need_activation=False,
        )
        self.right2 = nn.Sequential(
            ConvBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
                need_activation=False,
            ),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.conv = ConvBlock(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            need_activation=False,
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(right1, size=dsize, mode="bilinear", align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode="bilinear", align_corners=True)
        out = self.conv(left + right)
        return out

class SegmentHead(nn.Module):
    def __init__(self, in_channel, seghead_ratio, n_classes, dropout_rate=0.1):
        super(SegmentHead, self).__init__()
        b_channel = in_channel * seghead_ratio
        self.conv = ConvBlock(in_channel, b_channel, 3, stride=1)
        self.drop_out = nn.Dropout(dropout_rate)
        self.conv_out = nn.Conv2d(b_channel, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, size=None):
        fd = self.conv(x)
        fd = self.drop_out(fd)
        fd = self.conv_out(fd)
        if size:
            fd = F.interpolate(fd, size=size, mode="bilinear", align_corners=True)

        return fd

class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, seghead_ratio=6):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BilateralGuidedAggregationLayer()

        self.head = SegmentHead(128, seghead_ratio, n_classes)
        if self.training:
            self.aux2 = SegmentHead(16, seghead_ratio, n_classes)
            self.aux3 = SegmentHead(32, seghead_ratio, n_classes)
            self.aux4 = SegmentHead(64, seghead_ratio, n_classes)
            self.aux5_4 = SegmentHead(128, seghead_ratio, n_classes)

            self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        fd_d = self.detail(x)
        fd2, fd3, fd4, fd5_4, fd_s = self.segment(x)

        feat_head = self.bga(fd_d, fd_s)

        logits = self.head(feat_head, size)
        if self.training:
            logits_aux2 = self.aux2(fd2, size)
            logits_aux3 = self.aux3(fd3, size)
            logits_aux4 = self.aux4(fd4, size)
            logits_aux5_4 = self.aux5_4(fd5_4, size)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        else:
            return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, "last_bn") and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)