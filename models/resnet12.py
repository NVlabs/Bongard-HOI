# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch.nn as nn
import torch
from .models import register


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels, out_dim=512, inplanes=1):
        super().__init__()

        self.inplanes = inplanes  # 1 with 'L' or 3 with 'RGB' or 7 with non-meta-learning

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.layer5 = self._make_layer(channels[4])
        if out_dim != 512:
            self.fc = nn.Linear(channels[-1], out_dim)

        self.out_dim = out_dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        if self.out_dim != 512:
            x = self.fc(x)
        return x


@register('resnet12')
def resnet12(out_dim=512, inplanes=1, reduce_factor=2):
    assert 1 <= reduce_factor <= 32
    featmaps = [32 * 2 // reduce_factor, 64 * 2 // reduce_factor, 128 * 2 // reduce_factor,
                min(256 * 2 // reduce_factor, 256), min(512 * 2 // reduce_factor, 512)]
    out_dim = min(out_dim * 2 // reduce_factor, 128)
    print('resnet12 featmaps: {}, and out_dim: {}'.format(featmaps, out_dim))
    return ResNet12(featmaps, out_dim, inplanes)  # when image_res=512
    # return ResNet12([64, 128, 256, 512], out_dim)  # when image_res=256


@register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])
