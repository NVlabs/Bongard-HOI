# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn

import models
import utils
from .models import register

@register('rn_encoder')
class RelationalNetworkEncoder(nn.Module):
    def __init__(self, encoder, encoder_args={}):
        super(RelationalNetworkEncoder, self).__init__()

        # image encoder
        encoder = models.make(encoder, **encoder_args)
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.proj = nn.Conv2d(encoder.out_dim, encoder.out_dim // 2, kernel_size=1)

        # relational encoding
        self.g_mlp = nn.Sequential(
            nn.Linear(encoder.out_dim, encoder.out_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder.out_dim // 2, encoder.out_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder.out_dim // 2, encoder.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out_dim = encoder.out_dim // 2

    def forward(self, im):
        # BxCxHxW
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.proj(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # relational encoding
        b, c, h, w = x.shape
        hw = h * w
        # bxhwxc
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)

        # adapted from https://github.com/kimhc6028/relational-networks/blob/master/model.py
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b * 1 * hw * c)
        x_i = x_i.repeat(1, hw, 1, 1)  # (b * hw * hw  * c)
        x_j = torch.unsqueeze(x_flat, 2)  # (b * hw * 1 * c)
        x_j = x_j.repeat(1, 1, hw, 1)  # (b * hw * hw  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * hw * hw  * 2c)

        # reshape for passing through network
        x_full = x_full.view(b * hw * hw, -1)  # (b*hw*hw)*2c

        x_g = self.g_mlp(x_full)

        # reshape again and sum
        x_g = x_g.view(b, hw * hw, -1)

        x_g = x_g.sum(1)

        return x_g

if __name__ == '__main__':
    im = torch.rand((8, 3, 128, 128))

    model = RelationalNetworkEncoder(encoder='resnet50')
    x = model(im)
    print(x.shape)
