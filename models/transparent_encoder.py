# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler

import models
import utils
from .models import register


# CNN backbone then superpixels as patches
@register('transparent_superpixel_encoder')
class TransparentSuperpixelEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        super().__init__()

        # image encoder
        encoder = models.make(encoder)
        self.encoder = encoder
        self.out_dim = encoder.out_dim

    def forward(self, im, boxes=None, boxes_dim=None):
        img_shape = im.shape
        im = im.view(-1, *img_shape[-3:])
        num_im = im.size(0)

        feats = self.encoder(im)
        return feats