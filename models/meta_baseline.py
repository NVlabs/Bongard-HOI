# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        print('encoder: {}'.format(encoder))
        print('encoder_args: {}'.format(encoder_args))
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, **kwargs):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        if 'shot_boxes' in kwargs:
            assert 'query_boxes' in kwargs
            assert 'shot_boxes_dim' in kwargs
            assert 'query_boxes_dim' in kwargs
            x_shot = self.encoder(x_shot, kwargs['shot_boxes'], kwargs['shot_boxes_dim'])
        else:
            x_shot = x_shot.view(-1, *img_shape)
            x_shot = self.encoder(x_shot)

        if 'query_boxes' in kwargs:
            assert 'shot_boxes' in kwargs
            assert 'shot_boxes_dim' in kwargs
            assert 'query_boxes_dim' in kwargs
            x_query = self.encoder(x_query, kwargs['query_boxes'], kwargs['query_boxes_dim'])
        else:
            x_query = x_query.view(-1, *img_shape)
            x_query = self.encoder(x_query)

        # x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        # x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
            x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, way * query, feature_len]
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)  # [ep_per_batch, way * query, way]
        return logits
