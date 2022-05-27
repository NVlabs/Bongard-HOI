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


@register('cnn-baseline')
class CnnBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}):
        super().__init__()
        self.n_way = 2
        self.n_shot = 6
        self.encoder = models.make(encoder, **encoder_args)

        self.mlp = nn.Sequential(nn.Linear(self.encoder.out_dim * 2, self.encoder.out_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(self.encoder.out_dim, self.n_way))

        print('cnn-baseline')

    def forward(self, x_shot, x_query, **kwargs):
        ep_per_batch, n_way, n_shot = x_shot.shape[:-3]
        assert n_shot == self.n_shot and n_way == self.n_way
        c, w, h = x_shot.shape[-3:]
        assert c in [1, 3]  # greyscale

        x_shot_pos, x_shot_neg = x_shot[:, 0], x_shot[:, 1]  # [ep_per_batch, n_shot, 1, w, h]
        x_shot_pos = x_shot_pos.view(ep_per_batch, -1, w, h)  # [ep_per_batch, n_shot, w, h]
        x_shot_neg = x_shot_neg.view(ep_per_batch, -1, w, h)  # [ep_per_batch, n_shot, w, h]

        num_query_samples = x_query.size(1)  # n_query * n_way
        logits = []
        for i in range(num_query_samples):
            x_query_single = x_query[:, i]  # [ep_per_batch, 1, w, h]

            x_pos = torch.cat([x_shot_pos, x_query_single], dim=-3)  # [ep_per_batch, n_shot+1, w, h]
            x_neg = torch.cat([x_shot_neg, x_query_single], dim=-3)  # [ep_per_batch, n_shot+1, w, h]

            x_feat = self.encoder(torch.cat([x_pos, x_neg], dim=0))  # [2 * ep_per_batch, feat_len]
            x_pos_feat, x_neg_feat = x_feat[:ep_per_batch], x_feat[ep_per_batch:]  # [ep_per_batch, feat_len]

            x_feat = torch.cat([x_pos_feat, x_neg_feat], dim=-1)  # [ep_per_batch, feat_len * 2]
            logit = self.mlp(x_feat)  # [ep_per_batch, n_way]

            logits.append(logit.unsqueeze(1))
        logits = torch.stack(logits, dim=1)  # [ep_per_batch, n_query * n_way, n_way]

        return logits
