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


@register('wren')
class WReN(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='original'):
        super().__init__()
        self.n_way = 2
        self.n_shot = 6
        self.encoder = models.make(encoder, **encoder_args)

        self.feat_len = self.encoder.out_dim
        self.method = method

        self.mlp_g = nn.Sequential(nn.Linear(self.feat_len * 2, self.feat_len),
                                   nn.ReLU(),
                                   nn.Linear(self.feat_len, self.feat_len),
                                   nn.ReLU())

        if self.method == 'original':
            self.mlp_f = nn.Sequential(nn.Linear(self.feat_len, self.feat_len),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(self.feat_len, 1))
        elif self.method == 'modified':
            self.mlp_f = nn.Sequential(nn.Linear(self.feat_len * 2, self.feat_len),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(self.feat_len, 2))
        else:
            raise  Exception('method should be in [original, modified]')

        print('wren, {}'.format(method))

    def forward(self, x_shot, x_query, **kwargs):
        ep_per_batch, n_way, n_shot = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        assert n_shot == self.n_shot and n_way == self.n_way
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(ep_per_batch, n_way, n_shot, -1)  # [ep_per_batch, n_way, n_shot, feat_len]
        x_shot_pos, x_shot_neg = x_shot[:, 0, :, :], x_shot[:, 1, :, :]  # [ep_per_batch, n_way, n_shot, feat_len]
        x_query = x_query.view(*query_shape, -1)  # [ep_per_batch, n_way * n_query, feat_len]

        num_query_samples = x_query.size(1)  # n_way * n_query
        logits = []
        for i in range(num_query_samples):
            x_query_single = x_query[:, i:i + 1]  # [ep_per_batch, 1, feat_len]

            # positive
            emb_pos = torch.cat([x_shot_pos, x_query_single], dim=1)  # [ep_per_batch, n_shot + 1, feat_len]
            emb_pairs_pos = self.group_embeddings_batch(emb_pos)  # [ep_per_batch, num_pairs, 2 * feat_len]

            # negative
            emb_neg = torch.cat([x_shot_neg, x_query_single], dim=1)  # [ep_per_batch, n_shot + 1, feat_len]
            emb_pairs_neg = self.group_embeddings_batch(emb_neg)  # [ep_per_batch, num_pairs, 2 * feat_len]

            # relation
            emb_pairs = torch.cat([emb_pairs_pos, emb_pairs_neg], dim=1)  # [ep_per_batch, 2 * num_pairs, 2 * feat_len]
            emb_rels = self.mlp_g(emb_pairs.view(-1, self.feat_len * 2))  # [ep_per_batch * 2 * num_pairs, feat_len]
            emb_rels = emb_rels.view(ep_per_batch * n_way, -1, self.feat_len)  # [ep_per_batch * 2, num_pairs, feat_len]
            if self.method == 'original':
                logit = self.mlp_f(torch.sum(emb_rels, dim=1))  # [ep_per_batch * 2, 1]
            elif self.method == 'modified':
                emb_sums = torch.sum(emb_rels, dim=1).view(ep_per_batch, -1)  # [ep_per_batch, 2 * feat_len]
                logit = self.mlp_f(emb_sums)  # [ep_per_batch, 2]
            else:
                logit = None

            logit = logit.view(-1, n_way)  # [ep_per_batch, n_way]
            logits.append(logit.unsqueeze(1))

        logits = torch.stack(logits, dim=1)  # [ep_per_batch, n_query * n_way, n_way]

        return logits

    def group_embeddings_batch(self, embeddings):
        num_emb = self.n_shot + 1
        embeddings = embeddings.view(-1, num_emb, self.feat_len)

        emb_pairs = torch.cat(
            [embeddings.unsqueeze(1).expand(-1, num_emb, -1, -1),
             embeddings.unsqueeze(2).expand(-1, -1, num_emb, -1)],
            dim=-1).view(-1, num_emb ** 2, 2 * self.feat_len)

        use_indices = torch.tensor([i * num_emb + j for i in range(num_emb)
                                    for j in range(num_emb) if i != j])
        emb_pairs = emb_pairs[:, use_indices]  # [bs, num_pairs, 2 * feat_len]

        return emb_pairs
