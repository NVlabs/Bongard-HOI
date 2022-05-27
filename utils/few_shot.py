# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch


def split_shot_query(data, way, shot, query, ep_per_batch):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    assert data.dim() == 6  # [ep_per_batch, way, shot+query, C, image_size, image_size]
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def get_query_label(label, n_way, n_shot, n_query, ep_per_batch=1):
    label = label.view(ep_per_batch, n_way, n_shot + n_query)
    assert label.dim() == 3  # [ep_per_batch, way, shot+query]
    label_query = label.split([n_shot, n_query], dim=2)[1]
    label_query = label_query.contiguous().view(-1)
    return label_query


def make_nk_label(n_way, n_query, ep_per_batch=1):
    label = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1)
    label = label.repeat(ep_per_batch)  # ep_per_batch * n_way * n_query
    return label
