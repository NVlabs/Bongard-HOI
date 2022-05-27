# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import models
from utils.few_shot import make_nk_label
from .models import register


@register('snail')
class SnailFewShot(nn.Module):
    def __init__(self, encoder, encoder_args={}, dynamic_k=True):
        # N-way, K-shot (N=2, K=6)
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)

        self.N = 2
        self.K = 6
        num_channels = self.encoder.out_dim + self.N
        self.dynamic_k = dynamic_k

        num_filters = int(math.ceil(math.log(self.N * self.K + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.tc1 = TCBlock(num_channels, self.N * self.K + 1, 256)
        num_channels += num_filters * 256
        self.attention2 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.tc2 = TCBlock(num_channels, self.N * self.K + 1, 256)
        num_channels += num_filters * 256
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, self.N)

    def forward(self, x_shot, x_query, **kwargs):
        ep_per_batch, n_way, n_shot = list(x_shot.shape[:-3])
        img_shape = list(x_shot.shape[-3:])

        if kwargs.get('eval') is None:
            kwargs['eval'] = False

        if self.dynamic_k and not kwargs['eval']:
            raise NotImplementedError
            # Dynamically zero out up to K-1 training batches.
            x_shot = x_shot.view(ep_per_batch, -1, *img_shape)
            k = np.random.randint(0, n_shot - 1)
            x_shot[:, :k * n_way] = 0.
            x_shot = x_shot.reshape(ep_per_batch, n_way, n_shot, *img_shape)

        # x_shot = x_shot.view(-1, *img_shape)
        # x_last = x_query.view(-1, *img_shape)

        # x_tot = self.encoder(torch.cat([x_shot, x_last], dim=0))  # [bs * (n_way * n_shot + 1), n_feat]
        # x_tot = x_tot.view(ep_per_batch, n_way * n_shot + 1, -1)  # [bs, n_way * n_shot + 1, n_feat]

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
        x_tot = torch.cat((x_shot, x_query), dim=0)
        x_tot = x_tot.view(ep_per_batch, n_way * n_shot + 1, -1)  # [bs, n_way * n_shot + 1, n_feat]
        # x_shot = x_shot.view(*shot_shape, -1)
        # x_query = x_query.view(*query_shape, -1)

        labels_support = make_nk_label(n_way, n_shot, ep_per_batch)  # [bs * n_way * n_shot]
        labels_support = labels_support.cuda().unsqueeze(-1) # [bs * n_way * n_shot, 1]
        labels_support_onehot = torch.FloatTensor(labels_support.size(0), 2).cuda()

        labels_support_onehot.zero_()
        labels_support_onehot.scatter_(1, labels_support, 1)  # [bs * n_way * n_shot, n_way]
        labels_support_onehot = labels_support_onehot.view(ep_per_batch, -1, n_way)
        labels_query_zero = torch.Tensor(np.zeros((ep_per_batch, 1, n_way))).cuda()
        labels = torch.cat([labels_support_onehot, labels_query_zero], dim=1)  # [bs, n_way * n_shot + 1, n_way]

        x = torch.cat((x_tot, labels), dim=-1)  # [bs, n_way * n_shot + 1, n_feat + n_way]
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)  # [bs, n_way * n_shot + 1, n_way]
        return x[:, -1, :]  # [bs, n_way]


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation] # TODO: make this correct for different strides/padding


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg) # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i+1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length, 2))))])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).cuda()

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = torch.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2) # shape: (N, T, in_channels + value_size)


def labels_to_one_hot(labels):
    labels = labels.numpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs


def batch_for_few_shot(opt, x, y):
    seq_size = opt.num_cls * opt.num_samples + 1
    one_hots = []
    last_targets = []
    for i in range(opt.batch_size):
        one_hot, idxs = labels_to_one_hot(y[i * seq_size: (i + 1) * seq_size])
        one_hots.append(one_hot)
        last_targets.append(idxs[-1])
    last_targets = Variable(torch.Tensor(last_targets).long())
    one_hots = [torch.Tensor(temp) for temp in one_hots]
    y = torch.cat(one_hots, dim=0)
    x, y = Variable(x), Variable(y)
    x, y = x.cuda(), y.cuda()
    last_targets = last_targets.cuda()
    return x, y, last_targets
