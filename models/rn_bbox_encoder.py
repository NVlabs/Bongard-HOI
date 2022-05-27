# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn

# from torchvision.ops import roi_align
from detectron2.modeling.poolers import ROIPooler

import models
import utils
from .models import register

@register('rn_bbox_encoder')
class RelationalBBoxNetworkEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        super(RelationalBBoxNetworkEncoder, self).__init__()

        # image encoder
        encoder = models.make(encoder)
        self.encoder = encoder
        self.proj = nn.Conv2d(encoder.out_dim, encoder.out_dim // 2, kernel_size=1)

        # ROI Pooler
        self.roi_pooler = ROIPooler(
           output_size=7,
           scales=(1/32,), # TODO: this works for resnet50
           sampling_ratio=0,
           pooler_type='ROIAlignV2',
        )
        self.roi_processor = nn.Sequential(
            nn.Conv2d(encoder.out_dim // 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*7*7, 1024),
            nn.ReLU()
        )
        self.roi_processor_ln = nn.LayerNorm(1024)
        rn_in_planes = 1024 * 2

        # bbox coord encoding
        self.roi_processor_box = nn.Linear(4, 256)
        self.roi_processor_box_ln = nn.LayerNorm(256)
        rn_in_planes = (1024 + 256) * 2

        # relational encoding
        self.g_mlp = nn.Sequential(
            nn.Linear(rn_in_planes, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out_dim = rn_in_planes // 2

    def process_single_image_rois(self, roi_feats):
        # relational encoding
        M, C = roi_feats.shape
        b = 1
        # 1xMxC
        x_flat = roi_feats.unsqueeze(0)

        # adapted from https://github.com/kimhc6028/relational-networks/blob/master/model.py
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b * 1 * M * c)
        x_i = x_i.repeat(1, M, 1, 1)  # (b * M * M  * c)
        x_j = torch.unsqueeze(x_flat, 2)  # (b * M * 1 * c)
        x_j = x_j.repeat(1, 1, M, 1)  # (b * M * M  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * M * M  * 2c)

        # reshape for passing through network
        x_full = x_full.view(b * M * M, -1)  # (b*M*M)*2c

        x_g = self.g_mlp(x_full)

        # reshape again and sum
        x_g = x_g.view(b, M * M, -1)

        x_g = x_g.sum(1)
        return x_g

    def forward(self, im, boxes, boxes_dim=None, info_nce=False):
        # assert im.shape[0] == len(boxes), 'im: {} vs boxes: {}'.format(im.shape[0], len(boxes))
        img_shape = im.shape
        im = im.view(-1, *img_shape[-3:])
        # assert im.shape[0] == boxes_dim.shape[0], '{} vs {}'.format(im.shape, boxes_dim.shape)
        if boxes_dim is not None:
            boxes_dim_shape = boxes_dim.shape
            boxes_dim = boxes_dim.view(-1, *boxes_dim_shape[-1:])

        # BxCxHxW
        if info_nce:
            x, attn_v = self.encoder(im, info_nce)
        else:
            x = self.encoder(im)

        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)

        # RoI pooling/align
        all_boxes = []
        for boxes_i in boxes:
            all_boxes.extend(boxes_i)
        num_boxes = [boxes_i.tensor.shape[0] for boxes_i in all_boxes]

        roi_feats = self.roi_pooler([x], all_boxes)
        roi_feats = self.roi_processor(roi_feats)
        roi_feats = self.roi_processor_ln(roi_feats)
        # Add bbox pos features
        bbox_tensor = torch.cat([box.tensor for box in all_boxes]).to(roi_feats)
        # bbox coord normalization
        bbox_tensor[:, 0] = bbox_tensor[:, 0] / im.shape[3]
        bbox_tensor[:, 1] = bbox_tensor[:, 1] / im.shape[2]
        bbox_tensor[:, 2] = bbox_tensor[:, 2] / im.shape[3]
        bbox_tensor[:, 3] = bbox_tensor[:, 3] / im.shape[2]
        bbox_tensor = bbox_tensor * 2 - 1
        roi_box_feats = self.roi_processor_box_ln(self.roi_processor_box(bbox_tensor))
        roi_feats = torch.cat([roi_feats, roi_box_feats], dim=-1)

        feats_list = []
        start_idx = 0
        for num_boxes_i in num_boxes:
            end_idx = start_idx + num_boxes_i
            feats_i = self.process_single_image_rois(roi_feats[start_idx:end_idx])
            feats_list.append(feats_i)
            start_idx = end_idx
        assert end_idx == roi_feats.shape[0], '{} vs {}'.format(end_idx, roi_feats.shape[0])
        feats = torch.cat(feats_list, dim=0)
        # BxC
        if info_nce:
            return feats, attn_v
        else:
            return feats


if __name__ == '__main__':
    im = torch.rand((8, 3, 128, 128))

    model = RelationalBBoxNetworkEncoder(encoder='resnet50')
    x = model(im)
    print(x.shape)
