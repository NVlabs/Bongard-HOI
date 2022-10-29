# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import os
import json
from PIL import Image
import numpy as np
import glob
from PIL import ImageFilter
import random
import cv2
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from detectron2
from detectron2.structures import Boxes
from detectron2.data import transforms as T

from .datasets import register


@register('image-bongard-bbox')
class ImageBongard(Dataset):

    def __init__(self, use_gt_bbox=False, image_size=256, box_size=256, **kwargs):
        self.bong_size = kwargs.get('bong_size')
        if self.bong_size is None:
            self.bong_size = 7
        if box_size is None:
            box_size = image_size

        split_file_path = kwargs.get('split_file')
        assert split_file_path is not None
        bongard_problems = json.load(open(split_file_path, 'r'))
        self.bongard_problems = bongard_problems
        self.im_dir = kwargs.get('im_dir')
        assert self.im_dir is not None
        self.n_tasks = len(bongard_problems)

        # bounding boxes info
        # use ground-truth boxes if not provided
        bbox_file = kwargs.get('bbox_file')
        if not use_gt_bbox and bbox_file is not None:
            with open(bbox_file, 'rb') as f:
                self.boxes_data = pickle.load(f)
            self.det_thresh = kwargs.get('det_thresh')
            if self.det_thresh is None:
                self.det_thresh = 0.7
        else:
            self.boxes_data = None

        self.do_aug = 'augment' in kwargs or 'augment_plus' in kwargs

        self.pix_mean = (0.485, 0.456, 0.406)
        self.pix_std = (0.229, 0.224, 0.225)

        # detectron2-style data augmentation
        sample_style = 'range'
        augmentations = [T.ResizeShortestEdge(image_size, int(image_size * 2), sample_style)]
        if kwargs.get('augment') or kwargs.get('augment_plus'):
            augmentations.append(
                T.RandomFlip(
                    horizontal=True,
                    vertical=False,
                )
            )
        if kwargs.get('augment_plus'):
            self.photo_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        else:
            self.photo_aug = None
        self.augmentations = T.AugmentationList(augmentations)


    def get_crop_im(self, im_path, x1, y1, x2, y2, dilation=0.1, subject_dim=None, object_dim=None):
        im_path = os.path.join(self.im_dir, im_path)
        im = cv2.imread(im_path).astype(np.float32)
        # BGR to RGB
        im = im[:, :, ::-1]
        assert im is not None, im_path
        imh, imw, _ = im.shape

        if subject_dim is not None:
            sub_x1, sub_y1, sub_x2, sub_y2 = subject_dim
            im = cv2.rectangle(im, (sub_x1, sub_y1), (sub_x2, sub_y2), thickness=2, color=(0, 255, 0))
        if object_dim is not None:
            obj_x1, obj_y1, obj_x2, obj_y2 = object_dim
            im = cv2.rectangle(im, (obj_x1, obj_y1), (obj_x2, obj_y2), thickness=2, color=(0, 255, 0))

        h = y2 - y1
        w = x2 - x1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # dilation
        h = (1 + dilation) * h
        w = (1 + dilation) * w
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(imw, int(cx + w / 2))
        y2 = min(imh, int(cy + h / 2))
        crop_im = im[y1 : y2, x1 : x2]
        return crop_im, x1, y1, x2, y2

    def get_bbox(self, im, dim):
        imh, imw = im.height, im.width
        x1, y1, x2, y2 = dim
        if x1 <= 1 and y1 <= 1 and x2 <= 1 and y2 <= 1:
            x1 = imw * x1
            y1 = imh * y1
            x2 = imw * x2
            y2 = imh * y2
        return torch.Tensor(([x1, y1, x2, y2]))

    def get_detection_boxes_in_crop(self, image_id, x1, y1, x2, y2):
        all_boxes = self.boxes_data[image_id]['boxes']
        boxes = []
        for idx in range(all_boxes.shape[0]):
            x1_i, y1_i, x2_i, y2_i = all_boxes[idx]
            if x1_i >= x1 and y1_i >= y1 and x2_i <= x2 and y2_i <= y2:
                boxes.append(torch.Tensor([x1_i - x1, y1_i - y1, x2_i - x1, y2_i - y1]))
        if boxes:
            boxes = torch.stack(boxes, dim=0)
        else:
            # use the entire image if no detections found
            boxes = torch.Tensor([0, 0, x2 - x1, y2 - y1])
        return boxes

    def get_triplet_crop(self, scene_graph, tp_idx, image_id, im_dir='../OpenImages/validation', show_bbox=False):
        tp = scene_graph['triplets'][tp_idx]
        objects = scene_graph['objects']
        sub_x1, sub_y1, sub_x2, sub_y2 = objects[tp['subject']]['dimension']
        obj_x1, obj_y1, obj_x2, obj_y2 = objects[tp['object']]['dimension']
        assert sub_x2 > 1 and sub_y2 > 1
        assert obj_x2 > 1 and obj_y2 > 1

        x1 = min(sub_x1, obj_x1)
        y1 = min(sub_y1, obj_y1)
        x2 = max(sub_x2, obj_x2)
        y2 = max(sub_y2, obj_y2)

        im_path = os.path.join(im_dir, image_id)
        if not im_path.endswith('.jpg') and not im_path.endswith('.png'):
            im_path += '.jpg'
        if show_bbox:
            crop_im, x1, y1, x2, y2 = self.get_crop_im(im_path, x1, y1, x2, y2, subject_dim=objects[tp['subject']]['dimension'], object_dim=objects[tp['object']]['dimension'])
        else:
            crop_im, x1, y1, x2, y2 = self.get_crop_im(im_path, x1, y1, x2, y2)

        if self.boxes_data is None:
            # use ground-truth bounding boxes
            sub_bbox = torch.Tensor([sub_x1 - x1, sub_y1 - y1, sub_x2 - x1, sub_y2 - y1])
            obj_bbox = torch.Tensor([obj_x1 - x1, obj_y1 - y1, obj_x2 - x1, obj_y2 - y1])
            boxes = torch.stack((sub_bbox, obj_bbox), dim=0)
        else:
            boxes = self.get_detection_boxes_in_crop(image_id, x1, y1, x2, y2)

        return crop_im, boxes

    def get_image(self, crop_info):
        im_path = crop_info['im_path']
        x1, y1, x2, y2 = crop_info['crop_bbox']
        sub_bbox = crop_info['sub_bbox']
        obj_bbox = crop_info['obj_bbox']

        im_path = os.path.join(self.im_dir, im_path)
        im = cv2.imread(im_path).astype(np.float32)
        imh, imw, _ = im.shape
        # BGR to RGB
        im = im[:, :, ::-1]
        assert im is not None, im_path

        # fix image and annotation mismatch of openimages
        if "openimages" in im_path:
            x1, y1, x2, y2 = int(1.6 * x1), int(1.6 * y1), int(1.6 * x2), int(1.6 * y2)
            # if bounding box is out of the border, use the whole image
            if y1 > imh or y2 > imh or x1 > imw or x2 > imw:
                x1, y1, x2, y2 = int(0), int(0), imw, imh
            sub_bbox[0] = max(sub_bbox[0] * 1.6, int(0))
            sub_bbox[1] = max(sub_bbox[1] * 1.6, int(0))
            sub_bbox[2] = min(sub_bbox[2] * 1.6, imw)
            sub_bbox[3] = min(sub_bbox[3] * 1.6, imh)
            obj_bbox[0] = max(obj_bbox[0] * 1.6, int(0))
            obj_bbox[1] = max(obj_bbox[1] * 1.6, int(0))
            obj_bbox[2] = min(obj_bbox[2] * 1.6, imw)
            obj_bbox[3] = min(obj_bbox[3] * 1.6, imh)
        crop_im = im[y1:y2, x1:x2]
        assert crop_im.shape[0]*crop_im.shape[1] != 0, im_path

        if self.boxes_data is None:
            # use ground-truth bounding boxes
            sub_bbox = torch.Tensor(sub_bbox)
            obj_bbox = torch.Tensor(obj_bbox)
            boxes = torch.stack((sub_bbox, obj_bbox), dim=0)
        else:
            image_id = os.path.basename(im_path)
            boxes = self.get_detection_boxes_in_crop(image_id, x1, y1, x2, y2)

        # return self.transform(img), boxes
        aug_input = T.AugInput(crop_im, boxes=boxes)
        transforms = self.augmentations(aug_input)
        crop_im, boxes = aug_input.image, Boxes(aug_input.boxes)
        if self.photo_aug is not None:
            # color jittering of the input image
            crop_im = np.array(self.photo_aug(Image.fromarray(crop_im.astype(np.uint8))), dtype=np.float32)
        for i in range(3):
            crop_im[:, :, i] = (crop_im[:, :, i] / 255. - self.pix_mean[i]) / self.pix_std[i]
        crop_im = torch.as_tensor(np.ascontiguousarray(crop_im.transpose(2, 0, 1)))

        boxes_tensor = boxes.tensor
        boxes_dim = [
            (boxes_tensor[:, 2] + boxes_tensor[:, 0]) / 2 / imw,  # cx
            (boxes_tensor[:, 3] + boxes_tensor[:, 1]) / 2 / imh,  # cy
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) / imw, # width
            (boxes_tensor[:, 3] - boxes_tensor[:, 1]) / imh, # height
        ]
        boxes_dim = torch.stack(boxes_dim, dim=1)
        return crop_im, boxes, boxes_dim

    def pad_images(self, pos_ims, neg_ims):
        max_imh, max_imw = -1, -1
        for im_i in pos_ims:
            _, imh, imw = im_i.shape
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for im_i in neg_ims:
            _, imh, imw = im_i.shape
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, im_i in enumerate(pos_ims):
            pad_im_i = torch.zeros((3, max_imh, max_imw))
            _, imh, imw = im_i.shape
            pad_im_i[:, :imh, :imw] = im_i
            pos_ims[idx] = pad_im_i

        for idx, im_i in enumerate(neg_ims):
            pad_im_i = torch.zeros((3, max_imh, max_imw))
            _, imh, imw = im_i.shape
            pad_im_i[:, :imh, :imw] = im_i
            neg_ims[idx] = pad_im_i

        return pos_ims, neg_ims

    def __len__(self):
        return len(self.bongard_problems)

    def __getitem__(self, i):
        """
        (Pdb) x_shot.shape                                                                                    │
        torch.Size([16, 2, 6, 3, 256, 256])                                                                   │
        (Pdb) x_query.shape                                                                                   │
        torch.Size([16, 2, 3, 256, 256])                                                                      │
        (Pdb) label_query.shape                                                                               │
        torch.Size([32])
        """
        pos_info_list, neg_info_list, _ = self.bongard_problems[i]

        if self.do_aug:
            np.random.shuffle(pos_info_list)
            np.random.shuffle(neg_info_list)

        pos_ims, pos_boxes, pos_boxes_dim = [], [], []
        for pos_info_i in pos_info_list:
            im_i, boxes_i, boxes_dim_i = self.get_image(pos_info_i)
            pos_ims.append(im_i)
            pos_boxes.append(boxes_i)
            pos_boxes_dim.append(boxes_dim_i)

        neg_ims, neg_boxes, neg_boxes_dim = [], [], []
        for neg_info_i in neg_info_list:
            im_i, boxes_i, boxes_dim_i = self.get_image(neg_info_i)
            neg_ims.append(im_i)
            neg_boxes.append(boxes_i)
            neg_boxes_dim.append(boxes_dim_i)

        pos_ims, neg_ims = self.pad_images(pos_ims, neg_ims)

        pos_shot_ims = pos_ims[:-1]
        pos_query_im = pos_ims[-1]
        neg_shot_ims = neg_ims[:-1]
        neg_query_im = neg_ims[-1]

        pos_shot_boxes = pos_boxes[:-1]
        pos_query_boxes =  pos_boxes[-1]
        neg_shot_boxes = neg_boxes[:-1]
        neg_query_boxes = neg_boxes[-1]
        shot_boxes = pos_shot_boxes + neg_shot_boxes
        query_boxes = [pos_query_boxes, neg_query_boxes]

        pos_shot_boxes_dim = pos_boxes_dim[:-1]
        pos_query_boxes_dim = pos_boxes_dim[-1]
        neg_shot_boxes_dim = neg_boxes_dim[:-1]
        neg_query_boxes_dim = neg_boxes_dim[-1]
        shot_boxes_dim = torch.cat(
            (torch.cat(pos_shot_boxes_dim, dim=0), torch.cat(neg_shot_boxes_dim, dim=0)),
            dim=0
        )
        query_boxes_dim = torch.cat((pos_query_boxes_dim, neg_query_boxes_dim), dim=0)

        pos_shot_ims = torch.stack(pos_shot_ims, dim=0)
        neg_shot_ims = torch.stack(neg_shot_ims, dim=0)
        shot_ims = torch.stack((pos_shot_ims, neg_shot_ims), dim=0)
        query_ims = torch.stack((pos_query_im, neg_query_im), dim=0)
        query_labs = torch.Tensor([0, 1]).long()

        ret_dict = {
            'shot_ims': shot_ims,
            'shot_boxes': shot_boxes,
            'query_ims': query_ims,
            'query_boxes': query_boxes,
            'query_labs': query_labs,
            'shot_boxes_dim': torch.Tensor([0]), #shot_boxes_dim,
            'query_boxes_dim': torch.Tensor([0]), #query_boxes_dim
        }

        return ret_dict


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def collate_images_boxes_dict(batch):
    def _pad_tensor(tensor_list):
        max_imh, max_imw = -1, -1
        for tensor_i in tensor_list:
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, tensor_i in enumerate(tensor_list):
            pad_tensor_i = tensor_i.new_full(list(tensor_i.shape[:-2]) + [max_imh, max_imw], 0)
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            pad_tensor_i[..., :imh, :imw].copy_(tensor_i)
            tensor_list[idx] = pad_tensor_i
        return tensor_list

    keys = list(batch[0].keys())
    batched_dict = {}
    for k in keys:
        data_list = []
        for batch_i in batch:
            data_list.append(batch_i[k])
        if isinstance(data_list[0], torch.Tensor):
            if len(data_list[0].shape) > 1:
                data_list = _pad_tensor(data_list)
            data_list = torch.stack(data_list, dim=0)
        batched_dict[k] = data_list
    return batched_dict
