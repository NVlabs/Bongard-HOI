# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import os


datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    dataset = datasets[name](**kwargs)
    return dataset
