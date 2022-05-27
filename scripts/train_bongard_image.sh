#!/bin/bash

CFG=$1
OPTS=${@:2}

python train_meta_image_dist_bbox.py --config-file $CFG $OPTS 2>&1
