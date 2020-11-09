#!/usr/bin/env bash

# Predict
CUDA_VISIBLE_DEVICES=0 python predict.py \
--data_dir demo \
--pretrained_aanet check1channel/aanet_cats_inferno/aanet_latest.pth \
--feature_type aanet \
--feature_pyramid_network \
--no_intermediate_supervision

#--pretrained_aanet pretrained/aanet_kitti15-fb2a0d23.pth \
#--pretrained_aanet check1channel/aanet_cats/aanet_latest.pth \
