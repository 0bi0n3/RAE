#!/bin/bash

python src/stage1_sample.py \
--config configs/stage1/pretrained/DINOv2-B_512.yaml \
--image assets/UI_frame_0001.png \
--output reconstructed_output.png