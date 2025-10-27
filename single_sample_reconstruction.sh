#!/bin/bash

python src/stage1_sample.py \
--config configs/stage1/training/DINOv2-B_decXL_smol_test.yaml \
--image assets/UI_frame_0001.png \
--output test_UI_reconstructed_output.png