#!/bin/bash

# Load .env for WandB
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python src/stage1_sample_with_ckpt.py \
--config configs/stage1/training/DINOv2-B_decXL_smol_test.yaml \
--image assets/UI_frame_0003.png \
--output ftVIT_UI_recon_output3.png \
--ckpt UI_512_FDS_results/stage1/005-RAE/checkpoints/0075000.pt