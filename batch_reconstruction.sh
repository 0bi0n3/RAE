#!/bin/bash

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/stage1_sample_ddp.py \
  --config configs/stage1/pretrained/DINOv2-B_UI.yaml \
  --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/training_dataset_segments_10tutorials/ \
  --sample-dir recon_UI_samples2 \
  --num-samples 100 \
  # --image-size 224 \
  # --precision fp32