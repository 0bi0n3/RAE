#!/bin/bash

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/train_stage1.py \
  --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage1/training/DINOv2-B_decXL_smol_test.yaml \
  --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dummy_segs_output \
  --results-dir UI_test_results/stage1 \
  --image-size 256 --precision fp32 \
  --wandb