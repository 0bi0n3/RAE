#!/bin/bash

# Load .env for WandB
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/train_stage1.py \
  --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/training_dataset_segments_10tutorials \
  --results-dir UI_512_FDS_results/stage1 \
  --image-size 256 --precision fp32 \
  --wandb