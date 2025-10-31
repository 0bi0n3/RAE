#!/bin/bash

# Load .env for WandB
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    src/train_w_valid.py \
    --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_UI.yaml \
    --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/training_dataset_segments_10tutorials \
    --val-data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/validation_dataset_segments_2tutorials \
    --results-dir results/stage2_test \
    --image-size 256 \
    --precision bf16 \
    --wandb