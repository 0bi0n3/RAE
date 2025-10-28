#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKINg=0

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    src/train.py \
    --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \
    --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/training_dataset_segments_10tutorials \
    --results-dir results/stage2 \
    --precision bf16 \
    --wandb