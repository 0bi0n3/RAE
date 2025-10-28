#!/bin/bash
# Test with minimal steps first

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    src/train.py \
    --config configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B_UI.yaml \
    --data-path /mnt/welles/scratch/datasets/ImageNet1k \
    --results-dir results/stage2_test \
    --precision fp32 \
    --wandb