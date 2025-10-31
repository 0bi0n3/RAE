#!/bin/bash

# Load .env for WandB
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python src/metrics_run.py \
    --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B_UI.yaml \
    --checkpoint /mnt/hitchcock/scratch/oberon/RAE/results/stage2_test/020-DiTwDDTHead-Linear-velocity-none-bf16-acc1/checkpoints/0045000.pt \
    --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/validation_dataset_segments_2tutorials \
    --max-batches 10 \
    --save-samples \
    --output-dir quick_test_results