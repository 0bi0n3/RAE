#!/bin/bash

# Load .env for WandB
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/stage1_sample_ddp_with_ckpt.py \
  --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage1/pretrained/DINOv2-B_UI.yaml \
  --data-path /mnt/hitchcock/scratch/oberon/action_window_segments/dataset/training_dataset_segments_10tutorials/ \
  --sample-dir recon_UI_samples2 \
  --num-samples 100 \
  --ckpt /mnt/hitchcock/scratch/oberon/RAE/UI_512_FDS_results/stage1/005-RAE/checkpoints/0075000.pt
  # --image-size 224 \
  # --precision fp32