#!/bin/bash

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/stage1_sample_ddp.py \
  --config /mnt/hitchcock/scratch/oberon/RAE/configs/stage1/pretrained/DINOv2-B_512.yaml \
  --data-path /mnt/welles/scratch/datasets/ImageNet1k \
  --sample-dir recon_samples \
  --image-size 512 \
  --num-samples 100 \