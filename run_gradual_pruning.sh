#!/bin/bash

NUM_PROC=2
MASTER_PORT=29501

MODEL=deit_tiny_patch16_224 
EXP=AC_DC_MFAC_300_ep_${MODEL}

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_new.py \
    \
    /mnt/data/imagenet \
    \
    --config configs/deit_old.yaml \
    --sparseml-recipe recipes/gradual_pruning/vit_mfac_300_epochs.yaml \
    \
    --dataset imagenet \
    \
    --model ${MODEL} \
    --pretrained \
    \
    --experiment ${EXP} \
    \
    -b 512 \
    -vb 500 \
    --workers 16 \
    \
    --mfac-loader \
    --mfac-batch-size 512 \
    \
    --log-sparsity \
    --log-wandb \
    \
    --amp \
    \
    --checkpoint-freq 20 \
    --save-last \
    \
