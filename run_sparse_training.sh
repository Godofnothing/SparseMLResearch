#!/bin/bash

NUM_PROC=4
MASTER_PORT=29502

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training.py \
    \
    /nvmedisk/Datasets/ILSVRC/Data/CLS-LOC \
    \
    --config configs/gmlp_s16.yaml \
    --sparseml-recipe recipes/deit_small_patch16_224_sp=0.6.yaml \
    \
    --dataset imagenet \
    \
    --model deit_small_patch16_224 \
    \
    --experiment GM_600_epochs \
    \
    --log-wandb \
    \
    -b 256 \
    -vb 500 \
    --workers 16 \
    \
    --amp \
    \
    --log-sparsity \
    \