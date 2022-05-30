#!/bin/bash

NUM_PROC=2
MASTER_PORT=29501

MODEL=deit_tiny_patch16_224 
EXP=GPshort_OBS_50_ep_${MODEL}_lr=1e-4

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_old.py \
    \
    /nvmedisk/Datasets/ILSVRC/Data/CLS-LOC \
    \
    --config configs/deit_old.yaml \
    --sparseml-recipe recipes/gradual_pruning_short/vit_obs_50_epochs_sp=0.6_lr=1e-4.yaml \
    \
    --dataset imagenet \
    \
    --model ${MODEL} \
    --pretrained \
    \
    --experiment ${EXP} \
    \
    -b 256 \
    -vb 500 \
    --workers 16 \
    \
    --mfac-loader \
    --mfac-batch-size 256 \
    \
    --log-sparsity \
    \
    --amp \
    \
    --checkpoint-freq 2 \
    --save-last \
    \
