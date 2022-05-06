#!/bin/bash

NUM_PROC=2
MASTER_PORT=29504

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training.py \
    \
    /localhome/dkuznede/Datasets/ILSVRC \
    \
    --config configs/deit.yaml \
    --sparseml-recipe recipes/gradual_pruning/vit_mfac_300_epochs.yaml \
    \
    --dataset imagenet \
    \
    --model deit_tiny_patch16_224 \
    --pretrained \
    \
    --experiment MFAC_300_epochs \
    \
    -b 512 \
    -vb 500 \
    --workers 16 \
    \
    --log-wandb \
    \
    --mfac-loader \
    --mfac-batch-size 256 \
    \
    --log-sparsity \
    \
    --amp \
    \
    --load-calibration-images \
    --num-calibration-images 1000 \
    --path-to-labels data/imagenet_train_labels.npy \
    \
