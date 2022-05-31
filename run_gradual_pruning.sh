#!/bin/bash

NUM_PROC=4
MASTER_PORT=29501

# for das8gpu4
# DATA_DIR=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC 
# for gpu224
DATA_DIR=/localhome/dkuznede/Datasets/ILSVRC
# for zhores
# DATA_DIR=/gpfs/gpfs0/datasets/ImageNet/ILSVRC2012
# for ultramar
# DATA_DIR=/mnt/data/imagenet

MODEL=deit_small_patch16_224
EXP=GP_OBS_300_ep_${MODEL}_sp=0.4_0.9

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_old.py \
    \
    ${DATA_DIR} \
    \
    --config configs/deit_old.yaml \
    --sparseml-recipe recipes/gradual_pruning/obs/vit_obs_300_epochs_sp=0.40_0.90_block_size=64_damp=1e-6_num_grads=4096.yaml \
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
    --mfac-batch-size 128 \
    \
    --log-sparsity \
    --log-wandb \
    \
    --amp \
    \
    --checkpoint-freq 20 \
    --save-last \
    \
    --timeout 3600 \
    \
