#!/bin/bash

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
MASTER_PORT=29501
BATCH_SIZE_PER_GPU=$(( 1024 / ${NUM_PROC} ))

# for das8gpu4
DATA_DIR=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC 
# for gpu224
# DATA_DIR=/localhome/dkuznede/Datasets/ILSVRC
# for zhores
# DATA_DIR=/gpfs/gpfs0/datasets/ImageNet/ILSVRC2012
# for ultramar
# DATA_DIR=/mnt/data/imagenet

MODEL=deit_small_patch16_224
EXP=GP_300_ep_${MODEL}_obc+obs_sp=0.30_0.60_B=128_Ng=4096_D=1e-8_Nr=1

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_old.py \
    \
    ${DATA_DIR} \
    \
    --config configs/deit.yaml \
    --sparseml-recipe recipes/gradual_pruning/obc+obs/cyclic_exponential/vit_obc+obs_sp=0.30_0.60_B=128_Ng=4096_D=1e-8_Nr=1.yaml \
    \
    --dataset imagenet \
    \
    --model ${MODEL} \
    --pretrained \
    \
    --experiment ${EXP} \
    \
    -b ${BATCH_SIZE_PER_GPU} \
    -vb 500 \
    --workers 8 \
    \
    --aux-loader \
    --aux-batch-size 128 \
    \
    --split-qkv \
    \
    --spdy-loader \
    \
    --log-sparsity \
    --log-wandb \
    \
    --amp \
    \
    --checkpoint-freq 20 \
    --save-last \
    \
    --timeout 18000 \
    \
