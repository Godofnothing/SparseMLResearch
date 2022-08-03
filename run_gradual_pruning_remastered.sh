#!/bin/bash

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
MASTER_PORT=29502
BATCH_SIZE_PER_GPU=$(( 1024 / ${NUM_PROC} ))

# for das8gpu4
# DATA_DIR=/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC 
# for gpu224
# DATA_DIR=/localhome/dkuznede/Datasets/ILSVRC
# for zhores
# DATA_DIR=/gpfs/gpfs0/datasets/ImageNet/ILSVRC2012
# for ultramar
DATA_DIR=/mnt/data/imagenet

MODEL=deit_tiny_patch16_224
# EXP=OneShot+Finetune_${MODEL}_obc+obs_B=192_Ng=4096_D=1e-8_Nr=1_sp=0.4_max_lr=5e-5
EXP=debug_${MODEL}

torchrun \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_remastered.py \
    \
    --data_dir ${DATA_DIR} \
    \
    --config configs/light1.yaml \
    --sparseml-recipe recipes/debug/vit_3ep_obc+obs_B=192_Ng=4096_D=1e-8_Nr=1.yaml \
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
    --workers 16 \
    \
    --split-qkv \
    \
    --aux-loader \
    --aux-batch-size 128 \
    \
    --log-sparsity \
    --log-wandb \
    --log_param_histogram \
    \
    --amp \
    \
    --save-freq 20 \
    --save-last \
    \
    --timeout 3600 \
    \
    --output output/debug
