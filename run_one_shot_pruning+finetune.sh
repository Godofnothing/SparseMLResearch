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
EXP=OneShot+Finetune_vit_sp=0.40_B=192_Ng=4096_D=1e-6_Nr=1

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_old.py \
    \
    ${DATA_DIR} \
    \
    --config configs/deit_remastered.yaml \
    --sparseml-recipe recipes/one_shot+finetune/ovit/ovit_vit_sp=0.40_B=192_Ng=4096_D=1e-6_Nr=1.yaml \
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
    --output output/one_shot+finetune \
