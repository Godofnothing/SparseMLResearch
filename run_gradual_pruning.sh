#!/bin/bash

#SBATCH --job-name SparseML
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task 16
#SBATCH --time 24:00:00 # hh:mm:ss, walltime
#SBATCH --mem 160G
#SBATCH -w gn12

source /home/${USER}/.bashrc
source activate pysparse

NUM_PROC=4
MASTER_PORT=29501

MODEL=deit_tiny_patch16_224 
EXP=GPshort_GM_50_ep_${MODEL}_lr=1e-5

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training_old.py \
    \
    /gpfs/gpfs0/datasets/ImageNet/ILSVRC2012 \
    \
    --config configs/deit_old.yaml \
    --sparseml-recipe recipes/gradual_pruning_short/vit_gm_50_epochs_lr_max=1e-5.yaml \
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
    --log-wandb \
    \
    --amp \
    \
    --checkpoint-freq 2 \
    --save-last \
    \
