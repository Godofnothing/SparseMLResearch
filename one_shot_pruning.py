import os
import re
import timm
import torch
import wandb
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
#
from copy import deepcopy
from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import GradSampler
# 
from engine import val_epoch


def parse_args():
    parser = argparse.ArgumentParser('One shot pruning with AdaPrune finetuning.', add_help=False)
    # Model
    parser.add_argument('--model', default='deit_small_patch16_224', type=str)
    # Experiment
    parser.add_argument('--experiment', default='', type=str)
    # Path to data
    parser.add_argument('--data-dir', required=True, type=str)
    # Path to recipe
    parser.add_argument('--sparseml-recipe', required=True, type=str)
    # Loader params
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-vb', '--val_batch_size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--prefetch', default=2, type=int)
    # Sparsities
    parser.add_argument('--sparsities', nargs='+', required=True, type=float)
    # AdaPrune params
    parser.add_argument('--load-calibration-images', action='store_true')
    parser.add_argument('--num-calibration-images', default=1000, type=int)
    # Path to imagenet labels
    parser.add_argument('--path-to-labels', required=True, type=str)
    # Save arguments
    parser.add_argument('--save-dir', default='./output/one-shot', type=str, 
                        help='dir to save results')
    # Logging
    parser.add_argument('--log-wandb', action='store_true')
    parser.add_argument('--log-freq', default=100, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse args
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set num threads
    torch.set_num_threads(args.workers + 1)
    # init wandb
    if args.log_wandb:
        wandb.init(
            project="ImageNetOneShotPruning", 
            entity="spiridon_sun_rotator",
            name=f"OneShotPruning/{args.model}/{args.experiment}",
            config={
                "model" : args.model,
                "dataset" : "ImageNet",
                "calibration_images" : args.num_calibration_images
            }
        )
    # Data
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize,
    ])

    val_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(root=f'{args.data_dir}/train', transform=train_transforms)
    val_dataset = ImageFolder(root=f'{args.data_dir}/val', transform=val_transforms)

    # create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        prefetch_factor=args.prefetch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.val_batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        prefetch_factor=args.prefetch,
        pin_memory=True
    )

    # prepare calibration images
    if args.load_calibration_images:
        train_ids_all = range(len(train_dataset))
        train_labels_all = np.load(args.path_to_labels)
        train_ids, _, train_labels, _ =  train_test_split(
            train_ids_all, train_labels_all, stratify=train_labels_all, train_size=args.num_calibration_images)
        calibration_images = torch.stack([train_dataset[i][0] for i in train_ids], dim=0).to(device)

    # define for M-FAC pruner
    def mfac_data_generator(**kwargs):
        while True:
            for input, target in train_loader:
                input, target = input.to(device), target.to(device)
                yield [input], {}, target

    # dummy loss 
    loss_fn = nn.CrossEntropyLoss()
    # model
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device)
    # first evaluation
    val_acc = val_epoch(model, val_loader, loss_fn, device=device)['acc']
    print(f'Accuracy dense: {val_acc:.3f}')
    # make dir (if needed)
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_data = {
        'sparsity': args.sparsities, 'val/acc' : []
    }
    for sparsity in args.sparsities:
        print(f'Sparsity {sparsity:.3f}')
        model_sparse = deepcopy(model)
        # create sparseml manager
        manager = ScheduledModifierManager.from_yaml(args.sparseml_recipe)
        # update manager
        manager.modifiers[0].init_sparsity  = sparsity
        manager.modifiers[0].final_sparsity = sparsity
        # apply recipe
        optimizer = manager.apply(
            model_sparse, 
            grad_sampler={
                'data_loader_builder' : mfac_data_generator,
                'loss_function' : loss_fn,
            },
            teacher_model=model,
            finalize=True
        )
        # evaluate after AdaPrune
        val_acc = val_epoch(model_sparse, val_loader, loss_fn, device=device)['acc']
        # update experiment data
        experiment_data['val/acc'].append(val_acc)
        print(f'Test accuracy: {val_acc:.3f}')
        if args.log_wandb:
            wandb.log({'sparsity' : sparsity, 'val/acc': val_acc})

    with open(f'{args.save_dir}/experiment_data.pkl', 'wb') as fout:
        pickle.dump(experiment_data, fout, protocol=pickle.HIGHEST_PROTOCOL)   

    print('Finished!') 