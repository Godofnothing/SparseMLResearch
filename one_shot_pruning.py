import os
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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sparseml.pytorch.optim import ScheduledModifierManager
# 
from engine import val_epoch


def parse_args():
    parser = argparse.ArgumentParser('One shot pruning with AdaPrune finetuning.', add_help=False)
    # Model
    parser.add_argument('--model', default='deit_small_patch16_224', type=str)
    # Experiment
    parser.add_argument('--experiment', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
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
    # SPDY params
    parser.add_argument('--spdy_loader', action='store_true')
    parser.add_argument('--num_calibration_images', default=1024, type=int)
    # MFAC params
    parser.add_argument('--mfac_loader', action='store_true')
    # Save arguments
    parser.add_argument('--save-dir', default='./output/one-shot', type=str, 
                        help='dir to save results')
    # Logging
    parser.add_argument('--log-wandb', action='store_true')
    parser.add_argument('--log-freq', default=100, type=int)

    args = parser.parse_args()
    return args


def random_subset(dataset, num_samples: int):
    ids = np.random.permutation(len(dataset))[:num_samples]
    return Subset(dataset, ids)


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # seed all
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    if args.spdy_loader:
        calibration_dataset = ImageFolder(root=f'{args.data_dir}/train', transform=val_transforms)
        train_subset = random_subset(calibration_dataset, args.num_calibration_images)
        # calibration loader
        calibration_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True
        )
    else:
        calibration_loader = None

    # define for M-FAC pruner
    def mfac_data_generator(device=device, **kwargs):
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
    # define addtional args for SPDY
    if args.spdy_loader:
        spdy_kw = dict(loader=calibration_loader, loss_fn=loss_fn)
    else:
        spdy_kw = {}
    # define additional args for M-FAC/OBS
    if args.mfac_loader:
        mfac_kw = dict(
            grad_sampler = {
                'data_loader_builder' : mfac_data_generator, 
                'loss_function' : loss_fn,
            },
        )
    else:
        mfac_kw = {}

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
            teacher_model=model,
            **mfac_kw,
            **spdy_kw,
            finalize=True
        )
        # evaluate 
        val_acc = val_epoch(model_sparse, val_loader, loss_fn, device=device)['acc']
        # update experiment data
        experiment_data['val/acc'].append(val_acc)
        print(f'Test accuracy: {val_acc:.3f}')
        if args.log_wandb:
            wandb.log({'sparsity' : sparsity, 'val/acc': val_acc})

    with open(f'{args.save_dir}/experiment_data.pkl', 'wb') as fout:
        pickle.dump(experiment_data, fout, protocol=pickle.HIGHEST_PROTOCOL)   

    print('Finished!') 