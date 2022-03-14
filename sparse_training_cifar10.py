import timm
import wandb
import argparse
# torch imports
import torch
import torch.nn as nn
import torchvision.transforms as T
# dataloader imports
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
# scheduled modified manager
from sparseml.pytorch.optim import ScheduledModifierManager
# import train
from engine import train
# import resnet models
from models import resnet


def get_args_parser():
    parser = argparse.ArgumentParser('CIFAR10 Sparse Training', add_help=False)
    # Model
    parser.add_argument('--model', default='resnet20', type=str)
    # Path to recipe
    parser.add_argument('--recipe-path', required=True, type=str)
    # Logging
    parser.add_argument('--log-wandb', action='store_true')
    # Loss params
    parser.add_argument('--smoothing', default=0.0, type=float)
    # Dataloader parameters
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-vb', '--val_batch_size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    # Optimizer params
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    # Scheduler params
    parser.add_argument('--milestones', default=[82, 123], nargs="+", type=int,
                        help='Epochs on which lr is reduced')
    parser.add_argument('--decay-rate', default=0.1, type=float, 
                        help='Factor of lr reduction')
    # Save arguments
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='dir with results')
    return parser


if __name__ == '__main__':
    # parse args
    parser = get_args_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # init wandb
    if args.log_wandb:
        wandb.init(
            project="CIFAR10_sparse_training", 
            entity="spiridon_sun_rotator",
            name=f"AC/DC|CIFAR10|{args.model}",
            config={
                "model" : args.model,
                "dataset" : "CIFAR10"
            }
        )

    # define data   
    CIFAR10_MEAN, CIFAR10_STD = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)

    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    val_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    # create datasets
    train_dataset = CIFAR10(root='../Datasets/CIFAR10', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR10(root='../Datasets/CIFAR10', train=False, download=True, transform=val_transforms)
    # create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # model
    print(f'model: {args.model}')
    model = getattr(resnet, args.model)()
    # to device
    model = model.to(device)
    # define pruner
    manager = ScheduledModifierManager.from_yaml(args.recipe_path)
    # number of steps per epoch
    steps_per_epoch = len(train_loader)
    # optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # make a pruning modifier
    optimizer = manager.modify(model, optimizer, steps_per_epoch)
    # define loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    history = train(
        model,
        {"train" :  train_loader, "val" : val_loader},
        criterion=criterion, 
        optimizer=optimizer,
        num_epochs=manager.epoch_modifiers[0].end_epoch,
        device=device,
        log_wandb=args.log_wandb
    )

    # finalize the model
    manager.finalize(model)
