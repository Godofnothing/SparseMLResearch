""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import json
import yaml
import logging
import argparse
import colorama

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config, \
    Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, convert_splitbn_model
from timm.utils import setup_default_logging, random_seed, get_outdir, distribute_bn, \
     ModelEmaV2
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy, \
    BinaryCrossEntropy, LabelSmoothingCrossEntropy
# sparseml imports
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import save_model, load_model, load_optimizer, load_epoch, \
    load_loss_scaler, GradSampler

from utils.load_aux_data import load_calibration_images

# import wandb
try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# local imports
from utils.manager_utils import is_update_epoch
from utils.sparsity_utils import get_current_sparsity, get_sparsity_info
from utils.dist_utils import init_distributed_mode, setup_distributed
from utils.amp_utils import init_amp_mode, setup_amp
from utils.training_utils import train_one_epoch, valid_one_epoch, log_on_epoch
# import custom schedule
from optim import create_sam_optimizer, create_topk_sam_optimizer


def parse_args():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = argparse.ArgumentParser(description='Timm Training Config with SparseML integration',  
                                     add_help=False)  
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    # The main args parser
    parser = argparse.ArgumentParser(description='Sparse training with timm script')
    # SparseML recipe
    parser.add_argument('--sparseml-recipe', '-sp', required=True, type=str,
                        help='YAML config file with the sparsification recipe')

    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='validation batch size override (default: None)')

    # specific options for MFAC loader
    parser.add_argument('--mfac-loader', action='store_true',
                        help='whether to use separate loader for MFAC (default: False)')
    parser.add_argument('-mb', '--mfac-batch-size', type=int, default=None, metavar='N',
                        help='mfac batch size override (default: None)')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--sam', action='store_true',
                        help='Use sharpness-aware minimizer')
    parser.add_argument('--sam-rho', default=0.2, type=float,
                        help='Parameter rho in the sam optimizer')
    parser.add_argument('--sam-topk', default=0.0, type=float,
                        help='Keep only topk entries')
    parser.add_argument('--sam-global-sparsity', action='store_true',
                        help='Use global sparsity for SAM')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=int, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-freq', type=int, default=-1, metavar='N',
                        help='checkpointing frequency (default: no saving epoch checkpoints)')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing through model blocks/stages')

    # Logging args
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--log-sparsity', action='store_true', default=False,
                        help='whether to log sparsity on each pruning step')

    # AdaPrune calibration images args
    parser.add_argument('--load-calibration-images', action='store_true',
                        help='whether to use calibration images')
    parser.add_argument('--num-calibration-images', default=1000, type=int,
                        help='number of images used for calibration')
    parser.add_argument('--path-to-labels', default='./data/imagenet_train_labels.npy', type=str, 
                        help='path-to-imagenet-labels')

    # Whether to save last model
    parser.add_argument('--save-last', action='store_true', default=False,
                        help='Whether to save the last state of the model')
    
    config_args, remaining_args = config_parser.parse_known_args()
    # Do we have a config file to parse?
    args = parser.parse_known_args()
    # If there is config replace default 
    if config_args.config:
        with open(config_args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining_args)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    # torch to benchmark mode
    torch.backends.cudnn.benchmark = True
    # setup logging
    _logger = logging.getLogger('train')
    setup_default_logging()
    # parse args
    args, args_text = parse_args()
    init_distributed_mode(args, _logger)

    if args.log_wandb:
        if has_wandb and args.rank == 0:

            wandb.init(
                project=f"SparseTraining|{args.model}", 
                name=f"{args.experiment}",
                config=args
            )
    else: 
        if args.rank == 0:
            _logger.warning(
                colorama.Fore.RED + \
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`" + \
                colorama.Fore.RESET
            )

    use_amp = init_amp_mode(args, _logger)
    # seed everthing
    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint
    )

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if args.local_rank == 0:
        _logger.info(
            colorama.Fore.GREEN + \
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}' + \
            colorama.Fore.RESET
        )

    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(args.device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if use_amp == 'apex':
            from apex.parallel import convert_syncbn_model
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if args.sam and args.sam_topk == 0:
        optimizer = create_sam_optimizer(model, args)
    elif args.sam and args.sam_topk > 0:
        optimizer = create_topk_sam_optimizer(model, args)
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    loss_scaler, amp_autocast = setup_amp(model, optimizer, args, _logger)
    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        # load model checkpoint
        load_model(args.resume, model, fix_data_parallel=True)
        if args.rank == 0:
            _logger.info(f'Loading model from checkpoint {args.resume}')
        # load optimizer
        if not args.no_resume_opt:
            if args.rank == 0:
                _logger.info(f'Loading optimizer from checkpoint {args.resume}')
            load_optimizer(args.resume, optimizer, map_location=args.device)
            load_loss_scaler(args.resume, loss_scaler, map_location=args.device)
        resume_epoch = load_epoch(args.resume, map_location=args.device)
        if args.rank == 0:
            _logger.info(f'Starting training from {resume_epoch} epoch')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        # do not resume EMA

    setup_distributed(model, args, _logger)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)

    # create the train and eval datasets
    args.prefetcher = not args.no_prefetcher

    train_dataset = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split=args.train_split, 
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats
    )

    eval_dataset = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split=args.val_split, 
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size
    )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        train_dataset = AugMixDataset(train_dataset, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding
    )

    eval_loader = create_loader(
        eval_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem
    )

    # make separate MFAC-loader
    if args.mfac_loader:
        loader_mfac = create_loader(
            train_dataset,
            input_size=data_config['input_size'],
            batch_size=args.mfac_batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem
        )

        def mfac_data_generator(device=args.device):
            while True:
                for input, target in loader_mfac:
                    input, target = input.to(device), target.to(device)
                    yield [input], {}, target

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    val_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.mfac_loader:
        # create grad sampler with the created data laoder and loss function
        grad_sampler = GradSampler(mfac_data_generator(), val_loss_fn)

    #########################
    # Setup SparseML manager
    ############$############

    bonus_kw = {'grad_sampler' : grad_sampler} if args.mfac_loader else {}
    # prepare calibration images (if AdaPrune used)
    if args.load_calibration_images:
        calibration_images = load_calibration_images(train_dataset, args, _logger)
        bonus_kw['calibration_images'] = calibration_images

    manager = ScheduledModifierManager.from_yaml(args.sparseml_recipe)
    # wrap optimizer  
    optimizer = manager.modify(
        model, 
        optimizer, 
        steps_per_epoch=len(train_loader), 
        epoch=start_epoch, 
        distillation_teacher='self', 
        **bonus_kw
    ) 
    
    # override timm scheduler
    if any("LearningRate" in str(modifier) for modifier in manager.modifiers):
        lr_scheduler = None
        if manager.max_epochs:   
            num_epochs = manager.max_epochs
        if args.rank == 0:
            _logger.info(
                colorama.Fore.YELLOW + \
                "Disabling timm LR scheduler, managing LR using SparseML recipe" + \
                colorama.Fore.RESET
            )
            _logger.info(
                colorama.Fore.YELLOW + \
                f"Overriding max_epochs to {num_epochs} from SparseML recipe" + \
                colorama.Fore.RESET
            )
    else:
        if args.rank == 0:
            _logger.info(f'Scheduled epochs: {num_epochs}')

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric_dense  = 0.0
    best_metric_sparse = 0.0
    output_dir = None
    if args.rank == 0:
        assert args.experiment is not None, "One needs to set the name of experiment"
        exp_name = args.experiment
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False

        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, 
                model, 
                train_loader, 
                optimizer, 
                train_loss_fn, 
                args,
                _logger,
                lr_scheduler=lr_scheduler, 
                output_dir=output_dir, 
                amp_autocast=amp_autocast, 
                loss_scaler=loss_scaler,
                model_ema=model_ema, 
                mixup_fn=mixup_fn
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = valid_one_epoch(
                model, 
                eval_loader, 
                val_loss_fn, 
                args, 
                _logger,
                amp_autocast=amp_autocast
            )

            # get current average sparsity
            avg_sparsity = get_current_sparsity(manager, epoch)
            # evaluate EMA
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                # do not evaluate EMA if in sparse stage
                if avg_sparsity == 0.0:
                    ema_eval_metrics = valid_one_epoch(
                        model_ema.module, 
                        eval_loader, 
                        val_loss_fn, 
                        args, 
                        _logger,
                        amp_autocast=amp_autocast, 
                        log_suffix=' (EMA)'
                    )
                    eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                if args.sched == 'cosine_annealing':
                    lr_scheduler.step()
                else:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                log_on_epoch(
                    epoch, 
                    train_metrics, 
                    eval_metrics, 
                    filename=f'{output_dir}/summary.csv',
                    sparsity=avg_sparsity,
                    log_wandb=args.log_wandb,
                    write_header=(epoch == start_epoch)
                )

            if args.local_rank == 0:
                if epoch % args.checkpoint_freq == 0 and args.checkpoint_freq > 0:
                    save_model(
                        path=f'{output_dir}/{args.model}_epoch={epoch}.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )

                # log mean sparsity
                _logger.info(f'Mean sparsity: {avg_sparsity:.4f}')

                if args.log_sparsity and is_update_epoch(manager, epoch):
                    sparsity_info = json.loads(get_sparsity_info(model))
                    with open(f'{output_dir}/sparsity_distribution_epoch={epoch}.json', 'w') as outfile:
                        json.dump(sparsity_info, outfile)

                # save best sparse checkpoint
                if avg_sparsity > 0.0 and eval_metrics[eval_metric] > best_metric_sparse:
                    best_metric_sparse = eval_metrics[eval_metric]
                    best_epoch_sparse = epoch
                    save_model(
                        path=f'{output_dir}/{args.model}_sparse_best.pth',
                        model=model, 
                        optimizer=optimizer,
                        loss_scaler=loss_scaler, 
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )
                    _logger.info(f'New best sparse model on epoch {epoch} with accuracy {best_metric_sparse:.4f}')

                # save best dense checkpoint
                if avg_sparsity == 0.0 and eval_metrics[eval_metric] > best_metric_dense:
                    best_metric_dense = eval_metrics[eval_metric]
                    best_epoch_dense = epoch
                    save_model(
                        path=f'{output_dir}/{args.model}_dense_best.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )
                    _logger.info(f'New best dense model on epoch {epoch} with accuracy {best_metric_dense:.4f}')
    except KeyboardInterrupt:
        pass
    if best_metric_dense > 0.0 and args.local_rank == 0:
        _logger.info('*** Best metric (dense): {0} (epoch {1})'.format(best_metric_dense, best_epoch_dense))
    if best_metric_sparse > 0.0 and args.local_rank == 0:
        _logger.info('*** Best metric (sparse): {0} (epoch {1})'.format(best_metric_sparse, best_epoch_sparse))

    # save last (if requested)
    if args.save_last and args.local_rank == 0:
        save_model(
            path=f'{output_dir}/{args.model}_last.pth',
            model=model, 
            optimizer=optimizer, 
            loss_scaler=loss_scaler,
            epoch=epoch,
            use_zipfile_serialization_if_available=True,
            include_modifiers=True
        )    
        _logger.info(f'Saved last training state of the model')

    # finalize pruner
    manager.finalize(model)

    
if __name__ == '__main__':
    main()
