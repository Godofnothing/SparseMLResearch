import os
import yaml
import json
import torch
import logging
import argparse

import torch.nn as nn

from argparse import Namespace
from torch.nn.parallel import DistributedDataParallel

from timm.data import (
    create_dataset, 
    create_loader, 
    resolve_data_config,
    AugMixDataset
)
from timm.models import (
    create_model, 
    safe_model_name,
    convert_splitbn_model,
    load_checkpoint
)
from timm.utils import (
    random_seed, 
    set_jit_fuser,
    ModelEmaV2,
    distribute_bn,
    setup_default_logging
)
from timm.optim import (
    create_optimizer_v2, 
    optimizer_kwargs
)
from timm.scheduler import create_scheduler

# sparseml imports
from sparseml.pytorch.utils import (
    load_epoch,
    load_model,
    load_optimizer,
    load_loss_scaler,
    save_model
)
from sparseml.pytorch.optim import ScheduledModifierManager
# local import 
from utils.dist_utils import init_distributed
from utils.setup_utils import (
    setup_amp,
    setup_mixup,
    setup_loss_fn
)
from utils.training_utils import (
    train_one_epoch,
    validate
)
from utils.sparsity_utils import (
    get_current_sparsity, 
    get_sparsity_info
)
from utils.summary import update_summary
from utils.manager_utils import is_update_epoch
from optim import create_sam_optimizer

# using wandb for logging
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
# using memory_efficient_fusion if installed
try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False


def parse_args():
    # The first parser parses only the --config args
    config_parser = argparse.ArgumentParser(description='Timm Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                                help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    parser.add_argument('--data_dir', metavar='DIR', required=True,
                        help='path to dataset')
    group.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    group.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    group.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    group.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # SparseML parameters
    group = parser.add_argument_group('SparseML parameters')
    group.add_argument('--sparseml-recipe', required=True, type=str,
                        help='YAML config file with the sparsification recipe')
    group.add_argument('-mb', '--mfac-batch-size', type=int, default=None, metavar='N',
                        help='mfac batch size override (default: None)')
    group.add_argument('--mfac-loader', action='store_true',
                        help='whether to use separate loader for MFAC (default: False)')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    group.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    group.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='Validation batch size override (default: None)')
    group.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                        help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
    group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing through model blocks/stages')

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')
    group.add_argument('--sam', action='store_true',
                        help='Use sharpness-aware minimizer')
    group.add_argument('--sam-rho', default=0.2, type=float,
                        help='Parameter rho in the sam optimizer')
    group.add_argument('--sam-topk', default=0.0, type=float,
                        help='Keep only topk entries')
    group.add_argument('--sam-global-sparsity', action='store_true',
                        help='Use global sparsity for SAM')

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                        help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    group.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    # Distributed training params
    group = parser.add_argument_group('Distributed training parameters')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--timeout', default=1800, type=int,
                        help='url used to set up distributed training')
    parser.add_argument('--static_graph', action='store_true',
                        help='Whether to use static graph in DDP')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='Find unused parameters in DDP')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Logging params
    group = parser.add_argument_group('Logging parameters')
    group.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    group.add_argument('--log-sparsity', action='store_true', default=False,
                        help='Log sparsity distribution at pruning step')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    group.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    group.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
    group.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    group.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    group.add_argument('--output', default='./output', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    group.add_argument('--save-last', action='store_true', default=False,
                        help='Whether to save the last state of the model')
    group.add_argument('--save-freq', type=int, default=-1, metavar='N',
                        help='checkpointing frequency (default: no saving epoch checkpoints)')

    config_args, remaining_args = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            parser.set_defaults(**config)
    '''
    The main arg parser parses the rest of the args, the usual
    defaults will have been overridden if config file specified
    '''
    args = parser.parse_args(remaining_args)
    # Cache the args as a text string to save them in the output dir later
    text_args = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, text_args



# from timm.utils import setup_default_logging

if __name__ == '__main__':
    setup_default_logging()
    # parse args
    args, text_args = parse_args()
    # cudnn in benchmark mode
    torch.backends.cudnn.benchmark = True
    # init main logger
    main_logger = logging.getLogger('train')
    # init distributed mode
    init_distributed(args, main_logger)
    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.log_wandb:
        if has_wandb and args.rank == 0:
            wandb.init(
                project=f"SparseTraining|{args.model}", 
                name=f"{args.model}/{args.experiment}",
                config=args
            )
    elif args.rank == 0: 
        main_logger.warning(
            "You've requested to log metrics to wandb but package not found. "
            "Metrics not being logged to wandb, try `pip install wandb`"
        )

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
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
        args.num_classes = model.num_classes
    
    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if args.local_rank == 0:
        main_logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = args.aug_splits
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'

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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            main_logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
            )

    # run with torchscript
    if args.torchscript:
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)
    # init optimizer
    if args.sam:
        optimizer = create_sam_optimizer(model, args)
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # init AMP
    grad_scaler, amp_autocast = setup_amp(args, main_logger)

    # in case of resumed training
    start_epoch = 0
    if args.resume:
        # load model checkpoint
        load_model(args.resume, model, fix_data_parallel=True)
        if args.local_rank == 0:
            main_logger.info(f'Loading model from checkpoint {args.resume}')
        # load optimizer
        if not args.no_resume_opt:
            if args.local_rank == 0:
                main_logger.info(f'Loading optimizer from checkpoint {args.resume}')
            load_optimizer(args.resume, optimizer, map_location=args.device)
            if args.amp:
                load_loss_scaler(args.resume, grad_scaler)
        # load last epoch
        epoch = load_epoch(args.resume, map_location=args.device)
        if args.local_rank == 0:
            main_logger.info(f'Starting training from {epoch} epoch')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, 
            decay=args.model_ema_decay, 
            device='cpu' if args.model_ema_force_cpu else None
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
    
    # setup distributed training
    if args.distributed:
        if args.local_rank == 0:
            main_logger.info("Using native Torch DistributedDataParallel.")
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            broadcast_buffers=not args.no_ddp_bb,
            find_unused_parameters=args.find_unused_parameters
        )
        # NOTE: EMA model does not need to be wrapped by DDP

    # init scheduler
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # create the train and valid datasets
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

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        train_dataset = AugMixDataset(train_dataset, num_splits=num_aug_splits)

    valid_dataset = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split=args.val_split, 
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size
    )

    mixup_fn, collate_fn = setup_mixup(args)

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
        worker_seeding=args.worker_seeding,
    )

    valid_loader = create_loader(
        valid_dataset,
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
        pin_memory=args.pin_mem,
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
            distributed=False,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem
        )

        def mfac_data_generator(device=args.device, **kwargs):
            while True:
                for input, target in loader_mfac:
                    input, target = input.to(device), target.to(device)
                    yield [input], {}, target

    train_loss_fn, valid_loss_fn = setup_loss_fn(args)

    #########################
    # Setup SparseML manager
    ############$############

    manager_kw = {}
    if args.mfac_loader:
        manager_kw['grad_sampler'] = {
            'data_loader_builder' : mfac_data_generator,
            'loss_function' : valid_loss_fn
        }

    manager = ScheduledModifierManager.from_yaml(args.sparseml_recipe)
    # wrap optimizer  
    if args.amp:
        grad_scaler = manager.modify(
            model, 
            optimizer, 
            steps_per_epoch=len(train_loader), 
            wrap_optim=grad_scaler,
            epoch=start_epoch,
            **manager_kw
        ) 
    else:
        optimizer = manager.modify(
            model, 
            optimizer, 
            steps_per_epoch=len(train_loader), 
            epoch=start_epoch,
            **manager_kw
        ) 

    # override timm scheduler
    if any("LearningRate" in str(modifier) for modifier in manager.modifiers):
        lr_scheduler = None
        if manager.max_epochs:   
            num_epochs = manager.max_epochs
        if args.local_rank == 0:
            main_logger.info("Disabling timm LR scheduler, managing LR using SparseML recipe")
            main_logger.info(f"Overriding max_epochs to {num_epochs} from SparseML recipe")

    # set default
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch  = None
    decreasing  = True if 'loss' in eval_metric else False
    # set comparison
    if decreasing:
        is_better = lambda x, y: x < y 
    else:
        is_better = lambda x, y: x > y
    # setup directories for experiment
    output_dir = f'{args.output}/{args.experiment}'
    if args.local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/args.yaml', 'w') as f:
            f.write(text_args)

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
                lr_scheduler=lr_scheduler, 
                output_dir=output_dir,
                amp_autocast=amp_autocast, 
                grad_scaler=grad_scaler, 
                model_ema=model_ema, 
                mixup_fn=mixup_fn,
                logger=main_logger,
            )
            
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    main_logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(
                model, 
                valid_loader, 
                valid_loss_fn, 
                args, 
                amp_autocast=amp_autocast,
                logger=main_logger,
            )
            # get current mean sparsity
            mean_sparsity = get_current_sparsity(manager, epoch)
            # evaluate EMA
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                # do not evaluate EMA if in sparse stage
                if mean_sparsity == 0.0:
                    ema_eval_metrics = validate(
                        model_ema.module, 
                        valid_loader, 
                        valid_loss_fn, 
                        args, 
                        amp_autocast=amp_autocast, 
                        log_suffix=' (EMA)'
                    )
                    eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                lr_param = [param_group['lr'] for param_group in optimizer.param_groups]
                lr_m = sum(lr_param) / len(lr_param)
                # get current lr
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=f'{output_dir}/summary.csv',
                    write_header=(epoch == start_epoch),
                    log_wandb=args.log_wandb,
                    lr=lr_m,
                    sparsity=mean_sparsity
                )

            if args.local_rank == 0:
                if epoch % args.save_freq == 0 and args.save_freq > 0:
                    save_model(
                        path=f'{output_dir}/{args.model}_epoch={epoch}.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=grad_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )

                # log mean sparsity
                main_logger.info(f'Mean sparsity: {mean_sparsity:.4f}')

                if args.log_sparsity and is_update_epoch(manager, epoch):
                    sparsity_info = json.loads(get_sparsity_info(model))
                    with open(f'{output_dir}/sparsity_distribution_epoch={epoch}.json', 'w') as outfile:
                        json.dump(sparsity_info, outfile)
                    # reset current best metric
                    best_metric = None
                # save best checkpoint
                if best_metric is None or is_better(eval_metrics[eval_metric], best_metric):
                    best_metric = eval_metrics[eval_metric]
                    best_epoch = epoch
                    save_model(
                        path=f'{output_dir}/{args.model}_sparsity={mean_sparsity:.2f}_best.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=grad_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )
                    main_logger.info(f'New best model for sparsity {mean_sparsity:.2f} on epoch {epoch} with accuracy {best_metric:.4f}')
                
    except KeyboardInterrupt:
        pass
    else:
        pass
    # finalize manager
    manager.finalize(model)
    if args.local_rank == 0:
        if best_metric is not None:
            main_logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        main_logger.info('Training completed. Have a nice day!')
        wandb.finish()
