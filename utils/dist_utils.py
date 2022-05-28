import os
import sys
import torch
import torch.distributed as dist
# import NativeDDP
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex.parallel import DistributedDataParallel as ApexDDP
    has_apex = True
except ImportError:
    has_apex = False


__all__ = ['init_distributed_mode', 'setup_distributed']


def init_distributed_mode(args, _logger):
    args.rank        = 0
    args.distributed = False
    args.world_size  = 1
    # check if in distributed mode
    if 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        args.device = 'cuda:%d' % args.rank
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        if args.rank == 0:
            _logger.info(f'Training in distributed mode on {args.world_size} GPUs.')
        # greeting from workers
        _logger.info(f'Worker [{args.rank}]/[{args.world_size}] ready!')
    elif torch.cuda.is_available():
        args.device = 'cuda:0'
        _logger.info('Training with a single process on 1 GPUs.')
    else:
        _logger.info('Do not torture your poor CPU.')
        sys.exit(1)
    assert args.rank >= 0
    

def setup_distributed(model, args, _logger):
    # setup distributed training
    if args.distributed:
        if args.use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP
    return model
