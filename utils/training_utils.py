import csv
import time
import torch

# import wandb
try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

from collections import OrderedDict
from contextlib import suppress, nullcontext
# save image
from torchvision.utils import save_image
# timm imports
from timm.utils import AverageMeter, dispatch_clip_grad, reduce_tensor, accuracy
from timm.models import model_parameters

from .batchnorm_utils import enable_running_stats, disable_running_stats


def train_one_epoch(
    epoch, 
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    args,
    _logger,
    lr_scheduler=None, 
    output_dir=None, 
    amp_autocast=suppress,
    loss_scaler=None, 
    model_ema=None, 
    mixup_fn=None
):
    # define closure for SAM optimizer
    def closure():
        loss = loss_fn(model(input), target)
        loss.backward()
        return loss

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m  = AverageMeter()
    losses_m     = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # turn of batch norm stats if needed
        if args.sam:
            enable_running_stats(model)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            sync_context = model.no_sync() if args.sam else nullcontext
            with sync_context:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode    
                    )

            if args.sam:
                disable_running_stats(model)
                optimizer.step(closure)
            else:
                optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        lr = 0.0
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m)
                    )

                if args.save_images and output_dir:
                    save_image(
                        input,
                        fp=f'{output_dir}/train-batch-{batch_idx}.jpg',
                        padding=0,
                        normalize=True
                    )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg), ('lr', lr)])


def valid_one_epoch(
    model, 
    loader, 
    loss_fn, 
    args, 
    _logger,
    amp_autocast=suppress, 
    log_suffix=''
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics


def log_on_epoch(
    epoch, 
    train_metrics: OrderedDict, 
    eval_metrics: OrderedDict,
    filename: str, 
    sparsity: float = 0.0,
    log_wandb = False,
    write_header = False
):
    data = OrderedDict(epoch=epoch)
    # do not add 'lr' to train
    if train_metrics.get('lr'):
        data.update([('lr', train_metrics['lr'])])
    data.update([('train_' + k, v) for k, v in train_metrics.items() if k != 'lr'])
    data.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    data.update([('sparsity', sparsity)])

    if log_wandb:
        wandb.log(data)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=data.keys())
        if write_header: 
            dw.writeheader()
        dw.writerow(data)
