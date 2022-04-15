import torch

from .optimizers import SAM, TopKSAM


def create_sam_optimizer(model, args):
    if args.opt == 'sgd':
        optimizer = SAM(
            model.parameters(), 
            torch.optim.SGD,
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.opt == 'adam':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = SAM(
            model.parameters(), 
            torch.optim.Adam,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps
        )

    elif args.opt == 'adamw':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = SAM(
            model.parameters(), 
            torch.optim.AdamW,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps
        )
    else:
        raise NotImplementedError("Unfortunately this kind of optimizer is not supported")
    return optimizer


def create_topk_sam_optimizer(model, args):
    if args.opt == 'sgd':
        optimizer = TopKSAM(
            model.parameters(), 
            torch.optim.SGD,
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            topk=args.sam_topk,
            global_sparsity=args.sam_topk_global
        )
    elif args.opt == 'adam':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = TopKSAM(
            model.parameters(), 
            torch.optim.Adam,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps,
            topk=args.sam_topk,
            global_sparsity=args.sam_topk_global
        )

    elif args.opt == 'adamw':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = TopKSAM(
            model.parameters(), 
            torch.optim.AdamW,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps,
            topk=args.sam_topk,
            global_sparsity=args.sam_topk_global
        )
    else:
        raise NotImplementedError("Unfortunately this kind of optimizer is not supported")
    return optimizer