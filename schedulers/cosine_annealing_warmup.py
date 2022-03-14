import math


class CosineAnnealingWarmupRestarts:

    def __init__(
        self, 
        start_epoch: int,
        end_epoch: int,
        T_max: int, 
        update_frequency: float,
        init_warmup: int,
        period_warmup: int,
        warmup_lr_mult: float,
        min_lr_mult: float,
        lr_odd_mult: float = 1.0,
        lr_even_mult: float = 1.0,
        **kwargs
    ):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.T_max = T_max
        self.update_frequency = update_frequency
        self.init_warmup = init_warmup
        self.period_warmup = period_warmup
        self.warmup_lr_mult = warmup_lr_mult
        self.min_lr_mult = min_lr_mult
        self.lr_odd_mult = lr_odd_mult
        self.lr_even_mult = lr_even_mult

        
    def __call__(self, epoch):
        # clamp epoch
        epoch = min(epoch, self.T_max)
        # compute cosine annealing factor
        cosine_factor = self.min_lr_mult + \
            0.5 * (1 - self.min_lr_mult) * (1 + math.cos(math.pi * epoch / self.T_max))
        # compute warmup factor
        odd_iter = False
        if epoch >= self.start_epoch:
            # whether is compressed or decompressed stage
            odd_iter = ((epoch - self.start_epoch) // self.update_frequency) % 2 == 0 \
                 if epoch < self.end_epoch else True
            # subtract the offset from last ac/dc change
            epoch = max(epoch % self.update_frequency, epoch - self.end_epoch)
            # compute warmup factor
            linear_factor = self.warmup_lr_mult \
                + (1 - self.warmup_lr_mult) * (min(epoch, self.period_warmup) / max(self.period_warmup, 1))
            # compute phase factor
            phase_factor = self.lr_odd_mult if odd_iter else self.lr_even_mult
        else:
            # compute warmup factor
            linear_factor = self.warmup_lr_mult \
                + (1 - self.warmup_lr_mult) * (min(epoch, self.init_warmup) / max(self.init_warmup, 1))
            # compute phase factor
            phase_factor = 1.0

        lr_mult = linear_factor * cosine_factor * phase_factor
        return lr_mult
        