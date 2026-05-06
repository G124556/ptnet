import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_lr_init = warmup_lr_init
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + cosine_factor * (base_lr - self.min_lr)
            for base_lr in self.base_lrs
        ]
