""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

from .scheduler import Scheduler


class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 min_lr=0
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.min_lr = min_lr 
        
    def _get_lr(self, t):
        #an optimizer can have multiple parameter groups, each potentially having a different learning rate, e.g. optimizer = torch.optim.SGD([
        #     {'params': model.base.parameters(), 'lr': 0.01},
        #     {'params': model.classifier.parameters(), 'lr': 0.1}
        # ])
        #thats why here we return a  list of lr (for all bases(parts of model))
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [max(v * (self.decay_rate ** (t // self.decay_t)), self.min_lr) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
        

class TwoStepLRScheduler(Scheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr=0.001,
                 second_lr=0.0001,
                 warmup_epochs=20,
                #  t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 min_lr=0
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)
        self.warmup_epochs=warmup_epochs
        self.warmup_lr_init = lr
        self.second_lr = second_lr
        self.min_lr = min_lr 
        
    def _get_lr(self, t):
        if t < self.warmup_epochs:
            lrs = [  self.warmup_lr_init  for _ in self.base_values]
        else:
            lrs = [max(self.second_lr , self.min_lr) for _ in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        return self._get_lr(epoch)
   

    def get_update_values(self, num_updates: int):
        return self._get_lr(num_updates)

