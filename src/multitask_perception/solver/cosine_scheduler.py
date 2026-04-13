import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from multitask_perception.solver import registry


@registry.SCHEDULERS.register("CosineLR")
def CosineLR(cfg, optimizer, milestones):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.SOLVER.MAX_ITER, eta_min=0, last_epoch=-1
    )
    for param_group in scheduler.optimizer.param_groups:
        param_group["lr"] = param_group["lr"] / cfg.DATALOADER.BATCH_SIZE
        param_group["weight_decay"] = (
            param_group["weight_decay"] * cfg.DATALOADER.BATCH_SIZE
        )
    return scheduler


@registry.SCHEDULERS.register("WarmupCosineLR")
def WarmupCosineLR(cfg, optimizer, milestones):
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer=optimizer,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        eta_min=cfg.SOLVER.MIN_LR,
        T_0=cfg.SOLVER.MAX_ITER,
        T_mult=1,
        T_up=cfg.SOLVER.WARMUP_ITERS,
        gamma=1.0,
    )
    for param_group in scheduler.optimizer.param_groups:
        param_group["lr"] = param_group["lr"] / cfg.DATALOADER.BATCH_SIZE
        param_group["weight_decay"] = (
            param_group["weight_decay"] * cfg.DATALOADER.BATCH_SIZE
        )
    return scheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer,
        weight_decay,
        batch_size,
        eta_min,
        T_0,
        T_mult=1,
        T_up=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer:
            eta_min: Minimum learning rate learning rate.
            T_0: Number of steps of the first cycle.
            T_mult: A factor increases T_{i} after a restart. Default: 1.
            T_up: Number of steps of a linear warmup.
            gamma: A factor decreases eta_max_{i} after a restart. Default: 1.
            last_epoch: The index of last epoch. Default: -1.
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected positive integer T_up, but got {T_up}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min / batch_size
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        super().__init__(optimizer, last_epoch)
        self.eta_max = [base_lr / self.batch_size for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group["weight_decay"] = self.weight_decay * self.batch_size

        if self.T_cur == -1:
            return [self.eta_min for _ in self.eta_max]
        elif self.T_cur < self.T_up:
            return [
                (base_lr - self.eta_min) * self.T_cur / self.T_up + self.eta_min
                for base_lr in self.eta_max
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.eta_max
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = [
            base_lr * (self.gamma**self.cycle) / self.batch_size
            for base_lr in self.base_lrs
        ]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
