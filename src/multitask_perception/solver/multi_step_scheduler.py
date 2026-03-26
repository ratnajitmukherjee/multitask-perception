"""
This is the default multi-step scheduler that is typically used for training the models. Consists of:
- Initial LR (which is achieved after the warmup iterations typically 500 / 1000
- Drops LR by gammma value (for example 0.1) i.e. step2_LR = Initial_LR * Gamma Value after reaching a certain number
of iterations (i.e. Milestone)
- Drops LR again by gamma value upon eaching the next milestone
------------------------
Usage example: Max iterations: 400K iterations, Milestone 1: 280K, Milestone 2: 360K iterations
Initial LR: 1e-3, Milestone 1: 1e-4, Milestone 2: 1e-5

If you want smaller steps you can change the gamma value > 0.1 (for example 0.25 -> reduces by 1/4) and put more
milestones.
-------------------------
"""
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler

from multitask_perception.solver import registry


@registry.SCHEDULERS.register("WarmupMultiStepLR")
def WarmupMultiStepLR(cfg, optimizer, milestones):
    return WarmupMultiStepLR(
        optimizer=optimizer,
        milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
    )


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
