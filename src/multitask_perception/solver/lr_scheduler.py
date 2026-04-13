"""
List of all schedules which can be used in the Modular Training framework. Default is MultiStep Scheduler.
"""
from multitask_perception.solver import registry
from multitask_perception.solver.cosine_scheduler import CosineLR
from multitask_perception.solver.multi_step_scheduler import WarmupMultiStepLR
from multitask_perception.solver.polynomial_scheduler import PolynomialLR


__all__ = [
    "make_lr_scheduler",
    "WarmupMultiStepLR",
    "PolynomialLR",
    "CosineLR",  # this also includes CosineAnnealingLR
]


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return registry.SCHEDULERS[cfg.SOLVER.LR_SCHEDULER](cfg, optimizer, milestones)
