from multitask_perception.solver import registry
from multitask_perception.solver.adam_optimizer import ADAM_optimizer
from multitask_perception.solver.cosine_scheduler import CosineLR
from multitask_perception.solver.sgd_optimizer import SGD_optimizer


__all__ = ["make_optimizer", "SGD_optimizer", "ADAM_optimizer", "CosineLR"]


def make_optimizer(cfg, model, lr=None):
    return registry.SOLVERS[cfg.SOLVER.NAME](cfg, model, lr)
