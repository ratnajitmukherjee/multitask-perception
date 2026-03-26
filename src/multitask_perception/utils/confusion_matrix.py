"""
This Confusion Matrix class is used to calculate IoU for each class as well as mIoU for all segmentation heads.
This is like classification where IoU per class per instance is computed and then averaged per class and then
again averaged over all the classes. Therefore, this gives you IoU per class and then mIoU.
"""
import torch


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        epsilon = 1e-6
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / (h.sum(1) + epsilon)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + epsilon)
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )

    def get_metrics(self):
        acc_global, acc, iu = self.compute()
        acc_global = acc_global.item()
        mean_iou = iu.mean().item() * 100
        iu = (iu * 100).tolist()
        return acc_global, mean_iou, iu
