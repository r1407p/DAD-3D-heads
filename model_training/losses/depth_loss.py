from torch import nn, Tensor
from typing import List
import numpy as np

__all__ = ["DepthLoss"]
losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}

class DepthLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()

    def forward(self, predicted: List[Tensor], target: List[Tensor]) -> Tensor:
        # only calculate that depth is not inf
        valid_mask = target != np.inf
        predicted = predicted[valid_mask]
        target = target[valid_mask]
        return self.criterion(predicted, target)