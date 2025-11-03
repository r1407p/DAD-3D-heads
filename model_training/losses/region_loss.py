from torch import nn, Tensor
from typing import List

__all__ = ["RegionLoss"]
losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss, "bce": nn.BCELoss}

class RegionLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()

    def forward(self, predicted: List[Tensor], target: List[Tensor]) -> Tensor:
        return self.criterion(predicted, target)