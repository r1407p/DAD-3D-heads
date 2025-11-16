from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidSmoothL1Masked(nn.Module):
    def __init__(self, beta: float = 1.0, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        pred = pred[0]
        mask = target[1]
        target = target[0]
        pred_sig = torch.sigmoid(pred)
        if mask is None:
            mask = torch.ones_like(target)
        diff = F.smooth_l1_loss(pred_sig, target, beta=self.beta, reduction="none")
        diff = diff * mask
        if self.reduction == "sum":
            return diff.sum()
        elif self.reduction == "mean":
            denom = mask.sum().clamp_min(self.eps)
            return diff.sum() / denom
        else:
            return diff
