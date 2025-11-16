from torch import nn
import torch

__all__ = ["BCELoss"]

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce_loss(pred.sigmoid(), target)