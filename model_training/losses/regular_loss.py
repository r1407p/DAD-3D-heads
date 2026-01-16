from torch import nn, Tensor

class RegularLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        predicted = predicted[0]
        return predicted.norm(p=2, dim=1).mean()