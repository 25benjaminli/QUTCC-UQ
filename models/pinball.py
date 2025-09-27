import torch
import torch.nn as nn

class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
    
    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class BatchedPinballLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape
        assert quantiles.ndim == 1, "Quantiles tensor must be 1D"
        assert quantiles.shape[0] == pred.shape[0], "Quantiles tensor must have the same batch size as output"
        assert torch.all((0 < quantiles) & (quantiles < 1)), "Quantiles must be in the range (0, 1)"

        quantiles = quantiles.view([pred.shape[0]] + [1] * (pred.ndim - 1))
        error = pred - target

        loss = torch.where(error >= 0,
                           (1 - quantiles) * (error),
                           quantiles * -error)

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean': 
            loss = loss.mean()
        return loss

