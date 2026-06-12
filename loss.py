import torch
import torch.nn as nn


class StatisticalPathLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def weighted_wmse_loss(self, input, target, weight, reduction="mean"):
        loss = (weight.unsqueeze(-1).float() * (target - input)) ** 2

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss


TransformerLoss = StatisticalPathLoss
