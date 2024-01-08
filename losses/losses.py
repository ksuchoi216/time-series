import torch
import torch.nn as nn
import numpy as np


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, predictions, targets):
        value = (
            torch.abs(targets - predictions)
            / ((torch.abs(targets) + torch.abs(predictions)) / 2)
            * 100
        )
        smape = torch.mean(value)
        return smape


def SMAPE(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 200


def weighted_mse(alpha=1):
    def weighted_mse_fixed(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * alpha, 2.0)
        return grad, hess

    return weighted_mse_fixed


def build_loss_fn(lossfn_name):
    loss_fn = None
    if lossfn_name == "MSE":
        loss_fn = nn.MSELoss()

    return loss_fn
