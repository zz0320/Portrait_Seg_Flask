# -*- coding: utf-8 -*-
"""
# @file name  : focal_loss_binary.py
# @author     : zz0320
# @date       : 2022-4-21
# @brief      : 不调用crossEntropy的focal_loss -> 能完成0~1的loss 非one-hot 浮点类型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def binary_focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = .25,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(), target.size()))

    probs = torch.sigmoid(input)
    loss_tmp = - alpha * torch.pow((1. - probs + eps), gamma) * target * torch.log(probs + eps) \
                - (1 - alpha) * torch.pow(probs + eps, gamma) * (1. - target) * torch.log(1. - probs + eps)
    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, self.eps)
    # alpha 是哪一科
    # gamma 是刷题战术 专注于困难样本

if __name__ == '__main__':
    N = 1  # num_classes
    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
    loss_f = BinaryFocalLossWithLogits(**kwargs)
    input = torch.randn(1, N, 3, 5, requires_grad=True)
    target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    target.unsqueeze_(dim=1)
    loss = loss_f(input, target)
    print(input.shape, target, target.shape, loss)

    num_classes = 1
    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
    logits = torch.tensor([[[[6.325]]], [[[5.26]]], [[[87.49]]]])
    labels = torch.tensor([[[[1.]]], [[[1.]]], [[[0.]]]])
    loss = binary_focal_loss_with_logits(logits, labels, **kwargs)
    print(logits.shape, labels.shape, loss)