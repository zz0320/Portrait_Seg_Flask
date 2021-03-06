# -*- coding: utf-8 -*-
"""
# @file name  : cross_entropy.py
# @author     : zz0320
# @date       : 2022-4-21
# @brief      : 浮点数形式下的CE Loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CrossEntropyLossFloat(nn.Module):
    """
    浮点类型的CE实现，适用于标签是连续变量
    pytorch的原生CE 只适用于One-hot
    """
    def __init__(self):
        super(CrossEntropyLossFloat, self).__init__()

    def forward(self, inputs, target):
        assert inputs.shape == target.shape, "inputs.shape & target.shape must be equal! but got" \
                                             "inputs.shape:{}, target.shape:{}".format(inputs.shape, target.shape)
        log_prob = F.log_softmax(inputs, dim=1)          # 注意是要在分类的维度进行softmax
        return (-target * log_prob).sum(dim=1).mean()   # log_p * Q 然后再相加

if __name__ == '__main__':
    b, c, h, w = 1, 2, 2, 2
    label_index = torch.randint(1, c, (b, h, w))

    fake_out = torch.tensor([[[0.7311, 0.7311], [0.7311, 0.7311], [0.7311, 0.7311]],
                             [[0.2689, 0.2689], [0.2689, 0.2689], [0.2689, 0.2689]]], dtype=torch.float32
                            ).unsqueeze_(0)
    fake_label = torch.tensor([[[1, 1], [1, 1], [1, 1]],
                               [[0, 0], [0, 0], [0, 0]]], dtype=torch.float32
                              ).unsqueeze_(0)
    loss_f = CrossEntropyLossFloat()
    loss = loss_f(fake_out, fake_label)
    print(fake_out.shape, fake_label.shape, loss)


    fake_label = torch.tensor([0], dtype=torch.long)
    fake_out = torch.tensor([0.7311, 0.2689], dtype=torch.float32).unsqueeze_(0)
    loss_ce_f = nn.CrossEntropyLoss()
    loss_ce = loss_ce_f(fake_out, fake_label)
    print(fake_out.shape, fake_label.shape, loss_ce)
