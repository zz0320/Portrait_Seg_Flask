# -*- coding: utf-8 -*-
"""
# @file name  : dice_loss.py
# @author     : zz0320
# @date       : 2022-4-21
# @brief      : dice_loss
"""

import torch.nn as nn
from torch import torch



class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon


    def forward(self, predict, target):
        assert predict.size() == target.size()
        num = predict.size(0)

        # pred不需要转bool变量
        # 这里是soft dice loss 直接使用预测概率而不是使用阈值 或 将它们转换为二进制
        pred = torch.sigmoid(predict).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum() # 利用预测值与标签值相乘当交集
        union = (pred + targ).sum() # 相加当并集

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score

if __name__ == '__main__':

    fake_out = torch.tensor([7, 7, -5, -5], dtype=torch.float32) # pred会有一个sigmoid 所以结果loss并不高 很小
    fake_label = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss_f = DiceLoss()
    loss = loss_f(fake_out, fake_label)
    print(loss)
