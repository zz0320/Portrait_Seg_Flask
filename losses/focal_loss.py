# -*- coding: utf-8 -*-
"""
# @file name  : focal_loss.py
# @author     : zz0320
# @date       : 2022-4-21
# @brief      : 调用crossEntropy的focal_loss 只能完成0/1的loss one-hot
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class FocalLoss(nn.Module):
    """
    实现focal loss的代码 值得注意的是alpha可以用原先交叉熵损失函数里面的weight来代替
    然后再自己实现一个pt 设定一个gamma
    """

    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)  # 因CE中取了log 所以要exp回来  得到概率
        loss = ((1 - pt) ** self.gamma) * logpt  # (1 - pt) ^ gamma | weight CE自带 也就是alpha
        if self.size_average:  # 对一个batch里面的数据的loss是求和 还是 取平均
            return loss.mean()
        return loss.sum()


if __name__ == '__main__':
    target = torch.tensor([1], dtype=torch.long)
    gamma_lst = [0, 0.5, 1, 2, 5]
    loss_dict = {}
    for gamma in gamma_lst:
        focal_loss_func = FocalLoss(gamma=gamma)
        loss_dict.setdefault(gamma, [])

        for i in np.linspace(0.5, 10.0, num=30):
            outputs = torch.tensor([[5, i]], dtype=torch.float) # 制造不同概率的输出
            prob = F.softmax(outputs, dim=1) # Pytorch的CE自带softmax 所以想要知道具体的概率 需要用softmax
            loss = focal_loss_func(outputs, target)
            loss_dict[gamma].append((prob[0, 1].item(), loss.item()))

    for gamma, value in loss_dict.items():
        x_prob = [prob for prob, loss in value]
        y_loss = [loss for prob, loss in value]
        plt.plot(x_prob, y_loss, label="gamma" + str(gamma))

    plt.title("Focal Loss")
    plt.xlabel("probability of ground truth class")
    plt.ylabel("loss")
    plt.legend()
    plt.show()





