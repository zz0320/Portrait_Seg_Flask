# -*- coding: utf-8 -*-
"""
# @file name  : portrait_config.py
# @author     : zz0320
# @date       : 2022-04-20
# @brief      : 人像分割训练参数配置
"""

import torch
import torchvision.transforms as transforms
from easydict import EasyDict
import albumentations as A
import cv2


cfg = EasyDict() # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.is_fusion_data = True
cfg.is_ext_data = False
cfg.ext_num = 1800

# loss_f 的选择
cfg.loss_type = "BCE"
# cfg.loss_type = "BCE&dice"
# cfg.loss_type = "dice"
# cfg.loss_type = "focal"

# warm up and cosine decay
cfg.is_warmup = True
cfg.warmup_epoch = 1
cfg.lr_final = 1e-5
cfg.lr_warmup_init = 0

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 50

# cfg.crop_size = (360, 480)

# 梯度剪裁
cfg.hist_grad = True
cfg.is_clip = False
cfg.clip_value = 0.2

# batch size
cfg.train_bs = 4
cfg.valid_bs = 2
cfg.workers = 16

# 学习率等 超参数配置
cfg.exp_lr = False # 采用指数下降
cfg.lr_init = 0.01
cfg.factor = 0.1
cfg.milestones = [25, 45]
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

# log打印间隔
cfg.log_interval = 10

cfg.bce_pos_weight = torch.tensor(0.75)  # 36/48 = 0.75  [36638126., 48661074.])   36/(36+48) = 0.42

# 输入尺寸最短边
cfg.in_size = 224

norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
norm_std = (0.5, 0.5, 0.5)

# transform
cfg.tf_train = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])

cfg.tf_valid = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])


