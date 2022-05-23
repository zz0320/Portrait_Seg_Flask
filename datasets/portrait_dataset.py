# -*- coding: utf-8 -*-
"""
# @file name  : camvid_dataset.py
# @author     : zz0320
# @date       : 2022-4-16
# @brief      : Portrait数据集的Dataset定义
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import random
import cv2


class PortraitDataset2000(Dataset):
    """
    Deep Automatic Portait Matting 2000 数据集读取
    """
    cls_num = 2
    names = ['bg', 'portrait']

    def __init__(self, root_dir, transform=None, in_size=224):
        """
        初始化
        :param root_dir:
        :param transform:
        :param in_size:
        """
        super(PortraitDataset2000, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.label_path_list = list()
        self.in_size = in_size
        # 获取mask的path
        self._get_img_path()

    def __getitem__(self, index):
        """
        主要是这个函数 难点在于标签如何进行处理「标签并不是0/255 而是连续的」
        不能直接使用交叉熵损失函数了 自带的CE 不可用了 可以自己重写一个浮点型的
        可以用binaryLoss
        模型的输出 也是0-255「0～1」 之间的值了 使用sigmoid函数
        :param index:
        :return:
        """
        # step1：读取样本，得到ndarray形式
        path_label = self.label_path_list[index]
        path_img = path_label[:-10] + ".png"
        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_rgb = cv2.imread(path_label)  # 3个通道值是一样的 # 0-255，边界处是平滑的, 3个通道值是一样的

        # step2: 图像预处理
        if self.transform:
            transformed = self.transform(image=img_rgb, mask=msk_rgb)
            img_rgb = transformed['image']
            msk_rgb = transformed['mask']  # transformed后仍旧是连续的

        # step3：处理数据成为模型需要的形式 读取进来看一眼是0～1 还是0～255
        img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
        img_chw_tensor = torch.from_numpy(img_rgb).float()

        msk_gray = msk_rgb[:, :, 0]  # hwc --> hw
        msk_gray = msk_gray / 255.  # [0,255] scale [0,1] 连续变量
        label_tensor = torch.tensor(msk_gray, dtype=torch.float)  # 标签输出为 0-1之间的连续变量 ，shape=(224, 224)

        return img_chw_tensor, label_tensor


    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):
        """
        读取图片的路径信息
        :return:
        """
        file_list = os.listdir(self.root_dir)
        file_list = list(filter(lambda x:x.endswith("_matte.png"), file_list))
        path_list = [os.path.join(self.root_dir, name) for name in file_list]
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception("\nroot_dir:{} is a empty dir! Please checkout your path to images!".format(self.root_dir))
        self.label_path_list = path_list

class PortraitDataset34427(Dataset):
    """
    Deep Automatic Portrait Matting  34427，数据集读取
    ├─clip_img
    │  └─1803151818
    │      └─clip_00000000
    └─matting
        └─1803151818
            └─matting_00000000
    """
    cls_num = 2
    names = ["bg", "portrait"]

    def __init__(self, root_dir, transform=None, in_size=224, ext_num=None):
        super(PortraitDataset34427, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.in_size = in_size
        self.ext_num = ext_num

        # 获取mask的path
        self._get_img_path()

        # 截取部分数据
        if self.ext_num:
            self.img_info = self.img_info[:self.ext_num]

    def __getitem__(self, index):

        path_img, path_label = self.img_info[index]

        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_bgra = cv2.imread(path_label, cv2.IMREAD_UNCHANGED)  # [0-255] 是matting 融合的图像，需要提取alpha
        msk_gray = msk_bgra[:, :, 3]

        if self.transform:
            transformed = self.transform(image=img_rgb, mask=msk_gray)
            img_rgb = transformed['image']
            msk_gray = transformed['mask']

        img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
        msk_gray = msk_gray/255.  # [0-1]连续变量

        label_out = torch.tensor(msk_gray, dtype=torch.float)
        img_chw_tensor = torch.from_numpy(img_rgb).float()

        return img_chw_tensor, label_out

    def __len__(self):
        return len(self.img_info)

    def _get_img_path(self):
        img_dir = os.path.join(self.root_dir, "clip_img")
        img_lst, msk_lst = [], []

        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if not file.endswith(".jpg"):
                    continue
                if file.startswith("._"):
                    continue
                path_img = os.path.join(root, file)
                img_lst.append(path_img)

        for path_img in img_lst:
            path_msk = path_img.replace("clip_img", "matting"
                                        ).replace("clip_0", "matting_0"
                                                  ).replace(".jpg", ".png")
            if os.path.exists(path_msk):
                msk_lst.append(path_msk)
            else:
                print("path not found: {}\n path_img is: {}".format(path_msk, path_img))

        if len(img_lst) != len(msk_lst):
            raise Exception("\nimg info Error, img can't match with mask. got {} img, but got {} mask".format(
                len(img_lst), len(msk_lst)))
        if len(img_lst) == 0:
            raise Exception("\nroot_dir:{} is a empty dir! Please checkout your path to images!".format(self.root_dir))

        self.img_info = [(i, m) for i, m in zip(img_lst, msk_lst)]
        random.shuffle(self.img_info)


if __name__ == "__main__":
    mat_dir = r'/root/datasets/Po'  # 大样本数据集位置
    por_dir = r"/root/datasets/Portrait-dataset-2000/training"  # 小样本数据集位置

    mat_set = PortraitDataset34427(mat_dir)
    por_set = PortraitDataset2000(por_dir)
    all_set = ConcatDataset([mat_set, por_set])
    # train_loader = DataLoader(all_set, batch_size=1, shuffle=True, num_workers=0)

    train_loader = DataLoader(por_set, batch_size=1, shuffle=True, num_workers=0)
    print(len(train_loader))
    for i, sample in enumerate(train_loader):
        img, label = sample

        print(img.shape, label.shape)
        print(img)
        print(label)



