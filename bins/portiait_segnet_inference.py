# -*- coding: utf-8 -*-
"""
# @file name  : portiait_segnet_inference.py
# @author     : zz0320
# @date       : 2022-04-17
# @brief      : 人像分割 + 前向推理代码 + 结果输出
"""

import time
import cv2
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch
import numpy as np
import albumentations as A
from models.build_BiSeNet import BiSeNet
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--path_checkpoint',
    default=r"/Users/kenton/Downloads/Terminal_download/results/04-20_21-04/checkpoint_best.pkl",
    help = 'path to your dataset')
parser.add_argument('--path_img', default=r"/Users/kenton/Downloads/deeplearning_dataset/Portrait-dataset-2000/testing/00002.png",
                    help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"/Users/kenton/Downloads/deeplearning_dataset",
                    help="path to your dataset")
args = parser.parse_args()



if __name__ == '__main__':

    in_size = 512  # 224， 448 ， 336 ， 1024
    norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
    norm_std = (0.5, 0.5, 0.5)
    # 图片预处理
    transform = A.Compose([
        A.Resize(width=in_size, height=in_size),
        A.Normalize(norm_mean, norm_std),
    ])
    img_bgr = cv2.imread(args.path_img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb, mask=img_rgb)
    img_rgb = transformed['image']
    img_tensor = torch.tensor(np.array(img_rgb), dtype=torch.float).permute(2, 0, 1)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    # 模型加载
    model = BiSeNet(num_classes=1, context_path="resnet101")
    checkpoint = torch.load(args.path_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 推理
    with torch.no_grad():
        # 因为 CPU/GPU 是异步的 所以需要同步一次
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            s = time.time()
            outputs = model(img_tensor)
            # 等待GPU运算完成 然后再算时间
            torch.cuda.synchronize()
        else:
            s = time.time()
            outputs = model(img_tensor)
        print("{:.4f}s".format(time.time() - s))
        outputs = torch.sigmoid(outputs).squeeze(1)
        pre_label = outputs.data.cpu().numpy()[0]  # 取batch的第一个

    # 结果展示
    background = np.zeros_like(img_bgr, dtype=np.uint8)
    background[:] = 255
    alpha_bgr = pre_label
    alpha_bgr = cv2.cvtColor(alpha_bgr, cv2.COLOR_GRAY2BGR)
    h, w, c = img_bgr.shape
    alpha_bgr = cv2.resize(alpha_bgr, (w, h))
    # fusion
    result = np.uint8(img_bgr * alpha_bgr + background * (1 - alpha_bgr))
    out_img = np.concatenate([img_bgr, result], axis=1)


    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("result", out_img)
    # cv2.waitKey()
    # 适配服务器代码更改
    plt.imshow(out_img)
    plt.show()


