# -*- coding: utf-8 -*-
"""
# @file name  : bisenet_camera.py
# @author     : zz0320
# @date       : 2022-05-10
# @brief      : bisenet 调用摄像头进行inference
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import time
import cv2
import torch
import numpy as np
from models.build_BiSeNet import BiSeNet
import albumentations as A
from tools.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path_checkpoint', default=r"/Users/kenton/Downloads/Terminal_download/results/05-04_00-23/checkpoint_best.pkl",
                    help="path to your dataset")
parser.add_argument('--path_img', default=r"/Users/kenton/Downloads/Deeplearning_dataset/Portrait-dataset-2000/training/00001.png",
                    help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"/Users/kenton/Downloads/Deeplearning_dataset/Portrait-dataset-2000",
                    help="path to your dataset")
args = parser.parse_args()

if __name__ == '__main__':

    in_size = 512
    path_checkpoint = r"/Users/kenton/Downloads/Terminal_download/results/05-04_00-23/checkpoint_best.pkl"
    predictor = Predictor(path_checkpoint, device, backone_name="resnet101", tta=False)  # 边界

    # video_path = r"F:\terrace1 multi-person tracking results, color features vs deep features.mp4"
    video_path = 0
    vid = cv2.VideoCapture(video_path)  # 0表示打开视频，1; cv2.CAP_DSHOW去除黑边 '+ cv2.CAP_DSHOW'

    while True:
        return_value, frame_bgr = vid.read()  # 读视频每一帧
        if not return_value:
            raise ValueError("No image!")
        # —————————————————————————————————————————————————————————————————— #
        img_t, img_bgr = predictor.preprocess(frame_bgr, in_size=in_size)
        _, pre_label = predictor.predict(img_t)
        result = predictor.postprocess(img_bgr, pre_label, color="w", hide=False)
        # —————————————————————————————————————————————————————————————————— #
        cv2.imshow("result", result)
        # waitKey，参数是1，表示延时1ms；参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停
        # ord: 字符串转ASCII 数值
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()               # release()释放摄像头
    cv2.destroyAllWindows()     # 关闭所有图像窗口