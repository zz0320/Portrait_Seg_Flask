# -*- coding: utf-8 -*-
"""
# @file name  : analysis_badcase.py
# @author     : zz0320
# @date       : 2022-05-10
# @brief      : 推理portrait数据，并记录mIoU，用于分析badcase
"""

import numpy as np
import os
import torch
import cv2
from tools.predictor import Predictor
from tools.evalution_segmentaion import eval_semantic_segmentation


def get_img_list(root):
    """
    读取portrait2000的（图片路径，标签路径）
    :param root:
    :return:
    """
    file_list = os.listdir(root)
    file_list = list(filter(lambda x: x.endswith("_matte.png"), file_list))
    label_lst = [os.path.join(root, name) for name in file_list]
    img_lst = [string.replace("_matte.png", ".png") for string in label_lst]
    data_lst = [(path_img, path_label) for path_img, path_label in zip(img_lst, label_lst)]
    return data_lst


def process_label(path_label, threshold=0.78):
    msk_bgr = cv2.imread(path_label)  # 3个通道值是一样的 # 0-255，边界处是平滑的, 3个通道值是一样的
    msk_gray = cv2.cvtColor(msk_bgr, cv2.COLOR_BGR2GRAY)
    msk_gray_resize = cv2.resize(msk_gray, (in_size, in_size))
    msk_gray_resize = msk_gray_resize / 255.
    msk_gray_resize_binary = (msk_gray_resize > threshold).astype(np.int)
    return msk_gray_resize_binary


def get_iou(pred_mask_binary, label_binary):
    eval_metrix = eval_semantic_segmentation([pred_mask_binary], [label_binary], n_class=2)
    return eval_metrix["iou"][-1]


if __name__ == '__main__':

    set_name = "testing"   # training
    in_size = 512
    threshold = 0.78
    path_ckpt = r"/Users/kenton/Downloads/Terminal_download/results/05-04_00-23/checkpoint_best.pkl"
    # path_ckpt = r"G:\project_class_bak\results\seg_old_3\04-02_17-57-portrait-3w4-affine\checkpoint_best.pkl"
    # path_ckpt = r"F:\prj_class\bak\results\05-30_00-15\checkpoint_best.pkl"
    root_dir = r"/Users/kenton/Downloads/Deeplearning_dataset/Portrait-dataset-2000/{}".format(set_name)
    dir_name = os.path.dirname(path_ckpt)
    out_dir = os.path.join(os.path.dirname(path_ckpt), "{}_{}".format(set_name, in_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou_list = []

    # 获取img label path
    img_info_list = get_img_list(root_dir)
    # 初始化model
    predictor = Predictor(path_ckpt, device=device, tta=False)
    # for循环迭代
    for path_img, path_label in img_info_list:
        img_t, img_bgr = predictor.preprocess(path_img, in_size=in_size)  # 1.预处理
        _, pred_mask = predictor.predict(img_t)  # 2.推理获得mask;  pred_mask.shape == (h, w)
        out_img = predictor.postprocess(img_bgr, pred_mask, color="w")  # 3. 后处理，保存图片

        # 定义读label函数
        label_binary = process_label(path_label, threshold=threshold)
        pred_mask_binary = (pred_mask > threshold).astype(np.int)
        iou_single = get_iou(pred_mask_binary, label_binary)
        iou_list.append(iou_single)
        # 保存
        concat_img = np.concatenate([img_bgr, out_img], axis=1)
        name_img = "{:.3f}_".format(iou_single) + os.path.basename(path_img)
        path_out = os.path.join(out_dir, name_img)
        predictor.save_img(path_out, concat_img)

    print("mIoU: {}, length：{}".format(np.mean(iou_list), len(iou_list)))
