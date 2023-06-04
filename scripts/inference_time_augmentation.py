"""
ある時点でのモデルと1枚の推論用画像を受け取り，test_time_augmentationを行う関数
input: model, image
output: 10枚の推論結果の平均

augmentationの種類
1. vertical flip
2. horizontal flip
3. 90 degree rotation
4. 180 degree rotation
5. 270 degree rotation
6. gaussian noise
7. gaussian blur
9. random brightness
10. random contrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def test_time_augmentation(model, image):
    return

def vertical_flip(model, image):
    # 反転し，推論
    img = image.flip(1)
    # 推論結果をさらに反転
    return image.flip(2)