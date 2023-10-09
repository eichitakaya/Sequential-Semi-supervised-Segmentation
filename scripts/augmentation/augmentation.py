"""
ある時点でのモデルと1枚の推論用画像を受け取り，test_time_augmentationを行う関数
imageはtensor型のまま扱いたい

input: model, image
output: 11枚の推論結果の平均

augmentationの種類
0. なし
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
from PIL import Image
import torch_networks as networks
import random


def vertical_flip(image_tensor, target_tensor, p=0.5):
    # 0.5の確率で左右反転
    if random.random() < p:
        return image_tensor.flip(1), target_tensor.flip(1)
    else:
        return image_tensor, target_tensor

def horizontal_flip(image_tensor, target_tensor, p=0.5):
    # 0.5の確率で上下反転
    if random.random() < p:
        return image_tensor.flip(0), target_tensor.flip(0)
    else:
        return image_tensor, target_tensor

def rotate_90(image_tensor, p=0.5):
    # 90度回転
    return torch.rot90(image_tensor, 1, [2,3])

def rotate_180(image_tensor, p=0.5):
    # 180度回転
    return torch.rot90(image_tensor, 2, [2,3])

def rotate_270(image_tensor, p=0.5):
    # 270度回転
    return torch.rot90(image_tensor, 3, [2,3])

def augmentation(image_tensor, target_tensor):
    """_summary_

    Args:
        model (_type_): 任意のsegmentationモデル
        image_tensor (_type_): (1,1,H,W)のtensor
        method (_type_): "vote", "average"
    """
    # image_tensorと同じサイズのゼロ行列を作成
    augmented_tensor_image = torch.zeros_like(image_tensor)
    augmented_tensor_target = torch.zeros_like(target_tensor)
    
    for i in range(len(augmented_tensor_image)):
        image, target = image_tensor[i], target_tensor[i]
        image, target = horizontal_flip(image, target)
        augmented_tensor_image[i] = image
        augmented_tensor_target[i] = target
    
    return augmented_tensor_image, augmented_tensor_target

# 以下，テスト用
if __name__ == "__main__":
    # 以下ではPILを利用
    # takaya.jpgを読み込みtensorに変換
    image = Image.open("takaya.jpg")
    # グレースケールに変換
    image = image.convert("L")
    # 3チャンネルにそれぞれコピー
    image = np.array(image)
    image = np.stack([image, image, image], axis=2)
    # (3, H, W)に変換
    image = image.transpose(2,0,1)
    # torch.tensorに変換
    image = torch.tensor(image)
    result = augmentation(image, image)
    # resultの各チャンネルを画像として保存
    result = result[0].detach().numpy()
    result = result.astype(np.uint8)
    print(result.shape)
    for i in range(3):
        Image.fromarray(result[i]).save(f"result{i}.jpg")