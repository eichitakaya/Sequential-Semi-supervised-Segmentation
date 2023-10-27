"""
ある時点でのモデルと1枚の推論用画像を受け取り，test_time_augmentationを行う関数
imageはtensor型のまま扱いたい

input: model, image
output: 11枚の推論結果の平均

augmentationの種類
0. なし
1. right shift
2. left shift
3. + 10 degree rotation
4. - 10 degree rotation
5. gaussian noise
6. gaussian blur
7. high contrast
8. low contrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch_networks as networks
import random
import augmentation.transform_functions as transform_functions


def right_shift(image_tensor, target_tensor, shift_pixels, p=0.5):
    # 0.5の確率で右にshift_pixelsだけずらす
    if random.random() < p:
        image_tensor, target_tensor = transform_functions.right_shift(image_tensor, target_tensor, shift_pixels, inference=False)
    return image_tensor, target_tensor

def left_shift(image_tensor, target_tensor, shift_pixels, p=0.5):
    # 0.5の確率で左にshift_pixelsだけずらす
    if random.random() < p:
        image_tensor, target_tensor = transform_functions.left_shift(image_tensor, target_tensor, shift_pixels, inference=False)
    return image_tensor, target_tensor

def right_rotation(image_tensor, target_tensor, theta, p=0.5):
    # 0.5の確率で右に10度回転
    if random.random() < p:
        image_tensor, target_tensor = transform_functions.right_rotation(image_tensor, target_tensor, theta, inference=False)
    return image_tensor, target_tensor

def left_rotation(image_tensor, target_tensor, theta, p=0.5):
    # 0.5の確率で左に10度回転
    if random.random() < p:
        image_tensor, target_tensor = transform_functions.left_rotation(image_tensor, target_tensor, theta, inference=False)
    return image_tensor, target_tensor

def gaussian_noise(image_tensor, p=0.5):
    # 0.5の確率でガウシアンノイズを加える
    if random.random() < p:
        image_tensor = transform_functions.gaussian_noise(image_tensor)
    return image_tensor

def gaussian_blur(image_tensor, p=0.5):
    # 0.5の確率でガウシアンブラーをかける
    if random.random() < p:
        image_tensor = transform_functions.gaussian_blur(image_tensor)
    return image_tensor

def high_contrast(image_tensor, p=0.5):
    # 0.5の確率でコントラストを上げる
    if random.random() < p:
        image_tensor = transform_functions.high_contrast(image_tensor)
    return image_tensor

def low_contrast(image_tensor, p=0.5):
    # 0.5の確率でコントラストを上げる
    if random.random() < p:
        image_tensor = transform_functions.low_contrast(image_tensor)
    return image_tensor

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
        # 順番にaugmentation関数を通していく
        #image, target = right_shift(image, target, shift_pixels=10)
        #image, target = left_shift(image, target, shift_pixels=10)
        image, target = right_rotation(image, target, theta=10)
        image, target = left_rotation(image, target, theta=10)
        image = gaussian_noise(image)
        image = gaussian_blur(image)
        image = high_contrast(image)
        image = low_contrast(image)
        
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