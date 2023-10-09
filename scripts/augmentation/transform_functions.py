import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_networks as networks
import torchvision.transforms.functional as TF
import torchvision


# inference=Trueのときは，推論時のみの処理を行う．
def vertical_flip(image_tensor, target_tensor=None, inference=True):
    # inference=Falseのときは，target_tensorにも同様の処理を行う．
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 左右反転し，推論
        image = image_tensor.flip(3)
        target = target_tensor.flip(3)
        return image, target
    else:
        # 左右反転し，推論
        image = image_tensor.flip(3)
        return image

def horizontal_flip(image_tensor, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 上下反転し，推論
        image = image_tensor.flip(2)
        target = target_tensor.flip(2)
        return image, target
    else:
        # 上下反転し，推論
        image = image_tensor.flip(2)
        return image

def right_shift(image_tensor, shift_pixels, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # shift_pixelsだけ右にずらす
        image = image_tensor[:,:,:,:-shift_pixels]
        target = target_tensor[:,:,:,:-shift_pixels]
        # ずらした分だけ左側を0で埋める
        image = F.pad(image, (shift_pixels,0,0,0), mode="constant", value=0)
        target = F.pad(target, (shift_pixels,0,0,0), mode="constant", value=0)
        return image, target
    else:
        # shift_pixelsだけ右にずらす
        image = image_tensor[:,:,:,:-shift_pixels]
        # ずらした分だけ左側を0で埋める
        image = F.pad(image, (shift_pixels,0,0,0), mode="constant", value=0)
        return image

def left_shift(image_tensor, shift_pixels, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # shift_pixelsだけ左にずらす
        image = image_tensor[:,:,:,shift_pixels:]
        target = target_tensor[:,:,:,shift_pixels:]
        # ずらした分だけ右側を0で埋める
        image = F.pad(image, (0,shift_pixels,0,0), mode="constant", value=0)
        target = F.pad(target, (0,shift_pixels,0,0), mode="constant", value=0)
        return image, target
    else:
        # shift_pixelsだけ左にずらす
        image = image_tensor[:,:,:,shift_pixels:]
        # ずらした分だけ右側を0で埋める
        image = F.pad(image, (0,shift_pixels,0,0), mode="constant", value=0)
        return image

def right_rotation(image_tensor, theta, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # theta度右に回転
        image = image_tensor
        image = TF.rotate(image, -theta)
        target = target_tensor
        target = TF.rotate(target, -theta)
        return image, target
    else:
        # theta度右に回転
        image = image_tensor
        image = TF.rotate(image, -theta)
        return image

def left_rotation(image_tensor, theta, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # theta度左に回転
        image = image_tensor
        image = TF.rotate(image, theta)
        target = target_tensor
        target = TF.rotate(target, theta)
        return image, target
    else:
        # theta度左に回転
        image = image_tensor
        image = TF.rotate(image, theta)
        return image
    
def gaussian_noise(image_tensor):
    # ガウシアンノイズを加える
    image = image_tensor
    noise = torch.randn(image.size())
    print(noise)
    image = image + noise
    return image

def gaussian_blur(image_tensor, kernel_size=5, sigma=2):
    # ガウシアンブラーをかける
    image = image_tensor
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    image = blur(image)
    return image

def high_contrast(image_tensor, alpha = 1.05):
    # コントラストを変える(単にalphaをかけるだけでよい)
    image = image_tensor * alpha
    return image

def low_contrast(image_tensor, alpha = 0.95):
    # コントラストを変える(単にalphaをかけるだけでよい) 
    image = image_tensor * alpha
    return image
    