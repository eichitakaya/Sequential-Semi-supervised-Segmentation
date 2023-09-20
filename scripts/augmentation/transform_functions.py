import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_networks as networks
import torchvision.transforms.functional as TF

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

def right_shift(image_tensor, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 5pixel右にずらす
        image = image_tensor[:,:,:,:-5]
        target = target_tensor[:,:,:,:-5]
        # ずらした分だけ左側を0で埋める
        image = F.pad(image, (5,0,0,0), mode="constant", value=0)
        target = F.pad(target, (5,0,0,0), mode="constant", value=0)
        return image, target
    else:
        # 5pixel右にずらす
        image = image_tensor[:,:,:,:-5]
        # ずらした分だけ左側を0で埋める
        image = F.pad(image, (5,0,0,0), mode="constant", value=0)
        return image

def left_shift(image_tensor, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 5pixel左にずらす
        image = image_tensor[:,:,:,5:]
        target = target_tensor[:,:,:,5:]
        # ずらした分だけ右側を0で埋める
        image = F.pad(image, (0,5,0,0), mode="constant", value=0)
        target = F.pad(target, (0,5,0,0), mode="constant", value=0)
        return image, target
    else:
        # 5pixel左にずらす
        image = image_tensor[:,:,:,5:]
        # ずらした分だけ右側を0で埋める
        image = F.pad(image, (0,5,0,0), mode="constant", value=0)
        return image

def right_rotation(image_tensor, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 10度右に回転
        image = image_tensor
        image = TF.rotate(image, -10)
        target = target_tensor
        target = TF.rotate(target, -10)
        return image, target
    else:
        # 10度右に回転
        image = image_tensor
        image = TF.rotate(image, -10)
        return image

def left_rotation(image_tensor, target_tensor=None, inference=True):
    if not inference:
        # target_tensorがNoneの場合、errorを返す
        if target_tensor is None:
            raise ValueError("target_tensor is None")
        # 10度左に回転
        image = image_tensor
        image = TF.rotate(image, 10)
        target = target_tensor
        target = TF.rotate(target, 10)
        return image, target
    else:
        # 10度左に回転
        image = image_tensor
        image = TF.rotate(image, 10)
        return image

