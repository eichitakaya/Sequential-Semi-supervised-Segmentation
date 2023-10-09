import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_networks as networks
import torchvision.transforms.functional as TF
import augmentation.transform_functions as transform_functions

def vertical_flip(model, image_tensor):
    image = transform_functions.vertical_flip(image_tensor, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.vertical_flip(predict, inference=True)
    return predict

def horizontal_flip(model, image_tensor):
    image = transform_functions.horizontal_flip(image_tensor, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.horizontal_flip(predict, inference=True)
    return predict

def right_shift(model, image_tensor, shift_pixels):
    image = transform_functions.right_shift(image_tensor, shift_pixels, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.left_shift(predict, shift_pixels, inference=True)
    return predict

def left_shift(model, image_tensor, shift_pixels):
    image = transform_functions.left_shift(image_tensor, shift_pixels, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.right_shift(predict, shift_pixels, inference=True)
    return predict

def right_rotation(model, image_tensor, theta):
    image = transform_functions.right_rotation(image_tensor, theta, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.left_rotation(predict, theta, inference=True)
    return predict

def left_rotation(model, image_tensor, theta):
    image = transform_functions.left_rotation(image_tensor, theta, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.right_rotation(predict, theta, inference=True)
    return predict

def gaussian_noise(model, image_tensor):
    image = transform_functions.gaussian_noise(image_tensor)
    predict = image
    return predict

def gaussian_blur(model, image_tensor):
    image = transform_functions.gaussian_blur(image_tensor)
    predict = image
    return predict

def random_brightness(image_tensor):
    print(image_tensor.max(), image_tensor.min())
    image = transform_functions.random_brightness(image_tensor)
    #predict = model(image)
    predict = image
    print(predict.max(), predict.min())
    return predict

def random_contrast(image_tensor):
    print(image_tensor.max(), image_tensor.min())
    image = transform_functions.random_contrast(image_tensor)
    #predict = model(image)
    predict = image
    print(predict.max(), predict.min())
    return predict

model = networks.UNet()
# takaya.jpgをグレースケールで読み込みtensorに変換
image = cv2.imread("takaya.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = torch.from_numpy(image)
# float32に変換
image = image.float()
# 4次元に変換
image = image.unsqueeze(0)
image = image.unsqueeze(0)
#result = inference_time_augmentation(model, image, method="vote")
result = gaussian_noise(model, image)

# resultを画像として保存
result = result.detach().numpy()
result = result[0][0]
result = result * 255
result = result.astype(np.uint8)
cv2.imwrite("result.jpg", result)