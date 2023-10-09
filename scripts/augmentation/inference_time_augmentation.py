"""
ある時点でのモデルと1枚の推論用画像を受け取り，test_time_augmentationを行う関数
imageはtensor型のまま扱いたい

input: model, image
output: 9枚の推論結果の平均

augmentationの種類
0. なし
1. right shift
2. left shift
3. + 10 degree rotation
4. - 10 degree rotation
5. gaussian noise
6. gaussian blur
7. high contrast
8. row contrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_networks as networks
import torchvision.transforms.functional as TF
import augmentation.transform_functions as transform_functions

def vertical_flip(model, image_tensor):
    # 左右反転し，推論
    image = transform_functions.vertical_flip(image_tensor, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.vertical_flip(predict, inference=True)
    return predict

def horizontal_flip(model, image_tensor):
    # 上下反転し，推論
    image = transform_functions.horizontal_flip(image_tensor, inference=True)
    predict = model(image)
    # 推論結果をさらに反転
    predict = transform_functions.horizontal_flip(predict, inference=True)
    return predict

def right_shift(model, image_tensor, shift_pixels):
    # shift_pixels右にずらす
    image = transform_functions.right_shift(image_tensor, shift_pixels, inference=True)
    # 推論
    predict = model(image)
    # 推論結果をshift_pixels左にずらす
    predict = transform_functions.left_shift(predict, shift_pixels, inference=True)
    return predict

def left_shift(model, image_tensor, shift_pixels):
    # shift_pixels左にずらす
    image = transform_functions.left_shift(image_tensor, shift_pixels, inference=True)
    # 推論
    predict = model(image)
    # 推論結果をshift_pixels右にずらす
    predict = transform_functions.right_shift(predict, shift_pixels, inference=True)
    return predict

def right_rotation(model, image_tensor, theta):
    # theta度右に回転
    image = transform_functions.right_rotation(image_tensor, theta, inference=True)
    predict = model(image)
    # 推論結果をtheta度左に回転
    predict = transform_functions.left_rotation(predict, theta, inference=True)
    return predict

def left_rotation(model, image_tensor, theta):
    # theta度左に回転
    image = transform_functions.left_rotation(image_tensor, theta, inference=True)
    predict = model(image)
    # 推論結果をtheta度右に回転
    predict = transform_functions.right_rotation(predict, theta, inference=True)
    return predict

def gaussian_noise(model, image_tensor):
    # 画像にガウシアンノイズを加える
    image = transform_functions.gaussian_noise(image_tensor)
    predict = model(image)
    return predict

def gaussian_blur(model, image_tensor):
    # 画像にガウシアンブラーを加える
    image = transform_functions.gaussian_blur(image_tensor)
    predict = model(image)
    return predict

def high_contrast(model, image_tensor):
    # 画像のコントラストを上げる
    image = transform_functions.high_contrast(image_tensor)
    predict = model(image)
    return predict

def low_contrast(model, image_tensor):
    # 画像のコントラストを下げる
    image = transform_functions.low_contrast(image_tensor)
    predict = model(image)
    return predict

def inference_time_augmentation(model, image_tensor, device, method="vote"):
    """_summary_

    Args:
        model (_type_): 任意のsegmentationモデル
        image_tensor (_type_): (1,1,H,W)のtensor
        method (_type_): "vote", "average"
    """
    # 3枚の推論結果を格納するリスト
    predict_list = []
    # そのままの画像を推論
    predict_list.append(model(image_tensor))
    # 画像を左右反転したものを推論
    #predict_list.append(horizontal_flip(model, image_tensor))
    # 画像を上下反転したものを推論
    #predict_list.append(vertical_flip(model, image_tensor))
    # 画像を右に5pixelずらしたものを推論
    predict_list.append(right_shift(model, image_tensor, 5))
    # 画像を左に5pixelずらしたものを推論
    predict_list.append(left_shift(model, image_tensor, 5))
    # 画像を10度右に回転したものを推論
    predict_list.append(right_rotation(model, image_tensor, 10))
    # 画像を10度左に回転したものを推論
    predict_list.append(left_rotation(model, image_tensor, 10))
    # 画像にガウシアンノイズを加えたものを推論
    predict_list.append(gaussian_noise(model, image_tensor))
    # 画像にガウシアンブラーを加えたものを推論
    predict_list.append(gaussian_blur(model, image_tensor))
    # 画像のコントラストを上げたものを推論
    predict_list.append(high_contrast(model, image_tensor))
    # 画像のコントラストを下げたものを推論
    predict_list.append(low_contrast(model, image_tensor))
    
    # 9枚の推論結果の平均を取る
    if method == "average":
        predict = torch.mean(torch.stack(predict_list), dim=0)
        print("hogehogehoge")
        
    # 9枚の推論結果の多数決を取る
    elif method == "vote":
        # 9枚の推論結果を0,1に変換
        predict_list = [torch.where(predict_list[0] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[1] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[2] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[3] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[4] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[5] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[6] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[7] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[8] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))]
        
        # 9枚の推論結果の多数決を取る
        #print(predict_list)
        predict = torch.mode(torch.stack(predict_list), dim=0).values
    return predict

# 以下，テスト用
if __name__ == "__main__":
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
    result = vertical_flip(model, image)
    
    # resultを画像として保存
    result = result.detach().numpy()
    result = result[0][0]
    result = result * 255
    result = result.astype(np.uint8)
    cv2.imwrite("result.jpg", result)