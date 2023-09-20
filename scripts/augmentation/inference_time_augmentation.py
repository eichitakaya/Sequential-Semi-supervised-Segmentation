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
7. random brightness
8. random contrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_networks as networks
import torchvision.transforms.functional as TF



def vertical_flip(model, image_tensor):
    # 左右反転し，推論
    image = image_tensor.flip(3)
    predict = model(image)
    # 推論結果をさらに反転
    return predict.flip(3)

def horizontal_flip(model, image_tensor):
    # 上下反転し，推論
    image = image_tensor.flip(2)
    predict = model(image)
    # 推論結果をさらに反転
    return predict.flip(2)

def right_shift(model, image_tensor):
    # 5pixel右にずらす
    image = image_tensor[:,:,:,:-5]
    # ずらした分だけ左側を0で埋める
    image = F.pad(image, (5,0,0,0), mode="constant", value=0)
    # 推論
    predict = model(image)
    # 推論結果を5pixel左にずらす
    predict = predict[:,:,:,5:]
    # ずらした分だけ右側を0で埋める
    predict = F.pad(predict, (0,5,0,0), mode="constant", value=0)
    return predict

def left_shift(model, image_tensor):
    # 5pixel左にずらす
    image = image_tensor[:,:,:,5:]
    # ずらした分だけ右側を0で埋める
    image = F.pad(image, (0,5,0,0), mode="constant", value=0)
    # 推論
    predict = model(image)
    # 推論結果を5pixel右にずらす
    predict = predict[:,:,:,:-5]
    # ずらした分だけ左側を0で埋める
    predict = F.pad(predict, (5,0,0,0), mode="constant", value=0)
    return predict

def right_rotation(model, image_tensor):
    # 10度右に回転
    image = image_tensor
    image = TF.rotate(image, -10)
    predict = model(image)
    # 推論結果を10度左に回転
    predict = TF.rotate(predict, 10)
    return predict

def left_rotation(model, image_tensor):
    # 10度左に回転
    image = image_tensor
    image = TF.rotate(image, 10)
    predict = model(image)
    # 推論結果を10度右に回転
    predict = TF.rotate(predict, -10)
    return predict

def left_rotation(model, image_tensor):
    # 10度左に回転
    image = image_tensor
    predict = model(image)
    return predict

def inference_time_augmentation(model, image_tensor, device, method="average"):
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
    predict_list.append(right_shift(model, image_tensor))
    # 画像を左に5pixelずらしたものを推論
    predict_list.append(left_shift(model, image_tensor))
    # 画像を10度右に回転したものを推論
    predict_list.append(right_rotation(model, image_tensor))
    # 画像を10度左に回転したものを推論
    predict_list.append(left_rotation(model, image_tensor))
    
    # 5枚の推論結果の平均を取る
    if method == "average":
        predict = torch.mean(torch.stack(predict_list), dim=0)
        print("hogehogehoge")
        
    # 5枚の推論結果の多数決を取る
    elif method == "vote":
        # 5枚の推論結果を0,1に変換
        predict_list = [torch.where(predict_list[0] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[1] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)),
                        torch.where(predict_list[2] > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))]
        # 3枚の推論結果の多数決を取る
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
    result = right_rotation(model, image)
    
    # resultを画像として保存
    result = result.detach().numpy()
    result = result[0][0]
    result = result * 255
    result = result.astype(np.uint8)
    cv2.imwrite("result.jpg", result)