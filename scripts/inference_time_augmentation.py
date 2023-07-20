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
import cv2
import torch_networks as networks



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
    predict_list.append(horizontal_flip(model, image_tensor))
    # 画像を上下反転したものを推論
    predict_list.append(vertical_flip(model, image_tensor))
    
    # 3枚の推論結果の平均を取る
    if method == "average":
        predict = torch.mean(torch.stack(predict_list), dim=0)
        
    # 3枚の推論結果の多数決を取る
    elif method == "vote":
        # 3枚の推論結果を0,1に変換
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
    
    result = inference_time_augmentation(model, image, method="vote")
    
    # resultを画像として保存
    result = result.detach().numpy()
    result = result[0][0]
    result = result * 255
    result = result.astype(np.uint8)
    cv2.imwrite("result.jpg", result)