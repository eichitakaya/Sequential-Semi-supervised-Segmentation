# フォルダを受け取り，patient_xそれぞれについて全スライスのiouを計算
# 1つのcsvに全て書き込み，同フォルダに保存

import argparse
import glob
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calc_all(tp, pp):
    # TP, TN, FP, FN, Accuracy, Precision, Recall, DSC, IoUを計算する関数
    
    mask = pp != 0
    pp[mask] = 1
    mask = tp != 0
    tp[mask] = 1
    tn, fp, fn, tp = confusion_matrix(tp.flatten(), pp.flatten()).ravel() # TP, TN, FP, FNを計算
    presicion = tp / (tp + fp)
    recall = tp / (tp + fn)
    dice = tp / (tp + ((1/2)*(fp+fn)))
    iou = tp / (tp + fp + fn)
    return presicion, recall, dice, iou


def evaluation(folder_name):

    patients = glob.glob(f"../result/{folder_name}/patient*")

    dice_frame = []

    for patient in patients:
        predicts = glob.glob(patient + "/predict/*")
        dice_list = []
        for img_path in predicts:
            pp = np.array(Image.open(img_path))
            tp = np.array(Image.open(img_path.replace("predict", "target")))
            dice = calc_all(tp, pp)[2]
            dice_list.append(dice)
        dice_frame.append(dice_list)

    index = []
    for patient in patients:
        index.append(patient.split("/")[-1])
        
    dice_df = pd.DataFrame(dice_frame, index=index)
    
    # dice_dfを行毎に平均，標準偏差，最大値，最小値を計算
    dice_df["mean"] = dice_df.mean(axis=1)
    dice_df["std"] = dice_df.std(axis=1)
    dice_df["max"] = dice_df.max(axis=1)
    dice_df["min"] = dice_df.min(axis=1)
    
    dice_df.to_csv(f"../result/{folder_name}/statistics_per_patient.csv")

    # meanの列のみを取り出し，それらを新しいdfの行とする
    dice_df = dice_df["mean"]
    dice_df = pd.DataFrame(dice_df)
    # 平均，標準偏差，最大値，最小値を計算
    dice_df.loc["mean"] = dice_df.mean(axis=0)
    dice_df.loc["std"] = dice_df.std(axis=0)
    dice_df.loc["max"] = dice_df.max(axis=0)
    dice_df.loc["min"] = dice_df.min(axis=0)
    dice_df.to_csv(f"../result/{folder_name}/statistics_all.csv")

# テスト
if __name__ == "__main__":
    evaluation("heart")