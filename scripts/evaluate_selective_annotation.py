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


parser = argparse.ArgumentParser()
parser.add_argument("--folder_name", type=str)
args = parser.parse_args()

patients = glob.glob(f"../result/{args.folder_name}/*")

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

dice_df.to_csv(f"../result/{args.folder_name}/dice.csv")
