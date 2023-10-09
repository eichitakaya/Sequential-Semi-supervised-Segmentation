import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# folder_nameを受け取り，フォルダ内のpseudo labelを評価し，dfを出力
def make_dice_df(folder_name):

    patients = sorted(glob.glob(f"../result/{folder_name}/patient*"))

    dice_frame = []

    for patient in patients:
        predicts = sorted(glob.glob(patient + "/predict/*"))
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

    return dice_df


# patient毎のiou系列をpngに描画
def draw_iou_series(dice_df, folder_name):
    for patient in dice_df.index:
        plt.plot(dice_df.loc[patient])
        plt.title(patient)
        plt.xlabel("slice")
        plt.ylabel("dice")
        # dice_seriesフォルダがなければ作成
        if not os.path.exists(f"../result/{folder_name}/dice_series"):
            os.makedirs(f"../result/{folder_name}/dice_series")
        plt.savefig(f"../result/{folder_name}/dice_series/{patient}.png")
        plt.close()

def dice_statistics(dice_df, folder_name):
    # patient毎のiouの平均，標準偏差，最大値，最小値を計算
    dice_mean = dice_df.mean(axis=1)
    dice_std = dice_df.std(axis=1)
    dice_max = dice_df.max(axis=1)
    dice_min = dice_df.min(axis=1)

    # 全patientのiouの平均，標準偏差，最大値，最小値を計算
    dice_mean_mean = dice_mean.mean()
    dice_mean_std = dice_mean.std()
    dice_mean_max = dice_mean.max()
    dice_mean_min = dice_mean.min()

    # 統計情報をデータフレームにまとめ，csvに書き込み
    dice_statistics = pd.DataFrame([dice_mean_mean, dice_mean_std, dice_mean_max, dice_mean_min], index=["mean", "std", "max", "min"])
    dice_statistics.to_csv(f"../result/{folder_name}/dice_statistics.csv", header=False)
    
    
def evaluate_4S(folder_name):
    dice_df = make_dice_df(folder_name)
    # csvに書き込み
    dice_df.to_csv(f"../result/{folder_name}/dice.csv")
    # patient毎のiou系列をpngに描画
    draw_iou_series(dice_df, folder_name)
    dice_statistics(dice_df, folder_name)



if __name__ == "__main__":
    evaluate_4S("heart/max_variance/active_epoch10")