"""
データセットの準備に関する各種関数
"""

from PIL import Image
from PIL import ImageEnhance
import numpy as np
from chainer.datasets import split_dataset

#変換なし
def plain(filepath, ant=False, mouse=False, gt=False, mode="data", channels=3, width=512, height=512, devel=False):
    img = Image.open(filepath)
    """
    if devel == True:
        width=28
        height=28
    else:
        width=width
        height=height
    """
    if ant == True:
        #print("ant loading")
        #box = (880, 650, 1630, 1400)
        box = (920, 738, 1432, 1250)
        if devel==True:
            box = (1208, 1051, 1240, 1083)
        img = img.crop(box)
        #if mode == "data":
        #    print("contrast!")
        #    iec = ImageEnhance.Contrast(img)
        #    img = iec.enhance(1.2)
        if devel==False:
            #img = img.resize((width, height))
            a = 0
    if mouse == True:
        img = img.resize((width, height))

    if mode == "data":
        #print("raw loading")
        img = np.array(img, dtype=np.float32)
        #print(img.shape)
        #(channels, width, height)に変換
        img = img.reshape(channels, width, height).astype(np.float32)
        #0~1に正規化
        img = img/255.0
        #print("data")
    else:
        if ant == False and mouse==False:#ISBIかどうかを判定
            img = np.array(img, dtype=np.int32)
            mask = img != 255
            #mask = mask.astype(xp.int32)
            img[mask] = 1
            mask = img == 255
            img[mask] = 0
            #print("label")
            return img
        elif gt == False:
            #print("ant label loading")
            img = np.array(img, dtype=np.int32)
            mask = img != 0
            img[mask] = 1
            #mask = img == 255
            #img[mask] = 1
        elif gt == True:
            img = np.array(img, dtype=np.float32)
    return img

#回転
def rotate(filepath, angle, mode="data", channels=3, width=512, height=512):
    img = Image.open(filepath)

    # 任意の角度だけ回転
    img = img.rotate(angle)

    img = np.array(img, dtype=np.int32)

    if mode=="data":
        #(channels, width, height)に変換
        img = img.reshape(channels, width, height).astype(np.float32)
        #0~1に正規化
        img = img/255.0

    if mode=="label":
        mask = img == 255
        #mask = mask.astype(xp.int32)
        img[mask] = -1

    return img


def mirror(filepath, which, mode="data", channels=3, width=512, height=512):
    img = Image.open(filepath)

    # スイッチに応じて反転
    if which == "LR":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif which == "TB":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = np.array(img, dtype=np.int32)

    if mode=="data":
        #(channels, width, height)に変換
        img = img.reshape(channels, width, height).astype(np.float32)
        #0~1に正規化
        img = img/255.0

    if mode=="label":
        mask = img == 255
        #mask = mask.astype(xp.int32)
        img[mask] = -1

    return img



"""
def scale()

def translate()

def sheared()
"""


def set_data(n, initial=0, width=512, height=512, channels=1, data="isbi", devel=False, reverse=False):
    if devel==True:
        width=32
        height=32
    else:
        width=width
        height=height
    x_tensor = np.zeros((n-initial, channels, width, height)).astype(np.float32)
    t_tensor = np.zeros((n-initial, width, height)).astype(np.int32)
    g_tensor = np.zeros((n-initial, width, height)).astype(np.float32)

    for i in range(initial, n):
        #画像データの読み込み
        #"{0:04d}".format(1)
        if data == "isbi":
            print("called isbi")
            data_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(i) + ".tif"
            data_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"
            #あらかじめ作っておいたtensorに代入
            if reverse == True:
                x_tensor[i-initial] = plain(data_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
                t_tensor[i-initial] = plain(data_t_name, ant=False, mode="label", width=width, height=height)
            else:
                x_tensor[i-initial] = plain(data_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
                t_tensor[i-initial] = plain(data_t_name, ant=False, mode="label", width=width, height=height)


        if data == "ant":
            data_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"
            data_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(i) + ".tif"
            #あらかじめ作っておいたtensorに代入
            if devel == False:
                #print("notdevel")
                x_tensor[i-initial] = plain(data_x_name, ant=True, mode="data", channels=channels, width=width, height=height)
                t_tensor[i-initial] = plain(data_t_name, ant=True, mode="label", width=width, height=height)
                g_tensor[i-initial] = plain(data_t_name, ant=True, gt=True, mode="label", width=width, height=height)

            else:
                x_tensor[i-initial] = plain(data_x_name, ant=True, mode="data", channels=channels, width=width, height=height, devel=True)
                t_tensor[i-initial] = plain(data_t_name, ant=True, mode="label", width=width, height=height, devel=True)
                g_tensor[i-initial] = plain(data_t_name, ant=True, gt=True, mode="label", width=width, height=height, devel=True)





    return x_tensor, t_tensor, g_tensor


class Augment:
    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, index, rand_num=5, target=False):
        if target==True:
            img = Image.fromarray(np.uint8(self.tensor[index]))
        else:
            img = Image.fromarray(np.uint8(self.tensor[index][0]*255))
        num = rand_num
        #print(img)
        if num == 0:
            img = img.rotate(90)
        elif num == 1:
            img = img.rotate(180)
        elif num == 2:
            img = img.rotate(270)
        elif num == 3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif num == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif num == 5:
            img = img

        xx = np.array(img, dtype=np.float32)
        if target == False:
            xx = xx/255
        #print(xx)
        return xx


def set_ISBI():
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(30):
        train_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(i) + ".tif"
        train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"

        train_x = plain(train_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=False, mode="label", width=width, height=height)

        test_x_name = "../data/ISBI2012_experiment/test_volume30/data/test_volume30_" + "{0:04d}".format(i) + ".tif"
        test_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"
        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=False, mode="label", width=width, height=height)

        train_val.append((train_x, train_t))
        test.append((test_x, test_t))

    return train_val, test

def set_ANT():
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(251):
        train_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"
        train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(i) + ".tif"
        train_x = plain(train_x_name, ant=True, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=True, mode="label", width=width, height=height)
        train_val.append((train_x, train_t))

    for i in range(126):
        test_x_name = "../data/ant_experiment/test/test" + "{0:03d}".format(i) + ".tif"
        test_t_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=True, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=True, mode="label", width=width, height=height)
        test.append((test_x, test_t))
        
    return train_val, test

def set_ANT_semi(initial_labels, reverse=0, locally=0):
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(100):
        if i < initial_labels:
            if reverse == 1:
                train_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(99-i) + ".tif"
                train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(99-i) + ".tif"
            else:
                train_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"
                train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(i) + ".tif"
        else:
            if reverse == 1:
                train_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(99-i) + ".tif"
                train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(99) + ".tif"
            else:
                train_x_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"
                if locally == 0:
                    train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(0) + ".tif"
                else:
                    train_t_name = "../data/ant_experiment/label/label" + "{0:03d}".format(i) + ".tif"
                    
        train_x = plain(train_x_name, ant=True, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=True, mode="label", width=width, height=height)
        train_val.append((train_x, train_t))

    for i in range(126):
        test_x_name = "../data/ant_experiment/test/test" + "{0:03d}".format(i) + ".tif"
        test_t_name = "../data/ant_experiment/train/train" + "{0:03d}".format(i) + ".tif"        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=True, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=True, mode="label", width=width, height=height)
        test.append((test_x, test_t))
        
    return train_val, test


def set_ISBI_semi(initial_labels, reverse=0, locally=0):
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(30):
        if i < initial_labels:
            if reverse == 1:
                train_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(29-i) + ".tif"
                train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(29-i) + ".tif"
            else:
                train_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(i) + ".tif"
                train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"
        else:
            if reverse == 1:
                train_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(29-i) + ".tif"
                train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(29) + ".tif"
            else:
                train_x_name = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_" + "{0:04d}".format(i) + ".tif"
                if locally == 0:
                    train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(0) + ".tif"
                else:
                    train_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"

        train_x = plain(train_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=False, mode="label", width=width, height=height)

        test_x_name = "../data/ISBI2012_experiment/test_volume30/data/test_volume30_" + "{0:04d}".format(i) + ".tif"
        test_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".tif"
        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=False, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=False, mode="label", width=width, height=height)

        train_val.append((train_x, train_t))
        test.append((test_x, test_t))

    return train_val, test

def set_mouse1_semi(initial_labels, reverse=0, locally=0):
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(92):
        if i < initial_labels:
            if reverse == 1:
                train_x_name = "../data/mouse_stem_cell/01/t" + "{0:03d}".format(91-i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/01_GT/TRA/man_track" + "{0:03d}".format(91-i) + ".tif"
            else:
                train_x_name = "../data/mouse_stem_cell/01/t" + "{0:03d}".format(i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/01_GT/TRA/man_track" + "{0:03d}".format(i) + ".tif"
        else:
            if reverse == 1:
                train_x_name = "../data/mouse_stem_cell/01/t" + "{0:03d}".format(91-i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/01_GT/TRA/man_track" + "{0:03d}".format(91) + ".tif"
            else:
                train_x_name = "../data/mouse_stem_cell/01/t" + "{0:03d}".format(i) + ".tif"
                if locally == 0:
                    train_t_name = "../data/mouse_stem_cell/01_GT/TRA/man_track" + "{0:03d}".format(0) + ".tif"
                else:
                    train_t_name = "../data/mouse_stem_cell/01_GT/TRA/man_track" + "{0:03d}".format(i) + ".tif"
                
        train_x = plain(train_x_name, ant=False, mouse=True, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=False, mouse=True, mode="label", width=width, height=height)

        test_x_name = "../data/ISBI2012_experiment/test_volume30/data/test_volume30_" + "{0:04d}".format(0) + ".tif"
        test_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(0) + ".tif"
        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=False, mouse=True, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=False, mouse=True, mode="label", width=width, height=height)

        train_val.append((train_x, train_t))
        test.append((test_x, test_t))

    return train_val, test

def set_mouse2_semi(initial_labels, reverse=0, locally=0):
    channels = 1
    width = 512
    height = 512
    train_val = []
    test = []
    for i in range(92):
        if i < initial_labels:
            if reverse == 1:
                train_x_name = "../data/mouse_stem_cell/02/t" + "{0:03d}".format(91-i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/02_GT/TRA/man_track" + "{0:03d}".format(91-i) + ".tif"
            else:
                train_x_name = "../data/mouse_stem_cell/02/t" + "{0:03d}".format(i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/02_GT/TRA/man_track" + "{0:03d}".format(i) + ".tif"
        else:
            if reverse == 1:
                train_x_name = "../data/mouse_stem_cell/02/t" + "{0:03d}".format(91-i) + ".tif"
                train_t_name = "../data/mouse_stem_cell/02_GT/TRA/man_track" + "{0:03d}".format(91) + ".tif"
            else:
                train_x_name = "../data/mouse_stem_cell/02/t" + "{0:03d}".format(i) + ".tif"
                if locally == 0:
                    train_t_name = "../data/mouse_stem_cell/02_GT/TRA/man_track" + "{0:03d}".format(0) + ".tif"
                else:
                    train_t_name = "../data/mouse_stem_cell/02_GT/TRA/man_track" + "{0:03d}".format(i) + ".tif"

        train_x = plain(train_x_name, ant=False, mouse=True, mode="data", channels=channels, width=width, height=height)
        train_t = plain(train_t_name, ant=False, mouse=True, mode="label", width=width, height=height)

        test_x_name = "../data/ISBI2012_experiment/test_volume30/data/test_volume30_" + "{0:04d}".format(0) + ".tif"
        test_t_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(0) + ".tif"
        #実際にtest_tは存在しないので，ダミーでtrain_labelを代入しておく．
        test_x = plain(test_x_name, ant=False, mouse=True, mode="data", channels=channels, width=width, height=height)
        test_t = plain(test_t_name, ant=False, mouse=True, mode="label", width=width, height=height)

        train_val.append((train_x, train_t))
        test.append((test_x, test_t))

    return train_val, test

if __name__ == "__main__":
    x, t = set_data(1, initial=0, data="ant", devel=True)
