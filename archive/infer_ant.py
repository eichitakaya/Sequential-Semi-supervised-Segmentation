"""
任意の学習済みモデルで任意の枚数のテストデータに対する推論を出力する
※ aichanではなくローカルで実行する．
"""
import argparse
import datetime
import numpy as np
from PIL import Image
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import time
import DCN
import csv
import dataset as ds


parser = argparse.ArgumentParser(description='Chainer example: AIP')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')


args = parser.parse_args()

if args.gpu >= 0:
	cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


#使用するモデル
model = DCN.DCNAll()
chainer.serializers.load_npz("../models/Allepoch200_2.model", model)

#データセットの準備
n = 30
channels = 1
width = 512
height = 512

x_tensor = np.zeros((n, channels, width, height)).astype(np.float32)
t_tensor = np.zeros((n, channels, width, height)).astype(np.float32)
for i in range(n):
    #画像データの読み込み
    #"{0:04d}".format(1)
    data_x_name = "../data/test_volume30/data/test_volume30_" + "{0:04d}".format(i) + ".jpg"
    data_t_name = "../data/train_labels30/data/train_labels30_" + "{0:04d}".format(i) + ".jpg"
    img_x = ds.plain(data_x_name, "data", channels, width, height)
    img_t = ds.plain(data_t_name, "label", channels, width, height)


    #あらかじめ作っておいたtensorに代入
    x_tensor[i] = img_x
    t_tensor[i] = img_t

#GPUの確認
if args.gpu >= 0:
	cuda.get_device(args.gpu).use()
	model.to_gpu()


for i in range(n):
    xx = np.zeros((1, channels, width, height)).astype(np.float32)
    tt = np.zeros((channels, width, height)).astype(np.int32)
    xx[0] = x_tensor[i][0]
    tt[0] = t_tensor[i][0]
    #cudaに変換
    xx = xp.asarray(xx)
    tt = xp.asarray(tt)
    p1, p2, p3, fusion, predict = model(xx, tt, train=False)

    output = DCN.DCN_output(predict, 1)
    C1 = DCN.DCN_output(p1, 1)
    C2 = DCN.DCN_output(p2, 1)
    C3 = DCN.DCN_output(p3, 1)

    output = Image.fromarray(np.uint8(output))
    C1 = Image.fromarray(np.uint8(C1))
    C2 = Image.fromarray(np.uint8(C2))
    C3 = Image.fromarray(np.uint8(C3))

    output.save("../test_res/fusion/" + "{0:04d}".format(i) + ".jpg")
    C1.save("../test_res/C1/" + "{0:04d}".format(i) + ".jpg")
    C2.save("../test_res/C2/" + "{0:04d}".format(i) + ".jpg")
    C3.save("../test_res/C3/" + "{0:04d}".format(i) + ".jpg")

    print("saved " + "{0:04d}".format(i) + ".jpg" )
