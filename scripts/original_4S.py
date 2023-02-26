import os
import math
from data_loader import DataLoaderFor4S, predict2img
import torch_networks as networks
from torch_loss_functions import BCEDiceLoss
from torch import optim
import torch.nn as nn
import torch

import numpy as np

import csv

from PIL import Image

class SequentialSemiSupervisedSegmentation:
    def __init__(self, model, repeat_num, raw_model="", lr=0.001, _lambda=0.0005, M=3, epoch=10, batch=3, gpu_id=-1, dataset_name=0, scratch=0, pp=1, save_dir="", supervise=0, reverse=0, locally=0):
        # M == batch
        self.gpu_id = gpu_id

        self.epoch = epoch # エポックの初期値
        self.M = M # 初期枚数
        self.batch = M # ミニバッチ（1サイクルで学習に使う枚数）
        self.dataset_name = dataset_name # データセット名（spleen, heartなど）
        self.scratch = scratch # サイクルごとにモデルをリセットするかどうかのフラグ
        self.pp = pp # post processing
        self.save_dir = save_dir # 結果の出力先
        self.lr = lr # Adamのパラメータ
        self.repeat_num = repeat_num # 実験が何回目であるかを表す
        self.supervise = supervise # 半教師あり学習を行わない場合は1
        self.reverse = reverse # 逆向きのラベル伝播を行う場合は1
        self.locally = locally # ラベル伝播が完璧である場合をシミュレートする

        self.volumes = DataLoaderFor4S("heart")

        print("loaded dataset!")
        
        self.raw_model = raw_model
        self.model = model
        if self.gpu_id >= 0:
            self.model.to_gpu(self.gpu_id)
        print("loaded model!")
        
        self._lambda = _lambda
        self.optimizer = self.set_optimizer(self.model, self._lambda)
        print("set optimizer!")



    def set_optimizer(self, model, _lambda):
        #最適化手法にテコ入れする際は，ここをイジる．
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=_lambda)
        return optimizer
    
    def output_t(self, raw, target, predict, patient_id, slice_num):
        im_r = Image.fromarray(np.uint8(raw*255))
        im_t = Image.fromarray(np.uint8(target*255))
        im_p = Image.fromarray(np.uint8(predict*255))
        output = Image.new('L', (im_r.width*3, im_r.height))
        output.paste(im_r, (0, 0))
        output.paste(im_p, (im_r.width, 0))
        output.paste(im_t, (im_r.width*2, 0))
        os.makedirs(f"../result/patient_{patient_id}/target", exist_ok=True)
        os.makedirs(f"../result/patient_{patient_id}/predict", exist_ok=True)
        os.makedirs(f"../result/patient_{patient_id}/raw_predict_target", exist_ok=True)
        im_p.save(self.save_dir + "../result/" + f"patient_{patient_id}" + "/predict/" + "{0:03d}".format(slice_num) + ".png")
        im_t.save(self.save_dir + "../result/" + f"patient_{patient_id}" + "/target/" + "{0:03d}".format(slice_num) + ".png")
        output.save(self.save_dir + "../result/" + f"patient_{patient_id}" + "/raw_predict_target/" + "{0:03d}".format(slice_num) + ".png")
        print("saved output image!")

    def training(self, volume_id):
        self.X, self.T = self.volumes[volume_id]
        self.n = self.X.shape[0] # 1症例におけるスライス枚数
        print(self.n)
        training_model = self.model
        # 誤差関数の定義
        criterion = BCEDiceLoss()
        for i in range(self.n-self.M):#画像の枚数だけ繰り返し
            wc = 1
            if self.scratch == 1:
                training_model.to_cpu()
                training_model = self.raw_model()
                if self.gpu_id >= 0:
                    training_model.to_gpu(self.gpu_id)
                print("loaded model!")
                self.optimizer = self.set_optimizer(training_model, self._lambda)
                print("set optimizer!")
            
            if self.supervise == 0:
                # ここでselective annotation
                train_x = self.X[i:i+self.M]
                train_t = self.T[i:i+self.M]
            else:
                train_x = self.X[0:0+self.M]
                train_t = self.T[0:0+self.M]
            
            #xとtをTensorに変換
            train_x = torch.Tensor(train_x).float()
            train_t = torch.Tensor(train_t).float()
            epoch_i = int(self.epoch * ((1/2)**((i+1)-1)))
            for epoch in range(max(epoch_i, math.ceil(self.epoch / 100))):        
                # batch != Mの場合については別途考える必要あり        
                perm = np.random.permutation(self.M)
                
                batch_x = train_x[perm[0:self.batch]] / 255
                batch_t = train_t[perm[0:self.batch]]
                predict = training_model(batch_x)
                loss = criterion(predict, batch_t)
                print(loss)
                training_model.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            #推論でもバッチサイズの数だけ入力する必要がある．2以上の場合，最後の一つを未知とする．
            #ミニバッチ３枚のうち，最後が未知になるように調整する．
            #n=1のとき，学習せずに最後まで繰り返す．
            if self.n == 1:
                if self.dataset_num == 1:
                    data_n = 30
                if self.dataset_num == 0:
                    data_n = 100
                if self.dataset_num == 2 or self.dataset_num == 3:
                    data_n = 92
                for j in range(data_n - self.batch):
                    predict, loss = training_model(self.X[j+self.M-(self.batch-1):j+self.M+1], self.T[j+self.M-(self.batch-1):j+self.M+1])
                    with cupy.cuda.Device(self.gpu_id):
                        if self.dataset_num == 0:
                            add_t = DCN.DCN_output(inference=predict, batchsize=self.batch, p=0.5, ant=True, gpu_id=self.gpu_id)
                            if self.pp == 1:
                                add_t = chainer.cuda.to_cpu(add_t)
                                add_t = pp.opening(add_t)
                                print("post processed!")
                        if self.dataset_num == 1:
                            add_t = DCN.DCN_output(inference=predict, batchsize=self.batch, p=0.5, ant=True, gpu_id=self.gpu_id)
                            if self.pp == 1:
                                add_t = chainer.cuda.to_cpu(add_t)
                                add_t = pp.opening(add_t)
                                print("post processed!")
                        if self.dataset_num == 2 or self.dataset_num == 3:
                            add_t = DCN.DCN_output(inference=predict, batchsize=self.batch, p=0.5, ant=True, gpu_id=self.gpu_id)
                            if self.pp == 1:
                                add_t = chainer.cuda.to_cpu(add_t)
                                add_t = pp.opening(add_t)
                                print("post processed!")
                    # add_tを画像にして保存
                    self.output_t(predict=add_t, patient_id=volume_id, slice_num=j+self.M+1)
                    
            else:
                add_x = self.X[i+self.M-(self.batch-1):i+self.M+1]
                add_t = self.T[i+self.M-(self.batch-1):i+self.M+1]
                add_x = torch.Tensor(add_x) / 255
                add_t = torch.Tensor(add_t)
                predict = training_model(add_x)
                loss = criterion(predict, add_t)
                #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
                add_t = predict2img(predict)
                #ここに後処理
                # if self.pp == 1:
                #     add_t = chainer.cuda.to_cpu(add_t)
                #     add_t = pp.opening(add_t)
                #     print("post processed!")

                # add_tの最後のスライスを画像にして保存
                print(self.T.shape)
                self.output_t(raw=add_x[-1][0], target=self.T[i+self.M][0], predict=add_t[-1][0], patient_id=volume_id, slice_num=i+self.M+1)
                #ここで再びcupyに変換しないとエラーを吐く
                if self.supervise == 0:
                    if self.locally == 1:
                        print("a pseudo label was not added")
                    else:
                        self.T[i+self.M] = add_t[-1] /255
            print(i+1)


if __name__ == "__main__":
    model = networks.UNet()
    learn = SequentialSemiSupervisedSegmentation(model, 1)
    learn.training(3)