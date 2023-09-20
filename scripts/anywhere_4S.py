import os
import math
import active_selection
from data_loader import DataLoaderFor4S, predict2img
import augmentation.inference_time_augmentation as inference_time_augmentation
#from inference_time_augmentation import inference_time_augmentation
import augmentation.augmentation as augmentation
import torch_networks as networks
from torch_loss_functions import BCEDiceLoss
from torch import optim
import torch.nn as nn
import torch

import numpy as np

import csv

from PIL import Image

class SequentialSemiSupervisedSegmentation:
    def __init__(self, dataset, model, random_selection, raw_model="", lr=0.001, _lambda=0.0005, M=3, epoch=10, gpu_id=-1, scratch=0, save_dir="", supervise=0, locally=0, ita=0, epoch_decay=0):
        # M == batch
        self.gpu_id = gpu_id

        self.epoch = epoch # エポックの初期値
        self.M = M # 初期枚数
        self.batch = M # ミニバッチ（1サイクルで学習に使う枚数）
        self.dataset_name = dataset # データセット名（spleen, heartなど）
        self.scratch = scratch # サイクルごとにモデルをリセットするかどうかのフラグ
        self.save_dir = save_dir # 結果の出力先
        self.lr = lr # Adamのパラメータ
        self.supervise = supervise # 半教師あり学習を行わない場合は1
        self.locally = locally # ラベル伝播が完璧である場合をシミュレートする
        self.ita = ita # 推論時にデータ拡張を行う場合は1
        
        self.random_selection = random_selection

        self.dataset = dataset
        self.volumes = DataLoaderFor4S(self.dataset)
        self.epoch_decay = epoch_decay

        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        print("loaded dataset!")
        
        self.raw_model = raw_model
        self.model = model
        print("loaded model!")
        
        self._lambda = _lambda
        self.optimizer = self.set_optimizer(self.model, self._lambda)
        print("set optimizer!")



    def set_optimizer(self, model, _lambda):
        #最適化手法にテコ入れする際は，ここをイジる．
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=_lambda)
        return optimizer
    
    def output_t(self, raw, target, predict, patient_id, slice_num, folder_name):
        im_r = Image.fromarray(np.uint8(raw*255))
        im_t = Image.fromarray(np.uint8(target*255))
        im_p = Image.fromarray(np.uint8(predict*255))
        output = Image.new('L', (im_r.width*3, im_r.height))
        output.paste(im_r, (0, 0))
        output.paste(im_p, (im_r.width, 0))
        output.paste(im_t, (im_r.width*2, 0))
        print(folder_name)
        os.makedirs(f"../result/{folder_name}/patient_{patient_id}/target", exist_ok=True)
        os.makedirs(f"../result/{folder_name}/patient_{patient_id}/predict", exist_ok=True)
        os.makedirs(f"../result/{folder_name}/patient_{patient_id}/raw_predict_target", exist_ok=True)
        im_p.save(f"../result/{folder_name}/" + f"patient_{patient_id}" + "/predict/" + "{0:03d}".format(slice_num) + ".png")
        im_t.save(f"../result/{folder_name}/" + f"patient_{patient_id}" + "/target/" + "{0:03d}".format(slice_num) + ".png")
        output.save(f"../result/{folder_name}/" + f"patient_{patient_id}" + "/raw_predict_target/" + "{0:03d}".format(slice_num) + ".png")
        print("saved output image!")

    def training(self, volume_id):
        self.X, self.T = self.volumes[volume_id]
        self.n = self.X.shape[0] # 1症例におけるスライス枚数
        print(self.n)
        training_model = self.model
        training_model = training_model.to(self.device)
        # 誤差関数の定義
        self.criterion = BCEDiceLoss()
        
        # ここでselective annotation
        if self.random_selection == 0:
            selected_index = active_selection.calc_max_mean_group(self.X, self.M)[0]
            print(f"Actively selected {self.M} slices!")
        else:
            selected_index = active_selection.random_selection(self.n, self.M)[0]
            print(f"Randomly selected {self.M} slices!")
        print(f"selected {selected_index+1} / {self.n}") # 1始まりで，15 / 17
        train_x = self.X[selected_index:selected_index+self.M] # 0始まりで14, 15, 16
        train_t = self.T[selected_index:selected_index+self.M]
        
        # initial train
        train_x = torch.Tensor(train_x).float()
        train_t = torch.Tensor(train_t).float()
        for epoch in range(self.epoch):        
            # batch != Mの場合については別途考える必要あり        
            perm = np.random.permutation(self.M)
            
            batch_x = train_x[perm[0:self.batch]] / 255
            batch_t = train_t[perm[0:self.batch]]
            
            # ここでaugmentation
            batch_x, batch_t = augmentation(batch_x, batch_t)
            
            batch_x = batch_x.to(self.device)
            batch_t = batch_t.to(self.device)
                        
            predict = training_model(batch_x)
            loss = self.criterion(predict, batch_t)
            print(loss)
            training_model.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # いったんinitial modelを保存
        torch.save(training_model.state_dict(), '../model/initial_model.pth')
        
        # 選択されたM枚より前方へのラベル伝播（右端の場合は発動しない）
        if selected_index+self.M != self.n:
            self.forward_train(selected_index=selected_index, volume_id=volume_id)
        
        # 選択されたM枚より後方へのラベル伝播（左端の場合は発動しない）
        if selected_index != 0:
            self.backward_train(selected_index=selected_index, volume_id=volume_id)
        
            
    def forward_train(self, selected_index, volume_id):
        print("Start forward process!")
        # initial modelをload
        training_model = self.model
        training_model.load_state_dict(torch.load("../model/initial_model.pth"))
        training_model = training_model.to(self.device)
        # optimizerを初期化
        self.optimizer = self.set_optimizer(self.model, self._lambda)
        print("set optimizer!")
        
        # 新たな１枚を前方に追加
        add_x = self.X[selected_index+self.M] # 17 
        # 次元を増やす
        add_x = np.expand_dims(add_x, axis=0)
        add_x = torch.Tensor(add_x) / 255
        add_x = add_x.to(self.device)
        print(add_x.shape)
        if self.ita == 1:
            predict = inference_time_augmentation(training_model, add_x, device=self.device)
        else:
            predict = training_model(add_x)

        #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
        add_t = predict2img(predict)

        # add_tの最後のスライスを画像にして保存    
        self.output_t(raw=add_x[0][0].cpu(), target=self.T[selected_index+self.M][0], predict=add_t[0][0].cpu(), patient_id=volume_id, slice_num=selected_index+self.M, folder_name=f"{self.dataset_name}/{self.save_dir}")
        if self.supervise == 0:
            if self.locally == 1:
                print("a pseudo label was not added")
            else:
                self.T[selected_index+self.M] = add_t[0].cpu() /255
                print(f"added {selected_index+self.M+1}th target!")
        # 直前の追加スライスが最後であった場合，ループ回数は0になる  
        for index, i in enumerate(range(selected_index+1, self.n-self.M)):#前方に残された枚数だけ繰り返し
            train_x = self.X[i:i+self.M]
            train_t = self.T[i:i+self.M]
            #xとtをTensorに変換
            train_x = torch.Tensor(train_x).float()
            train_t = torch.Tensor(train_t).float()
            # epoch_decayが1の場合は，epochを1/2ずつ減らしていく
            if self.epoch_decay == 1:
                epoch_i = int(self.epoch * ((1/2)**((index+1))))
                epoch_i = max(epoch_i, math.ceil(self.epoch / 100))
            else:
                epoch_i = self.epoch
            for epoch in range(epoch_i):        
                # batch != Mの場合については別途考える必要あり        
                perm = np.random.permutation(self.M)
                
                batch_x = train_x[perm[0:self.batch]] / 255
                batch_t = train_t[perm[0:self.batch]]
                
                # ここでaugmentation
                batch_x, batch_t = augmentation(batch_x, batch_t)
                
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                
                
                
                predict = training_model(batch_x)
                loss = self.criterion(predict, batch_t)
                print(loss)
                training_model.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 未知のスライスを推論して追加
            # iが最後のスライスであった場合，追加は発動しない
            add_x = self.X[i+self.M]
            add_x = np.expand_dims(add_x, axis=0)
            add_x = torch.Tensor(add_x) / 255
            
            add_x = add_x.to(self.device)
            
            if self.ita == 1:
                predict = inference_time_augmentation(training_model, add_x, device=self.device)
            else:
                predict = training_model(add_x)
            #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
            add_t = predict2img(predict)

            # add_tの最後のスライスを画像にして保存
            self.output_t(raw=add_x[0][0].cpu(), target=self.T[i+self.M][0], predict=add_t[-1][0].cpu(), patient_id=volume_id, slice_num=i+self.M, folder_name=f"{self.dataset_name}/{self.save_dir}")
            print(f"added {i+self.M+1}th target!")
            #ここで再びcupyに変換しないとエラーを吐く
            if self.supervise == 0:
                if self.locally == 1:
                    print("a pseudo label was not added")
                else:
                    self.T[i] = add_t[-1].cpu() /255
    
    def backward_train(self, selected_index, volume_id):
        print("Start backward process!")
         # initial modelをload
        training_model = self.model
        training_model.load_state_dict(torch.load("../model/initial_model.pth"))
        training_model = training_model.to(self.device)
        # optimizerを初期化
        self.optimizer = self.set_optimizer(self.model, self._lambda)
        print("set optimizer!")
        # 最初の入力に使う１枚目の画像のindex
        start_num = selected_index - 1 # 11
        end_num = 0
        
        # 新たな１枚を前方に追加
        add_x = self.X[start_num:start_num+self.M] # 11, 12, 13
        add_t = self.T[start_num:start_num+self.M]
        add_x = torch.Tensor(add_x) / 255
        add_t = torch.Tensor(add_t)
        add_x = add_x.to(self.device)
        add_t = add_t.to(self.device)
        if self.ita == 1:
            predict = inference_time_augmentation(training_model, add_x, device=self.device)
        else:
            predict = training_model(add_x)

        #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
        add_t = predict2img(predict)

        # add_tの最後のスライスを画像にして保存
        self.output_t(raw=add_x[0][0].cpu(), target=self.T[selected_index-1][0], predict=add_t[0][0].cpu(), patient_id=volume_id, slice_num=selected_index-1, folder_name=f"{self.dataset_name}/{self.save_dir}")
        if self.supervise == 0:
            if self.locally == 1:
                print("a pseudo label was not added")
            else:
                self.T[selected_index-1] = add_t[-1].cpu() /255
                print(f"added {selected_index}th target!") # 39
        
        for index, i in enumerate(range(start_num-1, end_num-1, -1)):#後方に残された枚数だけ繰り返し
            train_x = self.X[i:i+self.M]# 38, 39, 40
            train_t = self.T[i:i+self.M]
            #xとtをTensorに変換
            train_x = torch.Tensor(train_x).float()
            train_t = torch.Tensor(train_t).float()
            
            # epoch_decayが1の場合は，epochを1/2ずつ減らしていく
            if self.epoch_decay == 1:
                epoch_i = int(self.epoch * ((1/2)**((index+1))))
                epoch_i = max(epoch_i, math.ceil(self.epoch / 100))
            else:
                epoch_i = self.epoch
                
            for epoch in range(epoch_i):        
                # batch != Mの場合については別途考える必要あり        
                perm = np.random.permutation(self.M)
                
                batch_x = train_x[perm[0:self.batch]] / 255
                batch_t = train_t[perm[0:self.batch]]
                # ここでaugmentation
                batch_x, batch_t = augmentation(batch_x, batch_t)
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                predict = training_model(batch_x)
                loss = self.criterion(predict, batch_t)
                print(loss)
                training_model.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 未知のスライスを推論して追加
            add_x = self.X[i:i+self.M]
            add_t = self.T[i:i+self.M]
            add_x = torch.Tensor(add_x) / 255
            add_t = torch.Tensor(add_t)
            add_x = add_x.to(self.device)
            add_t = add_t.to(self.device)
            if self.ita == 1:
                predict = inference_time_augmentation(training_model, add_x, device=self.device)
            else:
                predict = training_model(add_x)
            #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
            add_t = predict2img(predict)

            # add_tの最後のスライスを画像にして保存
            self.output_t(raw=add_x[0][0].cpu(), target=self.T[i][0], predict=add_t[0][0].cpu(), patient_id=volume_id, slice_num=i, folder_name=f"{self.dataset_name}/{self.save_dir}")
            print(f"added {i+1}th target!")
            #ここで再びcupyに変換しないとエラーを吐く
            if self.supervise == 0:
                if self.locally == 1:
                    print("a pseudo label was not added")
                else:
                    self.T[i] = add_t[-1].cpu() /255


if __name__ == "__main__":
    model = networks.UNet()
    learn = SequentialSemiSupervisedSegmentation(model, 1, 0)
    learn.training(4)