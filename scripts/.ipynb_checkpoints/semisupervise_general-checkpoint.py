import chainer
from chainer import cuda
from chainer.datasets import split_dataset
from chainer import optimizers

import dataset as ds

import DCN
import DCN_master

import numpy as np
import cupy

import csv

from PIL import Image

import postprocessing as pp

class SemisuperviesedLearning:
    def __init__(self, model, raw_model="", alpha=0.001, _lambda=0.0005, n=3, M=3, epoch=1, batch=3, gpu_id=-1, dataset_num=0, scratch=0, pp=1, save_dir=""):
        #M >= batchは必須
        self.gpu_id = gpu_id

        self.n = n
        self.epoch = epoch
        self.M = M 
        self.batch = batch
        self.dataset_num = dataset_num
        self.scratch = scratch
        self.pp = pp
        self.save_dir = save_dir
        self.alpha = alpha

        if self.dataset_num == 0:
            self.train251, self.test = ds.set_ANT_semi(self.batch)
            self.train, self.valid = split_dataset(self.train251, 250)
            #self.train_t, self.train_v = split_dataset(self.train, 3)
        if self.dataset_num == 1:
            self.train, self.valid = ds.set_ISBI_semi(self.batch)
        
        self.x, self.t = chainer.dataset.concat_examples(self.train, self.gpu_id)

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
        optimizer = optimizers.Adam(self.alpha)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(_lambda))
        return optimizer
    
    def output_t(self, predict, num, gpu_id):
        # DCN_outputの出力結果を入力とする
        predict = chainer.cuda.to_cpu(predict)
        if self.dataset_num == 0:
            output = Image.fromarray(np.uint8(predict))
            #output.save("../output_for_thesis/semisuper/" + "{0:03d}".format(num) + ".jpg")
            output.save(self.save_dir + "/result_img/" + "{0:03d}".format(num) + ".jpg")
        if self.dataset_num == 1:
            predict = predict - 255
            predict = predict * (-1)
            output = Image.fromarray(np.uint8(predict))
            #output.save("../output_for_thesis/semisuper_isbi/" + "{0:03d}".format(num) + ".jpg")
            output.save(self.save_dir + "/result_img/" + "{0:03d}".format(num) + ".jpg")

    def training(self):
        training_model = self.model
        for i in range(self.n):#画像の枚数だけ繰り返し
            wc = 1
            loss_list = []
            if self.scratch == 1:
                training_model.to_cpu()
                training_model = self.raw_model()
                if self.gpu_id >= 0:
                    training_model.to_gpu(self.gpu_id)
                print("loaded model!")
                self.optimizer = self.set_optimizer(training_model, self._lambda)
                print("set optimizer!")
            
            with cupy.cuda.Device(self.gpu_id):    
                train_x = self.x[i:i+self.M]
                train_t = self.t[i:i+self.M]
            
            for epoch in range(self.epoch):
                if epoch%10 == 0 and wc > 0.011:
                    wc = wc * 0.1
                    #wf = wf * 10
                
                perm = np.random.permutation(self.M)
                
                with cupy.cuda.Device(self.gpu_id):   
                    batch_x = train_x[perm[0:self.batch]]
                    batch_t = train_t[perm[0:self.batch]]
                
                predict, loss = training_model(batch_x, batch_t)
                #p1, p2, p3, fusion, predict, loss = self.model(self.x[i:i+self.batch], self.t[i:i+self.batch], wc=wc)
                print(loss.data)
                loss_list.append(loss.data)
                training_model.zerograds()
                loss.backward()
                self.optimizer.update()

                #100epoch毎にモデルを保存する
                if (epoch+1)%100 == 0:
                    training_model.to_cpu()
                    model_name = self.save_dir + "/epoch" + str(epoch+1) + ".model"
                    chainer.serializers.save_npz(model_name, training_model)
                    training_model.to_gpu(self.gpu_id)
    
            #推論でもバッチサイズの数だけ入力する必要がある．2以上の場合，最後の一つを未知とする．
            #ミニバッチ３枚のうち，最後が未知になるように調整する．
            predict, loss = training_model(self.x[i+self.M-(self.batch-1):i+self.M+1], self.t[i+self.M-(self.batch-1):i+self.M+1])
            #predict, loss = training_model(self.x[i+1:i+self.batch+1], self.t[i+1:i+self.batch+1])
            #p1, p2, p3, fusion, predict, loss = self.model(self.x[i+1:i+self.batch+1], self.t[i+1:i+self.batch+1])
        
            #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
            with cupy.cuda.Device(self.gpu_id):
                if self.dataset_num == 0:
                    add_t = DCN.DCN_output(inference=predict, batchsize=self.batch, p=0.5, ant=True, gpu_id=self.gpu_id)
                    #ここに後処理
                    if self.pp == 1:
                        add_t = chainer.cuda.to_cpu(add_t)
                        add_t = pp.opening(add_t)
                        print("post processed!")
                if self.dataset_num == 1:
                    add_t = DCN.DCN_output(inference=predict, batchsize=self.batch, p=0.5, ant=True, gpu_id=self.gpu_id)
                    #ここに後処理
                    if self.pp == 1:
                        add_t = chainer.cuda.to_cpu(add_t)
                        add_t = pp.opening(add_t)
                        print("post processed!")

            # add_tを画像にして保存
            self.output_t(predict=add_t, num=i+self.M+1, gpu_id=self.gpu_id)
            #ここで再びcupyに変換しないとエラーを吐く
            if self.gpu_id >= 0:
                add_t = chainer.cuda.to_gpu(add_t, device=self.gpu_id)
            with cupy.cuda.Device(self.gpu_id):
                self.t[i+self.M] = add_t /255

            # このイテレーションにおける各epochのlossを書き込み
            if i == 0:
                with open(self.save_dir + "/result.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(loss_list)
                print("saved loss_list!")
            else:
                with open(self.save_dir + "/result.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(loss_list)
                print("saved loss_list!")
        
            print(i+1)


if __name__ == "__main__":
    model = DCN_master.DCN_modify10()
    learn = SemisuperviesedLearning(model)
    learn.training()