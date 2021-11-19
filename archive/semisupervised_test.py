import chainer
from chainer import cuda
from chainer.datasets import split_dataset
from chainer import optimizers

import dataset as ds

import DCN
import DCN_master

class SemisuperviesedLearning:
    def __init__(self, model, _lambda=0.0005, n=3, epoch=1, batch=3, gpu_id=-1):
        self.train251, self.test = ds.set_ANT()
        self.train, self.valid = split_dataset(self.train251, 100)
        self.train_t, self.train_v = split_dataset(self.train, 3)
        self.x, self.t = chainer.dataset.concat_examples(self.train, gpu_id)

        self.n = n
        self.epoch = epoch
        self.batch = batch
        print("loaded dataset!")
        
        self.model = model
        if gpu_id >= 0:
            model.to_gpu(gpu_id)
        print("loaded model!")
        
        self.optimizer = self.set_optimizer(self.model, _lambda)
        print("set optimizer!")



    def set_optimizer(self, model, _lambda):
        #最適化手法にテコ入れする際は，ここをイジる．
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(_lambda))
        return optimizer
    
    def training(self):
        for i in range(self.n):#画像の枚数だけ繰り返し
            for epoch in range(self.epoch):
                #if epoch%10 == 0 and wc > 0.011:
                    #wc = wc * 0.1
                    #wf = wf * 10
    
                p1, p2, p3, fusion, predict, loss = self.model(self.x[i:i+self.batch], self.t[i:i+self.batch])
                self.model.zerograds()
                loss.backward()
                self.optimizer.update()
    
            #推論でもバッチサイズの数だけ入力する必要がある．2以上の場合，最後の一つを未知とする．
            p1, p2, p3, fusion, predict, loss = self.model(self.x[i+1:i+self.batch+1], self.t[i+1:i+self.batch+1])
        
            #１つだけ得られた新たな推論結果を，tの該当箇所(n+batch番目)に格納する．
            add_t = DCN.DCN_output(predict, self.batch)
            self.t[i+self.batch] = add_t /255
        
            print(i+1)


if __name__ == "__main__":
    model = DCN_master.DCN_modify10()
    learn = SemisuperviesedLearning(model)
    learn.training()