import numpy as np
np.random.seed(100)
import cupy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from chainer import link, Chain, optimizers, Variable
from PIL import Image
if chainer.cuda.available:
    import cupy as cp
    cp.random.seed(100)

# output function
def DCN_output(inference, batchsize, p=0.5, ant=False, gpu_id=-1):
    # 入力は2チャンネルのVariable
    with cupy.cuda.Device(gpu_id):
        inference = inference.data[batchsize-1][0] # 最後の一枚の１チャンネル目
        
        inference = inference * 255
        threshold = 255 * p

        if ant==False:
            mask255 = inference > threshold
            inference[mask255] = 255

            mask0 = inference <= threshold
            inference[mask0] = 0
        else:
            mask0 = inference > threshold
            inference[mask0] = 0

            mask255 = inference != 0
            inference[mask255] = 255
    
        if gpu_id < 0:
            inference = chainer.cuda.to_cpu(inference)
        output = inference

    return output

# Deep Contextual Networks
class DCNAll(Chain):
    def __init__(self):
        super(DCNAll, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bnorm1_1 = L.BatchNormalization(64),


            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
			conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
			conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
			conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
			conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),

            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
			conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
			conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.conv_pool3_1(p3))
        p3 = F.relu(self.conv_pool3_2(p3))

        fusion = p1 + p2 + p3

        loss = F.softmax_cross_entropy(fusion, t)
        predict = F.softmax(fusion)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------
class DCNAll_BN(Chain):
    def __init__(self):
        super(DCNAll_BN, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bnorm1_1 = L.BatchNormalization(64),
            bnorm1_2 = L.BatchNormalization(64),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bnorm2_1 = L.BatchNormalization(128),
            bnorm2_2 = L.BatchNormalization(128),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            bnorm3_1 = L.BatchNormalization(256),
            bnorm3_2 = L.BatchNormalization(256),
            bnorm3_3 = L.BatchNormalization(256),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bnorm4_1 = L.BatchNormalization(512),
            bnorm4_2 = L.BatchNormalization(512),
            bnorm4_3 = L.BatchNormalization(512),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
            conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up1 = L.BatchNormalization(2),
            bnorm_pool1_1 = L.BatchNormalization(2),
            bnorm_pool1_2 = L.BatchNormalization(2),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
            conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up2 = L.BatchNormalization(2),
            bnorm_pool2_1 = L.BatchNormalization(2),
            bnorm_pool2_2 = L.BatchNormalization(2),


            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
            conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
            conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up3 = L.BatchNormalization(2),
            bnorm_pool3_1 = L.BatchNormalization(2),
            bnorm_pool3_2 = L.BatchNormalization(2),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.bnorm_up1(self.upconv_pool1(h)))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))

        p2 = F.relu(self.bnorm_up2(self.upconv_pool2(h)))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))

        p3 = F.relu(self.bnorm_up3(self.upconv_pool3(h)))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        fusion = p1 + p2 + p3

        loss = F.softmax_cross_entropy(fusion, t)
        predict = F.softmax(fusion)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------
class DCNFirst(Chain):
    def __init__(self):
        super(DCNFirst, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bnorm1_1 = L.BatchNormalization(64),
            bnorm1_2 = L.BatchNormalization(64),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bnorm2_1 = L.BatchNormalization(128),
            bnorm2_2 = L.BatchNormalization(128),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
            conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up1 = L.BatchNormalization(2),
            bnorm_pool1_1 = L.BatchNormalization(2),
            bnorm_pool1_2 = L.BatchNormalization(2),


        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.upconv_pool1(h))
        # p1 = F.relu(self.conv_pool1_1(p1))
        # p1 = F.relu(self.conv_pool1_2(p1))

        #p2とp3も仮に置いておく
        p2 = p1
        p3 = p1

        fusion = p1

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNSecond(Chain):
    def __init__(self):
        super(DCNSecond, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),



            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
			conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
			conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        p1 = self.upconv_pool1(h)
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = self.upconv_pool2(h)
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))



        fusion = p1 + p2
        p3 = p1

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNC2(Chain):
    def __init__(self):
        super(DCNC2, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
			conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = self.upconv_pool2(h)
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))



        fusion = p2
        p1 = p2
        p3 = p2

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNC3(Chain):
    def __init__(self):
        super(DCNC3, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
			conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
			conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
			conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
			conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = self.upconv_pool3(h)
        p3_memo = F.relu(self.conv_pool3_1(p3))
        #print("p3_memo")
        p3 = F.relu(self.conv_pool3_2(p3_memo))

        fusion = p3
        p1 = p3
        p2 = p3

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNC2C3(Chain):
    def __init__(self):
        super(DCNC2C3, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
			conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
			conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
			conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),

            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
			conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
			conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))


        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = self.upconv_pool2(h)
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = self.upconv_pool3(h)
        p3 = F.relu(self.conv_pool3_1(p3))
        p3 = F.relu(self.conv_pool3_2(p3))

        fusion = p2 + p3

        p1 = p2

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNC1C3(Chain):
    def __init__(self):
        super(DCNC1C3, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
			conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
			conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
			conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
			conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
			conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
			conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
			conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
			conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),

            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
			conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
			conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        p1 = self.upconv_pool1(h)
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = self.upconv_pool3(h)
        p3 = F.relu(self.conv_pool3_1(p3))
        p3 = F.relu(self.conv_pool3_2(p3))

        fusion = p1 + p3
        p2 = p1

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict
#-------------------------------------------------------------------------------
class DCNBNAll(Chain):
    def __init__(self):
        super(DCNBNAll, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bnorm1_1 = L.BatchNormalization(64),
            bnorm1_2 = L.BatchNormalization(64),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bnorm2_1 = L.BatchNormalization(128),
            bnorm2_2 = L.BatchNormalization(128),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            bnorm3_1 = L.BatchNormalization(256),
            bnorm3_2 = L.BatchNormalization(256),
            bnorm3_3 = L.BatchNormalization(256),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bnorm4_1 = L.BatchNormalization(512),
            bnorm4_2 = L.BatchNormalization(512),
            bnorm4_3 = L.BatchNormalization(512),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
            conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up1 = L.BatchNormalization(2),
            bnorm_pool1_1 = L.BatchNormalization(2),
            bnorm_pool1_2 = L.BatchNormalization(2),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
            conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up2 = L.BatchNormalization(2),
            bnorm_pool2_1 = L.BatchNormalization(2),
            bnorm_pool2_2 = L.BatchNormalization(2),


            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
            conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
            conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up3 = L.BatchNormalization(2),
            bnorm_pool3_1 = L.BatchNormalization(2),
            bnorm_pool3_2 = L.BatchNormalization(2),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        fusion = p1 + p2 + p3

        if train==True:
            loss = F.softmax_cross_entropy(fusion, t)
            return p1, p2, p3, fusion, loss
        else:
            predict = F.softmax(fusion)
            print("predict")
            return p1, p2, p3, fusion, predict

#-------------------------------------------------------------------------------
class DCNcomplete(Chain):
    def __init__(self):
        super(DCNcomplete, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1)
            self.bnorm1_1 = L.BatchNormalization(64)
            self.bnorm1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.bnorm2_1 = L.BatchNormalization(128)
            self.bnorm2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.bnorm3_1 = L.BatchNormalization(256)
            self.bnorm3_2 = L.BatchNormalization(256)
            self.bnorm3_3 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bnorm4_1 = L.BatchNormalization(512)
            self.bnorm4_2 = L.BatchNormalization(512)
            self.bnorm4_3 = L.BatchNormalization(512)

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)
            

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.conv_pool3_1(p3))
        p3 = F.relu(self.conv_pool3_2(p3))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        #C = cp.array([[C1.data, C2.data, C3.data]], dtype=np.float32)
        #C = self.discount_w(C)
        #C = Variable(C.data[0][0])

        loss = C1 + C2 + C3 + fusion_loss

        predict = F.softmax(fusion)
        p1 = F.softmax(p1)
        p2 = F.softmax(p2)
        p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------
class DCNcompleteBN(Chain):
    def __init__(self):
        super(DCNcompleteBN, self).__init__(
            conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1),
            bnorm1_1 = L.BatchNormalization(64),
            bnorm1_2 = L.BatchNormalization(64),

            conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bnorm2_1 = L.BatchNormalization(128),
            bnorm2_2 = L.BatchNormalization(128),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            bnorm3_1 = L.BatchNormalization(256),
            bnorm3_2 = L.BatchNormalization(256),
            bnorm3_3 = L.BatchNormalization(256),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bnorm4_1 = L.BatchNormalization(512),
            bnorm4_2 = L.BatchNormalization(512),
            bnorm4_3 = L.BatchNormalization(512),

            upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1),
            conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up1 = L.BatchNormalization(2),
            bnorm_pool1_1 = L.BatchNormalization(2),
            bnorm_pool1_2 = L.BatchNormalization(2),

            upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2),
            conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1),
            conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up2 = L.BatchNormalization(2),
            bnorm_pool2_1 = L.BatchNormalization(2),
            bnorm_pool2_2 = L.BatchNormalization(2),


            upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3),
            conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0),
            conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0),
            bnorm_up3 = L.BatchNormalization(2),
            bnorm_pool3_1 = L.BatchNormalization(2),
            bnorm_pool3_2 = L.BatchNormalization(2),
        )
        self.train = True

    def __call__(self, x, t, train=True):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)

        loss = C1 + C2 + C3 + fusion_loss

        if train==True:
            predict = F.softmax(fusion)
            p1 = F.softmax(p1)
            p2 = F.softmax(p2)
            p3 = F.softmax(p3)
            return p1, p2, p3, fusion, predict, loss
        else:
            # 出力としてlossとpredictのどちらが正しいか分からないので，両方出しておく
            predict = F.softmax(fusion)
            p1 = F.softmax(p1)
            p2 = F.softmax(p2)
            p3 = F.softmax(p3)
            print("predict")
            return p1, p2, p3, fusion, predict

#-------------------------------------------------------------------------------------------------------------

class DCNcomplete_w_He(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(DCNcomplete_w_He, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm1_1 = L.BatchNormalization(64)
            self.bnorm1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1, initialW=initializer)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm2_1 = L.BatchNormalization(128)
            self.bnorm2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm3_1 = L.BatchNormalization(256)
            self.bnorm3_2 = L.BatchNormalization(256)
            self.bnorm3_3 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm4_1 = L.BatchNormalization(512)
            self.bnorm4_2 = L.BatchNormalization(512)
            self.bnorm4_3 = L.BatchNormalization(512)

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1, initialW=initializer)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2, initialW=initializer)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3, initialW=initializer)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0, initialW=initializer)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)
            
    def __call__(self, x, t, wc=1, wf=1, train=True):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.conv_pool3_1(p3))
        p3 = F.relu(self.conv_pool3_2(p3))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        

        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------

class DCNcomplete_w_He_BN(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(DCNcomplete_w_He_BN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm1_1 = L.BatchNormalization(64)
            self.bnorm1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1, initialW=initializer)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm2_1 = L.BatchNormalization(128)
            self.bnorm2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm3_1 = L.BatchNormalization(256)
            self.bnorm3_2 = L.BatchNormalization(256)
            self.bnorm3_3 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm4_1 = L.BatchNormalization(512)
            self.bnorm4_2 = L.BatchNormalization(512)
            self.bnorm4_3 = L.BatchNormalization(512)

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1, initialW=initializer)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2, initialW=initializer)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 16, stride=8, pad=3, initialW=initializer)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0, initialW=initializer)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)
            
    def __call__(self, x, t, wc=1, wf=1, train=True):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        

        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------

class DCN_modified(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(DCN_modified, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(  1,  64, 3, stride=1, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D( 64,  64, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm1_1 = L.BatchNormalization(64)
            self.bnorm1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D( 64, 128, 3, stride=1, pad=1, initialW=initializer)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm2_1 = L.BatchNormalization(128)
            self.bnorm2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm3_1 = L.BatchNormalization(256)
            self.bnorm3_2 = L.BatchNormalization(256)
            self.bnorm3_3 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=initializer)
            self.bnorm4_1 = L.BatchNormalization(512)
            self.bnorm4_2 = L.BatchNormalization(512)
            self.bnorm4_3 = L.BatchNormalization(512)

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 4, stride=2, pad=1, initialW=initializer)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize= 8, stride=4, pad=2, initialW=initializer)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1, initialW=initializer)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 48, stride=8, pad=19, initialW=initializer)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0, initialW=initializer)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0, initialW=initializer)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)
            
    def __call__(self, x, t, wc=1, wf=1, train=True):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))

        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))

        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))

        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        

        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss


#----------------------------------------------------------------------------------------------------------------------------

class UNet(Chain):

    def __init__(self, n_class=2):
        super(UNet, self).__init__()
        with self.init_scope():
            self.n_class = n_class

            self.enco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
            self.enco1_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

            self.enco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.enco2_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)

            self.enco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.enco3_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            self.enco4_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.enco4_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

            self.enco5_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)

            self.deco6_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)
            self.deco6_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

            self.deco7_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.deco7_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            self.deco8_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.deco8_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)

            self.deco9_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.deco9_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
            self.deco9_3 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

            self.final_layer = L.Convolution2D(None, n_class, ksize=1)

            self.bn1_1 = L.BatchNormalization(  64)
            self.bn1_2 = L.BatchNormalization(  64)

            self.bn2_1 = L.BatchNormalization( 128)
            self.bn2_2 = L.BatchNormalization( 128)

            self.bn3_1 = L.BatchNormalization( 256)
            self.bn3_2 = L.BatchNormalization( 256)

            self.bn4_1 = L.BatchNormalization( 512)
            self.bn4_2 = L.BatchNormalization( 512)

            self.bn5_1 = L.BatchNormalization(1012)

            self.bn6_1 = L.BatchNormalization(1012)
            self.bn6_2 = L.BatchNormalization( 512)

            self.bn7_1 = L.BatchNormalization( 512)
            self.bn7_2 = L.BatchNormalization( 256)

            self.bn8_1 = L.BatchNormalization( 256)
            self.bn8_2 = L.BatchNormalization( 128)

            self.bn9_1 = L.BatchNormalization( 128)
            self.bn9_2 = L.BatchNormalization(  64)
            self.bn9_3 = L.BatchNormalization(  64)

    def __call__(self, x, t): #x = (batchsize, 3, 360, 480)
        #if LRN:
        #    x = F.local_response_normalization(x) #Needed for preventing from overfitting

        h1_1 = F.relu(self.bn1_1(self.enco1_1(x)))
        h1_2 = F.relu(self.bn1_2(self.enco1_2(h1_1)))

        pool1 = F.max_pooling_2d(h1_2, 2, stride=2, return_indices=False) #(batchsize,  64, 180, 240)

        h2_1 = F.relu(self.bn2_1(self.enco2_1(pool1)))
        h2_2 = F.relu(self.bn2_2(self.enco2_2(h2_1)))
        pool2 = F.max_pooling_2d(h2_2, 2, stride=2, return_indices=False) #(batchsize, 128,  90, 120) 

        h3_1 = F.relu(self.bn3_1(self.enco3_1(pool2)))
        h3_2 = F.relu(self.bn3_2(self.enco3_2(h3_1)))
        pool3 = F.max_pooling_2d(h3_2, 2, stride=2, return_indices=False) #(batchsize, 256,  45,  60) 

        h4_1 = F.relu(self.bn4_1(self.enco4_1(pool3)))
        h4_2 = F.relu(self.bn4_2(self.enco4_2(h4_1)))
        pool4 = F.max_pooling_2d(h4_2, 2, stride=2, return_indices=False) #(batchsize, 256,  23,  30) 

        h5_1 = F.relu(self.bn5_1(self.enco5_1(pool4)))

        up5 = F.unpooling_2d(h5_1, ksize=2, stride=2, outsize=(pool3.shape[2], pool3.shape[3]))
        h6_1 = F.relu(self.bn6_1(self.deco6_1(F.concat((up5, h4_2)))))
        h6_2 = F.relu(self.bn6_2(self.deco6_2(h6_1)))

        up6 = F.unpooling_2d(h6_2, ksize=2, stride=2, outsize=(pool2.shape[2], pool2.shape[3]))
        h7_1 = F.relu(self.bn7_1(self.deco7_1(F.concat((up6, h3_2)))))
        h7_2 = F.relu(self.bn7_2(self.deco7_2(h7_1)))

        up7 = F.unpooling_2d(h7_2, ksize=2, stride=2, outsize=(pool1.shape[2], pool1.shape[3]))
        h8_1 = F.relu(self.bn8_1(self.deco8_1(F.concat((up7, h2_2)))))
        h8_2 = F.relu(self.bn8_2(self.deco8_2(h8_1)))

        up8 = F.unpooling_2d(h8_2, ksize=2, stride=2, outsize=(x.shape[2], x.shape[3])) #x = (batchsize, 128, 360, 480)
        h9_1 = F.relu(self.bn9_1(self.deco9_1(F.concat((up8, h1_2)))))
        h9_2 = F.relu(self.bn9_2(self.deco9_2(h9_1)))
        h9_3 = F.relu(self.bn9_3(self.deco9_3(h9_2)))

        h = self.final_layer(h9_3)
        
        loss = F.softmax_cross_entropy(h, t)
        
        predict = F.softmax(h)
        return predict, loss