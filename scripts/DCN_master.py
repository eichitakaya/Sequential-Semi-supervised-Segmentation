import numpy as np
np.random.seed(100)
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from chainer import link, Chain, optimizers, Variable
from PIL import Image
if chainer.cuda.available:
    import cupy as cp
    cp.random.seed(100)

#-------------------------------------------------------------------------------------------------------------
class DCN123(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN123, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify1(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify1, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=4, stride=4, pad=0)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 10, stride=8, pad=0)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify2(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify2, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=4, stride=4, pad=0)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 20, stride=8, pad=5)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify3(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify3, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize= 2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 20, stride=8, pad=5)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify4(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify4, self).__init__()
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

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 10, stride=8, pad=0)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify5(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify5, self).__init__()
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

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize= 20, stride=8, pad=5)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify6(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify6, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify7(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify7, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=10, stride=4, pad=3)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=18, stride=8, pad=4)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
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
class DCN_modify8(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify8, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=6, stride=4, pad=1)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=16, stride=8, pad=3)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN_modify9(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify9, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=10, stride=4, pad=3)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=16, stride=8, pad=3)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN_modify10(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify10, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=6, stride=4, pad=1)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=16, stride=8, pad=3)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN_modify11(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify11, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=6, stride=2, pad=2)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=16, stride=8, pad=3)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN_modify12(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify12, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=14, stride=8, pad=2)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
class DCN_modify13(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify13, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=12, stride=8, pad=1)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
class DCN_modify5_2(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify5_2, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=4, stride=2, pad=1)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=18, stride=8, pad=4)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN_modify3_2(Chain):
    def __init__(self):
        self.vis_p1_1 = 0
        self.vis_p1_2 = 0
        self.vis_p2_1 = 0
        self.vis_p2_2 = 0
        self.vis_p3_1 = 0
        self.vis_p3_2 = 0
        super(DCN_modify3_2, self).__init__()
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

            self.upconv_pool1 = L.Deconvolution2D(128, 2, ksize=2, stride=2, pad=0)
            self.conv_pool1_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool1_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up1 = L.BatchNormalization(2)
            self.bnorm_pool1_1 = L.BatchNormalization(2)
            self.bnorm_pool1_2 = L.BatchNormalization(2)

            self.upconv_pool2 = L.Deconvolution2D(256, 2, ksize=8, stride=4, pad=2)
            self.conv_pool2_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=1)
            self.conv_pool2_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up2 = L.BatchNormalization(2)
            self.bnorm_pool2_1 = L.BatchNormalization(2)
            self.bnorm_pool2_2 = L.BatchNormalization(2)


            self.upconv_pool3 = L.Deconvolution2D(512, 2, ksize=18, stride=8, pad=4)
            self.conv_pool3_1 = L.Convolution2D( 2, 2, 3, stride=1, pad=0)
            self.conv_pool3_2 = L.Convolution2D( 2, 2, 1, stride=1, pad=0)
            self.bnorm_up3 = L.BatchNormalization(2)
            self.bnorm_pool3_1 = L.BatchNormalization(2)
            self.bnorm_pool3_2 = L.BatchNormalization(2)

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        p1 = F.relu(self.upconv_pool1(h))
        self.vis_p1_1 = p1
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        self.vis_p1_2 = p1
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        p2 = F.relu(self.upconv_pool2(h))
        self.vis_p2_1 = p2
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        self.vis_p2_2 = p2
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        p3 = F.relu(self.upconv_pool3(h))
        self.vis_p3_1 = p3
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        self.vis_p3_2 = p3
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)


        loss = wc*(C1+C2+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss#-------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------
class DCN123NoBN(Chain):
    def __init__(self):
        self.vis_h1 = 0
        self.vis_h2 = 0
        self.vis_h3 = 0
        super(DCN123NoBN, self).__init__()
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        self.vis_h1 = h
        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.conv_pool1_1(p1))
        p1 = F.relu(self.conv_pool1_2(p1))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        self.vis_h2 = h
        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.conv_pool2_1(p2))
        p2 = F.relu(self.conv_pool2_2(p2))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        self.vis_h3 = h
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
class DCN1(Chain):
    def __init__(self):
        super(DCN1, self).__init__()
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h
        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        fusion = p1
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p2 = p1
        p3 = p1

        loss = fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------
class DCN2(Chain):
    def __init__(self):
        super(DCN2, self).__init__()
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

    def __call__(self, x, t):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        vis_h2 = h
        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        fusion = p2
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p3 = p2
        p1 = p2

        loss = fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------
class DCN3(Chain):
    def __init__(self):
        super(DCN3, self).__init__()
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

    def __call__(self, x, t):
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        vis_h2 = h

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        vis_h3 = h
        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        fusion = p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p1 = p3
        p2 = p3

        loss = fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------

class DCN12(Chain):
    def __init__(self):
        super(DCN12, self).__init__()
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h
        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        vis_h2 = h
        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        fusion = p1 + p2
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p3 = p1

        loss = wc*(C1+C2) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss

#-------------------------------------------------------------------------------------------------------------
class DCN13(Chain):
    def __init__(self):
        super(DCN13, self).__init__()
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h
        p1 = F.relu(self.upconv_pool1(h))
        p1 = F.relu(self.bnorm_pool1_1(self.conv_pool1_1(p1)))
        p1 = F.relu(self.bnorm_pool1_2(self.conv_pool1_2(p1)))

        C1 = F.softmax_cross_entropy(p1, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        vis_h2 = h
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        vis_h3 = h
        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p1 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p2 = p1

        loss = wc*(C1+C3) + wf*fusion_loss

        predict = F.softmax(fusion)
        #p1 = F.softmax(p1)
        #p2 = F.softmax(p2)
        #p3 = F.softmax(p3)
        return p1, p2, p3, fusion, predict, loss
#-------------------------------------------------------------------------------------------------------------
class DCN23(Chain):
    def __init__(self):
        super(DCN23, self).__init__()
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

    def __call__(self, x, t, wc=1, wf=1):
        wc = wc
        x, t = chainer.Variable(x), chainer.Variable(t)
        h = F.relu(self.bnorm1_1(self.conv1_1(x)))
        h = F.relu(self.bnorm1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bnorm2_1(self.conv2_1(h)))
        h = F.relu(self.bnorm2_2(self.conv2_2(h)))
        vis_h1 = h

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm3_1(self.conv3_1(h)))
        h = F.relu(self.bnorm3_2(self.conv3_2(h)))
        h = F.relu(self.bnorm3_3(self.conv3_3(h)))
        vis_h2 = h
        p2 = F.relu(self.upconv_pool2(h))
        p2 = F.relu(self.bnorm_pool2_1(self.conv_pool2_1(p2)))
        p2 = F.relu(self.bnorm_pool2_2(self.conv_pool2_2(p2)))

        C2 = F.softmax_cross_entropy(p2, t)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.bnorm4_1(self.conv4_1(h)))
        h = F.relu(self.bnorm4_2(self.conv4_2(h)))
        h = F.relu(self.bnorm4_3(self.conv4_3(h)))
        vis_h3 = h
        p3 = F.relu(self.upconv_pool3(h))
        p3 = F.relu(self.bnorm_pool3_1(self.conv_pool3_1(p3)))
        p3 = F.relu(self.bnorm_pool3_2(self.conv_pool3_2(p3)))

        C3 = F.softmax_cross_entropy(p3, t)

        fusion = p2 + p3
        fusion_loss = F.softmax_cross_entropy(fusion, t)
        p1 = p2

        loss = wc*(C2+C3) + wf*fusion_loss

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
