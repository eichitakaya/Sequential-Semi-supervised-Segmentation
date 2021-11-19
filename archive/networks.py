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


class DCN123(Chain):
    def __init__(self):
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
        return predict, loss

#------------------------------------------------------------
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