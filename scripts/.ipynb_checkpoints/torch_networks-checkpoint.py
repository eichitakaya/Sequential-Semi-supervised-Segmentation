import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, n_class=2, input_channel=1, output_channel=1):
        super(UNet, self).__init__()
        self.n_class = n_class
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        
        self.enco1_1 = nn.Conv2d(self.input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.enco1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.enco2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enco2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.enco3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enco3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.enco4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.enco4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.enco5_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.enco5_2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.deco6_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.deco6_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.deco7_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deco7_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.deco8_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deco8_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.deco9_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deco9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.final_layer = nn.Conv2d(64, self.output_channel, kernel_size=1)

        self.bn1_1 = nn.BatchNorm2d(  64)
        self.bn1_2 = nn.BatchNorm2d(  64)

        self.bn2_1 = nn.BatchNorm2d(  128)
        self.bn2_2 = nn.BatchNorm2d(  128)

        self.bn3_1 = nn.BatchNorm2d(  256)
        self.bn3_2 = nn.BatchNorm2d(  256)

        self.bn4_1 = nn.BatchNorm2d(  512)
        self.bn4_2 = nn.BatchNorm2d(  512)

        self.bn5_1 = nn.BatchNorm2d(  1024)
        self.bn5_2 = nn.BatchNorm2d(  512)

        self.bn6_1 = nn.BatchNorm2d(  512)
        self.bn6_2 = nn.BatchNorm2d(  256)

        self.bn7_1 = nn.BatchNorm2d(  256)
        self.bn7_2 = nn.BatchNorm2d(  128)

        self.bn8_1 = nn.BatchNorm2d(  128)
        self.bn8_2 = nn.BatchNorm2d(  64)

        self.bn9_1 = nn.BatchNorm2d(  64)
        self.bn9_2 = nn.BatchNorm2d(  64)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  

    def forward(self, x): #x = (batchsize, 3, 360, 480)
        #if LRN:
        #    x = F.local_response_normalization(x) #Needed for preventing from overfitting

        h1_1 = F.relu(self.bn1_1(self.enco1_1(x)))
        h1_2 = F.relu(self.bn1_2(self.enco1_2(h1_1)))
        pool1, pool1_indice = F.max_pool2d(h1_2, 2, stride=2, return_indices=True) #(batchsize,  64, 180, 240)

        h2_1 = F.relu(self.bn2_1(self.enco2_1(pool1)))
        h2_2 = F.relu(self.bn2_2(self.enco2_2(h2_1)))
        pool2, pool2_indice = F.max_pool2d(h2_2, 2, stride=2, return_indices=True) #(batchsize, 128,  90, 120) 

        h3_1 = F.relu(self.bn3_1(self.enco3_1(pool2)))
        h3_2 = F.relu(self.bn3_2(self.enco3_2(h3_1)))
        pool3, pool3_indice = F.max_pool2d(h3_2, 2, stride=2, return_indices=True) #(batchsize, 256,  45,  60) 

        h4_1 = F.relu(self.bn4_1(self.enco4_1(pool3)))
        h4_2 = F.relu(self.bn4_2(self.enco4_2(h4_1)))
        pool4, pool4_indice = F.max_pool2d(h4_2, 2, stride=2, return_indices=True) #(batchsize, 256,  23,  30) 

        h5_1 = F.relu(self.bn5_1(self.enco5_1(pool4)))
        h5_2 = F.relu(self.bn5_2(self.enco5_2(h5_1)))
        
        up5 = F.max_unpool2d(h5_2, pool4_indice, kernel_size=2, stride=2, output_size=(pool3.shape[2], pool3.shape[3]))
        h6_1 = F.relu(self.bn6_1(self.deco6_1(torch.cat((up5, h4_2), dim=1))))
        h6_2 = F.relu(self.bn6_2(self.deco6_2(h6_1)))

        up6 = F.max_unpool2d(h6_2, pool3_indice, kernel_size=2, stride=2, output_size=(pool2.shape[2], pool2.shape[3]))
        h7_1 = F.relu(self.bn7_1(self.deco7_1(torch.cat((up6, h3_2), dim=1))))
        h7_2 = F.relu(self.bn7_2(self.deco7_2(h7_1)))

        up7 = F.max_unpool2d(h7_2, pool2_indice, kernel_size=2, stride=2, output_size=(pool1.shape[2], pool1.shape[3]))
        h8_1 = F.relu(self.bn8_1(self.deco8_1(torch.cat((up7, h2_2), dim=1))))
        h8_2 = F.relu(self.bn8_2(self.deco8_2(h8_1)))

        up8 = F.max_unpool2d(h8_2, pool1_indice, kernel_size=2, stride=2, output_size=(x.shape[2], x.shape[3])) #x = (batchsize, 128, 360, 480)
        h9_1 = F.relu(self.bn9_1(self.deco9_1(torch.cat((up8, h1_2), dim=1))))
        h9_2 = F.relu(self.bn9_2(self.deco9_2(h9_1)))

        h = self.final_layer(h9_2)
        #print(h.shape)
        #print(t.shape)
        predict = h
        #loss = 	nn.BCEWithLogitsLoss(h, t)
        
        #predict = nn.Softmax(h)
        return torch.sigmoid(predict)
