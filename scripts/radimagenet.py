import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import torch_networks as networks
import torch_loss_functions as lf

from data_loader import DataSetFor2DSegmentation, GroupWiseSplit

from original_transforms import add_axis

transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(add_axis)])


spleen = GroupWiseSplit(target_volume="spleen", val_ratio=0.3, group_shuffle=False)
spleen_train = DataSetFor2DSegmentation(spleen.train_groups, target_focus=True, transform=transform)
spleen_val = DataSetFor2DSegmentation(spleen.val_groups, target_focus=True, transform=transform)

# パラメータ設定
train_size = len(spleen_train)
val_size = len(spleen_val)
channels = 1
width = 320
height = 320
epoch_n = 100
batchsize = 10

train_loader = DataLoader(spleen_train, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(spleen_val, batch_size=batchsize)

def img_output(inference,  p=0.5, ant=False, gpu_id=-1):
    # 入力は2チャンネルのVariable
    inference = inference.data[0] # 最後の一枚の１チャンネル目
    
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
    
        output = inference

    return output


if torch.cuda.is_available():
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id))
    
# モデルの定義
model = networks.UNet()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model = model.to("cuda")

# optimizerの準備
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

# 誤差関数の定義
criterion = lf.BCEDiceLoss()
#criterion = nn.CrossEntropyLoss()

# 訓練ループ

train_loss_list = []
val_loss_list = []

for epoch in range(epoch_n):
    train_loss_add = 0
    model.train()
    for i, data in enumerate(train_loader):
        x, t = data
        x = torch.tensor(x)
        t = torch.tensor(t).float()
        
        x = x.to("cuda")
        t = t.to("cuda")
        
        predict = model(x)
        
        loss = criterion(predict, t)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_add += loss.data
        model.eval()
        demo_pred = model(demo_img)
        demo_pred = demo_pred[3].to("cpu").detach().numpy().copy() * 255
        plt.imsave("../result/predicted_imgs/iter_" + str(epoch * (int(train_size/batchsize)) + i) +".png", np.rot90(demo_pred[0]), cmap="gray")
        model.train()
        
    loss_mean = train_loss_add / int(train_size/batchsize)
    print("epoch" + str(epoch+1))
    print("train_loss:" + str(loss_mean))
    train_loss_list.append(loss_mean)
    
    # validation
    model.eval()
    val_loss_add = 0
    num = 0
    for i, data in enumerate(val_loader):
            
        #cudaに変換
        x, t = data
        x = torch.tensor(x)
        t = torch.tensor(t).float()
        x = x.to("cuda")
        t = t.to("cuda")
        predict = model(x)
        
        loss = criterion(predict, t)
        val_loss_add += loss.data
    n_data = 6
    row = 1
    col = 3

    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(12,8))

    #fig.suptitle("学習経過")
    
    #img_x = x[3].to("cpu").detach().numpy().copy()
    #img_t = t[3].to("cpu").detach().numpy().copy()
    #pred = predict[3].to("cpu").detach().numpy().copy() * 255

#     ax[0].set_title("raw_image")
#     ax[0].axes.xaxis.set_visible(False)
#     ax[0].axes.yaxis.set_visible(False)
#     ax[0].imshow(img_x[0], cmap="gray")

#     ax[1].set_title("target")
#     ax[1].axes.xaxis.set_visible(False)
#     ax[1].axes.yaxis.set_visible(False)
#     ax[1].imshow(img_t[0], cmap="gray")

#     ax[2].set_title("predict")
#     ax[2].axes.xaxis.set_visible(False)
#     ax[2].axes.yaxis.set_visible(False)
#     ax[2].imshow(pred[0], cmap="gray")
#    plt.savefig("../result/demo/epoch_" + str(epoch+1) + ".png")
#    plt.imsave("../result/predicted_imgs/epoch_" + str(epoch+1) + ".png", np.rot90(pred[0]), cmap="gray")
    loss_mean = val_loss_add / int(val_size/batchsize)
    val_loss_list.append(loss_mean)
    print("val_loss:" + str(loss_mean))