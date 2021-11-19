from PIL import Image
import dataset
import numpy as np
import cupy as cp
import math
import DCN
import dataset as ds
import cv2
import chainer
import networks

img_path = "../output_for_thesis/for_journal/ant_few_unet2/evaluate_img_19400/model19400_" 

def iou_calc(tp, pp):
    mask = pp == 0
    pp[mask] = 0
    mask = pp != 0
    pp[mask] = 1
    
    intersection = tp + pp
    mask=intersection !=2
    intersection[mask] = 0
    intersection = intersection / 2
    
    iou = intersection.sum() / (tp.sum() + pp.sum() - intersection.sum())
    
    return iou

def calc_all(tp, pp):
    mask = pp != 0
    pp[mask] = 1
    tn, fp, fn, tp = confusion_matrix(tp.flatten(), pp.flatten()).ravel()
    presicion = tp / (tp + fp)
    recall = tp / (tp + fn)
    dice = tp / (tp + ((1/2)*(fp+fn)))
    iou = tp / (tp + fp + fn)
    return presicion, recall, dice, iou

def iou_calc2(tp, pp):
    pp /= 255
    pp = pp - 1
    mask = pp != 0
    pp[mask] = 1 #予測画像の
    #print(tp)
    #print(pp.sum())
    intersection = tp + pp
    mask=intersection !=2
    intersection[mask] = 0
    intersection = intersection / 2
    
    iou = intersection.sum() / (tp.sum() + pp.sum() - intersection.sum())
    
    return iou

def to_array01(img):
    array = np.array(img) / 255
    return array

def to_img(array):
    img = Image.fromarray(np.array(array*255, dtype=np.uint8))
    return img

def opening(img):
    array = to_array01(img)
    kernel = np.ones((3,3),np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
    img = to_img(array)
    return img

def closing(img):
    array = to_array01(img)
    kernel = np.ones((2,2),np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel)
    img = to_img(array)
    return img


def evaluate(img_path, dataset, n, trained_model="", postprocess=False, gpu_id=0):
    #n=1の場合，教師あり学習のみを用いることを意味する
    iou_list = []
    if dataset == 0:
        if n == 1:
            slices = 97
            #ここでいったん最後まで推論しておく必要がある．
            for i in range(slices):
                test_img_path = "../data/ant_experiment/train/train"
                target_img_path = "../data/ant_experiment/label/label"
                infer(model_path=trained_model, test_img_path=test_img_path, target_img_path=target_img_path, save_path=img_path, img_index=i+3, dataset=dataset, gpu_id=gpu_id)
        else:
            slices = n
        for i in range(slices):
            tp_name = "../data/ant_experiment/label/label" + "{0:03d}".format(i+(100-slices)) + ".tif"
            tp = ds.plain(tp_name, ant=True, mode="label", channels=1, width=512, height=512)
            img = Image.open(img_path + "/result_img/" + "{0:03d}".format(i+(100-slices+1)) + ".jpg")
            if postprocess == 1:
                img = opening(img)
            img = np.array(img, dtype=np.float64)
            mask = img <= 200
            img[mask] = 0
            pp = img
            iou = iou_calc(tp, pp)
            iou_list.append(iou)
         
    if dataset == 1:
        if n == 1:
            slices = 27
            for i in range(slices):
                test_img_path = "../data/ISBI2012_experiment/train_volume30/data/train_volume30_"
                target_img_path = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_"
                infer(model_path=trained_model, test_img_path=test_img_path, target_img_path=target_img_path, save_path=img_path, img_index=i+3, dataset=dataset, gpu_id=gpu_id)
        else:
            slices = n
        for i in range(slices):
            tp_name = "../data/ISBI2012_experiment/train_labels30/data/train_labels30_" + "{0:04d}".format(i+(30-slices)) + ".jpg"
            tp = ds.plain(tp_name, ant=False, mode="label", channels=1, width=512, height=512)
            img = Image.open(img_path + "/result_img/" + "{0:03d}".format(i+(30-slices+1)) + ".jpg")
            if postprocess == 1:
                img = opening(img)
            img = np.array(img, dtype=np.float64)
            mask = img <= 200
            img[mask] = 0
            pp = img
            iou = iou_calc2(tp, pp)
            iou_list.append(iou)
        
    return sum(iou_list) / len(iou_list), iou_list


def infer(model_path, test_img_path, target_img_path, img_index, img_save=True, save_path="", dataset=0, gpu_id=0):
    gpu_id = gpu_id
    chainer.cuda.get_device(gpu_id).use()
    model = networks.UNet()
    model.to_gpu()
    chainer.serializers.load_npz(model_path, model)
    if dataset == 0:
        ant = True
        data_x_name = test_img_path + "{0:03d}".format(img_index)+ ".tif"
        data_t_name = target_img_path + "{0:03d}".format(img_index) + ".tif"
    else:
        ant = False
        data_x_name = test_img_path + "{0:04d}".format(img_index)+ ".jpg"
        data_t_name = target_img_path + "{0:04d}".format(img_index) + ".jpg"
    img_x = ds.plain(data_x_name, ant=ant, mode="data", channels=1, width=512, height=512)
    img_t = ds.plain(data_t_name, ant=ant, mode="label", channels=1, width=512, height=512)
    xx = np.zeros((1, 1, 512, 512)).astype(np.float32)
    tt = np.zeros((1, 512, 512)).astype(np.int32)
    xx[0] = img_x
    tt[0] = img_t
    xx = cp.asarray(xx)
    tt = cp.asarray(tt)
    #print(xx.shape)
    with chainer.using_config("train", False):
        predict, loss = model(xx, tt)
    output = DCN.DCN_output(predict, batchsize=1, p=0.5, ant=ant, gpu_id=gpu_id)
    output = chainer.cuda.to_cpu(output)
    #print(output.shape)
    output3 = np.copy(output)
        
    if img_save==True:
        #output *= 255
        #print(output)
        output = Image.fromarray(np.uint8(output))
        output.save(save_path + "/result_img/" + "{0:03d}".format(img_index+1) + ".jpg")