import cv2
import numpy as np
import pylab as pl

def otsu(arr):
    # 方法2 （OpenCVで実装）
    ret, th = cv2.threshold(arr, 0, 255, cv2.THRESH_OTSU)
    # 結果を出力
    #print(th.min(), th.max())
    return th

def fractal_dimension(arr):
    # 入力は2次元のグレースケール行列
    if arr.ndim == 3:
        arr = arr[0]
    arr = otsu(arr)
    # finding all the non-zero pixels
    #print(arr.max(), arr.min())
    pixels=[]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] != 0:
                pixels.append((i,j))
    
    Lx=arr.shape[1]
    Ly=arr.shape[0]
    pixels=pl.array(pixels)
    
    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales=np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
    fd = -coeffs[0]
    
    return fd

#img = cv2.imread("koch.png", cv2.IMREAD_GRAYSCALE)

#gray = otsu(img)

#print(fractal_dimension(img))