from PIL import Image
import dataset
import numpy as np
import math
import DCN
import dataset as ds
import cv2


def to_array01(img):
    array = np.array(img) / 255
    return array

def to_img(array):
    img = Image.fromarray(np.array(array*255, dtype=np.uint8))
    return img

def opening(img):
    array = to_array01(img)
    kernel = np.ones((3,3),np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel) * 255
    return array

def closing(img):
    array = to_array01(img)
    kernel = np.ones((2,2),np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel) * 255
    return array