import numpy as np

def add_axis(arr):
    arr_new = np.expand_dims(arr, 0)
    return arr_new
