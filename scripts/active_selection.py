import numpy as np
import pandas as pd
import random
from fractal_analysis import fractal_dimension

def select_init(volume) -> np.ndarray:
    
    return X, Selected_T, GT

def calc_fractal_dimension_series(volume):
    dimension_list = []
    print(volume.shape)
    for i in range(volume.shape[0]):
        dimension_list.append(fractal_dimension(volume[i]))
    series = pd.Series(dimension_list)
    return series

def calc_max_variance_group(volume, window_size):
    fractal_dimensions = calc_fractal_dimension_series(volume)
    variance_series = fractal_dimensions.rolling(window_size).var()
    max_var_index = variance_series.idxmax()
    max_variance_group = list(range(max_var_index-(window_size-1), max_var_index+1))
    
    return max_variance_group

def calc_max_mean_group(volume, window_size):
    fractal_dimensions = calc_fractal_dimension_series(volume)
    variance_series = fractal_dimensions.rolling(window_size).mean()
    max_mean_index = variance_series.idxmax()
    max_mean_group = list(range(max_mean_index-(window_size-1), max_mean_index+1))
    
    return max_mean_group

def random_selection(n, window_size):
    selected = random.randint(0, n-window_size)
    random_selected_list = list(range(selected, selected+window_size))
    return random_selected_list