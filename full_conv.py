import numpy as np
from scipy.signal import convolve2d

def full_conv(in_matrix, w_matrix):
    zin = in_matrix.shape
    zw = w_matrix.shape
    
    in_aux = np.zeros((zin[0] + (zw[0] - 1) * 2, zin[1] + (zw[1] - 1) * 2))
    in_aux[zw[0] - 1:zw[0] - 1 + zin[0], zw[1] - 1:zw[1] - 1 + zin[1]] = in_matrix
    
    x = convolve2d(in_aux, w_matrix, mode='valid')
    
    return x
