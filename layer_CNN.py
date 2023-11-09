import numpy as np

#Inicializa los valores de las variables de la capa de la red neuronal convolucional
def layer_CNN(N, M, D, size_in, size_filter, gain): #Eliminar reduction
    ns = size_in - size_filter + 1
    
    X = np.zeros((size_in, size_in, D, N))
    Y = np.zeros((ns, ns, M, N)) #(20x20x10x1)
    W = gain * 2 * np.random.rand(size_filter, size_filter, D, M) - gain * 1
    B = gain * 2 * np.random.rand(M, 1) - gain * 1
    
    Reg_pool = None
    # if reduction > 0:
    #     Reg_pool = np.zeros((size_in, size_in, 3, D, N))
        
    return X, Y, W, B, #Reg_pool
