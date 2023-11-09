import numpy as np

#Capas completamente conectadas
def layer_FCC(in_dim, out_dim, gain):
    X = np.zeros((in_dim, 1)) #Entrada
    Y = np.zeros((out_dim, 1)) #Salida
    W = gain * 2 * np.random.rand(out_dim, in_dim) - gain # Filtro convolucional (Peso)
    B = gain * 2 * np.random.rand(out_dim, 1) - gain # Bias (Desviacion)
    
    
    # print(f'X {X} \n Y {Y} \n W {W} \n B {B}')
    
    return X, Y, W, B
