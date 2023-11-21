import os
import matplotlib.pyplot as plt
import numpy as np
from layer_CNN import layer_CNN
from layer_FCC import layer_FCC
from full_conv import full_conv
# import view_dataset
from scipy.signal import convolve2d
import pandas as pd
import random
import math
import pickle

depurar = 1

# Funcion switch para las etiquetas de las imagenes de entrada
def switch_yd(yd):
    yd_trt_dict = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    
    return yd_trt_dict.get(yd, None)


# Funcion que carga la mlist
def loadDataset(fileName, samples):
    # Cargar el conjunto de datos CSV
    train_data = pd.read_csv(fileName, header=None)
    
    # Obtener etiquetas y características
    y = np.array(train_data.iloc[:samples, 0])
    x = np.array(train_data.iloc[:samples, 1:])

    return x, y

ruta_archivo_train = "C:\\Users\\luisc\\Downloads\\Luis\\Luis_personal\\Semestre_11\\Almacenamiento de datos\\Proyecto_CIFAR100\\mnist_train.csv"
ruta_archivo_test = "C:\\Users\\luisc\\Downloads\\Luis\\Luis_personal\\Semestre_11\\Almacenamiento de datos\\Proyecto_CIFAR100\\mnist_test.csv"
# Intenta cargar los datos de entrenamiento
try:
    Amnist, y = loadDataset(ruta_archivo_train, 60000)
    print("Archivos de datos cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los archivos de datos: {str(e)}")
    # Vuelve a ejecutar la función loadDataset si ocurre un error
    Amnist, y = loadDataset(ruta_archivo_train, 60000)
    print("Archivos de datos cargados nuevamente.")    

# Intenta cargar los datos de test
try:
    Bmnist, y_test = loadDataset(ruta_archivo_test, 10000)
    print("Archivos de datos cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los archivos de datos: {str(e)}")
    # Vuelve a ejecutar la función loadDataset si ocurre un error
    Bmnist, y_test = loadDataset(ruta_archivo_test, 10000)
    print("Archivos de datos cargados nuevamente.")

# Variable initialization
LR = 1e-2  # Initial learning rate
MT = 100   # Loss function modifier for backpropagation algorithm
c1 = 1e-5  # Constant for adjusting the output layer of the CNN

test_set = 0     # Flag to select between training the CNN or only test the CNN
sobre_train = 0  # Flag to train the CNN from the actual kernels and synaptic weights instead of random values


if test_set == 0:
    if sobre_train == 0:
        # cnn_D0, cnn_M0 = 1, 10
        # cnn_D1, cnn_M1 = cnn_M0, 10
        # cnn_D2, cnn_M2 = cnn_M1, 10
        cnn_M0 = 10
        cnn_M1 = 10
        cnn_M2 = 10
        cnn_D0 = 1
        cnn_D1 = cnn_M0
        cnn_D2 = cnn_M1

        X0, Y0, W0, B0 = layer_CNN(1, cnn_M0, cnn_D0, 28, 9, 1) #Se elimino el penultimo parametro
        X1, Y1, W1, B1 = layer_CNN(1, cnn_D1, cnn_M1, 20, 5, 1) #Se elimino R1
        X2, Y2, W2, B2 = layer_CNN(1, cnn_D2, cnn_M2, 16, 3, 1) #Se elimino R2

        #Modificamos los 10 campos de b2
        # for(i) in range(10):
        #     B0[i] = 0
        #     B1[i] = 0
        #     B2[i] = 0
            
        #Capas completamente conectadas
        X3, Y3, W3, B3 = layer_FCC(1960, 100, 1)
        X4, Y4, W4, B4 = layer_FCC(100, 100, 1)
        X5, Y5, W5, B5 = layer_FCC(100, 10, 1)

# Storage variables
Z1 = np.zeros((28, 28, 1))  # Variable used for displaying the input image in the training of the CNN

# print(Z1)

if test_set == 0:
    # Iterations for training the CNN
    iteraciones = 500000
else:
    # Iterations for test after training the CNN
    iteraciones = 5000

E = np.zeros(iteraciones)  # Loss function of the training set
Etest = np.zeros(iteraciones)  # Loss function of the test set
yCNN = np.zeros((10, iteraciones))  # Output of the CNN
yDPN = np.zeros((10, iteraciones))  # Desired output
sourc = np.zeros((iteraciones, 2))  # Input image source

#Comenzamos el ciclo de entrenamiento

#Bucle principal
temporal = 1

# for K in range(1, iteraciones + 1):
for K in range(0, iteraciones):

    # Incremento de la tasa de aprendizaje
    # LR += 0.2000e-07
    
    # Reiniciar el contador try_nan
    try_nan = 0
    
    # Obtener la imagen de entrada desde el conjunto de entrenamiento de forma aleatoria
    while try_nan < 1:
        # print("entrenamiento")
        sp = int(math.floor(np.random.rand(1, 1) * (60e3 - 1)) + 1)  # Genera un número aleatorio entre 1 y 10e3
        # print(sp)
        digit = Amnist[sp-1]   # Obtiene la imagen de entrada (sp-1) porque los indices empiezan en 0
        X0 = digit.reshape(28, 28)
        #Normalizamos 
        X0 = X0 / 255 #--mejora
        
        #Obtenemos la etiqueta de la imagen de entrada
        yd = y[sp-1] # (sp-1) porque los indices empiezan en 0
        
        if depurar == 1: 
            yd = 0 #--mejora
        
        # switch de la etiqueta de la imagen de entrada del 1 al 10
        yd_trt = switch_yd(yd)

        if yd_trt is not None:
            # print(yd_trt)
            prueba = 1
        else:
            print("Invalid yd value")
        
        YD = yd_trt
        YD = np.reshape(YD, (-1,1)) #Convertimos YD a una dimension de 10x1 --Mejora
        
        try_nan=1; #Terminamos el while
        
    
    try_nan = 0; #Reiniciamos el contador try_nan
    
    #Obtener la imagen de entrada del set de test por seleccion aleatoria
    while try_nan < 1:
        # print("test")
        sp_test = int(math.floor(np.random.rand(1, 1) * (10e3 - 1)) + 1)  # Genera un número aleatorio entre 1 y 60e3
        # print(sp_test)
        digit = Bmnist[sp_test-1]   # Obtiene la imagen de entrada (sp-1) porque los indices empiezan en 0
        X0_test = digit.reshape(28, 28)
        #Normalizamos 
        X0_test = X0_test / 255 #--mejora
        
        #Obtenemos la etiqueta de la imagen de entrada
        yd = y_test[sp_test-1] # (sp-1) porque los indices empiezan en 0
        
        if depurar == 1:
            yd = 0 #--mejora
        
        # switch de la etiqueta de la imagen de entrada del 1 al 10
        yd_trt = switch_yd(yd)

        if yd_trt is not None:
            # print(yd_trt)
            prueba = 1
        else:
            print("Invalid yd value")
        
        YD_test = yd_trt
        YD_test = np.reshape(YD_test, (-1,1)) #Convertimos YD a una dimension de 10x1 --Mejora
        
        try_nan=1; #Terminamos el while
        


    # Asignar el valor de sp a sourc en la fila K y columna 1 (índices basados en 0)
    sourc[K][0] = sp
    
    # Asignar el valor de sp_test a sourc en la fila K y columna 2 (índices basados en 0)
    sourc[K][1] = sp_test
    
    # Test run of the CNN
    
    ############# DEPURAR #############    
    
    if depurar == 1:
        Xop = np.zeros((28,28))
        #X0 test
        # 
        #           Ciclos for para probar el código
        #
        for i in range(28):
            for j in range(28):
                Xop[i,j] = i + j * 28
        X0_test = Xop
        X0 = Xop

        cfil = 3e-4
        c1 = c1*1e-3
        Wop = np.zeros((9,9,1,10))
        for k1 in range(1):
            for p1 in range(10):
                for i1 in range(9):
                    for j1 in range(9):
                        Wop[i1,j1,k1,p1] = i1 + 8 * j1
        W0 = Wop*cfil

        W1p = np.zeros((5,5,10,10))
        for k1 in range(10):
            for p1 in range(10):
                for i1 in range(5):
                    for j1 in range(5):
                        W1p[i1,j1,k1,p1] = i1 + 4 * j1
        W1 = W1p*cfil

        W2p = np.zeros((3,3,10,10))
        for k1 in range(10):
            for p1 in range(10):
                for i1 in range(3):
                    for j1 in range(3):
                        W2p[i1,j1,k1,p1] = i1 + 2 * j1
        W2 = W2p*cfil

        W3p = np.zeros((100,1960))
        for ki in range (100):
            for kj in range(1960):
                W3p[ki,kj] = ki + kj*10
        W3 = W3p*cfil

        W4p = np.zeros((100,100))
        for ki in range (100):
            for kj in range(100):
                W4p[ki,kj] = ki + kj*10
        W4 = W4p*cfil

        W5p = np.zeros((10,100))
        for ki in range (10):
            for kj in range(100):
                W5p[ki,kj] = ki + kj*10
        W5 = W5p*cfil
        B0 = 0*B0
        B1 = 0*B1
        B2 = 0*B2
        B3 = 0*B3
        B4 = 0*B4
        B5 = 0*B5

        W0 = W0 - 0.0108
        W1 = W1 - 0.003
        W2 = W2 - 9.0000e-04
        W3 = W3 - 2.9534
        W4 = W4 - 0.1633
        W5 = W5 - 0.1498

        # Prueba dos pesos
        W0p = np.zeros((9*9*1*10,1))
        W1p = np.zeros((5*5*10*10,1))
        W2p = np.zeros((3*3*10*10,1))
        W3p = np.zeros((100*1960,1))
        W4p = np.zeros((100*100,1))
        W5p = np.zeros((10*100,1))

        for k1 in range(9*9*1*10):
            W0p[k1] = k1
        for k1 in range(5*5*10*10):
            W1p[k1] = k1
        for k1 in range(3*3*10*10):
            W2p[k1] = k1
        for k1 in range(100*1960):
            W3p[k1] = k1
        for k1 in range(100*100):
            W4p[k1] = k1
        for k1 in range(10*100):
            W5p[k1] = k1

        W0 = np.reshape(W0p,(9,9,1,10),order='F')
        W0 = W0*cfil
        W1 = np.reshape(W1p,(5,5,10,10),order='F')
        W1 = W1*cfil
        W2 = np.reshape(W2p,(3,3,10,10),order='F')
        W2 = W2*cfil
        W3 = np.reshape(W3p,(100,1960),order='F')
        W3 = W3*cfil
        W4 = np.reshape(W4p,(100,100),order='F')
        W4 = W4*cfil
        W5 = np.reshape(W5p,(10,100),order='F')
        W5 = W5*cfil

        W0 = W0 - 0.1213
        W1 = W1 - 0.3748
        W2 = W2 - 0.1348
        W3 = W3 - 29.3998
        W4 = W4 - 1.4998
        W5 = W5 - 0.1498
        #print(W2[0:3,0:3,4,4])
        #break
        #'''

    plt.imshow(X0, cmap='gray')
    plt.title(f'Primera imagen {sp}')
    # plt.show()

    #Primera capa
    Y0 = np.reshape(Y0, (20,20,10))
    X1 = np.reshape(X1, (20,20,10))
        
    contador = 0
        
    # Recorremos un bucle en el rango de cnn_M0 (10)
    for km in range(cnn_M0): # cnn_M0 = 10
        # Creamos una matriz de ceros 'sm1' con las mismas dimensiones que Y0[:, :, 0]
        # sm1 = np.zeros_like(Y0[:, :, 0]) #20x20x1
        sm1 = np.zeros([20,20])
        
        # Recorremos un bucle en el rango de cnn_D0
        for kd in range(cnn_D0): # cnn_D0 = 1
            # ----------------------------------------------- #
            # Creamos una matriz de ceros 'am1' de tamaño 9x9
            am1 = np.zeros((9, 9))

            # Recorremos bucles anidados para llenar 'am1' con valores de 'W0'
            for q1 in range(9):
                for q2 in range(9):
                    # Copiamos valores de 'W0' con inversión en indices
                    am1[q1, q2] = W0[8 - q1, 8 - q2, kd, km] # 9x9x1x10
            # ----------------------------------------------- #            

            # Realizamos la convolución 2D entre 'x' y 'am1' en modo 'valid'
            sm1 += convolve2d(X0, am1, mode='valid')
            
            # if(contador == 9):
            #     plt.imshow(sm1, cmap='gray')
            #     plt.title(f'Primera Capa SN {km}')
            #     plt.show()

            
            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            # sm1 = (sm1 - np.min(sm1))/(np.max(sm1)-np.min(sm1)) #Normalizacion se mueve al inicio
            
            if(contador == 9):
                plt.imshow(sm1, cmap='gray')
                plt.title(f'Primera Capa N {km}')
                # plt.show()

            contador = contador + 1

        #Convertimos Y0 a una dimension de 20x20x10
        # Y0 = Y0.reshape(20,20,10)
        
        #Convertimos X1 a una dimension de 20x20x10
        # X1 = X1.reshape(20,20,10)
        
            
        Y0[:, :, km] = np.maximum(sm1 + B0[km], 0)
        X1[:, :, km] = Y0[:, :, km]

    contador = 0;
    
    #normalizacion 
    sm1 = (sm1 - np.min(sm1))/(np.max(sm1)-np.min(sm1))    
    
    
    #Segunda capa 

    for km in range(cnn_M1): 
        #inicializamos sm1 con una matriz de ceros de 16x16
        sm1 = np.zeros([16,16])

        for kd in range(cnn_D1): 
            am1 = np.zeros((5, 5))

            for q1 in range(5):
                for q2 in range(5):
                    am1[q1, q2] = W1[4 - q1, 4 - q2, kd, km]

            sm1 += convolve2d(X1[:, :, kd-1], am1, 'valid')

            # if(contador == 99):
            #     plt.imshow(sm1, cmap='gray')
            #     plt.title(f'Segunda Capa SN {km}')
            #     plt.show()


            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            sm1 = (sm1 - np.min(sm1))/(np.max(sm1)-np.min(sm1))
            
            if(contador == 99):
                plt.imshow(sm1, cmap='gray')
                plt.title(f'Segunda Capa N {km}')
                # plt.show()

            #aumentamos el contador 
            contador = contador + 1

        
        #Convertimos Y0 a una dimension de 16x16x10
        Y1 = Y1.reshape(16,16,10)
        
        #Convertimos X1 a una dimension de 16x16x10
        X2 = X2.reshape(16,16,10)

        # Calcula Y1[:,:,km] = max(sm1 + B1(km), 0)
        Y1[:, :, km-1] = np.maximum(sm1 + B1[km-1], 0)
        X2[:, :, km-1] = Y1[:, :, km-1]

    contador = 0
    
    #Tercera capa

    for km in range(cnn_M2):
        #inicializamos sm2 con una matriz de ceros de 14x14
        sm2 = np.zeros([14,14])

        for kd in range(cnn_D2): # cnn_D2 = 10
            am2 = np.zeros((3, 3))

            for q1 in range(3):
                for q2 in range(3):
                    am2[q1, q2] = W2[2 - q1, 2 - q2, kd, km]

            sm2 += convolve2d(X2[:, :, kd-1], am2, 'valid')

            # if(contador == 99):
            #         plt.imshow(sm2, cmap='gray')
            #         plt.title(f'Tercera Capa SN {km}')
            #         plt.show()


            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            sm2 = (sm2 - np.min(sm2))/(np.max(sm2)-np.min(sm2))
            
            if(contador == 99):
                plt.imshow(sm2, cmap='gray')
                plt.title(f'Tercera Capa N {km}')
                # plt.show()

            #aumentamos el contador
            contador = contador + 1
            
        # Aplicamos reshape a Y2 para que tenga una dimension de 14x14x10
        Y2 = Y2.reshape(14,14,10)

        # Calcula Y2[:,:,km] = max(sm2 + B2(km), 0)
        Y2[:, :, km-1] = np.maximum(sm2 + B2[km-1], 0)

    # Variable que quieres guardar
    Y2_Pack = {'Y2': Y2}

    
    # # Abrir un archivo en modo escritura binaria
    # with open('Salida_Y2.pkl', 'wb') as archivo:
    #     # Guardar la variable en el archivo
    #     pickle.dump(Y2_Pack, archivo)

    # # print("Probando pickle")

    # # Abrir el archivo en modo lectura binaria
    # with open('Salida_Y2.pkl', 'rb') as archivo:
    #     # Cargar la variable desde el archivo
    #     mi_variable = pickle.load(archivo)
    #     # print(mi_variable)  # Imprime: {'clave': 'valor'}

    
    plt.imshow(Y2[:, :, km-1], cmap='gray')
    plt.title(f'Y2 {km}')
    # plt.show()

    # Convertir Y2 a una dimension de 1960x1
    X3 = Y2.reshape((1960, 1), order='F') #1960x1
    
    X3g = X3 #1960x1
    W3g = W3 #100x1960
    Y3g = np.dot(W3g, X3g) #100x1
    Y3 = Y3g
    Y3 = np.maximum(Y3 + 1.0 * B3, 0) #Elimina los negativos
    
    # Capa 4
    X4 = Y3
    X4g = X4
    W4g = W4
    Y4g = np.dot(W4g, X4g) #100x1
    Y4 = Y4g
    Y4 = np.maximum(Y4 + 1.0 * B4, 0)
    
    # Capa 5
    X5 = Y4
    X5g = X5
    W5g = W5
    Y5g = np.dot(W5g, X5g)
    Y5 = Y5g
    
    # Normalización sigmoidal en la capa 5 (asumiendo c1 y B5)
    Y5 = np.exp(c1 * (Y5 + B5))
    Y5 /= np.sum(Y5)
    
    # Calcular el error en función de YD y Y5
    E[K] = 0.5 * np.mean((YD - Y5)**2)

    # Test CNN
    # print("Test CNN")

    plt.imshow(X0_test, cmap='gray')
    plt.title(f'Primera imagen Test {sp_test}')
    # plt.show()

    #Primera capa
    contador = 0
        
    # Recorremos un bucle en el rango de cnn_M0 (10)
    for km in range(cnn_M0): # cnn_M0 = 10
        # Creamos una matriz de ceros 'sm1' con las mismas dimensiones que Y0[:, :, 0]
        # sm1 = np.zeros_like(Y0[:, :, 0]) #20x20x1
        sm1 = np.zeros([20,20])
        
        # Recorremos un bucle en el rango de cnn_D0
        for kd in range(cnn_D0): # cnn_D0 = 1
            # ----------------------------------------------- #
            # Creamos una matriz de ceros 'am1' de tamaño 9x9
            am1 = np.zeros((9, 9))

            # Recorremos bucles anidados para llenar 'am1' con valores de 'W0'
            for q1 in range(9):
                for q2 in range(9):
                    # Copiamos valores de 'W0' con inversión en indices
                    am1[q1, q2] = W0[8 - q1, 8 - q2, kd, km] # 9x9x1x10
            # ----------------------------------------------- #            

            # Realizamos la convolución 2D entre 'x' y 'am1' en modo 'valid'
            sm1 += convolve2d(X0_test, am1, mode='valid')
            
            # plt.imshow(sm1, cmap='gray')
            # plt.show()

            # if(contador == 9):
            #     plt.imshow(sm1, cmap='gray')
            #     plt.title(f'Primera Capa SN {km}')
            #     plt.show()

            
            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            sm1 = (sm1 - np.min(sm1))/(np.max(sm1)-np.min(sm1))
            
            # if(contador == 9):
            #     plt.imshow(sm1, cmap='gray')
            #     plt.title(f'Primera Capa Test N {km}')
            #     # plt.show()
                
            contador = contador + 1
            
        #Convertimos Y0 a una dimension de 20x20x10
        Y0 = Y0.reshape(20,20,10)
        
        # print("X1")
        # print(X1.shape)
        # print(X1)
        
        #Convertimos X1 a una dimension de 20x20x10
        X1 = X1.reshape(20,20,10)
        
        Y0[:, :, km] = np.maximum(sm1 + B0[km], 0)
        X1[:, :, km] = Y0[:, :, km]

        
    #Segunda capa
    contador = 0;

    for km in range(cnn_M1): 
        #inicializamos sm1 con una matriz de ceros de 16x16
        sm1 = np.zeros([16,16])

        for kd in range(cnn_D1): 
            am1 = np.zeros((5, 5))

            for q1 in range(5):
                for q2 in range(5):
                    am1[q1, q2] = W1[4 - q1, 4 - q2, kd, km]
            
            sm1 += convolve2d(X1[:, :, kd-1], am1, 'valid')

            # if(contador == 99):
            #     plt.imshow(sm1, cmap='gray')
            #     plt.title(f'Segunda Capa SN {km}')
            #     plt.show()

            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            sm1 = (sm1 - np.min(sm1))/(np.max(sm1)-np.min(sm1))
            
            if(contador == 99):
                plt.imshow(sm1, cmap='gray')
                plt.title(f'Segunda Capa Test N {km}')
                # plt.show()
            
            #aumentamos el contador 
            contador = contador + 1

        #Convertimos Y0 a una dimension de 16x16x10
        Y1 = Y1.reshape(16,16,10)
        
        #Convertimos X1 a una dimension de 16x16x10
        X2 = X2.reshape(16,16,10)

        # Calcula Y1[:,:,km] = max(sm1 + B1(km), 0)
        Y1[:, :, km-1] = np.maximum(sm1 + B1[km-1], 0)
        X2[:, :, km-1] = Y1[:, :, km-1]
        
    
    #Tercera Capa
    contador = 0
    
    for km in range(cnn_M2):
        #inicializamos sm2 con una matriz de ceros de 14x14
        sm2 = np.zeros([14,14])

        for kd in range(cnn_D2): # cnn_D2 = 10
            am2 = np.zeros((3, 3))

            for q1 in range(3):
                for q2 in range(3):
                    am2[q1, q2] = W2[2 - q1, 2 - q2, kd, km]

            sm2 += convolve2d(X2[:, :, kd-1], am2, 'valid')

            # if(contador == 99):
            #         plt.imshow(sm2, cmap='gray')
            #         plt.title(f'Tercera Capa SN {km}')
            #         plt.show()


            #Normalizamos sm1 = (sm1 -min(sm1))/(max(sm1)-min(sm1))
            sm2 = (sm2 - np.min(sm2))/(np.max(sm2)-np.min(sm2))
            
            if(contador == 99):
                plt.imshow(sm2, cmap='gray')
                plt.title(f'Tercera Capa Test N {km}')
                # plt.show()

            #aumentamos el contador
            contador = contador + 1
            
        # Aplicamos reshape a Y2 para que tenga una dimension de 14x14x10
        Y2 = Y2.reshape(14,14,10)

        # Calcula Y2[:,:,km] = max(sm2 + B2(km), 0)
        Y2[:, :, km-1] = np.maximum(sm2 + B2[km-1], 0)

    plt.imshow(Y2[:, :, km-1], cmap='gray')
    plt.title(f'Y2 Test {km}')
    # plt.show()

    # Convertir Y2 a una dimension de 1960x1
    X3 = Y2.reshape((1960, 1), order='F')
    
    X3g = X3
    W3g = W3
    Y3g = np.dot(W3g, X3g) #100x1    
    Y3 = Y3g
    Y3 = np.maximum(Y3 + 1.0 * B3, 0) #Elimina los negativos

    # Capa 4
    X4 = Y3 #100x1
    X4g = X4
    W4g = W4
    Y4g = np.dot(W4g, X4g) #100x1
    Y4 = Y4g
    Y4 = np.maximum(Y4 + 1.0 * B4, 0)

    # Capa 5
    X5 = Y4
    X5g = X5
    W5g = W5
    Y5g = np.dot(W5g, X5g)
    Y5 = Y5g
    
    # Y5 = (1 + np.exp(c1 * (-Y5 - B5))) ** -1
    Y5_past = Y5
    Y5 = np.exp(c1 * (Y5 + B5)) / np.sum(np.exp(c1 * (Y5 + B5)))

    #if sl == 1:
    YD_neg = YD_test
    Y5_neg = Y5

    Etest[K] = 0.5 * np.mean((YD_test - Y5) ** 2)
    
    Y5 = Y5.reshape(10)
    
    yCNN[:, K] = Y5 #Salida de la CNN (Resultado)
    yDPN[:, K] = YD_test #Resultado real

    # Visualization of the training process
    # print("if")
    if (K) % 1000 == 999: #Entra cada 1000 iteraciones
        print((K ) % 1000)
        Q1 = E[K - 999:K]
        Q2 = Etest[K - 999:K]
        
        print("K-999")
        print(K - 999)
        
        print("Q1")
        print(Q1)
        
        print("Q2")
        print(Q2)

    #Propagacion del error
    # print("Propagacion del error")

    # Back propagation error
    if test_set == 0:
        dE5 = (Y5 - YD_test) * MT # 10x1
        
        dF5 = c1 * Y5 * (1 - Y5) # 10x1
        
        dC5 = dE5 * dF5 # 10x1
        
        #Aplicamos reshape a DC5 para que tenga una dimension de 10x1
        dC5 = dC5.reshape(10,1) # 10x1
        
        dC5g = dC5 # 10x1
        #dW5 = -LR * np.dot(X5.T, dC5)
        dW5 = -LR * np.dot(dC5, X5.T) #  10x1 * 1x100  O 100x1 * 1x10
        dB5 = -LR * dC5 # 10x1
        
        dE4g = np.dot(W5g.T, dC5g)  # propagacion  # 100x10(T) 10x1
        dE4 = dE4g #100x1
        dF4 = np.sign(Y4) #(100x1)
        dC4 = dE4 * dF4 #100x1
        
        dC4g = dC4 #100x1
        
        dW4g = np.dot(dC4g, X4g.T) #100x1 * 1x100
        dW4 = dW4g #100x100
        dW4 = -LR * dW4 #100x100
        dB4 = -LR * dC4 #100x1
        
        dE3g = np.dot(W4g.T, dC4g) #100x100(T) 100x1 = 100x1
        dE3 = dE3g #100x1
        dF3 = np.sign(Y3) #100x1
        dC3 = dE3 * dF3 #100x1
        
        dC3g = dC3 #100x1
        
        dW3g = np.dot(dC3g, X3g.T) #100x1 * 1x1960
        dW3 = dW3g #100x1960
        dW3 = -LR * dW3 #100x1960
        dB3 = -LR * dC3 #100x1 

        dE2fg = np.dot(W3g.T, dC3g) #1960x100 100x1
        dE2f = dE2fg #1960x1
        
        dE2 = dE2f.reshape((14, 14, 10), order='F') #14x14x10
        dF2 = np.sign(Y2) #14x14x10 
        dC2 = dE2 * dF2 #14x14x10
        
        dW2 = np.zeros_like(W2) #3x3x10x10
        dB2 = np.zeros_like(B2) #10x1
        
        for km in range(cnn_M2): #10
            # Inicializar dCs2 como una matriz de ceros
            dCs2 = np.zeros((14, 14))
            
            # Rellenar dCs2 con valores de dC2
            for q1 in range(14):
                for q2 in range(14):
                    dCs2[q1, q2] = dC2[13 - q1, 13 - q2, km]
            
            for kd in range(cnn_D2): # cnn_D2 = 10
                # Realizar la operación de convolución y actualizar dW2
                dW2[:, :, kd, km] = -LR * convolve2d(X2[:, :, kd-1], dCs2, mode='valid') #3x3x10x10
            
            # Calcular y actualizar dB2
            dB2[km] = -LR * np.sum(dCs2) #10x1
            
        dE1p = np.zeros_like(X2) #16x16x10

        for kd in range(cnn_D2): # cnn_D2 = 10
            aq1 = np.zeros_like(dE1p[:, :, 0])
            # temp = np.zeros_like(dE1p[:, :, 0])
            
            for km in range(cnn_M2): # cnn_M2 = 10
                aq1 += convolve2d(dC2[:, :, km], W2[:, :, kd, km], mode="full") #16x16x10
                # temp += full_conv(dC2[:, :, km], W2[:, :, kd, km])
                # #Comparamos aq1 con temp
                # if np.array_equal(aq1, temp):
                #     print("Son iguales")
                # else:
                #print("No son iguales")
            dE1p[:, :, kd] = aq1 #16x16x10

        dE1 = dE1p #16x16x10
                
        dF1 = np.sign(Y1) #16x16x10
        dC1 = dE1 * dF1 #16x16x10

        dW1 = np.zeros_like(W1) #5x5x10x10
        dB1 = np.zeros_like(B1) #10x1

        for km in range(cnn_M1): #10
            # Inicializar dCs1 como una matriz de ceros
            dCs1 = np.zeros((16, 16))

            for q1 in range(16):
                for q2 in range(16):
                    dCs1[q1, q2] = dC1[15 - q1, 15 - q2, km] #16x16x10

            for kd in range(cnn_D1): # cnn_D1 = 10
                # Realizar la operación de convolución y actualizar dW1
                dW1[:, :, kd, km] = -LR * convolve2d(X1[:, :, kd], dCs1, mode='valid') #5x5x10x10

            # Calcular y actualizar dB1
            dB1[km] = -LR * np.sum(dCs1)

        dE0p = np.zeros_like(X1) #20x20x10

        for kd in range(cnn_D1): # cnn_D1 = 10
            aq0 = np.zeros_like(dE0p[:, :, 0]) #20x20
            for km in range(cnn_M1): # cnn_M1 = 10
                aq0 += convolve2d(dC1[:, :, km], W1[:, :, kd, km], mode="full") #20x20
            dE0p[:, :, kd] = aq0 #20x20x10

        dE0 = dE0p #20x20x10
        
        dF0 = np.sign(Y0) #20x20x10
        dC0 = dE0 * dF0 #20x20x10

        dW0 = np.zeros_like(W0) #9x9x1x10
        dB0 = np.zeros_like(B0) #10x1

        for km in range(cnn_M0): #10
            dCs0 = np.zeros((20, 20))
            for q1 in range(20):
                for q2 in range(20):
                    dCs0[q1, q2] = dC0[19 - q1, 19 - q2, km]
            for kd in range(cnn_D0): # cnn_D0 = 1
                dW0[:, :, kd, km] = -LR * convolve2d(X0, dCs0, mode='valid')
            dB0[km] = -LR * np.sum(dCs0)
            
        if np.isnan(dW0).any():
            print('NaN')
            break

        if np.isnan(dW1).any():
            print('NaN')
            break

        if np.isnan(dW2).any():
            print('NaN')
            break

        if np.isnan(dW3).any():
            print('NaN')
            break

        if np.isnan(dW4).any():
            print('NaN')
            break

        if np.isnan(dW5).any():
            print('NaN')
            break

        W5 += dW5 #10x100
        B5 += dB5 #10x1

        W4 += dW4 #100x100
        B4 += dB4 #100x1

        W3 += dW3 #100x1960
        B3 += dB3 #100x1

        W2 += dW2 #3x3x10x10
        B2 += dB2 #10x1

        W1 += dW1 #5x5x10x10
        B1 += dB1 #10x1

        W0 += dW0 #9x9x1x10 
        B0 += dB0 #10x1


# # Tomar una imagen de ejemplo
# digit = x[0]
# digit_pixels = digit.reshape(28, 28)

# # Visualizar la imagen
# plt.imshow(digit_pixels, cmap='gray')
# plt.show()

# W=1/100*np.ones((10,10));

# convolve2d(digit_pixels, W, mode='valid')

# plt.imshow(convolve2d(digit_pixels, W, mode='valid'), cmap='gray')
# plt.show()