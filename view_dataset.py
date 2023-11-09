import numpy as np
import scipy.io

# Read dataset
Amnist = scipy.io.loadmat('Amnist.mat')
index_A = scipy.io.loadmat('index_A.mat')
Bmnist_V2 = scipy.io.loadmat('Bmnist_V2.mat')
index_B_V2 = scipy.io.loadmat('index_B_V2.mat')

# Create label_A
label_A = np.zeros((10_000, 1))
# print(index_A['index_A'])
contador = 0;
for i in range(10):
    for j in range(int(index_A['index_A'][i])):
        contador = contador + 1
        label_A[j] = int(i)
        print(f'label_A[',contador,'] = ' , label_A[j])

ready_dataset = 1