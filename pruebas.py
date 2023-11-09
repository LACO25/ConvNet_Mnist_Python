import numpy as np

rs = np.array([[16, 2, 3, 13],
                [5, 11, 10, 8],
                [9, 7, 6, 12],
                [4, 14, 15, 1]])


output = rs.reshape((8, 2), order='F')

print(output)

# Definir las matrices W3g y X3g (asegúrate de que tengan dimensiones compatibles)
W3g = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

X3g = np.array([2, 3, 4])

# Realizar la multiplicación de matrices
Y3g = np.dot(W3g, X3g)

print("Resultado")
# Mostrar el resultado
print(Y3g)
