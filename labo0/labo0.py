import numpy as np



def esCuadrada(matriz):
    return len(matriz) == len(matriz[0])


m = np.array([[1,2],[5,4]])

print(esCuadrada(m))

m = np.array([[1,2],[5,4],[2,3]])

print(esCuadrada(m))

