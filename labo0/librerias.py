import numpy as np



def esCuadrada(matriz):
    return len(matriz) == len(matriz[0])


def triangSup(A):
    res = np.array(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i >= j:
                res[i][j] = 0

    return res


def diagonal(A):
    res = np.array(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i != j:
                res[i][j] = 0

    return res

def triangInf(A):
    res = np.array(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i <= j:
                res[i][j] = 0

    return res

def traza(A):
    res = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                res += A[i][j] 

    return res


def traspuesta(A):
    res = []

    for j in range(len(A[0])):
        fila = []
        for i in range(len(A)):
            fila.append(A[i][j])
        res.append(fila)

    return np.array(res)

def esSimetrica(A):
    res = True

    for i in range(len(A)):
        for j in range(len(A)):
            if(A[i][j] != A[j][i]):
                res = False

    return res


def calcularAx(A,x):
    res = []

    for i in range(len(A)):
        calc = 0
        for j in range(len(A[i])):
            calc += A[i][j]*x[j]
        res.append(calc)

    return res

def intercambiarFilas(A,i,j):
    aux = np.array(A[i])

    A[i] = A[j]

    A[j] = aux


def sumarFilaMultiplo(A,i,j,s):

    for k in range(len(A[i])):
        A[i][k] = A[i][k] + A[j][k]*s


def esDiagonalmenteDominante(A):
    res = True

    for i in range(len(A)):
        if (not filaDominante(A[i],i)):
            res = False

    return res

def filaDominante(fila,n):
    pivote = abs(fila[n])
    sum = 0
    for i in range(len(fila)):
        if i != n:
            sum += abs(fila[i])

    return pivote > sum

def matrizCirculante(v):
    filas = [v]

    for i in range(1,len(v)):
        filas.append(filaCirculante(v,i))

    return np.array(filas)

def filaCirculante(v,n):
    fila = []
    i = len(v) - n
    while(len(fila) < len(v)):
     fila.append(v[i])
     i += 1
     if i >= len(v):
         i = 0
    return fila


def matrizVandermonde(v):
    filas = []
    for i in range(1,len(v)+1):
        fila = []
        for j in range(len(v)):
            fila.append(v[j]**(i-1))
        filas.append(fila)
    
    return np.array(filas)


def matrizHilbert(n):
    filas = []
    for i in range(n):
        fila = []
        for j in range(n):
            fila.append(1/(i+j+1))
        filas.append(fila)
    return np.array(filas)