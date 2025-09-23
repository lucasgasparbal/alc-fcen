import alc

### Funciones L05-QR
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    nops = 0
    if A.shape[0] != A.shape[1]:
        return None, None
        if retorna_nops:
            return None, None, nops
        
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    n = A.shape[0]
    norma = alc.norma(A[:,0],2) # n sumas + n productos y 1 raiz cuadrada
    
    nops += 2*n+1
    
    Q[:,0] = A[:,0]/norma # n divisiones
    R[0,0] = norma
    
    nops += n
    
    for i in range(1,n):
        Q[:,i] = A[:,i]
        
        for j in range(i):
            
            R[j,i] = np.dot(Q[:,j].T,Q[:,i]) # n productos y n sumas
            if alc.sonIguales(R[j,i], 0,tol):
                R[j,i] = 0
                
            nops += 2*n
            
            Q[:,i] = Q[:,i] -(R[j,i]*Q[:,j]) # n productos y n restas
            
            nops += 2*n
        
            R[i,i] = alc.norma(Q[:,i],2) # n sumas + n productos y 1 raiz cuadrada
        
            nops += 2*n+1
        
        
        Q[:,i] = Q[:,i]/R[i,i] #n divisiones
    
        nops += n
    
    if retorna_nops:
            return Q, R, nops
    else:
            return Q, R
        
            
#Para las matrices H recordar H = I-2uu^t            

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)