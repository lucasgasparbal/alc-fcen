#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 11:19:13 2025

@author: Estudiante
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 19:54:44 2025

@author: lucas
"""
import numpy as np


def error(a,b):
    x = np.float64(a)
    y = np.float64(b)
    
    return abs(x-y)
    
    


def sonIguales(a, b, atol=1e-10):
    return error(a,b) < atol

def norma(x,p):
    if p == 'inf':
        return np.max(abs(x))

    sum = np.float64(0)
    for elem in x:
        sum += abs(elem**p)
        
    return sum ** (1/p)

def normaliza(X,p):
    res = []
    for x in X:
        res.append(x/norma(x,p))
    
        
    return res


def normaMatMC(A,q,p,Np):
    n = A[0].size
    maximo = 0
    xMax = None
    for i in range(Np):
        x = np.random.randn(n)
        xNormalizado = x / norma(x, p)
        normaInducida = norma(A @ xNormalizado, q)
        if normaInducida > maximo:
            maximo = normaInducida
            xMax = xNormalizado
            
    return [maximo,xMax]


def normaExacta(A,p):
    maximo = 0
    if p == 1:
        
        for i in range(A.shape[0]):
            suma = np.sum(abs(A[:,i]))
            if suma > maximo:
                maximo = suma
        return maximo
    elif p == 'inf':
        for i in range(A.shape[1]):
            suma = np.sum(abs(A[i,:]))
            if suma > maximo:
                maximo = suma
        return maximo
    
    
def condMC(A,p,Np):
    inversa = np.linalg.inv(A)
    normaA = normaMatMC(A,p,p,Np)
    normaInversa = normaMatMC(inversa,p,p,Np)
    return normaA[0] * normaInversa[0]

def condExacta(A,p):
    inversa = np.linalg.inv(A)
    normaA = normaExacta(A, p)
    normaInversa = normaExacta(inversa, p)
    return normaA * normaInversa

def calculaLU(A):
        cant_op = 0
        m=A.shape[0]
        n=A.shape[1]
        Ac = A.copy().astype(np.float64)
        
        if m!=n:
            return None, None, 0
    
    
        L = np.zeros(A.shape)
        U = np.zeros(A.shape)
        
        cant_op = 0
        n = A.shape[0]
        for i in range(n):
            if np.isclose(Ac[i,i],0):
                return None, None, 0
            for j in range(i+1,n):
                multiplicador = Ac[j,i]/Ac[i,i] #1 division
                filaMult = multiplicador*Ac[i,i:] #n-i multiplicaciones
                Ac[j,i:] = Ac[j,i:] - filaMult # n-i restas
                Ac[j,i] = multiplicador
                cant_op += 1+2*(n-(i+1))
                
            L[i,i] = 1
            L[i+1:,i] = Ac[i+1:,i]
            U[i,i:] = Ac[i,i:]
                
        
        return L, U, cant_op

def resolverLy(L,b):
    n = L.shape[0]
    y = np.zeros(n).astype(np.float64)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] = y[i] - L[i,j]*y[j]
        y[i] = y[i] / L[i,i]
     
    return y
         

def resolverUx(U,y):
    n = U.shape[0]
    x = np.zeros(n).astype(np.float64)
    for i in range(n-1,-1,-1):
        x[i] = y[i]
        for j in range(n-1,i,-1):
            x[i] -= U[i,j]*x[j]
        x[i] = x[i] / U[i,i]
     
    return x

def res_tri(A,b,inferior=True):
    if inferior:
        return resolverLy(A, b)
    else:
        return resolverUx(A, b)
    

def inversa(A):
    L, U, c = calculaLU(A)
    if L is None:
        return None
    n = A.shape[0]
    res = np.zeros(A.shape)
    for i  in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        y = res_tri(L,ei)
        x = res_tri(U,y,False)
        res[i] = x
    
    return res.T

A = np.array([
    [2, 3, 1],
    [2, 1, 0],
    [0, 0, 1]
])

def calculaLDV(A):
    L, U, count = calculaLU(A)
    if L is None:
        return None, None, None, 0
    V,D, count2 = calculaLU(U.T)
    
    return L, D, V.T, count + count2



def esSimetrica(A, atol=1e-10):
    res = True

    for i in range(len(A)):
        for j in range(len(A)):
            if(not sonIguales(A[i,j],A[j,i])):
                res = False

    return res

def esSDP(A,atol=1e-10):
    if not esSimetrica(A,atol):
        return False
    
    L, D, V, count = calculaLDV(A)
    if L is None:
        return False
    res = True
    for i in range(A.shape[0]):
        if(not D[i,i] > 0):
            res = False
            
    return res
    pass


