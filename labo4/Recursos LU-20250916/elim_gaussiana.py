#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
import alc as alc
import matplotlib.pyplot as plt

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy().astype(np.float64)
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    
    cant_op = 0
    n = A.shape[0]
    for i in range(n):
        
        for j in range(i+1,n):
            multiplicador = Ac[j,i]/Ac[i,i] #1 division
            filaMult = multiplicador*Ac[i,i:]
            Ac[j,i:] = Ac[j,i:] - filaMult # n-i restas y n-i multiplicaciones
            Ac[j,i] = multiplicador
            cant_op += 1+2*(n-i)
            
        L[i,i] = 1
        L[i+1:,i] = Ac[i+1:,i]
        U[i,i:] = Ac[i,i:]
                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op
#%% 1b
def matrizAleatoria(n):
    return np.random.randn(n,n)

ns = [2,3,4,5,6,7,8,9,10,15,20,50,100]
normas = []
erroresRelativos = []
counts = []
for n in ns:
    A = matrizAleatoria(n)
    L, U, count = elim_gaussiana(matrizAleatoria(n))
    
    Ac = L @ U
    normaError = alc.normaExacta(A-Ac,'inf')
    normaA = alc.normaExacta(A,'inf')
    erroresRelativos.append(normaError/normaA)
    counts.append(count)
    normas.append(alc.normaExacta(A-Ac,'inf'))
    
print(counts)
fig, ax = plt.subplots()
ax.set_title("Modulo de error sobre valor de n")
ax.set_xlabel('n')
ax.set_ylabel('||A-Ac||')
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(ns,normas)
plt.show()

fig, ax = plt.subplots()
ax.set_title("Error por cantidad de operaciones")
ax.set_xlabel('cantidad de operaciones')
ax.set_ylabel('Error relativo')
ax.plot(counts,erroresRelativos)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
    
#%% 2
def matrizB(n):
    B = np.zeros([n,n])
    
    for i in range(n):
        for j in range(n):
            if i == j or j == (n-1):
                B[i,j] = 1
            elif i > j:
                B[i,j] = -1
    
    return B
                

ns = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,50,100,150,200,250]
normaU = []
counts = []
for n in ns:
    B = matrizB(n)
    L, U, count = elim_gaussiana(B)
    normaU.append(alc.normaExacta(U,'inf'))
    counts.append(count)

print(normaU)
print(counts)
fig, ax = plt.subplots()
ax.set_title("Cantidad de operaciones sobre n")
ax.set_xlabel('n')
ax.set_ylabel('cantidad de operaciones')
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(ns,counts)

plt.show()    

#cant operaciones en funcion de n cant(n) = (n-1)*n*(4*n+7)//6


#%%3

#a

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

A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
])

b = np.array([8, -11, -3])

L, U, count = elim_gaussiana(A)

y = resolverLy(L,b)
sol = resolverUx(U,y) #deberia ser 2,3,-1


#%% 4

def calculaLDV(A):
    L, U, count = elim_gaussiana(A)
    V,D, count2 = elim_gaussiana(U.T)
    
    return L, D, V.T

A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
])

L, U, count =elim_gaussiana(A)

print(L)
print(U)
print(U.T)
print("---LDV---")
L, D, V = calculaLDV(A)

print(L)
print(D)
print(V)

    
#%%
def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
