#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
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
            Ac[j,i:] = Ac[j,i:] - multiplicador*Ac[i,i:] # n-i restas y n-i multiplicaciones
            Ac[j,i] = multiplicador
            cant_op += 1+2*(n-i)
            
        L[i,i] = 1
        L[i+1:,i] = Ac[i+1:,i]
        U[i:,i:] = Ac[i:,i:]
                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op


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
    
    
