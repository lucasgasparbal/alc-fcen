#%% imports
import numpy as np
import matplotlib.pyplot as plt
import alc

#%% ejercicio 1 a
A1 = np.array([[1/2,1/2],
               [1/2,1/2]])

A2 = np.array([[1,1],
              [0,1]])

A3 = np.array([[0,1],
              [-1,0]])


def calcularFk(A,k,v):

    w = A@v
    normaW = alc.norma(w,2)
    if normaW > 0:
        w = w / normaW
    
    for i in range(k-1):
        w = A@v
        normaW = alc.norma(w,2)
        if normaW > 0:
            w = w / normaW

    return w

matrices = ["A1","A2","A3"]
for A in [A1,A2,A3]:
    
    randoms = []
    colores = ['red','blue','green','yellow','black','cyan','purple','peru','violet']
    for k in range(5):
        randoms.append(np.random.randn(2))

    vectores = [(1/np.sqrt(2))*np.array([1,1]),
                (1/np.sqrt(2))*np.array([1,-1]),
                np.array([1,0]),np.array([0,1])]
    
    fig, ax = plt.subplots()
    ax.plot()
    i = 0
    for v in randoms+vectores:
        resultados = []
        for k in [1,5,10,50,100]:
            resultados.append(calcularFk(A,k,v))
        
        print(resultados)
        coords = [v]+resultados
        coords = np.array(coords)
        ax.scatter(coords[:,0],coords[:,1],color=colores[i],alpha=0.5)
        ax.plot(coords[:,0],coords[:,1],color=colores[i],alpha=0.5)
        ax.set_title("fk para A = "+str(A))
        ax.set_xlim((-2,2))
        ax.set_ylim((-2,2))
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        i += 1

# %% 1b

def metpot2k(A,tol=1e-15,K = 1000):

    n = A.shape[0]
    v = np.random.randn(n)
    vmonio = calcularFk(A,2,v)
    e = alc.productoEscalar(vmonio,vmonio)
    k = 0
    while abs(e-1) > tol and k < K:
        v = vmonio
        vmonio = calcularFk(A,2,vmonio)
        e = alc.productoEscalar(vmonio,vmonio)
        k += 1
    
    autovalor = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio))

    return v, autovalor, k, e-1
