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


def calcularFk(A,k,v,atol=1e-12):

    w = A @ v
    normaW = alc.norma(w,2)
    if normaW > 0:
        w = w / normaW
    
    for i in range(k-1):
        w = alc.productoMatricial(A,w,atol)
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
    e = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)
    eAnt = 0
    k = 0
    while abs(e-eAnt) > tol and k < K:
        v = vmonio
        vmonio = calcularFk(A,2,vmonio)
        eAnt = e
        e = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)    
        k += 1
    
    autovalor = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)

    return v, autovalor, k, e-1


print(metpot2k(np.eye(3),1e-3))

# %% 1c

for A in [A1,A2,A3]:
    print(metpot2k(A))

# %% tests metpot2k

# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95

# %%
