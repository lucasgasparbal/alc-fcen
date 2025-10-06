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

    w = alc.productoMatricial(A,v,atol)
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
    e = alc.productoEscalar(vmonio,v,tol)
    eAnt = 0
    k = 0
    while abs(e-1) > tol and k < K:
        v = vmonio
        vmonio = calcularFk(A,2,v)
        e = alc.productoEscalar(vmonio,v,tol)    
        k += 1
    
    autovalor = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)

    return v, autovalor, k, abs(e-1)


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
print("metodo de la potencia OK!")



# %% ejercicio 2
rng = np.random.default_rng()
C = rng.normal(scale=15,size=(100,100))

A = (1/2)*(C+C.T)

def metpot2kConEstimacionAutovalor(A,tol=1e-15,K = 1000):

    n = A.shape[0]
    v = np.random.randn(n)
    vmonio = calcularFk(A,2,v)
    e = alc.productoEscalar(vmonio,v,tol)
    autovalores = []
    eAnt = 0
    k = 0
    while abs(e-1) > tol and k < K:
        v = vmonio
        vmonio = calcularFk(A,2,v)
        e = alc.productoEscalar(vmonio,v,tol)
        autovalor = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)
        autovalores.append(autovalor)    
        k += 1
    
    autovalor = alc.productoEscalar(vmonio,alc.productoMatricial(A,vmonio,tol),tol)
    autovalores.append(autovalor) 

    return v, autovalor, k, abs(e-1), autovalores

B = A + 500*np.eye(100)

fig, ax = plt.subplots()
ax.set_title("aproximacion al autovalor")
ax.set_xlabel("valor de paso k")
ax.set_ylabel("error entre autovalor y valor estimado")
ax.set_xlim(0,500)
eigvals = np.linalg.eigvals(B)
maxEigVal = np.max(abs(eigvals))
eigvals = np.partition(eigvals, -2)
EigVal2 = np.max(abs(eigvals))

for i in range(10):
    v,autoval,k,_,aproximaciones = metpot2kConEstimacionAutovalor(B)
    errores = []
    ks = range(k+1)
    for l in aproximaciones:
        errores.append(abs(autoval-l))
    erroresEstimados = []
    for k in ks:
        erroresEstimados.append((2*errores[k]*np.log(abs(EigVal2)/abs(maxEigVal)))+errores[0])

    ax.plot(ks,errores)
    ax.plot(ks,erroresEstimados,linestyle='dashed')
    
    

# %% ejercicio 3

def diagRH(A,tol=1e-15,K=1e5):
    v1, l1,_,_ = metpot2k(A,tol,K)
    n = A.shape[0]
    u = np.eye(n)[0]-v1
    u = u/alc.norma(u,2)
    if n == 2:
        S = np.eye(n)-2*alc.uuT(u)
        D = A-2*alc.vwT(u,alc.productoMatricial(u,A))
        D = D-2*alc.vwT(alc.productoMatricial(D,u),u)
    else:
        B = A-2*alc.vwT(u,alc.productoMatricial(u,A))
        #H = np.eye(n)-2*alc.uuT(u)
        B= B-2*alc.vwT(alc.productoMatricial(B,u),u)
        Amonio = B[1:n,1:n]
        Smonio, Dmonio = diagRH(Amonio,tol,K)
        D = np.zeros((n,n))
        D[0,0] = l1
        D[1:n,1:n] = Dmonio
        S = np.zeros((n,n))
        S[0,0] = 1
        S[1:n,1:n] = Smonio
        S = S-2*alc.vwT(u,alc.productoMatricial(u,S))

    return S, D


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = alc.normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95

print("diagRH OK!")# %% 1c

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
print("metodo de la potencia OK!")

# %% 2d

tols = [1e-3,1e-5,1e-8,1e-10,1e-12,1e-15]
errores = []
for tol in tols:
    errores = []
    for i in range(100):

        rng = np.random.default_rng()
        C = rng.normal(scale=15,size=(10,10))
        A = (1/2)*(C+C.T)
        S, D = diagRH(A,tol)
        errores.append(alc.normaExacta(A-(S@D@S.T),'inf'))

    fig, ax = plt.subplots()
    ax.set_title("histograma de errores para tol = "+str(tol))

    ax.hist(errores)

    plt.show()


# %%
