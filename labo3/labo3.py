import numpy as np
import matplotlib.pyplot as plt
#%% ejercicio 1

#a
def norma(x,p):
    if p == 'inf':
        return np.max(abs(x))

    sum = np.float64(0)
    for elem in x:
        sum += abs(elem**p)
        
    return sum ** (1/p)

#b

def normaliza(X,p):
    res = []
    for x in X:
        res.append(x/norma(x,p))
    
        
    return res

def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)
    
def puntos():
    x = np.linspace(-1.5,1.5,50)
    y = np.linspace(-1.5,1.5,50)
    X, Y = np.meshgrid(x,y)
    
    return np.vstack([X.ravel(),Y.ravel()])

puntos = puntos()

fig, ax = plt.subplots()

for p in [1,2,5,10,100,200,'inf']:
   cumplen =  np.array(normaliza(puntos.T, p)).T
   fig, ax = plt.subplots()
   ax.scatter(cumplen[0,:],cumplen[1,:])
   ax.set_title("norma p = "+str(p))
   plt.show()
   

#%% ejercicio 2

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

def estimarNorma(A,q,p):
    normas = []
    for i in range(100):
        normas.append(normaMatMC(A,q,p,1000))
    
    return normas
#%% 2b
I = np.array([[1,0],
              [0,1]])
dosbi = estimarNorma(np.array([[1,0],[0,1]]),2,1)
dosbii = estimarNorma(np.array([[1,0],[0,1]]),1,2)
dosbiii = estimarNorma(I,2,'inf')
dosbiv = estimarNorma(I,'inf',2)
dosbv = estimarNorma(np.array([[0,-1],[1,0]]),2,2)
dosbvi = estimarNorma(np.array([[1,0],[0,0]]),2,2)
dosbvii = estimarNorma(np.array([[1,0],[0,0]]),2,2)
dosbvii = estimarNorma(np.array([[1,0],[0,0]]),2,'inf')
dosbviii = estimarNorma(np.array([[10,10],[0,0]]),2,'inf')

#%% 2c
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
    


#%% ejercicio 3

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


def variaPerc(b,perc):
    copia = []
    for i in range(b.size):
        variacion = (1-perc/100)+np.random.rand()*((1+perc/100)-(1-perc/100))
        copia.append(b[i]*variacion)
        
    return np.array(copia)

print(variaPerc(np.array([1,0]),23))

#%% 3b

b = np.array([1,0])
for p in [1,2,10,25,'inf']:
    print("---norma "+str(p)+"---")
    print("comun: "+str(norma(b,p)))
    print("variada 10%: " +str(norma(variaPerc(b,10),p)) )
    print("variada 25%: " +str(norma(variaPerc(b,25),p)) )
    print("variada 3%: " +str(norma(variaPerc(b,3),p)) )

#%% variacion matricial
def variacionMatricial(A,p,perc,Np):
    
    errorxs = []
    errorbs = []
    bDeMaxError = []
    maxError = 0
    for i in range(Np):
        b = np.random.randn(A.shape[1])
        bvar = variaPerc(b,perc)
        x = np.linalg.solve(A,b)
        xvar = np.linalg.solve(A,bvar)
        errorx = norma(x-xvar,p)/norma(x,p)
        errorb = norma(b-bvar,p)/norma(b,p)
        errorxs.append(errorx)
        errorbs.append(errorb)
        if errorx > maxError:
            maxError = errorx
            bDeMaxError = [b,bvar]
            
        cond = condMC(A, p, Np)
        
    return [cond,errorxs,errorbs,bDeMaxError]


As = [np.array([[1,-1],
                [1,1]]),
      np.array([[1000,1/1000],
                  [0,1000]]),
      np.array([[501,499],
                [500,500]]),
      np.array([[1/1000,1000],[0,1000]])]


for k in range(len(As)):
    for p in [1,2,"inf"]:
        for perc in [1,5,10]:
            resultados = variacionMatricial(As[k], p, perc, 100)
            erroresx = resultados[1]
            erroresb = resultados[2]
            valores = []
            for i in range(len(erroresx)):
                valores.append(erroresx[i]/erroresb[i])
            fig, ax = plt.subplots()
            ax.set_title("Matriz A"+str(k)+" p = "+str(p)+" perc = "+str(perc))
            ax.hist(valores)
            print(resultados[0])
            ax.axvline(resultados[0],c='red')
            plt.show()
            
        

            
#%%
# Tests norma
assert np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2))
assert np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10))
assert norma(np.random.rand(10), 2) <= np.sqrt(10)
assert norma(np.random.rand(10), 2) >= 0

# Tests normaliza
assert ([np.allclose(norma(x, 2), 1) for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2)])
assert ([not np.allclose(norma(x, 2), 1) for x in normaliza([np.array([1] * k) for k in range(1, 11)], 1)])
assert ([np.allclose(norma(x, 'inf'), 1) for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf')])

# Test normaMC

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

#%% Tests normaExacta

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

#%% Test condMC

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))


# Test condExacta

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))
