#%%

import numpy as np

#labo1
def error(a,b):
    x = np.float64(a)
    y = np.float64(b)
    
    return abs(x-y)
    
    


def sonIguales(a, b, atol=1e-10):
    return error(a,b) < atol

#labo2

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
    
#labo3
    
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

#labo4

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


#labo 5

def productoEscalar(x,y,atol=1e-12):
    
    if x.shape != y.shape:
        return None
    
    n  = x.shape[0]
    
    suma = np.float64(0)
    for i in range(n):
        termino = x[i]*y[i]
        if abs(termino) >= atol:
            suma += x[i]*y[i]
        
    if abs(suma) < atol:
        suma = 0    
        
    return suma

def Ax(A,x,atol=1e-12):
    n , m = A.shape
    if m != x.size:
        return None
    
    b = np.zeros(n)

    for i in range(n):
        b[i] = productoEscalar(A[i],x,atol)

    return b

    
def vTA(v,A,atol=1e-12):
    n = v.size
    if n != A.shape[0]:
        return None
    
    res = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        res[i] = productoEscalar(v,A[:,i],atol)
    
    return res
        



def productoMatricial(A,B,atol=1e-12):
    if len(A.shape) == 1:
        #asumo que es un vector fila
        return vTA(A,B)
    if len(B.shape) == 2:
        q, r = B.shape
    else:
       return Ax(A,B,atol)
    n,m = A.shape
    

    if m != q:
        return None
    
    res = np.zeros((n,r))
    
    for i in range(n):
        for j in range(r):
            res[i,j] = productoEscalar(A[i], B[:,j],atol)
    
    return res

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    nops = 0
    if A.shape[0] != A.shape[1]:
        return None, None
        if retorna_nops:
            return None, None, nops
        
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    n = A.shape[0]
    norma2 = norma(A[:,0],2) # n sumas + n productos y 1 raiz cuadrada
    
    nops += 2*n+1
    
    Q[:,0] = A[:,0]/norma2 # n divisiones
    R[0,0] = norma2
    
    nops += n
    
    for i in range(1,n):
        Q[:,i] = A[:,i]
        
        for j in range(i):
            
            R[j,i] = productoEscalar(Q[:,j].T,Q[:,i]) # n productos y n sumas
            if sonIguales(R[j,i], 0,tol):
                R[j,i] = 0
                
            nops += 2*n
            
            Q[:,i] = Q[:,i] -(R[j,i]*Q[:,j]) # n productos y n restas
            
            nops += 2*n
        
        R[i,i] = norma(Q[:,i],2) # n sumas + n productos y 1 raiz cuadrada
        
        nops += 2*n+1
        
        
        Q[:,i] = Q[:,i]/R[i,i] #n divisiones
    
        nops += n
    
    if retorna_nops:
            return Q, R, nops
    else:
            return Q, R

def uuT(u,tol=1e-12):
    return vwT(u,u,tol)

def vwT(v,w,tol=1e-12):
    n = v.shape[0]

    if n != w.shape[0]:
        return None

    res = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            res[i,j] = v[i]*w[j]
            if abs(res[i,j]) < tol:
                res[i,j] = 0
            
    return res

def check_tol(A,tol=1e-12):
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            if abs(A[i,j]) < tol:
                A[i,j] = 0

def QR_con_HH(A,tol=1e-12):
    m, n = A.shape
    if m < n:
        return None, None
    
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(n-1):
        x = R[k:,k]
        alfa = np.sign(x[0])*norma(x,2)
        u = x -alfa*np.eye(x.shape[0])[0]
        normaU = norma(u,2)
        if normaU > tol:
            u = u / normaU
            
            
            '''
            Haciendo producto por bloques entre Hkmonio = [[Ik, 0],  y R =[[R1, R2],
                                                           [0, Hk]]        [0, R4]]
            
            Hkmonio @ R = [[R1, R2],
                           [0,Hk*R4]]
            '''
            if k == 0:
                R = R -2*vwT(u,productoMatricial(u,R,tol),tol) #productoMatricial(Hk, R,tol)
                check_tol(R,tol)
            else:
                R4 = R[k:,k:]
                R4= R4 -2*vwT(u,productoMatricial(u,R4,tol),tol) #productoMatricial(Hk, R[k:,k:],tol) #R4
                check_tol(R4,tol)
            
            '''
            Haciendo producto por bloques entre Hkmonio.T = [[Ik, 0],  y Q =[[Q1, Q2],
                                                           [0, Hk.T]]        [Q3, Q4]]
            
            Q @ Hkmonio.T = [[Q1, Q2@Hk.T],
                             [Q3,Q4@Hk.T]
            '''
            
            
            if k == 0:
                Q = Q - 2*vwT(productoMatricial(Q,u,tol),u,tol) #productoMatricial(Q, Hk.T,tol)
                check_tol(Q,tol)
            else:
                Q2 = Q[:k,k:]
                Q4 = Q[k:,k:]
                if k == 1:
                    Q2 = Q2 -2*productoEscalar(Q2[0],u,tol)*u.T # si k1 Q2 es un vector fila y Q2*u es escalar
                else:
                    Q2 = Q2 -2*vwT(productoMatricial(Q2,u,tol),u,tol) #productoMatricial(Q[:k,k:], Hk.T,tol) #Q2
                Q4 =  Q4 -2*vwT(productoMatricial(Q4,u,tol),u,tol) #productoMatricial(Q[k:,k:], Hk.T,tol) #Q4
                check_tol(Q2,tol)
                check_tol(Q4,tol)
    
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A,tol)
    elif metodo == 'GS':
        return QR_con_GS(A,tol)
    else:
        return None, None

#labo 6

def calcularFk(A,k,v,atol=1e-12):

    w = productoMatricial(A,v,atol)
    normaW = norma(w,2)
    if normaW > 0:
        w = w / normaW
    
    for i in range(k-1):
        w = productoMatricial(A,w,atol)
        normaW = norma(w,2)
        if normaW > 0:
            w = w / normaW

    return w

def metpot2k(A,tol=1e-15,K = 1000):

    n = A.shape[0]
    v = np.random.randn(n)
    vmonio = calcularFk(A,2,v)
    e = productoEscalar(vmonio,v,tol)
    k = 0
    while abs(e-1) > tol and k < K:
        v = vmonio
        vmonio = calcularFk(A,2,v)
        e = productoEscalar(vmonio,v,tol)    
        k += 1
    
    autovalor = productoEscalar(vmonio,productoMatricial(A,vmonio,tol),tol)

    return v, autovalor, k, abs(e-1)

def diagRH(A,tol=1e-15,K=1e5):
    v1, l1,_,_ = metpot2k(A,tol,K)
    n = A.shape[0]
    u = np.eye(n)[0]-v1
    u = u/norma(u,2)
    if n == 2:
        S = np.eye(n)-2*uuT(u)
        D = A-2*vwT(u,productoMatricial(u,A))
        D = D-2*vwT(productoMatricial(D,u),u)
        check_tol(D)
    else:
        B = A-2*vwT(u,productoMatricial(u,A))
        B= B-2*vwT(productoMatricial(B,u),u)
        check_tol(B)
        Amonio = B[1:n,1:n]
        Smonio, Dmonio = diagRH(Amonio,tol,K)
        D = np.zeros((n,n))
        D[0,0] = l1
        D[1:n,1:n] = Dmonio
        S = np.zeros((n,n))
        S[0,0] = 1
        S[1:n,1:n] = Smonio
        S = S-2*vwT(u,productoMatricial(u,S))

    return S, D

#%% Tests L05-QR:


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
Q2h,R2h = QR_con_HH(A2)
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

print("Labo5 OK!")
# Test L06-metpot2k, Aval

import numpy as np

#### TESTEOS
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
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95

print("LABO 6 OK!")

# %%
