#%%

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

def productoEscalar(x,y,atol=1e-12):
    
    if x.shape != y.shape:
        return None
    
    n  = x.shape[0]
    
    suma = np.float64(0)
    for i in range(n):
        termino = x[i]*y[i]
        if abs(termino) >= atol:
            suma += x[i]*y[i]
        
        
        
    return suma
    
def productoMatricial(A,B,atol=1e-12):
    n,m = A.shape
    q,r = B.shape
    if m != q:
        return None
    
    res = np.zeros((n,r))
    
    for i in range(n):
        for j in range(r):
            res[i,j] = productoEscalar(A[i], B[:,j],atol)
            if abs(res[i,j]) < atol:
                res[i,j] = 0
    
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

def uuT(u):
    n = u.shape[0]
    res = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            res[i,j] = u[i]*u[j]
            
    return res


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
            
            Hk = np.eye(m-k)-2*uuT(u) # dimension de Hk m-k x m-k
            
            '''
            Haciendo producto por bloques entre Hkmonio = [[Ik, 0],  y R =[[R1, R2],
                                                           [0, Hk]]        [0, R4]]
            
            Hkmonio @ R = [[R1, R2],
                           [0,Hk*R4]]
            '''
            if k == 0:
                R = productoMatricial(Hk, R,tol)
            else:
                R[k:,k:] = productoMatricial(Hk, R[k:,k:],tol) #R4
            
            '''
            Haciendo producto por bloques entre Hkmonio.T = [[Ik, 0],  y Q =[[Q1, Q2],
                                                           [0, Hk.T]]        [Q3, Q4]]
            
            Q @ Hkmonio.T = [[Q1, Q2@Hk.T],
                             [Q3,Q4@Hk.T]
            '''
            if k == 0:
                Q = productoMatricial(Q, Hk.T,tol)
            else:
                Q[:k,k:] = productoMatricial(Q[:k,k:], Hk.T,tol) #Q2
                Q[k:,k:] = productoMatricial(Q[k:,k:], Hk.T,tol) #Q4
    
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A,tol)
    elif metodo == 'GS':
        return QR_con_GS(A,tol)
    else:
        return None, None
    

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
# %%
