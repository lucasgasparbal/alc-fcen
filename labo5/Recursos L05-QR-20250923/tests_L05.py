import alc

import numpy as np

import matplotlib.pyplot as plt
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
        
#%% 1b
def inversaQR(A):
    Q, R = QR_con_GS(A)
    if Q is None:
        return None
    n = A.shape[0]
    res = np.zeros(A.shape)
    for i  in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        y = np.linalg.solve(Q,ei)
        x = alc.res_tri(R,y,False)
        res[i] = x
    
    return res.T
def A_aleatoria(n):
    res = np.zeros([n,n])
    
    for i in range(n):
        for j in range(n):
            res[i,j] = np.float64(-1 + 2*np.random.random())
    
    return res
A = A_aleatoria(5)
L, U , nopsLU = alc.calculaLU(A)
while L is None:
    A = A_aleatoria(5)
    L, U , nopsLU = alc.calculaLU(A)
    
Q, R, nopsQR = QR_con_GS(A,retorna_nops=True)

AinvLU = alc.inversa(A)
AinvQR = inversaQR(A)
print("---linalg---")
print(np.linalg.inv(A))
print("---QR---")
print(AinvQR)
print("---LU---")
print(AinvLU)
ELU = A @ AinvLU
EQR = A @ AinvQR
In = np.eye(5,5)
print("cant LU: "+ str(nopsLU))
print("cant QR: "+ str(nopsQR))
normaLU = alc.normaExacta(ELU-In, 'inf')
normaQR = alc.normaExacta(EQR-In, 'inf')
print("---EQR---")
print(EQR)
print("---ELU---")
print(ELU)
print(normaLU)
print(normaQR)

ns = [2,5,10,20,50,100]

for n in ns:
    erroresLU = []
    erroresQR = []
    for i in range(100):
        A = A_aleatoria(n)
        L, U , nopsLU = alc.calculaLU(A)
        while L is None:
            A = A_aleatoria(n)
            L, U , nopsLU = alc.calculaLU(A)
            
        Q, R, nopsQR = QR_con_GS(A,retorna_nops=True)
        AinvLU = alc.inversa(A)
        AinvQR = inversaQR(A)
        ELU = A @ AinvLU
        EQR = A @ AinvQR
        In = np.eye(n,n)
        normaLU = alc.normaExacta(ELU-In, 'inf')
        normaQR = alc.normaExacta(EQR-In, 'inf')
        erroresLU.append(normaLU)
        erroresQR.append(normaQR)
    fig, ax = plt.subplots()
    
    ax.set_title("Error al calcular inversa con n = " +str(n))
    
    ax.hist(erroresLU,alpha=0.5,label="con LU")
    ax.hist(erroresQR,alpha=0.5,label="con QR")
    ax.legend()
    plt.show()
#%% 1c
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
        
        if(not alc.sonIguales(R[i,i], 0,tol)):
            Q[:,i] = Q[:,i]/R[i,i] #n divisiones
        else:
            R[i,i] = np.float64(0)
            
        nops += n
    Qs = []
    Rs = []
    for i in range(n):
        
        if not R[i][i] == 0:
            Qs.append(Q[:,i])
            Rs.append(R[i])
    
    QTreducida = np.array(Qs)
    Rreducida = np.array(Rs)
            
    if retorna_nops:
            return QTreducida.T, Rreducida, nops
    else:
            return QTreducida.T, Rreducida
    

A = np.array([[1,1,1],
              [1,1,1],
              [1,1,1]])

Q, R = QR_con_GS_A_LD(A)



#%%            
#Para las matrices H recordar H = I-2uu^t

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
            
def uuT(u):
    n = u.shape[0]
    res = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            res[i,j] = u[i]*u[j]
            
    return res


def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    m, n = A.shape
    if m < n:
        return None, None
    
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(n-1):
        x = R[k:,k]
        alfa = np.sign(x[0])*alc.norma(x,2)
        u = x -alfa*np.eye(x.shape[0])[0]
        normaU = alc.norma(u,2)
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
                R[k:,:k] = productoMatricial(Hk, R[k:,:k],tol) #R3
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

A = np.array([[4,0,0],
              [3,1,0],
              [0,0,3]])

Q, R = QR_con_HH(A)            
#%% 2b
def inversaQRHH(A):
    Q, R = QR_con_HH(A)
    if Q is None:
        return None
    n = A.shape[0]
    res = np.zeros(A.shape)
    for i  in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        y = np.linalg.solve(Q,ei)
        x = alc.res_tri(R,y,False)
        res[i] = x
    
    return res.T


ns = [2,5,10,20,50,100]

for n in ns:
    
    erroresLU = []
    erroresQRGS = []
    erroresQRHH = []
    erroresQGS = []
    erroresQHH = []
    
    for i in range(100):
        A = A_aleatoria(n)
        L, U , nopsLU = alc.calculaLU(A)
        while L is None:
            A = A_aleatoria(n)
            L, U , nopsLU = alc.calculaLU(A)
            
        Q, R, nopsQR = QR_con_GS(A,retorna_nops=True)
        Qh, Rh = QR_con_HH(A)
        AinvLU = alc.inversa(A)
        AinvQR = inversaQR(A)
        AinvQRHH = inversaQRHH(A)
        ELU = A @ AinvLU
        EQR = A @ AinvQR
        EQRHH = A @ AinvQRHH
        In = np.eye(n,n)
        normaLU = alc.normaExacta(ELU-In, 'inf')
        normaQR = alc.normaExacta(EQR-In, 'inf')
        normaQRHH = alc.normaExacta(EQRHH-In, 'inf')
        normaQGS = alc.normaExacta((Q.T@Q)-In,'inf')
        normaQHH = alc.normaExacta((Qh.T@Qh)-In,'inf')
        erroresLU.append(normaLU)
        erroresQRGS.append(normaQR)
        erroresQRHH.append(normaQRHH)
        erroresQGS.append(normaQGS)
        erroresQHH.append(normaQHH)
    
    
    fig, ax = plt.subplots()
    
    ax.set_title("Error al calcular inversa con n = " +str(n))
    
    ax.hist(erroresLU,alpha=0.5,label="con LU")
    ax.hist(erroresQRGS,alpha=0.5,label="con QRGS")
    ax.hist(erroresQRHH,alpha=0.5,label="con QRHH")
    ax.legend()
    plt.show()
    
    fig, ax = plt.subplots()
    
    ax.set_title("Error sobre Q con n = " +str(n))
    ax.hist(erroresQGS,alpha=0.5,label="con QRGS")
    ax.hist(erroresQHH,alpha=0.5,label="con QRHH")
    ax.legend()
    plt.show()
    

#%% 2c    
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    ELU
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A,tol)
    elif metodo == 'GS':
        return QR_con_GS(A,tol)
    else:
        return None, None

# Tests L05-QR:


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