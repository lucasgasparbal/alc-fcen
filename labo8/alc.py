import numpy as np
#labo1
def error(a,b):
    x = np.float64(a)
    y = np.float64(b)
    return abs(x-y)

def error_relativo(x,y):
    return error(x,y)/(abs(x))

def sonIguales(a, b, atol=1e-10):
    return error(a,b) < atol

def matricesIguales(A,B):
    if A.shape != B.shape:
        return False
    return np.allclose(A, B)

#labo2
def rota(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def escala(s):
    
    n = len(s)
    
    m = np.zeros([n,n])
    
    for i in range(n):
        m[i][i] = s[i]
        
    return m

def rota_y_escala(theta, s):
    
    return escala(s) @ rota(theta)

def afin(theta, s, b):
    
    bloqueA = rota_y_escala(theta, s)
    
    res =  np.zeros([3,3])
    
    for i in range(2):
        for j in range(2):
            res[i][j] = bloqueA[i][j]
            
    for i in range(2):
        res[i][2] = b[i]
    
    res[2][2] = 1
    
    return res

def trans_afin(v,theta,s,b):
        
    vTres = np.array([v[0],v[1],1])

    resTres = afin(theta,s,b) @ vTres  
    
    return np.array([resTres[0],resTres[1]])

#labo3
def norma(x,p):
    if p == 'inf':
        return np.max(abs(x))

    sum = np.float64(0)
    for elem in x:
        val = elem**p
        sum += abs(val)
        
    return sum ** (1/p)

def normaliza(X,p):
    res = []
    for x in X:
        nor = norma(x,p)
        if nor != 0:
            res.append(x/nor)
    
        
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

#LABO 5

def productoEscalar(x,y,atol=1e-12):
    
    if x.shape != y.shape:
        return None
    
    n  = x.shape[0]
    
    suma = np.float64(0)
    for i in range(n):
        if abs(x[i]) >= atol and abs(y[i]) >= atol:
            suma += x[i]*y[i]
    
    return suma

def vTA(v,A,atol=1e-12):
    n = v.size
    if n != A.shape[0]:
        return None
    
    res = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        res[i] = productoEscalar(v,A[:,i],atol)
    
    return res

def Ax(A,x,atol=1e-12):
    n , m = A.shape
    if m != x.size:
        return None
    
    b = np.zeros(n)

    for i in range(n):
        b[i] = productoEscalar(A[i],x,atol)

    return b

def check_tol_vector(v,tol=1e-15):
    n = v.size
    for i in range(n):
        if v[i] < tol:
            v[i] = 0

def check_tol(A,tol=1e-15):
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            if abs(A[i,j]) < tol:
                A[i,j] = 0
    
def productoMatricial(A,B,atol=1e-12):
    if len(A.shape) == 1:
        #asumo que es un vector fila
        return vTA(A,B,atol)
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

    res = np.zeros((n,n),np.float64)
    
    for i in range(n):
        for j in range(n):
            res[i,j] = v[i]*w[j]
            if abs(res[i,j]) < tol:
                res[i,j] = 0
            
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

#LABO 6
def prod_interno(p,q):
    sumador = 0
    for i in range(len(p)):
        sumador += p[i]*q[i]
    return sumador #esta al cuadrado
    
def f_A(A,v,tol=1e-15):
    w_prima = Ax(A,v,tol)
    norma_w_prima2 =productoEscalar(w_prima,w_prima,tol) #||w_prima||^2
    norma_w_prima = np.sqrt(norma_w_prima2)
    res = np.zeros_like(v)
    if norma_w_prima >= tol:
        res = w_prima / norma_w_prima
    return res

def multMatrizXColumna(A,c):
    n,m= A.shape
    res = np.zeros(n)
    for i in range(n):
        for j in range(m):
            res[i]+= A[i][j]*c[j]
    return res

def metpot(A,tol=1e-15,K=1000):
    n,n = A.shape
    v = np.random.randn(n)
    v_moño1 = f_A(A,v,tol)
    v_moño = f_A(A,v_moño1,tol)
    e = productoEscalar(v_moño,v,tol)
    k = 0
    cont_iteraciones = 0
    while(abs(e-1) >tol and k < K):
        v = v_moño
        v_moño1 = f_A(A,v,tol)
        v_moño= f_A(A,v_moño1,tol)
        e = productoEscalar(v_moño,v,tol)
        cont_iteraciones +=1
        k+=1
    lambdaADevolver = productoEscalar(v_moño,Ax(A,v_moño,tol),tol)
    return [v,lambdaADevolver,cont_iteraciones]

def diagRH(A,tol=1e-15,K=1e5):
    v1, l1,_ = metpot(A,tol,K)
    n = A.shape[0]
    e1 = np.eye(n)[0]
    u = e1-v1
    nor = norma(u,2)
    if nor < tol:
        u = e1
    else:
        u = u/norma(u,2)
    if n == 2:
        S = np.eye(n)-2*uuT(u,tol)
        check_tol(S)
        D = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        D = D-2*vwT(productoMatricial(D,u,tol),u,tol)
        check_tol(D)
    else:
        B = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        check_tol(B)
        B= B-2*vwT(productoMatricial(B,u,tol),u,tol)
        check_tol(B)
        Amonio = B[1:n,1:n]
        Smonio, Dmonio = diagRH(Amonio,tol,K)
        D = np.zeros((n,n),dtype=np.float64)
        D[0,0] = l1
        D[1:n,1:n] = Dmonio
        S = np.zeros((n,n),dtype=np.float64)
        S[0,0] = 1
        S[1:n,1:n] = Smonio
        S = S-2*vwT(u,productoMatricial(u,S,tol),tol)
        check_tol(S)

    return S, D


def diagRHSVD(A,tol=1e-15,K=1e5,cant_autovals = 'max'):
    if cant_autovals == 'max':
        S, D = _diagRHSVDMax(A,tol,K)
    else:
        S, D = _diagRHSVD(A,cant_autovals,tol,K)
    n = D.shape[0]
    i = 0
    while i < n and abs(D[i,i]) >= tol:
        i += 1
    
    return S[:,:i], D[:i,:i] #tengo que recortar las columnas de S porque uso la traspuesta


def _diagRHSVDMax(A,tol=1e-15,K=1e5):
    v1, l1,_ = metpot(A,tol,K)
    if l1 < tol:
        return np.zeros(A.shape), np.zeros(A.shape)
    n = A.shape[0]
    u = np.eye(n,dtype=np.float64)[0]-v1
    normaU = norma(u,2)
    if normaU < tol:
        u = v1 #v1 es e1
    else:
        u = u/normaU
    if n == 2:
        S = np.eye(n)-2*uuT(u,tol)
        check_tol(S)
        D = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        D = D-2*vwT(productoMatricial(D,u,tol),u,tol)
        check_tol(D)
    else:
        B = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        check_tol(B)
        B= B-2*vwT(productoMatricial(B,u,tol),u,tol)
        check_tol(B)
        Amonio = B[1:n,1:n]
        Smonio, Dmonio = _diagRHSVDMax(Amonio,tol,K)
        D = np.zeros((n,n),dtype=np.float64)
        D[0,0] = l1
        D[1:n,1:n] = Dmonio
        S = np.zeros((n,n),dtype=np.float64)
        S[0,0] = 1
        S[1:n,1:n] = Smonio
        S = S-2*vwT(u,productoMatricial(u,S,tol),tol)
        check_tol(S)

    return S, D

def _diagRHSVD(A,cant_autovals,tol=1e-15,K=1e5):
    v1, l1,_ = metpot(A,tol,K)
    if l1 < tol or cant_autovals == 0:
        return np.zeros(A.shape), np.zeros(A.shape)
    n = A.shape[0]
    u = np.eye(n,dtype=np.float64)[0]-v1
    normaU = norma(u,2)
    if normaU < tol:
        u = v1 #v1 es e1
    else:
        u = u/normaU
    if n == 2:
        S = np.eye(n)-2*uuT(u,tol)
        check_tol(S)
        D = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        D = D-2*vwT(productoMatricial(D,u,tol),u,tol)
        if cant_autovals == 1:
            D[1,1] = 0
        check_tol(D)
    else:
        B = A-2*vwT(u,productoMatricial(u,A,tol),tol)
        check_tol(B)
        B= B-2*vwT(productoMatricial(B,u,tol),u,tol)
        check_tol(B)
        Amonio = B[1:n,1:n]
        Smonio, Dmonio = _diagRHSVD(Amonio,cant_autovals-1,tol,K)
        D = np.zeros((n,n),dtype=np.float64)
        D[0,0] = l1
        D[1:n,1:n] = Dmonio
        S = np.zeros((n,n),dtype=np.float64)
        S[0,0] = 1
        S[1:n,1:n] = Smonio
        S = S-2*vwT(u,productoMatricial(u,S,tol),tol)
        check_tol(S)

    return S, D
       
def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    A = np.random.rand(n,n)
    listaNormalizados = normaliza(A.T,1)
    res = np.array(listaNormalizados)
    return res.T

def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    
    A = np.random.rand(n,n)
    for i in range(n):
        for j in range(n):
            if A[i,j] > thres:
                A[i,j] = 0
            else:
                A[i,j] = 1

    zeroDeRn = np.zeros(n)
    for i in range(n):
        col = A.T[i]
        if (col == zeroDeRn).all():
            A[i,i] = 1

    listaNormalizados = normaliza(A.T,1)
    res = np.array(listaNormalizados)
    return res.T

#Labo 8

def diagonal(A):
    m, n = A.shape
    if m != n:
        return None
    res = []
    for i in range(n):
        res.append(A[i,i])
    
    return np.array(res)

def traspuesta(A):
    n, m = A.shape
    At = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            At[i,j] = A[j,i]
    
    return At

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    Ahermetiana = productoMatricial(A.T,A)
    S, D = diagRH(Ahermetiana)
    n = Ahermetiana.shape[0]

    primerIndiceNulo = n
    i = 0
    while i < n and D[i,i] >= tol:
        i += 1
    
    primerIndiceNulo = i
    autovectores = []
    for i in range(primerIndiceNulo,n):
        autovectores.append(S[:,i])

    return np.array(autovectores).T

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    dict = {}
    if len(listado) == 0:
        return dict, (m_filas,n_columnas)
    

    filas = listado[0]
    columnas = listado[1]
    valores = listado[2]
    for i in range(len(filas)):
        if valores[i] >= tol:
            dict[(filas[i],columnas[i])] = valores[i]
    
    return dict, (m_filas,n_columnas)

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    dict, dims = A
    res = np.zeros(dims[0])
    values = []
    for i in range(dims[0]):
        sum = 0
        for j in range(v.size):
            if (i,j) in dict:
                valor_posible = dict[(i,j)]*v[j]
                sum += valor_posible
        values.append(sum)
    
    return np.array(values)

# Tests L08
def svd_reducida1(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m,n = A.shape
    
    if n >= m: #mas columnas que filas, se resuelve para A
        Asim = productoMatricial(A.T,A,tol)
    else: #mas filas que columnas, se resuelve para A.T
        Asim = productoMatricial(A,A.T,tol)

    V, D = diagRHSVD(Asim,tol,cant_autovals = k)
    B = productoMatricial(A,V,tol)

    columnasU = normaliza(B.T,2)
    U = np.array(columnasU).T

    sig = np.sqrt(diagonal(D))

    if n < m:
        return V, sig, U
    else:
        return U, sig, V
    
def diagonal(A):
    m, n = A.shape
    if m != n:
        return None
    res = []
    for i in range(n):
        res.append(A[i,i])
    
    return np.array(res)
# Tests L08
def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m,n = A.shape
    
    if n > m: #mas columnas que filas, se resuelve para A.T
        Asim = productoMatricial(A,A.T,tol)
        A = A.T
    else: #mas filas que columnas, se resuelve para A
        Asim = productoMatricial(A.T,A,tol)

    V, D = diagRHSVD(Asim,tol,cant_autovals = k)
    B = productoMatricial(A,V,tol)

    columnasU = normaliza(B.T,2)
    U = np.array(columnasU).T

    sig = np.sqrt(diagonal(D))

    if n > m:
        return V.T, sig, U
    else:
        return U, sig, V