from alc import *

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
        return V, sig, U
    else:
        return U, sig, V


    
    

    
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
     for n in [2,5,10,20]:
         k = 1
         for _ in range(10):
             A = genera_matriz_para_test(m,n)
             test_svd_reducida_mn(A)
             print("BIEN "+str((m,n))+" - "+str(k)+"/10")
             k += 1 


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for k in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)
        print("BIEN CON NUCLEO DIM "+str(tam_nucleo) + " - "+str(k+1)+"/10")

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso k='+str(k)+')'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso k='+str(k)+')'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso k='+str(k)+')'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso k='+str(k)+')'
    assert len(hS) == k, 'Tamaño de hS incorrecto'
print("BIEN DIMENSIONES")
