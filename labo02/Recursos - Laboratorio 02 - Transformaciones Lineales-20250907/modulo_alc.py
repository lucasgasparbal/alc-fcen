import numpy as np

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


# Tests para rota
assert(np.allclose(rota(0), np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1], [1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]])))

# Tests para escala
assert(np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(escala([1, 1, 1]), np.eye(3)))
assert(np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])))

# Tests para rotayescala
assert(np.allclose( rota_y_escala(0, [2, 3])   , np.array([[2, 0], [0, 3]])))
assert(np.allclose(rota_y_escala(np.pi/2, [1, 1]), np.array([[0, -1], [1, 0]])))
assert(np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]])))

# Tests para afin
assert(np.allclose(
    afin(0, [1, 1], [1, 2]),
    np.array([[1, 0, 1],
              [0, 1, 2],
              [0, 0, 1]])))

assert(np.allclose(afin(np.pi/2, [1, 1], [0, 0]),
    np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])))

assert(np.allclose(afin(0, [2, 3], [1, 1]),
    np.array([[2, 0, 1],
              [0, 3, 1],
              [0, 0, 1]])))

# Tests para trans_afin
assert(np.allclose(
    trans_afin(np.array([1, 0]), np.pi/2, [1, 1], [0, 0]),
    np.array([0, 1])
))

assert(np.allclose(
    trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
    np.array([2, 3])
))

assert(np.allclose(
    trans_afin(np.array([1, 0]), np.pi/2, [3, 2], [4, 5]),
    np.array([4, 7])
))
    

