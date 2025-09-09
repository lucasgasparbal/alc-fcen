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
    x = np.linspace(-1.5,1.5,250)
    y = np.linspace(-1.5,1.5,250)
    X, Y = np.meshgrid(x,y)
    
    return 

puntos = puntos()

fig, ax = plt.subplots()

for p in [1,2,5,10,100,200]:
    cumplen = []
    for x in puntos:
        if norma(x,p) == 1:
            cumplen.append(x)
    
    ax.plot(cumplen)

# Tests norma
assert np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2))
assert np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10))
assert norma(np.random.rand(10), 2) <= np.sqrt(10)
assert norma(np.random.rand(10), 2) >= 0

# Tests normaliza
assert ([np.allclose(norma(x, 2), 1) for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2)])
assert ([not np.allclose(norma(x, 2), 1) for x in normaliza([np.array([1] * k) for k in range(1, 11)], 1)])
assert ([np.allclose(norma(x, 'inf'), 1) for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf')])
