import numpy as np
import matplotlib.pyplot as plt

#e1

print(0.3+0.25) #res = 0.55, esperado 0.55

print(0.3-0.25)
#resultado = 0.04999999999999999, esperado 0.05

print(0*2+2**-2)
#mantisa: 0, exponente: -2

print(0*2+2**-2+2**-5+2**-6+2**-9+2**-10+2**-13+2**-14+2**-17)
#no se puede llegar a 0.3 con base 2 y mantisa finita


#e2

print((np.sqrt(2)**2)-2) #da 4.44*10**-16, debería dar 0


cienValores = np.arange(10**14,10**16,(10**16-10**14)/100)

def funcion1(x):
    return (np.sqrt(2*(x**2)+1)-1)

def funcion2(x):
    return ((2*x**2)/(np.sqrt(2*(x**2)+1)-1))

res1 = []
res2 = []
diferencia = []
for x in cienValores:
    y1 = funcion1(x)
    y2 = funcion2(x)
    
    res1.append(y1)
    res2.append(y2)
    diferencia.append(y1-y2)
    
i = 9
ticks = [cienValores[0]]    
for x in range(0,10):
    ticks.append(cienValores[i])
    i += 10
    

fig , ax = plt.subplots()
ax.set(xticks=ticks,yticks=ticks)


ax.plot(cienValores,res1,color='red',marker='o')
plt.tight_layout()
plt.show()
fig , ax = plt.subplots()
ax.set(xticks=ticks,yticks=ticks)

ax.plot(cienValores,res2,color='blue',marker='o')
plt.tight_layout()
plt.show()


fig , ax = plt.subplots()
ax.set(xticks=ticks)

ax.plot(cienValores,diferencia,color='green',marker='o')
plt.tight_layout()
plt.show()

# en general, la diferencia es a favor de la funcion 2, lo que indicaría
# que tiende a sobreestimar mas que la función 1

#e3

def sucesion(i):
    if i == 1:
        return np.sqrt(2)
    else: 
        return ((sucesion(i-1)**2)/np.sqrt(2))
    
    
res = []

val = np.sqrt(2)
for i in range(1,101):
    res.append(val)
    val = (val*val)/np.sqrt(2)
    

plt.plot(res)


#%% e 4


n=7
s = np.float32(0)
for i in range(1,10**n+1):
    s = s +np.float32(1/i)
print("suma = ", s)



s = np.float32(0)
for i in range(1,5*10**n+1):
    s = s +np.float32(1/i)
print("suma = ", s)
    
n=7
s = np.float64(0)
for i in range(1,10**n+1):
    s = s +np.float32(1/i)
print("suma = ", s)



s = np.float64(0)
for i in range(1,5*10**n+1):
    s = s +np.float32(1/i)
print("suma = ", s) 



#%% e5

def matricesIguales(A , B):
    if A.shape != B.shape:
        return False
    else:
        res = True
        for i in range(len(A)):
            for j in range(len(B)):
                if not np.isclose(A[i][j],B[i][j]):
                    res = False
                    
        return res
    
A = np.array([[np.float32(4),np.float32(2),np.float32(1)],
              [np.float32(2),np.float32(7),np.float32(9)],
              [np.float32(0),np.float32(5),np.float32(22/3)]])

L = np.array([[np.float32(1),np.float32(0),np.float32(0)],
              [np.float32(0.5),np.float32(1),np.float32(0)],
              [np.float32(0),np.float32(5/6),np.float32(1)]])
U = np.array([[np.float32(4),np.float32(2),np.float32(1)],
              [np.float32(0),np.float32(6),np.float32(8.5)],
              [np.float32(0),np.float32(0),np.float32(0.25)]])

C = L @ U
print(matricesIguales(A, C))
print(A[2][2])
print(C[2][2])
print(A[2][2] == C[2][2])


#%% E6

def esSimetrica(A):
    res = True

    for i in range(len(A)):
        for j in range(len(A)):
            if(A[i][j] != A[j][i]):
                res = False

    return res

A = np.array(np.random.rand(4,4))


print(esSimetrica(A))
print(esSimetrica(A.T@A))
print(esSimetrica(A.T@((A*0.25)/0.25)))
print(esSimetrica(A.T@((A*0.2)/0.2)))
