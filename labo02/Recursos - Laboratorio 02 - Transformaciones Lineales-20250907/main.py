import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = []
    ############### Insert code here!! ######################3    
    xy = T @ wz
    ############### Insert code here!! ######################3
    return xy

          
def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)

def circulo(r,centroX,centroY):
    t = np.linspace(0,2*np.pi,100)
    x = r*np.cos(t) + centroX
    y = r*np.sin(t) + centroY
    circulo = np.vstack((x,y))
    return circulo


def cizalla(c,d):
    return np.array([[1,c],[d,1]])

def rotacion(phi):
    return np.array([[np.cos(phi),-np.sin(phi)],
                     [np.sin(phi),np.cos(phi)]])
def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    T = pd.read_csv('T.csv', header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')
    
    # ejercicio 1
    T = np.array([[1/2,0],[0,1/2]])

    Tinverted = np.array([[2,0],[0,2]])
    
    vistform(T,wz,'ejercicio 1, reducir a la mitad')
    vistform(Tinverted,wz,'ejercicio 1,inversa de reducir a la mitad')
    
    # ejercicio 2
        
    T = np.array([[2,0],[0,3]])
    canonicos = np.array([[1,0],[0,1]])
    aleatorio = np.random.rand(2,1)
    circ = circulo(1,0,0)
    vistform(T,canonicos,'ejercicio 2 vectores canonicos')
    vistform(T,aleatorio,'ejercicio 2 vector aleatorio')
    vistform(T,circ,'ejercicio 2 circulo')
    
    #ejercicio 3
    
    #3b
    for b in [0,0.25,0.5,0.75,1]:
        T = np.array([[0,b],[0,1]])
        vistform(T,wz,'deformacion con b = '+str(b))
        
    #la transformacion agrupa los puntos en una recta y la rota con respecto al eje x
        
    #3c
    for b in [0,0.25,0.5,0.75,1]:
         T = np.array([[0,b],[b,1]])
         vistform(T,wz,'deformacion con b = c = '+str(b))
    
    #toma la figura entera y la rota y distorciona hasta la recta y crecientemente con el valor de b
    
    #ejercicio 4
    
    for phi in[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]:
        vistform(rotacion(phi),wz,'rotacion con phi = '+str(phi))
    
    #la rotación se diferencia con la de cizalla porque no deforma al objeto
    
    #ejercicio 5
    
    T = np.array([[5/2,1/2],
                  [1/2,5/2]
                  ])
    vistform(T,circulo(1,0,0),'Rotacion y Reescalamiento')
        
    
    
if __name__ == "__main__":
    main()
