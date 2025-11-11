import numpy as np
from alc import *


A = np.random.random((8,6))
hU,hS,hV = svd_reducida(A,k=5)