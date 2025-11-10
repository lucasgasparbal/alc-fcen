import numpy as np
from alc import *


# %%
A = np.array([[3,0,0,0,0],[0,1,0,0,0]])

hU, hS, hV = svd_reducida(A)