from utils import *; importlib.reload(sys.modules['utils']); from utils import *
import numpy as np
from scipy.ndimage import maximum_filter

harris = np.load('data/quiz06/harris.npy', allow_pickle=True).item()
C00 = harris['g*(I_x^2)']
C11 = harris['g*(I_y^2)']
C01 = harris['g*(I_x I_y)']

print(repr(C00))
print(repr(C11))
print(repr(C01))

k = 0.06
tau = 516

a = C00
b = C11
c = C01
r = a*b - c**2 - k*(a + b)**2

M = r > tau
footprint = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=bool)
Nmax = maximum_filter(r, footprint=footprint, mode='constant', cval=-np.inf)
c = np.where(M & (r > Nmax))
c = np.array(c)