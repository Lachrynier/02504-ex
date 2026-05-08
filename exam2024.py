from utils import *; importlib.reload(sys.modules['utils']); from utils import *

import cv2
import numpy as np

# Q1
R = cv2.Rodrigues(np.array([0.2, 0.2, -0.1]))[0]
t = np.array([-0.08, 0.01, 0.03])

Q = np.array([-0.38, 0.1, 1.32])

K = np.array([
    [1400, 0, 750],
    [0, 1400, 520.0],
    [0, 0, 1]
])

q = Pi(K @ np.column_stack((R, t)) @ PiInv(Q))
print(f'Q1: {q}')

# Q2
p1 = np.array([[ 1.45349587e+02, -1.12915131e-01, 1.91640565e+00, -6.08129962e-01],
[ 1.05603820e+02, 5.62792554e-02, 1.79040110e+00, -2.32182177e-01]])
p2 = np.array([[ 1.3753556, -1.77072961, 2.94511795, 0.04032374],
[ 0.30936653, 0.37172814, 1.44007577, -0.03173825]])

# q1 = PiInv(p1)
# q2 = PiInv(p2)

H = hest(p1, p2)
H = H / H[0, 0]

print(f'Q2:\n{repr(H)}')
print(np.linalg.norm(p1 - Pi(H @ PiInv(p2)), axis=0))

# Q
