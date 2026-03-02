from utils import *; importlib.reload(sys.modules['utils']); from utils import *
from itertools import product


Q = np.array(list(product((0, 1), repeat=3)), dtype=float).T

f = 1000
dx = 1920 // 2
dy = 1080 // 2
K = np.array([
    [f, 0, dx],
    [0, f, dy],
    [0, 0, 1]
])

s = np.sqrt(0.5); R = np.array([
    [s, -s, 0],
    [s,  s, 0],
    [0,  0, 1]
])
t = np.array([0., 0., 10.])[:,None]

# E4.1
P = K @ np.column_stack((R, t))
q = P @ PiInv(Q)

# E4.2
P_est = pest(q, PiInv(Q))
q_est = P_est @ PiInv(Q)
e_proj = np.mean(np.linalg.norm(Pi(q_est) - Pi(q), axis=0))
print(f'reprojection error (no normalization): {e_proj}')

P_est = pest(q, PiInv(Q), normalize=True)
q_est = P_est @ PiInv(Q)
e_proj = np.mean(np.linalg.norm(Pi(q_est) - Pi(q), axis=0))
print(f'reprojection error (normalization): {e_proj}')